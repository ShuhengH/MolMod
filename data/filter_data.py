from rdkit import Chem
from rdkit.Chem import FilterCatalog, rdMolDescriptors
import multiprocessing
from tqdm import tqdm
input_smi = "cleaned_smiles.smi"
output_smi = "filtered_unique_output.smi"
allowed_atoms = {'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I'}
def init_catalog():
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.ALL)  # All rules
    return FilterCatalog.FilterCatalog(params)
def process_smiles_chunk(smiles_chunk):
    catalog = init_catalog()
    result = set()
    for line in smiles_chunk:
        line = line.strip()
        if not line:
            continue
        try:
            mol = Chem.MolFromSmiles(line)
            if not mol:
                continue
            # Check atom validity
            if not all(atom.GetSymbol() in allowed_atoms for atom in mol.GetAtoms()):
                continue
            # Filter by catalog rules
            if catalog.HasMatch(mol):
                continue
            # Filter by heavy atom count
            if rdMolDescriptors.CalcNumHeavyAtoms(mol) > 15:
                continue
            cano_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
            result.add(cano_smiles)
        except Exception as e:
            print(f"Error processing line: {line}\n{e}")
    return result
def main():
    num_cpus = multiprocessing.cpu_count() - 1 or 1
    print(f"Using {num_cpus} CPUs...")
    with open(input_smi, 'r') as infile:
        smiles_list = infile.readlines()
    chunk_size = len(smiles_list) // num_cpus + 1
    smiles_chunks = [smiles_list[i:i + chunk_size] for i in range(0, len(smiles_list), chunk_size)]
    with multiprocessing.Pool(num_cpus) as pool:
        jobs = [pool.apply_async(process_smiles_chunk, (chunk,)) for chunk in smiles_chunks]
        all_results = set()
        for job in tqdm(jobs):
            all_results.update(job.get())
    with open(output_smi, 'w') as outfile:
        for smi in all_results:
            outfile.write(smi + '\n')
    print(f"Filtering complete! Retained {len(all_results)} molecules, saved to {output_smi}")
if __name__ == '__main__':
    main()