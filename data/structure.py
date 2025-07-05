import pandas as pd
from collections import Counter
from rdkit import Chem
import csv

# Use the functions already defined
def has_ring(mol):
    """Check if molecule contains rings"""
    return 1 if mol.GetRingInfo().NumRings() > 0 else 0

def count_benzene_rings(mol):
    """Count number of benzene rings"""
    benzene_pattern = Chem.MolFromSmarts('c1ccccc1')
    return len(mol.GetSubstructMatches(benzene_pattern))

def count_heterocycles(mol):
    """Count number of heterocycles - using method compatible with latest RDKit version"""
    # Total number of rings
    total_rings = mol.GetRingInfo().NumRings()
    
  
    # Symmetrized Smallest Set of Smallest Rings
    rings = mol.GetSSSR() if hasattr(mol, 'GetSSSR') else Chem.GetSymmSSSR(mol)
    
    # Estimate carbon-only rings count
    carbon_only_rings = 0
    for ring in rings:
        is_carbon_only = True
        for atom_idx in ring:
            atom = mol.GetAtomWithIdx(atom_idx)
            if atom.GetSymbol() != 'C':
                is_carbon_only = False
                break
        if is_carbon_only:
            carbon_only_rings += 1
    
    return total_rings - carbon_only_rings

def count_atom_types(mol):
    """Count specific atom types"""
    atom_counter = Counter([atom.GetSymbol() for atom in mol.GetAtoms()])
    return {
        'N_count': atom_counter.get('N', 0),
        'O_count': atom_counter.get('O', 0)
    }

def extract_molecular_features(smiles):
    """Extract molecular features from SMILES string"""
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        return None
    
    # Basic molecular features
    features = {
        'smiles': smiles,
        'has_ring': has_ring(mol),
        'benzene_rings': count_benzene_rings(mol),
        'heterocycles': count_heterocycles(mol),
    }
    
    # Add atom counts
    atom_counts = count_atom_types(mol)
    features.update(atom_counts)
    
    return features

# Function to process CSV files
def process_smiles_csv(input_file, output_file, smiles_column='SMILES'):
    """
    Process CSV file containing SMILES strings, calculate molecular features and save results
    
    Parameters:
    input_file (str): Input CSV file path
    output_file (str): Output CSV file path
    smiles_column (str): Column name containing SMILES, default is 'SMILES'
    """
    # Read CSV file
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    if smiles_column not in df.columns:
        print(f"Error: Column '{smiles_column}' not found in CSV file")
        return
    
    # Create results list
    results = []
    
    # Process each SMILES string
    for index, row in df.iterrows():
        smiles = row[smiles_column]
        
        # Extract features
        features = extract_molecular_features(smiles)
        
        if features is None:
            print(f"Warning: Cannot parse SMILES: {smiles}")
            continue
        
        # If needed to preserve other columns from original CSV, add them to features dictionary
        for column in df.columns:
            if column != smiles_column and column not in features:
                features[column] = row[column]
        
        results.append(features)
    
    # Convert results to DataFrame and save
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")