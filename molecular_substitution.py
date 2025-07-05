"""
Molecular substitution utilities for fragment-based drug design
"""

from rdkit import Chem
from rdkit.Chem import AllChem


def re_li(smiles):
    """Replace * with [Li] in SMILES"""
    return smiles.replace('*', '[Li]')


def re_group(scaffold_smiles, fragment_library):
    """
    Replace [Li] positions in scaffold with fragments
    
    Args:
        scaffold_smiles: molecule with [Li] markers
        fragment_library: list of fragments (or single fragment)
    
    Returns:
        list of product SMILES
    """
    # get the scaffold molecule
    mol = Chem.MolFromSmiles(scaffold_smiles)
    if not mol:
        return []
    
    # handle single fragment input
    if isinstance(fragment_library, str):
        fragment_library = [fragment_library]
    
    products = []
    
    # try each fragment
    for frag in fragment_library:
        frag = frag.strip()
        
        # remove the connecting C
        if frag.startswith('C'):
            group = frag[1:]
        else:
            group = frag
        
        # set up the reaction
        rxn_smarts = f'[*:1][Li]>>[*:1]{group}'
        
        try:
            rxn = AllChem.ReactionFromSmarts(rxn_smarts)
            results = rxn.RunReactants((mol,))
            
            # collect valid products
            for res in results:
                prod = res[0]
                try:
                    Chem.SanitizeMol(prod)
                    products.append(Chem.MolToSmiles(prod))
                except:
                    pass  # skip bad molecules
                
        except:
            pass  # skip problematic fragments
    
    return products


