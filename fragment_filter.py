"""
Fragment filter for molecule generation
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
import os
import time

class MoleculeFragmentFilter:
    """Filter molecules from pre-generated CSV"""
    
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = None
        self.load_fragments()
    
    def load_fragments(self):
        """Load the CSV file"""
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")
        
        self.df = pd.read_csv(self.csv_path)
        print(f"Loaded {len(self.df)} fragments")
        
        # debug info
        print("Columns:", self.df.columns.tolist())
        
        # check if we have all needed columns
        needed = ['smiles', 'has_ring', 'benzene_rings', 'heterocycles', 
                  'N_count', 'O_count', 'Lipo_pred', 'BBB_pred', 
                  'AqSol_pred', 'LD50_pred']
        missing = [col for col in needed if col not in self.df.columns]
        if missing:
            print(f"Warning: Missing columns: {missing}")
    
    def filter_molecules(self, 
                        optimization_type: str,
                        selected_parameter: str,
                        rings_option: str = None,
                        benzene_count: int = None,
                        hetero_count: int = None,
                        n_count: int = None,
                        o_count: int = None,
                        num_samples: int = 100,
                        use_random: bool = True) -> List[str]:
        """
        Filter molecules based on criteria
        
        Returns list of SMILES matching the conditions
        """
        df = self.df.copy()
        
        # filter valid molecules first
        if 'Valid_Molecule' in df.columns:
            df = df[df['Valid_Molecule'] == 1]
        
        if optimization_type == "only_admet":
            # just filter by ADMET
            df = self._filter_by_admet(df, selected_parameter)
            
        elif optimization_type == "str_and_admet":
            # structure filters first
            if rings_option is not None:
                has_ring = 1 if rings_option == "yes" else 0
                df = df[df['has_ring'] == has_ring]
            
            if benzene_count is not None:
                df = df[df['benzene_rings'] == benzene_count]
            
            if hetero_count is not None:
                df = df[df['heterocycles'] == hetero_count]
            
            if n_count is not None:
                df = df[df['N_count'] == n_count]
            
            if o_count is not None:
                df = df[df['O_count'] == o_count]
            
            # then ADMET
            df = self._filter_by_admet(df, selected_parameter)
        
        # if we don't have enough, relax the filters
        if len(df) < num_samples:
            print(f"Only found {len(df)} molecules, relaxing filters...")
            df = self._relax_filters(optimization_type, selected_parameter, 
                                    rings_option, benzene_count, hetero_count, 
                                    n_count, o_count, num_samples)
        
        # sample the molecules
        if len(df) > num_samples:
            if use_random:
                # truly random
                df = df.sample(n=num_samples)
            else:
                # semi-random for reproducibility
                seed = int(time.time()) % 10000
                df = df.sample(n=num_samples, random_state=seed)
        
        return df['smiles'].tolist()
    
    def filter_molecules_with_data(self, 
                        optimization_type: str,
                        selected_parameter: str,
                        rings_option: str = None,
                        benzene_count: int = None,
                        hetero_count: int = None,
                        n_count: int = None,
                        o_count: int = None,
                        num_samples: int = 100,
                        use_random: bool = True) -> pd.DataFrame:
        """Same as filter_molecules but returns full DataFrame"""
        df = self.df.copy()
        
        # filter valid molecules
        if 'Valid_Molecule' in df.columns:
            df = df[df['Valid_Molecule'] == 1]
        
        if optimization_type == "only_admet":
            df = self._filter_by_admet(df, selected_parameter)
            
        elif optimization_type == "str_and_admet":
            # structure filters
            if rings_option is not None:
                has_ring = 1 if rings_option == "yes" else 0
                df = df[df['has_ring'] == has_ring]
            
            if benzene_count is not None:
                df = df[df['benzene_rings'] == benzene_count]
            
            if hetero_count is not None:
                df = df[df['heterocycles'] == hetero_count]
            
            if n_count is not None:
                df = df[df['N_count'] == n_count]
            
            if o_count is not None:
                df = df[df['O_count'] == o_count]
            
            # then ADMET
            df = self._filter_by_admet(df, selected_parameter)
        
        # relax if needed
        if len(df) < num_samples:
            print(f"Only {len(df)} molecules, relaxing...")
            df = self._relax_filters(optimization_type, selected_parameter, 
                                    rings_option, benzene_count, hetero_count, 
                                    n_count, o_count, num_samples)
        
        # sample
        if len(df) > num_samples:
            if use_random:
                df = df.sample(n=num_samples)
            else:
                seed = int(time.time()) % 10000
                df = df.sample(n=num_samples, random_state=seed)
        
        return df
    
    def _filter_by_admet(self, df: pd.DataFrame, selected_parameter: str) -> pd.DataFrame:
        """Apply ADMET filters"""
        if selected_parameter == "logP":
            # good logP range for oral drugs
            return df[(df['Lipo_pred'] >= -1.0) & (df['Lipo_pred'] <= 2.0)]
            
        elif selected_parameter == "logS":
            # good solubility
            return df[df['AqSol_pred'] > -2.0]
            
        elif selected_parameter == "BBB":
            # can pass blood-brain barrier
            return df[df['BBB_pred'] == 1]
            
        elif selected_parameter == "LD50":
            # low toxicity
            return df[df['LD50_pred'] > 2.5]
            
        elif selected_parameter == "logP_BBB":
            # both logP and BBB
            return df[(df['Lipo_pred'] >= 1.0) & (df['Lipo_pred'] <= 3.0) & (df['BBB_pred'] == 1)]
            
        elif selected_parameter == "logP_logS":
            # both logP and solubility
            return df[(df['Lipo_pred'] >= 1.0) & (df['Lipo_pred'] <= 3.0) & (df['AqSol_pred'] > -2.0)]
            
        elif selected_parameter == "logP_LD50":
            # both logP and toxicity
            return df[(df['Lipo_pred'] >= -1.0) & (df['Lipo_pred'] <= 3.0) & (df['LD50_pred'] > 2.5)]
        
        return df
    
    def _relax_filters(self, optimization_type: str, selected_parameter: str,
                      rings_option: str, benzene_count: int, hetero_count: int,
                      n_count: int, o_count: int, num_samples: int) -> pd.DataFrame:
        """Relax filters when we don't have enough molecules"""
        df = self.df.copy()
        
        # valid molecules only
        if 'Valid_Molecule' in df.columns:
            df = df[df['Valid_Molecule'] == 1]
        
        # relax structure filters
        if optimization_type == "str_and_admet":
            if rings_option is not None:
                has_ring = 1 if rings_option == "yes" else 0
                df = df[df['has_ring'] == has_ring]
            
            # allow +/- 2 for counts
            if benzene_count is not None:
                df = df[
                    (df['benzene_rings'] >= max(0, benzene_count - 2)) & 
                    (df['benzene_rings'] <= benzene_count + 2)
                ]
            
            if hetero_count is not None:
                df = df[
                    (df['heterocycles'] >= max(0, hetero_count - 2)) & 
                    (df['heterocycles'] <= hetero_count + 2)
                ]
            
            if n_count is not None:
                df = df[
                    (df['N_count'] >= max(0, n_count - 2)) & 
                    (df['N_count'] <= n_count + 2)
                ]
            
            if o_count is not None:
                df = df[
                    (df['O_count'] >= max(0, o_count - 2)) & 
                    (df['O_count'] <= o_count + 2)
                ]
        
        # relax ADMET ranges
        if selected_parameter == "logP":
            df = df[(df['Lipo_pred'] >= 0.0) & (df['Lipo_pred'] <= 5.0)]
        elif selected_parameter == "logS":
            df = df[df['AqSol_pred'] > -5.0]
        elif selected_parameter == "LD50":
            df = df[df['LD50_pred'] > 2.0]
        elif selected_parameter == "logP_BBB":
            df = df[(df['Lipo_pred'] >= 0.0) & (df['Lipo_pred'] <= 5.0)]
        elif selected_parameter == "logP_logS":
            df = df[(df['Lipo_pred'] >= 0.0) & (df['Lipo_pred'] <= 5.0)]
        elif selected_parameter == "logP_LD50":
            df = df[(df['Lipo_pred'] >= 0.0) & (df['Lipo_pred'] <= 5.0) & (df['LD50_pred'] > 2.0)]
        
        return df

# helper function for easy use
def generate_molecules_from_csv(csv_path: str, optimization_type: str, selected_parameter: str,
                               rings_option: str = None, benzene_count: int = None,
                               hetero_count: int = None, n_count: int = None,
                               o_count: int = None, num_samples: int = 100,
                               use_random: bool = True) -> List[str]:
    """
    Get molecules from CSV based on filters
    
    use_random: True = fully random, False = time-based seed
    """
    # create filter and get molecules
    filter = MoleculeFragmentFilter(csv_path)
    
    selected = filter.filter_molecules(
        optimization_type=optimization_type,
        selected_parameter=selected_parameter,
        rings_option=rings_option,
        benzene_count=benzene_count,
        hetero_count=hetero_count,
        n_count=n_count,
        o_count=o_count,
        num_samples=num_samples,
        use_random=use_random
    )
    
    return selected