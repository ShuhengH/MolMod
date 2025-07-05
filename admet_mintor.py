import torch
import pandas as pd
import numpy as np
import pickle
import os
from DeepPurpose import CompoundPred
# import data_merge
import data_merge
import model_download_patch
model_download_patch.patch_model_loading()
def predict_admet_from_smiles(smiles_list):
    """
    Predict ADMET properties for a list of SMILES strings.
    
    Args:
        smiles_list (list): List of SMILES strings
        
    Returns:
        pandas.DataFrame: DataFrame containing original SMILES and all ADMET predictions
    """
    # Create temporary SMILES file
    temp_smi_path = "temp_input.smi"
    temp_csv_path = "temp_smiles.csv"
    
    try:
        # Save SMILES list to temporary file
        with open(temp_smi_path, 'w') as f:
            for smi in smiles_list:
                f.write(f"{smi}\n")
        
        # Convert SMILES to CSV format
        data_merge.general_admet_col(temp_smi_path, temp_csv_path)
        
        # Read the data
        data = pd.read_csv(temp_csv_path)
        data_smiles = data["SMILES"]
        data_smiles = pd.DataFrame(data_smiles)
        
        # Initialize results DataFrame
        data_add_score = data.copy()
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Dictionary to store all model configurations
        model_configs = {
            # Absorption
            'Caco2': {
                'encoding': 'rdkit_2d_normalized',
                'model_path': "best-model/BayesianRidge-caco2_Rmodel/model.pickle",
                'is_classification': False,
                'use_ml': True
            },
            'HIA': {
                'encoding': 'GIN_AttrMasking',
                'model_path': "best-model/GIN_AttrMasking-HIA_Hou_Cmodel",
                'is_classification': True,
                'use_ml': False
            },
            'Pgp': {
                'encoding': 'GIN_AttrMasking',
                'model_path': "best-model/GIN_AttrMasking-Pgp_Cmodel",
                'is_classification': True,
                'use_ml': False
            },
            'Bioav': {
                'encoding': 'RDKit2D',
                'model_path': "best-model/RDKit2D-Bioav_Cmodel",
                'is_classification': True,
                'use_ml': False
            },
            'Lipo': {
                'encoding': 'GIN_ContextPred',
                'model_path': "best-model/GIN_ContextPred-Lipo_Rmodel",
                'is_classification': False,
                'use_ml': False
            },
            'AqSol': {
                'encoding': 'AttentiveFP',
                'model_path': "best-model/AttentiveFP-AqSol_Rmodel",
                'is_classification': False,
                'use_ml': False
            },
            
            # Distribution
            'BBB': {
                'encoding': 'GIN_ContextPred',
                'model_path': "best-model/GIN_ContextPred-BBB_Cmodel",
                'is_classification': True,
                'use_ml': False
            },
            'PPBR': {
                'encoding': 'rdkit_2d_normalized',
                'model_path': "best-model/SVR_PPBR_Rmodel/model.pickle",
                'is_classification': False,
                'use_ml': True
            },
            'VD': {
                'encoding': 'rdkit_2d_normalized',
                'model_path': "best-model/SVR-VD_Rmodel/model.pickle",
                'is_classification': False,
                'use_ml': True
            },
            
            # Metabolism
            'CYP2D6-I': {
                'encoding': 'GIN_ContextPred',
                'model_path': "best-model/GIN_ContextPred-CYP2D6-I_Cmodel-0.714",
                'is_classification': True,
                'use_ml': False
            },
            'CYP3A4-I': {
                'encoding': 'GIN_ContextPred',
                'model_path': "best-model/GIN_ContextPred-CYP3A4-I_Cmodel-1000",
                'is_classification': True,
                'use_ml': False
            },
            'CYP2C9-I': {
                'encoding': 'GIN_ContextPred',
                'model_path': "best-model/GIN_ContextPred-CYP2C9-I_Cmodel-0.802",
                'is_classification': True,
                'use_ml': False
            },
            'CYP2D6-S': {
                'encoding': 'GIN_ContextPred',
                'model_path': "best-model/GIN_ContextPred-CYP2D6-S_Cmodel-1500",
                'is_classification': True,
                'use_ml': False
            },
            'CYP3A4-S': {
                'encoding': 'CNN',
                'model_path': "best-model/CNN-CYP3A4-S_Cmodel",
                'is_classification': True,
                'use_ml': False
            },
            'CYP2C9-S': {
                'encoding': 'GIN_ContextPred',
                'model_path': "best-model/GIN_ContextPred-CYP2C9-S_Cmodel-1500",
                'is_classification': True,
                'use_ml': False
            },
            'CYP2C19-I': {
                'encoding': 'GIN_ContextPred',
                'model_path': "best-model/GIN_ContextPred-CYP2C19-I_Cmodel-500",
                'is_classification': True,
                'use_ml': False
            },
            'CYP1A2-I': {
                'encoding': 'GIN_ContextPred',
                'model_path': "best-model/GIN_ContextPred-cyp1a2_Cmodel-500",
                'is_classification': True,
                'use_ml': False
            },
            
            # Excretion
            'Half-Life': {
                'encoding': 'rdkit_2d_normalized',
                'model_path': "best-model/SVR-Half-life_Rmodel/model.pickle",
                'is_classification': False,
                'use_ml': True
            },
            'CL-Micro': {
                'encoding': 'GIN_ContextPred',
                'model_path': "best-model/GIN_ContextPred_CL-Micro_Rmodel",
                'is_classification': False,
                'use_ml': False
            },
            'CL-Hepa': {
                'encoding': 'GIN_ContextPred',
                'model_path': "best-model/GIN_ContextPred_CL-Hepa_Rmodel",
                'is_classification': False,
                'use_ml': False
            },
            
            # Toxicity
            'hERG': {
                'encoding': 'RDKit2D',
                'model_path': "best-model/RDKit2D-hERG_Cmodel",
                'is_classification': True,
                'use_ml': False
            },
            'AMES': {
                'encoding': 'GIN_AttrMasking',
                'model_path': "best-model/GIN_AttrMasking-ames_Cmodel",
                'is_classification': True,
                'use_ml': False
            },
            'DILI': {
                'encoding': 'GIN_AttrMasking',
                'model_path': "best-model/GIN_AttrMasking-dili_Cmodel-1000-0.894",
                'is_classification': True,
                'use_ml': False
            },
            'LD50': {
                'encoding': 'GIN_AttrMasking',
                'model_path': "best-model/GIN_AttrMasking-ld50_Rmodel",
                'is_classification': False,
                'use_ml': False
            },
            'Carcinogens': {
                'encoding': 'CNN',
                'model_path': "best-model/CNN-carcinogens_Cmodel",
                'is_classification': True,
                'use_ml': False
            },
            'ClinTox': {
                'encoding': 'GIN_AttrMasking',
                'model_path': "best-model/AttentiveFP-clintox_Cmodel-500",
                'is_classification': True,
                'use_ml': False
            },
            'Skin-Reaction': {
                'encoding': 'GIN_AttrMasking',
                'model_path': "best-model/GIN_ContextPred-skin_reaction_Cmodel-500",
                'is_classification': True,
                'use_ml': False
            }
        }
        
        # Helper function for ML model prediction
        def predict_ml_model(model_path, test_data):
            with open(model_path, "rb") as file:
                model = pickle.load(file)
            return model.predict(test_data)
        
        # Helper function for deep learning model prediction
        def predict_dl_model(model_path, data, is_classification):
            model = CompoundPred.model_pretrained(model_path)
            pred = model.predict(data)
            if is_classification:
                return np.asarray([1 if i else 0 for i in (np.asarray(pred) >= 0.5)])
            return pred
        
        # Iterate through all models and make predictions
        for property_name, config in model_configs.items():
            try:
                if config['use_ml']:
                    # For ML models (e.g., SVR, BayesianRidge)
                    test_data = data_merge.smiles_ml_encode(data, data_smiles, config['encoding'])
                    predictions = predict_ml_model(config['model_path'], test_data)
                else:
                    # For deep learning models
                    data_encoding = data_merge.smiles_encode(data_smiles, config['encoding'])
                    data['drug_encoding'] = data_encoding
                    predictions = predict_dl_model(config['model_path'], data, config['is_classification'])
                
                data_add_score[f"{property_name}_pred"] = predictions
                
            except Exception as e:
                print(f"Error predicting {property_name}: {str(e)}")
                
        return data_add_score
        
    finally:
        # Clean up temporary files
        for temp_file in [temp_smi_path, temp_csv_path]:
            if os.path.exists(temp_file):
                os.remove(temp_file)

# Example usage:
# if __name__ == "__main__":
#     # Example SMILES list
#     example_smiles = [
#         "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(N)(=O)=O)C(F)(F)F",
#         "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5"
#     ]
    
#     # Run predictions
#     results = predict_admet_from_smiles(example_smiles)
#     print(results)
    
#     # Save results
#     results.to_csv("admet_predictions.csv", index=False)