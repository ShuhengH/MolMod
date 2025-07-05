import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import math
import re
import json
import argparse
import os
import sys
from rdkit import Chem
from rdkit.Chem import QED, Crippen, Draw
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.rdMolDescriptors import CalcTPSA
from rdkit.Chem import RDConfig
from rdkit.Chem import rdFMCS
from rdkit.Chem import Draw
from moses.utils import get_mol
# Add SA scorer path
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
# Import model and related functions
from model import GPT, GPTConfig
from utils import check_novelty, sample, canonic_smiles
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from molecular_substitution import re_li, re_group
from prediction import ParallelADMETPredictor
from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd
from draw import plot_hexagon_radar
from fragment_filter import MoleculeFragmentFilter, generate_molecules_from_csv

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/molecule_images", StaticFiles(directory="molecule_images"), name="molecule_images")
app.mount("/ketcher", StaticFiles(directory="ketcher"), name="ketcher")

context = "C"
predictor = ParallelADMETPredictor(use_gpu=False, batch_size=128, num_workers=4)

# Regular expression pattern for splitting SMILES strings
pattern = "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
regex = re.compile(pattern)

stoi = json.load(open('smiles.json', 'r'))
itos = {i: ch for ch, i in stoi.items()}
print("Character mapping dictionary loaded")

# Ensure image directory exists
if not os.path.exists("molecule_images"):
    os.makedirs("molecule_images")

# Model Manager Class
class ModelManager:
    def __init__(self):
        self.models = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model configuration mapping
        self.model_configs = {
            "only_admet": GPTConfig(vocab_size=31, block_size=64, num_props=1, 
                                  n_layer=8, n_head=8, n_embd=256, 
                                  scaffold=None, scaffold_maxlen=0,
                                  lstm=False, lstm_layers=0),
            
            "str_and_admet": GPTConfig(vocab_size=31, block_size=64, num_props=6, 
                                      n_layer=8, n_head=8, n_embd=256, 
                                      scaffold=None, scaffold_maxlen=0,
                                      lstm=False, lstm_layers=0)
        }
        
        # Map parameter names to weight files
        self.weight_files = {
            # Single parameter optimization - ADMET only
            "only_admet_logP": "./weights/finetuned_model_logp_only.pt",
            "only_admet_logS": "./weights/finetuned_model_logs_only.pt",
            "only_admet_BBB": "./weights/finetuned_model_bbb_only.pt",
            "only_admet_LD50": "./weights/finetuned_model_ld50_only.pt",
            
            # Multi-parameter optimization - ADMET only
            "only_admet_logP_BBB": "./weights/finetuned_model_logp_bbb.pt",
            "only_admet_logS_liver": "./weights/finetuned_model_logs_liver.pt",
            
            # Single parameter optimization - structure and ADMET
            "str_and_admet_logP": "./weights/finetuned_model_logp_str.pt",
            "str_and_admet_logS": "./weights/finetuned_model_logs_str.pt",
            "str_and_admet_BBB": "./weights/finetuned_model_bbb_str.pt",
            "str_and_admet_LD50": "./weights/finetuned_model_ld50_str.pt",
            
            # Multi-parameter optimization - structure and ADMET
            "str_and_admet_logP_BBB": "./weights/finetuned_model_logp_bbb_str.pt",
        }
    
    def get_model_key(self, optimization_type, parameter):
        """Generate model key from optimization type and parameter"""
        return f"{optimization_type}_{parameter}"
    
    def load_model(self, optimization_type, parameter):
        """Load model, initialize if not exists"""
        model_key = self.get_model_key(optimization_type, parameter)
        
        if model_key not in self.models:
            # Check if weight file exists
            if model_key not in self.weight_files:
                raise ValueError(f"Model weight file not found: {model_key}")
            
            # Get correct configuration
            config_key = optimization_type
            if config_key not in self.model_configs:
                raise ValueError(f"Model configuration not found: {config_key}")
            
            # Create model and load weights
            model = GPT(self.model_configs[config_key])
            model.load_state_dict(torch.load(self.weight_files[model_key], map_location=self.device))
            model.to(self.device)
            self.models[model_key] = model
        
        return self.models[model_key]

model_manager = ModelManager()

# Helper functions
def calc_molecular_descriptors(smiles):
    """Calculate molecular descriptors"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return pd.Series([None] * 5, index=["mol_weight", "tpsa", "h_donors", "h_acceptors", "rot_bonds"])
    
    return pd.Series({
        "mol_weight": round(Descriptors.MolWt(mol), 2),
        "tpsa": round(Descriptors.TPSA(mol), 2),
        "h_donors": int(Descriptors.NumHDonors(mol)),
        "h_acceptors": int(Descriptors.NumHAcceptors(mol)),
        "rot_bonds": int(Descriptors.NumRotatableBonds(mol))
    })

def generate_molecules(custom_conditions, device, model):
    """Generate molecules based on given conditions"""
    all_unique_molecules = []
    unique_smiles_set = set()
    
    for condition in custom_conditions:
        prop_values = condition
        
        # Prepare initial context
        x = torch.tensor([stoi[s] for s in regex.findall(context)], dtype=torch.long)[None,...].repeat(10, 1).to(device)
        
        # Prepare conditions
        p = torch.tensor([prop_values]).repeat(10, 1).to(device)
        
        # Sample molecules using the model
        y = sample(model, x, 20, temperature=1, sample=True, top_k=5, prop=p, scaffold=None)
        
        # Process generated molecules
        for gen_mol in y:
            completion = ''.join([itos[int(i)] for i in gen_mol if i < 30])
            completion = completion.replace('<', '')
            
            # Validate molecule
            mol = get_mol(completion)
            if mol:
                # Calculate canonical SMILES for deduplication
                canon_smi = canonic_smiles(completion)
                if canon_smi and canon_smi not in unique_smiles_set:
                    # Only add unique molecules not seen before
                    unique_smiles_set.add(canon_smi)
                    all_unique_molecules.append(completion)
    
    return all_unique_molecules

def process_results(result):
    """Process result data, add molecular descriptors"""
    # Calculate molecular descriptors
    if 'drug_encoding' in result.columns:
        result = result.drop(columns=['drug_encoding'])
    descriptor_df = result["SMILES"].apply(calc_molecular_descriptors)
    
    # Merge back to original result table
    result_df = pd.concat([result, descriptor_df], axis=1)
    
    # Round decimal values
    numeric_cols = ['logs', 'ld50', 'logp']
    for col in numeric_cols:
        if col in result_df.columns:
            result_df[col] = result_df[col].round(2)
    
    return result_df

def generate_images(result_df):
    """Generate structure and radar images for molecules"""
    for idx, row in result_df.iterrows():
        smiles = row['SMILES']
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is not None:
            # Generate unique filename with timestamp
            timestamp = int(time.time() * 1000)  # Millisecond timestamp
            
            # Generate molecule structure image
            img_filename = f"molecule_{idx}_{timestamp}.png"
            img_path = os.path.join("molecule_images", img_filename)
            img = Draw.MolToImage(mol, size=(300, 300))
            img.save(img_path)
            
            # Generate radar chart
            radar_image_filename = f"molecule_{idx}_radar_{timestamp}.png"
            radar_image_path = os.path.join("molecule_images", radar_image_filename)
            
            # Extract properties
            mol_weight = row['mol_weight']
            logp = row['logp']
            tpsa = row['tpsa']
            h_donors = row['h_donors']
            h_acceptors = row['h_acceptors']
            rot_bonds = row['rot_bonds']
            
            # Define normalization max and min values
            properties = [mol_weight, logp, tpsa, h_donors, h_acceptors, rot_bonds]
            max_values = [500, 5, 150, 10, 10, 10]
            min_values = [0, -5, 0, 0, 0, 0]
            
            # Normalize properties
            properties_normalized = [(prop - min_val) / (max_val - min_val) 
                                    for prop, min_val, max_val in zip(properties, min_values, max_values)]
            
            # Define labels
            labels = ['Mw', 'LogP', 'TPSA', 'H Donors', 'H Acceptors', 'Rotatable Bonds']
            
            # Generate radar chart
            plot_hexagon_radar(properties_normalized, labels, save_path=radar_image_path)
            
            # Store image paths
            result_df.at[idx, 'molecule_image_path'] = f"/molecule_images/{img_filename}"
            result_df.at[idx, 'radar_image_path'] = f"/molecule_images/{radar_image_filename}"
    
    # Convert to JSON format
    json_data = result_df.to_dict(orient='records')
    return json_data

# Routes
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("generate_page.html", {
        "request": request,
        "molecules": [],  # Provide empty array instead of None
        "optimization_type": "",
        "selected_parameter": ""
    })

@app.get("/generate_page", response_class=HTMLResponse)
async def generate_page(request: Request):
    return templates.TemplateResponse("generate_page.html", {"request": request})

@app.get("/about", response_class=HTMLResponse)
async def about_page(request: Request):
    return templates.TemplateResponse("About.html", {"request": request})

@app.get("/help", response_class=HTMLResponse)
async def help_page(request: Request):
    return templates.TemplateResponse("Help.html", {"request": request})

@app.get("/contact", response_class=HTMLResponse)
async def contact_page(request: Request):
    return templates.TemplateResponse("Contact.html", {"request": request})

@app.post("/generate_json")
async def generate_json(
    smiles: str = Form(None),
    optimization_mode: str = Form(...),
    optimization_type: str = Form(...),
    selected_parameter: str = Form(...),
    rings_option: str = Form(None),
    benzene_count: int = Form(None),
    hetero_count: int = Form(None),
    n_count: int = Form(None),
    o_count: int = Form(None),
):
    try:
        # Clean image directory
        image_dir = "molecule_images"
        for filename in os.listdir(image_dir):
            file_path = os.path.join(image_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error cleaning file: {e}")
        
        # Process SMILES input
        mod_smiles = re_li(smiles)
        mod_mol = Chem.MolFromSmiles(mod_smiles)
        
        # Filter molecule fragments from CSV file
        csv_path = "db_v2.csv"  # Please modify to your actual file path
        
        print(f"Filtering molecules from CSV...")
        print(f"Optimization type: {optimization_type}")
        print(f"Selected parameter: {selected_parameter}")
        
        # Use generate_molecules_from_csv to get SMILES list
        all_unique_molecules = generate_molecules_from_csv(
            csv_path=csv_path,
            optimization_type=optimization_type,
            selected_parameter=selected_parameter,
            rings_option=rings_option,
            benzene_count=benzene_count,
            hetero_count=hetero_count,
            n_count=n_count,
            o_count=o_count,
            num_samples=100
        )
        
        print(f"Filtered {len(all_unique_molecules)} molecule fragments from CSV")
        
        # Process generated molecules
        result = re_group(mod_smiles, all_unique_molecules)
        
        print(f"Got {len(result)} molecules after recombination")
        
        # Predict ADMET properties for all recombined molecules
        predictor.filter_properties(['Lipo', 'AqSol', 'BBB', 'LD50'])
        results = predictor.predict_admet_from_smiles(result, chunk_size=200)
        results = results.rename(columns={
            "AqSol_pred": "logs",
            "LD50_pred": "ld50",
            "Lipo_pred": "logp",
            "BBB_pred": "bbb"
        })
        
        # Process results, add molecular descriptors
        result_df = process_results(results)
        
        # Generate images with better error handling
        for idx, row in result_df.iterrows():
            smiles = row['SMILES']
            mol = Chem.MolFromSmiles(smiles)
            
            if mol is not None:
                # Generate unique filename with timestamp
                timestamp = int(time.time() * 1000)
                
                # Find maximum common substructure
                try:
                    mcs_result = rdFMCS.FindMCS([mod_mol, mol], timeout=5)
                    if mcs_result and mcs_result.smartsString:
                        mcs_mol = Chem.MolFromSmarts(mcs_result.smartsString)
                        
                        # Get matching atom indices
                        match_atoms = mol.GetSubstructMatch(mcs_mol) if mcs_mol else []
                        matched_atoms_set = set(match_atoms)
                        
                        # Get unmatched atoms
                        unmatched_atoms = [i for i in range(mol.GetNumAtoms()) if i not in matched_atoms_set]
                        
                        # Get bonds connecting unmatched atoms
                        unmatched_bonds = []
                        for bond in mol.GetBonds():
                            begin = bond.GetBeginAtomIdx()
                            end = bond.GetEndAtomIdx()
                            if begin in unmatched_atoms or end in unmatched_atoms:
                                unmatched_bonds.append(bond.GetIdx())
                        
                        # Draw molecule with highlighting
                        img = Draw.MolToImage(
                            mol,
                            size=(400, 300),
                            highlightAtoms=unmatched_atoms,
                            highlightBonds=unmatched_bonds,
                            highlightColor=(1.0, 0.5, 0.4)
                        )
                    else:
                        # If MCS fails, draw normal image
                        img = Draw.MolToImage(mol, size=(400, 300))
                except Exception as e:
                    # If error occurs, draw normal image
                    print(f"MCS error for molecule {idx}: {e}")
                    img = Draw.MolToImage(mol, size=(400, 300))
                
                # Save image
                img_filename = f"molecule_{idx}_{timestamp}.png"
                img_path = os.path.join("molecule_images", img_filename)
                img.save(img_path)
                
                # Generate radar chart
                radar_image_filename = f"molecule_{idx}_radar_{timestamp}.png"
                radar_image_path = os.path.join("molecule_images", radar_image_filename)
                
                # Extract properties with error handling
                try:
                    mol_weight = row['mol_weight']
                    logp = row['logp']
                    tpsa = row['tpsa']
                    h_donors = row['h_donors']
                    h_acceptors = row['h_acceptors']
                    rot_bonds = row['rot_bonds']
                    
                    # Generate radar chart
                    properties = [mol_weight, logp, tpsa, h_donors, h_acceptors, rot_bonds]
                    max_values = [500, 5, 150, 10, 10, 10]
                    min_values = [0, -5, 0, 0, 0, 0]
                    
                    properties_normalized = [(prop - min_val) / (max_val - min_val) 
                                            for prop, min_val, max_val in zip(properties, min_values, max_values)]
                    
                    labels = ['Mw', 'LogP', 'TPSA', 'H Donors', 'H Acceptors', 'Rotatable Bonds']
                    plot_hexagon_radar(properties_normalized, labels, save_path=radar_image_path)
                    
                    # Store image paths
                    result_df.at[idx, 'molecule_image_path'] = f"/molecule_images/{img_filename}"
                    result_df.at[idx, 'radar_image_path'] = f"/molecule_images/{radar_image_filename}"
                except Exception as e:
                    print(f"Error generating radar chart for molecule {idx}: {e}")
                    # Set default values if error occurs
                    result_df.at[idx, 'molecule_image_path'] = f"/molecule_images/{img_filename}"
                    result_df.at[idx, 'radar_image_path'] = None
        
        # Optimize data structure
        result_df['Label'] = result_df.index
        
        # Ensure all floats have two decimal places
        float_cols = ['logp', 'logs', 'ld50', 'mol_weight', 'tpsa']
        for col in float_cols:
            if col in result_df.columns:
                result_df[col] = result_df[col].round(2)
        
        # Convert integer fields
        int_cols = ['h_donors', 'h_acceptors', 'rot_bonds', 'bbb', 'Label']
        for col in int_cols:
            if col in result_df.columns:
                result_df[col] = result_df[col].fillna(0).astype(int)
        
        # Convert to JSON format
        json_data = result_df.to_dict(orient='records')
        
        print(f"Returning {len(json_data)} molecule results")
        
        # Return JSON data
        return {
            "status": "success",
            "molecules": json_data,
            "optimization_type": optimization_type,
            "selected_parameter": selected_parameter
        }
    
    except Exception as e:
        # Exception handling
        error_message = f"Error occurred during generation: {str(e)}"
        print(error_message)
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": error_message}