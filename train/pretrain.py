import pandas as pd
import argparse
from utils import set_seed
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
import re
import pandas as pd
import argparse
from utils import set_seed
import numpy as np
import wandb
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.cuda.amp import GradScaler

from model import GPT, GPTConfig
from trainer import Trainer, TrainerConfig
from dataset import SmileDataset
import math
from sklearn.model_selection import train_test_split
from utils import SmilesEnumerator
import re
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--run_name', type=str,
                        help="name for wandb run", required=False)
    parser.add_argument('--debug', action='store_true',
                        default=False, help='debug')
    # in moses dataset, on average, there are only 5 molecules per scaffold
    parser.add_argument('--scaffold', action='store_true',
                        default=False, help='condition on scaffold')
    parser.add_argument('--lstm', action='store_true',
                        default=False, help='use lstm for transforming scaffold')
    parser.add_argument('--data_name', type=str, default='cleaned_smiles',
                        help="name of the dataset to train on", required=False)
    # parser.add_argument('--property', type=str, default = 'qed', help="which property to use for condition", required=False)
    parser.add_argument('--props', nargs="+", default=['Lipo'],
                        help="properties to be used for condition", required=False)
    parser.add_argument('--num_props', type=int, default = 0, help="number of properties to use for condition", required=False)
    # parser.add_argument('--prop1_unique', type=int, default = 0, help="unique values in that property", required=False)
    parser.add_argument('--n_layer', type=int, default=8,
                        help="number of layers", required=False)
    parser.add_argument('--n_head', type=int, default=8,
                        help="number of heads", required=False)
    parser.add_argument('--n_embd', type=int, default=256,
                        help="embedding dimension", required=False)
    parser.add_argument('--max_epochs', type=int, default=10,
                        help="total epochs", required=False)
    parser.add_argument('--batch_size', type=int, default=512,
                        help="batch size", required=False)
    parser.add_argument('--learning_rate', type=float,
                        default=6e-4, help="learning rate", required=False)
    parser.add_argument('--lstm_layers', type=int, default=0,
                        help="number of layers in lstm", required=False)
    parser.add_argument('--ckpt_path', type=str, default='./weights/logp.pt',
                        help="path to save checkpoint", required=False)

    args = parser.parse_args()

    set_seed(42)
    
    # Change data reading part to read from H5 file
    file_path = './' + args.data_name + '.h5'
    
    try:
        with h5py.File(file_path, 'r') as f:
            # Print H5 file keys for debugging
            print(f"Keys in H5 file: {list(f.keys())}")
            
            # First read SMILES data
            if 'smiles' in f:
                smiles_data = [s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in f['smiles'][:]]
            else:
                print("Error: 'smiles' dataset not found in H5 file")
                exit(1)
                
            # Create a dictionary to store all property data
            props_dict = {'smiles': smiles_data}
            
            # Read user-specified property columns
            for prop in args.props:
                if prop in f:
                    props_dict[prop] = f[prop][:]
                else:
                    print(f"Warning: Property '{prop}' not found in H5 file")
            
            # Create DataFrame
            data = pd.DataFrame(props_dict)
    except Exception as e:
        print(f"Error reading H5 file: {e}")
        # If H5 file reading fails, try reading CSV or SMI file as fallback
        try:
            data = pd.read_csv('./' + args.data_name + '.smi', sep='\s+', header=None)
            # Name the columns
            data.columns = ['smiles'] + [f'prop_{i}' for i in range(1, len(data.columns))]
            print("Successfully read SMI file as fallback.")
        except:
            try:
                data = pd.read_csv('data/' + args.data_name + '.csv')
                print("Successfully read CSV file as fallback.")
            except Exception as e2:
                print(f"Failed to read any data file: {e2}")
                exit(1)

    # Data preprocessing
    data = data.dropna(axis=0).reset_index(drop=True)
    print("Data preview:")
    print(data.head())
    
    # Ensure column names are lowercase
    data.columns = data.columns.str.lower()
    
    # Split train and validation sets
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    
    # Get SMILES
    smiles = train_data['smiles']
    vsmiles = val_data['smiles']
    
    # Get property data
    if args.props:
        # Check all specified properties are in the data
        missing_props = [p for p in args.props if p.lower() not in train_data.columns]
        if missing_props:
            print(f"Warning: The following properties are not in the dataset: {missing_props}")
            # Only use existing properties
            args.props = [p for p in args.props if p.lower() in train_data.columns]
        
        # Convert to lowercase to match dataframe column names
        props_lower = [p.lower() for p in args.props]
        prop = train_data[props_lower].values.tolist()
        vprop = val_data[props_lower].values.tolist()
        num_props = len(props_lower) if args.num_props == 0 else args.num_props
        print(f"Using properties: {props_lower}")
    else:
        prop = []
        vprop = []
        num_props = 0
        print("No properties specified for condition.")

    # SMILES processing part
    pattern = "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)

    lens = [len(regex.findall(i.strip()))
              for i in (list(smiles.values) + list(vsmiles.values))]
    max_len = max(lens)
    print('Max len: ', max_len)

    smiles = [i + str('<')*(max_len - len(regex.findall(i.strip())))
                for i in smiles]
    vsmiles = [i + str('<')*(max_len - len(regex.findall(i.strip())))
                for i in vsmiles]

    whole_string =["#","(",")","-","1","2","3","4","5","6","7","<","=","B","Br","C","Cl","F","I","N","O","P","S","[PH]","[SH]","[nH]","c","n","o","p","s"]
    print(f"Vocabulary size: {len(whole_string)}")
    
    # Print selected property information
    if num_props > 0:
        print(f"Selected {num_props} properties for conditioning:")
        for i, p in enumerate(args.props[:num_props]):
            print(f"  {i+1}. {p}")
    train_dataset = SmileDataset(args, smiles, whole_string, max_len, prop=prop, aug_prob=0, scaffold=None, scaffold_maxlen= 0, use_scaffold=False)
    valid_dataset = SmileDataset(args, vsmiles, whole_string, max_len, prop=vprop, aug_prob=0, scaffold=None, scaffold_maxlen= 0, use_scaffold=False)
    mconf = GPTConfig(train_dataset.vocab_size, train_dataset.max_len, num_props=num_props,  # args.num_props,
                        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, scaffold=args.scaffold, scaffold_maxlen=0,
                        lstm=args.lstm, lstm_layers=args.lstm_layers)

    model = GPT(mconf)

    tconf = TrainerConfig(max_epochs=args.max_epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
                            lr_decay=True, warmup_tokens=0.1*len(train_data)*max_len, final_tokens=args.max_epochs*len(train_data)*max_len,
                            num_workers=10, ckpt_path=f'./weights/{args.run_name}.pt', block_size=train_dataset.max_len, generate=False)

    wandb.init(project="scaffold_constrained", name=args.run_name)
    trainer = Trainer(model, train_dataset, valid_dataset,
                        tconf, train_dataset.stoi, train_dataset.itos)
    df = trainer.train(wandb)