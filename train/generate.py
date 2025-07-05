#!/usr/bin/env python3
"""
Generate molecules using trained GPT model
Usage: python generate.py --model_weight model.pt --output results.csv
"""

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import json
import re
import math
import os
import sys

from rdkit import Chem
from rdkit.Chem import QED, Crippen, RDConfig
from rdkit.Chem.rdMolDescriptors import CalcTPSA
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

from model import GPT, GPTConfig
from utils import check_novelty, sample, canonic_smiles
from moses.utils import get_mol


def main():
    parser = argparse.ArgumentParser()
    
    # Required arguments
    parser.add_argument('--model_weight', type=str, required=True,
                        help='path to model weights')
    parser.add_argument('--output', type=str, required=True,
                        help='output CSV file')
    
    # Model config
    parser.add_argument('--n_layer', type=int, default=8)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--n_embd', type=int, default=256)
    parser.add_argument('--vocab_size', type=int, default=31)
    parser.add_argument('--block_size', type=int, default=64)
    
    # Generation settings
    parser.add_argument('--gen_size', type=int, default=10000,
                        help='number of molecules to generate')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=30)
    parser.add_argument('--top_p', type=float, default=0.95)
    
    # Property conditioning
    parser.add_argument('--num_props', type=int, default=2,
                        help='number of properties')
    parser.add_argument('--conditions', type=str, default='[[2.0,2.0]]',
                        help='property conditions as JSON')
    
    # Optional: dataset for novelty check
    parser.add_argument('--data_csv', type=str, default=None,
                        help='dataset CSV for novelty calculation')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
    
    # Load vocabulary
    if not os.path.exists('smiles.json'):
        print("Error: smiles.json not found")
        return
        
    stoi = json.load(open('smiles.json', 'r'))
    itos = {i: ch for ch, i in stoi.items()}
    print(f"Vocab size: {len(stoi)}")
    
    # Setup regex
    pattern = r"(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    
    # Parse conditions
    try:
        conditions = json.loads(args.conditions)
    except:
        print(f"Error parsing conditions: {args.conditions}")
        return
    
    # Load model
    print(f"Loading model from {args.model_weight}")
    
    mconf = GPTConfig(
        vocab_size=args.vocab_size,
        block_size=args.block_size,
        num_props=args.num_props,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        scaffold=False,
        scaffold_maxlen=0,
        lstm=False,
        lstm_layers=0
    )
    
    model = GPT(mconf)
    model.load_state_dict(torch.load(args.model_weight, map_location='cpu'))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Model on {device}")
    
    # Load reference data if provided
    train_smiles = None
    if args.data_csv and os.path.exists(args.data_csv):
        print(f"Loading reference data from {args.data_csv}")
        ref_data = pd.read_csv(args.data_csv)
        if 'smiles' in ref_data.columns:
            train_smiles = set(ref_data['smiles'].dropna())
        elif 'SMILES' in ref_data.columns:
            train_smiles = set(ref_data['SMILES'].dropna())
    
    # Generate molecules
    context = "C"
    gen_iter = math.ceil(args.gen_size / args.batch_size)
    
    all_results = []
    
    for cond in conditions:
        print(f"\nGenerating with condition: {cond}")
        molecules = []
        
        for _ in tqdm(range(gen_iter)):
            # Prepare input
            x = torch.tensor([stoi[s] for s in regex.findall(context)], 
                           dtype=torch.long)[None,...].repeat(args.batch_size, 1).to(device)
            
            # Prepare condition
            if args.num_props == 1:
                p = torch.tensor([cond]).repeat(args.batch_size, 1).to(device)
            else:
                p = torch.tensor([cond]).repeat(args.batch_size, 1).to(device)
            
            # Sample
            y = sample(model, x, args.block_size,
                      temperature=args.temperature,
                      sample=True,
                      top_k=args.top_k,
                      top_p=args.top_p,
                      prop=p,
                      scaffold=None)
            
            # Process outputs
            for gen_mol in y:
                completion = ''.join([itos[int(i)] for i in gen_mol if int(i) < len(itos)])
                completion = completion.replace('<', '')
                mol = get_mol(completion)
                if mol:
                    molecules.append((mol, completion))
        
        # Calculate properties
        results = []
        for mol, smi in molecules:
            try:
                results.append({
                    'smiles': Chem.MolToSmiles(mol),
                    'qed': round(QED.qed(mol), 3),
                    'sas': round(sascorer.calculateScore(mol), 3),
                    'logp': round(Crippen.MolLogP(mol), 3),
                    'tpsa': round(CalcTPSA(mol), 2),
                    'condition': str(cond)
                })
            except:
                continue
        
        df = pd.DataFrame(results)
        
        # Calculate metrics
        if len(df) > 0:
            canon = [canonic_smiles(s) for s in df['smiles']]
            unique = list(set(canon))
            
            validity = len(df) / (args.batch_size * gen_iter)
            uniqueness = len(unique) / len(df)
            
            if train_smiles:
                novelty = check_novelty(unique, train_smiles) / 100
            else:
                novelty = -1
            
            print(f"Valid: {validity:.3f}, Unique: {uniqueness:.3f}, Novel: {novelty:.3f}")
            
            df['validity'] = validity
            df['uniqueness'] = uniqueness
            df['novelty'] = novelty
            
            all_results.append(df)
    
    # Save results
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        try:
            final_df.to_csv(args.output, index=False)
            print(f"\nSaved {len(final_df)} molecules to {args.output}")
        except Exception as e:
            print(f"Error saving to {args.output}: {e}")
            # Try saving to current directory as backup
            backup_file = f"generated_molecules_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
            final_df.to_csv(backup_file, index=False)
            print(f"Saved to backup file: {backup_file}")
    else:
        print("No molecules generated!")


if __name__ == '__main__':
    main()