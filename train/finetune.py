from utils import check_novelty, sample, canonic_smiles
from dataset import SmileDataset
from rdkit.Chem import QED
from rdkit.Chem import Crippen
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit import Chem
import math
from tqdm import tqdm
import wandb
import argparse
from model import GPT, GPTConfig
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from rdkit.Chem.rdMolDescriptors import CalcTPSA
from trainer import Trainer, TrainerConfig
from dataset import SmileDataset
from sklearn.model_selection import train_test_split

import seaborn as sns
import moses
from moses.utils import get_mol
from utils import SmilesEnumerator
import re
import json
from rdkit.Chem import RDConfig
import json

import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))

import sascorer


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
    parser.add_argument('--data_name', type=str, default='data_momo5',
                        help="name of the dataset to train on", required=False)
    # parser.add_argument('--property', type=str, default = 'qed', help="which property to use for condition", required=False)
    parser.add_argument('--props', nargs="+", default=['has_ring', 'benzene_rings', 'heterocycles','N_count', 'O_count','Lipo_pred','AqSol_pred'],
                        #'has_ring','benzene_rings','heterocycles','N_count','O_count','LD50_pred','AqSol_pred'
                        help="properties to be used for condition", required=False)
    parser.add_argument('--num_props', type=int, default = 7, help="number of properties to use for condition", required=False)
    # parser.add_argument('--prop1_unique', type=int, default = 0, help="unique values in that property", required=False)
    parser.add_argument('--n_layer', type=int, default=8,
                        help="number of layers", required=False)
    parser.add_argument('--n_head', type=int, default=8,
                        help="number of heads", required=False)
    parser.add_argument('--n_embd', type=int, default=256,
                        help="embedding dimension", required=False)
    parser.add_argument('--max_epochs', type=int, default=10,
                        help="total epochs", required=False)
    parser.add_argument('--batch_size', type=int, default = 512, help="batch size", required=False)
    parser.add_argument('--gen_size', type=int, default = 10000, help="number of times to generate from a batch", required=False)
    parser.add_argument('--vocab_size', type=int, default = 31, help="vocabulary size", required=False)  # Modified comment
    parser.add_argument('--block_size', type=int, default = 64, help="block size", required=False)  # Modified comment
    parser.add_argument('--lstm_layers', type=int, default=0,
                        help="number of layers in lstm", required=False)
    
 
    parser.add_argument('--pretrained_weights', type=str, default='weights/logp.pt',
                        help="path to pretrained model weights", required=False)
    

    parser.add_argument('--save_path', type=str, default='weights/logp_finetune.pt',
                        help="path to save fine-tuned model weights", required=False)

    args = parser.parse_args()

    context = "C"

    data = pd.read_csv('data/' + args.data_name + '.csv')
    data = data.dropna(axis=0).reset_index(drop=True)
    print(data)

    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    train_data = train_data.reset_index(drop=True)
    val_data = val_data.reset_index(drop=True)
    prop = train_data[args.props].values.tolist()
    vprop = val_data[args.props].values.tolist()
    smiles = train_data['smiles']
    vsmiles = val_data['smiles']

    pattern = "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)

    stoi = json.load(open(f'smiles.json', 'r'))

    #itos = { i:ch for i,ch in enumerate(chars) }
    itos = { i:ch for ch,i in stoi.items() }
    print(stoi.keys())
    print(itos)
    lens = [len(regex.findall(i.strip()))
            for i in (list(smiles.values) + list(vsmiles.values))]
    max_len = max(lens)
    print('Max len: ', max_len)
    whole_string = ["#","(",")","-","1","2","3","4","5","6","7","<","=","B","Br","C","Cl","F","I","N","O","P","S","[PH]","[SH]","[nH]","c","n","o","p","s"]
    
    # Update vocab_size to actual size
    args.vocab_size = len(stoi)  # Use actual vocabulary size
    
    train_dataset = SmileDataset(args,smiles, whole_string, max_len, prop=prop, aug_prob=0, scaffold=None, scaffold_maxlen=0, use_scaffold=False)
    valid_dataset = SmileDataset(args, vsmiles, whole_string, max_len, prop=vprop, aug_prob=0, scaffold=None, scaffold_maxlen=0, use_scaffold=False)

    num_props = len(args.props)  # Use actual number of properties

    mconf = GPTConfig(args.vocab_size, args.block_size, num_props=num_props,  # Use variables instead of hard-coding
                      n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, scaffold=args.scaffold, scaffold_maxlen=0,
                      lstm=args.lstm, lstm_layers=args.lstm_layers)
    
    model = GPT(mconf)

    # 使用命令行参数指定的权重路径
    print(f'Loading pretrained weights from: {args.pretrained_weights}')
    model.load_state_dict(torch.load(args.pretrained_weights))
    model.to('cuda')
    print('Model loaded')

    model.train()

    # Freeze embedding layer
    for param in model.tok_emb.parameters():
        param.requires_grad = False

    # Freeze first 4 transformer blocks
    for param in model.blocks[:4].parameters():
        param.requires_grad = False

    tconf = TrainerConfig(
        max_epochs=4,  # Fewer epochs for fine-tuning
        batch_size=256,
        learning_rate=2e-4,  # Use smaller learning rate for fine-tuning
        lr_decay=True,
        warmup_tokens=0.1 * len(train_data) * max_len,
        final_tokens=4 * len(train_data) * max_len,  # Use actual max_epochs
        num_workers=10,
        ckpt_path=args.save_path,  # 使用命令行参数指定的保存路径
        block_size=train_dataset.max_len,  # Use actual max_len
        generate=False
    )

    # 根据run_name或者自动生成wandb运行名称
    wandb.init(project="scaffold_constrained", name=wandb_name)
    
    trainer = Trainer(model, train_dataset, valid_dataset, tconf, train_dataset.stoi, train_dataset.itos)
    df = trainer.train(wandb)