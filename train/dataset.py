import torch
from torch.utils.data import Dataset
from utils import SmilesEnumerator
import numpy as np
import re
import math

class SmileDataset(Dataset):
    def __init__(self, args, data, content, block_size, aug_prob=0.5, prop=None, scaffold=None, scaffold_maxlen=None, use_scaffold=True):
        chars = sorted(list(set(content)))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d smiles, %d unique characters.' % (data_size, vocab_size))
    
        self.stoi = {ch:i for i,ch in enumerate(chars)}
        self.itos = {i:ch for i,ch in enumerate(chars)}
        self.max_len = block_size
        self.vocab_size = vocab_size
        self.data = data
        self.prop = prop
        self.sca = scaffold
        self.scaf_max_len = scaffold_maxlen
        self.debug = args.debug
        self.tfm = SmilesEnumerator()
        self.aug_prob = aug_prob
        self.use_scaffold = use_scaffold  # 新增参数，控制是否使用scaffold
    
    def __len__(self):
        if self.debug:
            return math.ceil(len(self.data) / (self.max_len + 1))
        else:
            return len(self.data)
    def __getitem__(self, idx):
        smiles, prop = self.data[idx], self.prop[idx] if self.prop is not None else None
        smiles = smiles.strip()
    
        # 处理scaffold，仅当use_scaffold为True时
        if self.use_scaffold and self.sca is not None:
            scaffold = self.sca[idx].strip()
        else:
            scaffold = ""  # 当不使用scaffold时，设置为空字符串
    
        # SMILES增强
        p = np.random.uniform()
        if p < self.aug_prob:
            smiles = self.tfm.randomize_smiles(smiles)
    
        # 正则表达式分词
        pattern = "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)
    
        # 处理SMILES序列
        smiles_tokens = regex.findall(smiles)
        if len(smiles_tokens) > self.max_len:
            smiles_tokens = smiles_tokens[:self.max_len]
        else:
            smiles_tokens += ["<"] * (self.max_len - len(smiles_tokens))
    
        # 处理scaffold序列，仅当use_scaffold为True时
        # 如果不使用scaffold，直接返回空的 tensor
        if self.use_scaffold and self.sca is not None:
            scaffold_tokens = regex.findall(scaffold)
            if len(scaffold_tokens) > self.scaf_max_len:
                scaffold_tokens = scaffold_tokens[:self.scaf_max_len]
            else:
                scaffold_tokens += ["<"] * (self.scaf_max_len - len(scaffold_tokens))
            sca_dix = [self.stoi[s] for s in scaffold_tokens]
            sca_tensor = torch.tensor(sca_dix, dtype=torch.long)
        else:
            # 直接返回一个空的tensor
            # 强制转换为整数
            sca_tensor = torch.zeros(self.scaf_max_len, dtype=torch.long) if self.scaf_max_len else torch.tensor([], dtype=torch.long)


    
        # 转换为索引
        dix = [self.stoi[s] for s in smiles_tokens]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
    
        # 处理属性值
        if prop is not None:
            prop_tensor = torch.tensor(prop, dtype=torch.float)
        else:
            prop_tensor = torch.tensor([], dtype=torch.float)
    
        return x, y, prop_tensor, sca_tensor

   