# MolMod: Fragment-Based Molecular Modification Platform

A transformer-based deep learning platform for site-specific molecular optimization through fragment replacement.

**Web Demo**: http://molmod.tdd-lab.com/

## Overview

![MolMod Platform](docs/images/platform_overview.jpg)

*Figure: (a) Client-server architecture with RESTful API. (b) Three-step workflow: molecular input with modification site marking, property selection, and result visualization.*

## Features

- **Site-specific modification**: Mark specific positions for fragment replacement
- **Property optimization**: Target multiple ADMET properties (LogP, LogS, BBB, LD50)
- **High scaffold retention**: ≥99.99% preservation of core structures
- **Web-based interface**: No installation required for basic usage
- **Batch generation**: Generate thousands of optimized candidates

## Requirements

```
# Core dependencies
torch>=1.11.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.0.0
h5py>=3.11.0
rdkit>=2024.3.0
molsets>=0.3.1  # moses
tqdm>=4.66.0

# For ADMET prediction models
dgl>=0.9.1
dgllife>=0.3.2
PyTDC>=0.4.1

# For web interface (optional)
fastapi>=0.111.0
uvicorn>=0.33.0
jinja2>=3.1.0
python-multipart>=0.0.20

# For experiment tracking (optional)
wandb>=0.15.0
```

## Installation

```bash
# Create environment
conda create -n molmod python=3.8
conda activate molmod

# Install PyTorch (adjust CUDA version as needed)
pip install torch==1.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html

# Install dependencies
pip install -r requirements.txt

# Install RDKit (if not included in requirements)
conda install -c conda-forge rdkit
```

## Data Sources

The MolMod platform uses curated molecular datasets from established chemical databases:

### Pre-training Data
- **ZINC20 Database**: Large-scale molecular library containing over 1.3 billion drug-like compounds
  - **Website**: https://zinc20.docking.org/

### Fine-tuning Data  
- **ChEMBL Database**: Manually curated database of bioactive molecules with drug-like properties
  - **Website**: https://www.ebi.ac.uk/chembl/

⚠️ **Note**: Both databases are publicly available. Please cite the original sources when using this platform in your research.

## Quick Start

All training scripts are located in the `train/` directory. Example shell scripts are provided in the repository root for convenience.

### 1. Pre-training

**Unconditional generation** (no property constraints):
```bash
python train/pretrain.py \
  --run_name unconditional_model \
  --data_name cleaned_smiles \
  --num_props 0 \
  --n_layer 8 \
  --n_head 8 \
  --n_embd 256 \
  --batch_size 512 \
  --learning_rate 6e-4 \
  --max_epochs 10
```

**Multi-property conditioning**:
```bash
python train/pretrain.py \
  --run_name multi_prop_model \
  --data_name cleaned_smiles \
  --props Lipo LD50 \
  --num_props 2 \
  --n_layer 8 \
  --n_head 8 \
  --n_embd 256 \
  --batch_size 512 \
  --learning_rate 6e-4 \
  --max_epochs 10
```

⚠️ **Note**: Make sure `--num_props` matches the number of properties in `--props`.

**Using the provided script**:
```bash
bash pretrain.sh
```

### 2. Fine-tuning

Fine-tune using pre-trained weights with the new parameter options:

```bash
python train/finetune.py \
  --run_name props_finetune \
  --pretrained_weights weights/logp.pt \
  --save_path weights/logp_finetune.pt \
  --data_name data/fin_data \
  --props Lipo_pred \
  --num_props 1 \
  --n_layer 8 \
  --n_head 8 \
  --n_embd 256 \
  --batch_size 256 \
  --max_epochs 4
```

**Multi-property fine-tuning example**:
```bash
python train/finetune.py \
  --run_name multi_props_finetune \
  --pretrained_weights weights/multi_prop_model.pt \
  --save_path weights/multi_prop_finetune.pt \
  --data_name data/fin_data \
  --props Lipo_pred LD50_pred AqSol_pred \
  --num_props 3 \
  --n_layer 8 \
  --n_head 8 \
  --n_embd 256 \
  --batch_size 256 \
  --max_epochs 4
```

**Using the provided script**:
```bash
bash finetune.sh
```

⚠️ **Important**: 
- Make sure `--pretrained_weights` points to an existing pre-trained model
- The `--save_path` parameter specifies where to save the fine-tuned model
- Ensure your data file path includes the `data/` prefix if needed

### 3. Generation

**Single property generation**:
```bash
python train/generate.py \
  --model_weight weights/logp.pt \
  --output results/logp.csv \
  --gen_size 10000 \
  --n_layer 8 \
  --n_head 8 \
  --n_embd 256 \
  --batch_size 512 \
  --num_props 1 \
  --conditions "[[2.5]]" \
  --temperature 1.0 \
  --top_k 30 \
  --top_p 0.95
```

**Multi-property generation**:
```bash
python train/generate.py \
  --model_weight weights/logp_ld50.pt \
  --output results/logp_ld50_results.csv \
  --n_layer 8 \
  --n_head 8 \
  --n_embd 256 \
  --num_props 2 \
  --conditions "[[2.0, 2.5]]" \
  --gen_size 10000 \
  --batch_size 512 \
  --temperature 1.0 \
  --top_k 30 \
  --top_p 0.95
```

**Using the provided script**:
```bash
bash generate.sh
```

**For HPC/SLURM environments**:
The `generate.sh` script includes SLURM job configuration for cluster environments:
```bash
sbatch generate.sh  # Submit as SLURM job
```


⚠️ **Note**: 
- Ensure your data file name (without extension) matches the `--data_name` parameter in training scripts
- For data files in subdirectories, include the full path: `--data_name data/fin_data`

## Web Application

### Online Access
Visit http://molmod.tdd-lab.com/ for the web interface.

### Usage Workflow
The web interface follows a simple three-step process (see figure above):
1. **Step 1**: Input molecule with marked modification sites (*) using the molecular editor
2. **Step 2**: Select optimization parameters and number of molecules to generate  
3. **Step 3**: Choose target properties (LogP, LogS, LD50, BBB) and structural constraints
4. **Submit**: Generate molecules and view results with property profiles

### Local Deployment
```bash
uvicorn api:app --reload
# Access at http://localhost:8000
```
### Running the Scripts
```bash
# Make scripts executable
chmod +x *.sh

# Run individual scripts
bash pretrain.sh     # Pre-training
bash finetune.sh     # Fine-tuning
bash generate.sh     # Generation (or sbatch generate.sh for SLURM)
```

### Customizing Scripts
Edit the shell scripts to:
- Change model parameters (`--n_layer`, `--n_head`, `--n_embd`)
- Modify training conditions (`--props`, `--conditions`)
- Adjust data paths and output locations
- Update SLURM job configurations for your cluster

⚠️ **Important Notes**:
- Ensure `--num_props` always matches the actual number of properties
- Verify file paths exist before running scripts
- For multi-property models, conditions array length must match `--num_props`
