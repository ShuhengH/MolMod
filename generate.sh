#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=13
#SBATCH --gpus-per-node=1 
#SBATCH -p gpu
#SBATCH -t 800:00:00
#SBATCH -o output12
module load anaconda3
conda activate py38
nvidia-smi
# single property

python train/generate.py --model_weight weights/logp.pt --output results/logp.csv --gen_size 10000 --n_layer 8 --n_head 8 --n_embd 256 --batch_size 512 --num_props 1 --conditions "[[2.5]]" --temperature 1.0 --top_k 30 --top_p 0.95

# # multi property

# python train/generate.py --model_weight weights/logp_ld50.pt --output results/logp_ld50_results.csv --n_layer 8 --n_head 8 --n_embd 256 --num_props 3 --conditions "[[2.0, 2.5]]" --gen_size 10000 --batch_size 512 --temperature 1.0 --top_k 30 --top_p 0.95
 



