
# uncondtion
python train/pretrain.py --run_name unconditional_model --data_name cleaned_smiles --num_props 0 --n_layer 8 --n_head 8 --n_embd 256 --batch_size 512 --learning_rate 6e-4 --max_epochs 10
# multi condition 
python train/pretrain.py --run_name multi_prop_model --data_name cleaned_smiles --props Lipo LD50 --num_props 2 --n_layer 8 --n_head 8 --n_embd 256 --batch_size 512 --learning_rate 6e-4 --max_epochs 10