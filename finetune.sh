

#Fine-tune with the matching pre-trained model for the corresponding properties

python train/finetune.py --run_name props_finetune --pretrained_weights weights/logp.pt --save_path weights/logp_finetune.pt --data_name data/fin_data --props Lipo_pred --num_props 1 --n_layer 8 --n_head 8 --n_embd 256 --batch_size 256 --max_epochs 4