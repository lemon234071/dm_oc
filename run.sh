python3 weighting_lm.py --pretrain_epochs 30 --epochs 0 --batch_size 64 --warmup_steps 600 \
        --w_init 1. --w_decay 5. --norm_fn linear --gradient_accumulation_steps 2 \
        --save_dir daily_bl --learning_rate 5e-4 --lr_schedule None  #--model_checkpoint pytorch_models/gpt2/
#python3 weighting_lm.py --epochs 10 --batch_size 16 --val_batch_size 16 --max_val_step 1 --warmup_steps 600 \
#      --w_init 1. --w_decay 5. --norm_fn linear --gradient_accumulation_steps 8 \
#      --save_dir daily_weighting 
#python3 weighting_lm.py --pretrain_epochs 1 --epochs 9 --batch_size 16 --val_batch_size 16 --warmup_steps 600 \
#      --w_init 1. --w_decay 5. --norm_fn linear --gradient_accumulation_steps 8 \
#      --save_dir daily_1-9
# python3 weighting_lm.py --infer_from "$1" --pretrained --top_k 1 --outpath infer.txt
