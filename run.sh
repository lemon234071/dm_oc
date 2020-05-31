python3 weighting_lm.py --pretrain_epochs 10 --epochs 0 --batch_size 64 \
        --w_init 1. --w_decay 5. --norm_fn linear --gradient_accumulation_steps 4 \
        --save_dir daily_bl
# python3 weighting_lm.py --epochs 10 --batch_size 64 \
#       --w_init 1. --w_decay 5. --norm_fn linear --gradient_accumulation_steps 4 \
#       --save_dir daily_weighting
# python3 weighting_lm.py --pretrain_epochs 1 --epochs 9 --batch_size 64 \
#       --w_init 1. --w_decay 5. --norm_fn linear --gradient_accumulation_steps 4 \
#       --save_dir daily_1-9
