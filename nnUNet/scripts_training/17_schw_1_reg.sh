#!/bin/bash
task=33
task_folder='33_3d'
name='schw_1_reg'

# python3 ../preprocess.py --data /data/private_data/schw/ --task 33 --ohe --exec_mode training --results /data/private_data/schw/

# mkdir /results/schw_results
mkdir /results/schw_results/$name
# mkdir /results/schw_results/$name/fold-0
# echo Training $name fold-0!
# --resume_training --ckpt_path /results/schw_results/schw_1_reg/fold-0/checkpoints/last.ckpt 

# export CUDA_VISIBLE_DEVICES=1 && python3 ../main.py --no_back_in_output --data /data/private_data/schw/${task}_3d --results /results/schw_results/$name/fold-0 --resume_training --ckpt_path /results/schw_results/schw_1_reg/fold-0/checkpoints/last.ckpt --deep_supervision --depth 6 --filters 64 96 128 192 256 384 512 --min_fmap 2 --scheduler --learning_rate 0.0003 --epochs 100 --nfolds 3 --fold 0 --amp --gpus 1 --task $task --save_ckpt


# mkdir /results/schw_results/$name/fold-1
# echo Training $name fold-1!

# export CUDA_VISIBLE_DEVICES=1 && python3 ../main.py --no_back_in_output --data /data/private_data/schw/${task}_3d --results /results/schw_results/$name/fold-1 --resume_training --ckpt_path /results/schw_results/schw_1_reg/fold-1/checkpoints/last.ckpt --deep_supervision --depth 6 --filters 64 96 128 192 256 384 512 --min_fmap 2 --scheduler --learning_rate 0.0003 --epochs 100 --nfolds 3 --fold 1 --amp --gpus 1 --task $task --save_ckpt

mkdir /results/schw_results/$name/fold-2
echo Training $name fold-2!

export CUDA_VISIBLE_DEVICES=1 && python3 ../main.py --no_back_in_output --data /data/private_data/schw/${task}_3d --results /results/schw_results/$name/fold-2 --deep_supervision --depth 6 --filters 64 96 128 192 256 384 512 --min_fmap 2 --scheduler --learning_rate 0.0003 --epochs 100 --nfolds 3 --fold 2 --amp --gpus 1 --task $task --save_ckpt

echo End Training!

# echo Save predicts $name fold-0!
# mkdir /results/schw_infer/$name

# export CUDA_VISIBLE_DEVICES=1 && python ../main.py --exec_mode predict --task ${task}_3d --data /data/private_data/schw/${task}_3d --no_back_in_output --dim 3 --fold 0 --nfolds 3 --ckpt_path /results/schw_results/$name/fold-0/checkpoints/best*.ckpt --results /results/schw_infer/$name --amp --tta --save_preds

# python metrics_1class.py --path_to_pred /results/schw_infer/$name/*fold=0_tta --path_to_target  /data/private_data/schw/$name/labels --out /results/schw_infer/$name/metrics_${name}_fold-0.csv


# echo Save predicts $name fold-1!

# export CUDA_VISIBLE_DEVICES=1 && python ../main.py --exec_mode predict --task ${task}_3d --data /data/private_data/schw/${task}_3d --no_back_in_output --dim 3 --fold 1 --nfolds 3 --ckpt_path /results/schw_results/$name/fold-1/checkpoints/best*.ckpt --results /results/schw_infer/$name --amp --tta --save_preds

# python metrics_1class.py --path_to_pred /results/schw_infer/$name/*fold=1_tta --path_to_target  /data/private_data/schw/$name/labels --out /results/schw_infer/$name/metrics_${name}_fold-1.csv

echo Save predicts $name fold-2!

export CUDA_VISIBLE_DEVICES=1 && python ../main.py --exec_mode predict --task ${task}_3d --data /data/private_data/schw/${task}_3d --no_back_in_output --dim 3 --fold 2 --nfolds 3 --ckpt_path /results/schw_results/$name/fold-2/checkpoints/best*.ckpt --results /results/schw_infer/$name --amp --tta --save_preds

python metrics_1class.py --path_to_pred /results/schw_infer/$name/*fold=2_tta --path_to_target  /data/private_data/schw/$name/labels --out /results/schw_infer/$name/metrics_${name}_fold-2.csv
