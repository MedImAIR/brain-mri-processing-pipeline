#!/bin/bash
task=34
task_folder='34_3d'
name='bgpd_1reg'

# python3 ../preprocess.py --data /data/private_data/ --task $task --ohe --exec_mode training --results /data/private_data/

# mkdir /results/bgpd_results/$name
# mkdir /results/bgpd_results/$name/fold-0
# echo Training $name fold-0!

export CUDA_VISIBLE_DEVICES=1 && python3 ../main.py --no_back_in_output --data /data/private_data/${task}_3d --resume_training --ckpt_path /results/bgpd_results/$name/fold-0/checkpoints/last.ckpt  --results /results/bgpd_results/$name/fold-0 --deep_supervision --depth 6 --filters 64 96 128 192 256 384 512 --min_fmap 2 --scheduler --learning_rate 0.0003 --epochs 70 --nfolds 3 --fold 0 --amp --gpus 1 --task $task --patience 20 --save_ckpt 


echo End Training!

echo Save predicts $name fold-0!
mkdir /results/bgpd_infer/$name

export CUDA_VISIBLE_DEVICES=1 && python ../main.py --no_back_in_output  --exec_mode predict --task ${task}_3d --data /data/private_data/${task}_3d --dim 3 --fold 0 --nfolds 3 --ckpt_path /results/bgpd_results/$name/fold-0/checkpoints/best*.ckpt --results /results/bgpd_infer/$name --amp --tta --save_preds


mkdir /results/bgpd_results/$name/fold-1
echo Training $name fold-1!

export CUDA_VISIBLE_DEVICES=1 && python3 ../main.py --no_back_in_output --data /data/private_data/${task}_3d --resume_training --ckpt_path /results/bgpd_results/$name/fold-1/checkpoints/last.ckpt --results /results/bgpd_results/$name/fold-1 --deep_supervision --depth 6 --filters 64 96 128 192 256 384 512 --min_fmap 2 --scheduler --learning_rate 0.0003 --epochs 70 --nfolds 3 --fold 1 --amp --gpus 1 --task $task --patience 20 --save_ckpt

 echo Save predicts $name fold-1!

export CUDA_VISIBLE_DEVICES=1 && python ../main.py --no_back_in_output  --exec_mode predict --task ${task}_3d --data /data/private_data/${task}_3d --dim 3 --fold 1 --nfolds 3 --ckpt_path /results/bgpd_results/$name/fold-1/checkpoints/best*.ckpt --results /results/bgpd_infer/$name --amp --tta --save_preds


mkdir /results/bgpd_results/$name/fold-2
echo Training $name fold-2!

export CUDA_VISIBLE_DEVICES=1 && python3 ../main.py --no_back_in_output  --data /data/private_data/${task}_3d --results /results/bgpd_results/$name/fold-2 --deep_supervision --depth 6 --filters 64 96 128 192 256 384 512 --min_fmap 2 --scheduler --learning_rate 0.0003 --epochs 70 --nfolds 3 --fold 2 --amp --gpus 1 --task $task --patience 20 --save_ckpt

echo Save predicts $name fold-2!

export CUDA_VISIBLE_DEVICES=1 && python ../main.py --no_back_in_output  --exec_mode predict --task ${task}_3d --data /data/private_data/${task}_3d --dim 3 --fold 2 --nfolds 3 --ckpt_path /results/bgpd_results/$name/fold-2/checkpoints/best*.ckpt --results /results/bgpd_infer/$name --amp --tta --save_preds

