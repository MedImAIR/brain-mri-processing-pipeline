mkdir /results/gbm_results/1_reg_aug
mkdir /results/gbm_results/1_reg_aug/fold-0
echo Training fold-0!

export CUDA_VISIBLE_DEVICES=1 && python3 ../main.py --brats --data /data/private_data/21_3d --results /results/gbm_results/1_reg_aug/fold-0 --deep_supervision --depth 6 --filters 64 96 128 192 256 384 512 --min_fmap 2 --scheduler --learning_rate 0.0003 --epochs 100 --nfolds 3 --fold 0 --amp --gpus 1 --task 21 --save_ckpt


mkdir /results/gbm_results/1_reg_aug/fold-1
echo Training fold-1!

export CUDA_VISIBLE_DEVICES=1 && python3 ../main.py --brats --data /data/private_data/21_3d --results /results/gbm_results/1_reg_aug/fold-1 --deep_supervision --depth 6 --filters 64 96 128 192 256 384 512 --min_fmap 2 --scheduler --learning_rate 0.0003 --epochs 100 --nfolds 3 --fold 1 --amp --gpus 1 --task 21 --save_ckpt

mkdir /results/gbm_results/1_reg_aug/fold-2
echo Training fold-2!

export CUDA_VISIBLE_DEVICES=1 && python3 ../main.py --brats --data /data/private_data/21_3d --results /results/gbm_results/1_reg_aug/fold-2 --deep_supervision --depth 6 --filters 64 96 128 192 256 384 512 --min_fmap 2 --scheduler --learning_rate 0.0003 --epochs 100 --nfolds 3 --fold 2 --amp --gpus 1 --task 21 --save_ckpt

echo End Training!

echo Save predicts fold-0!
mkdir /results/gbm_infer/1_reg_aug

export CUDA_VISIBLE_DEVICES=1 && python ../main.py --exec_mode predict --task 21_3d --data /data/private_data/21_3d --brats --dim 3 --fold 0 --nfolds 3 --ckpt_path /results/gbm_results/1_reg_aug/fold-0/checkpoints/best*.ckpt --results /results/gbm_infer/1_reg_aug --amp --tta --save_preds

python metrics.py --path_to_pred /results/gbm_infer/1_reg_aug/*fold=0_tta --path_to_target  /data/private_data/gbm_1_reg_train/labels --out /results/gbm_infer/1_reg_aug/metrics_gbm_1_reg_aug_fold-0.csv


echo Save predicts fold-1!

export CUDA_VISIBLE_DEVICES=1 && python ../main.py --exec_mode predict --task 21_3d --data /data/private_data/21_3d --brats --dim 3 --fold 1 --nfolds 3 --ckpt_path /results/gbm_results/1_reg_aug/fold-1/checkpoints/best*.ckpt --results /results/gbm_infer/1_reg_aug --amp --tta --save_preds

python metrics.py --path_to_pred /results/gbm_infer/1_reg_aug/*fold=1_tta --path_to_target  /data/private_data/gbm_1_reg_train/labels --out /results/gbm_infer/1_reg_aug/metrics_gbm_1_reg_aug_fold-1.csv

# echo Save predicts fold-2!

export CUDA_VISIBLE_DEVICES=1 && python ../main.py --exec_mode predict --task 21_3d --data /data/private_data/21_3d --brats --dim 3 --fold 2 --nfolds 3 --ckpt_path /results/gbm_results/1_reg_aug/fold-2/checkpoints/best*.ckpt --results /results/gbm_infer/1_reg_aug --amp --tta --save_preds

python metrics.py --path_to_pred /results/gbm_infer/1_reg_aug/*fold=2_tta --path_to_target  /data/private_data/gbm_1_reg_train/labels --out /results/gbm_infer/1_reg_aug/metrics_gbm_1_reg_aug_fold-2.csv
