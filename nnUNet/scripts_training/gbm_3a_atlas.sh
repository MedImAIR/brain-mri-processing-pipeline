# python3 ../preprocess.py --data /data/private_data/ --task 20 --ohe --exec_mode training --results /data/private_data/
mkdir /results/gbm_results/gbm_3a_atlas
mkdir /results/gbm_results/gbm_3a_atlas/fold-0
echo Training fold-0!

export CUDA_VISIBLE_DEVICES=0 && python3 ../main.py --brats --data /data/private_data/20_3d --results /results/gbm_results/gbm_3a_atlas/fold-0 --deep_supervision --depth 6 --filters 64 96 128 192 256 384 512 --min_fmap 2 --scheduler --learning_rate 0.0003 --epochs 100 --nfolds 3 --fold 0 --amp --gpus 1 --task 20 --save_ckpt


mkdir /results/gbm_results/gbm_3a_atlas/fold-1
echo Training fold-1!

export CUDA_VISIBLE_DEVICES=0 && python3 ../main.py --brats --data /data/private_data/20_3d --results /results/gbm_results/gbm_3a_atlas/fold-1 --deep_supervision --depth 6 --filters 64 96 128 192 256 384 512 --min_fmap 2 --scheduler --learning_rate 0.0003 --epochs 100 --nfolds 3 --fold 1 --amp --gpus 1 --task 20 --save_ckpt

mkdir /results/gbm_results/gbm_3a_atlas/fold-2
echo Training fold-2!

export CUDA_VISIBLE_DEVICES=0 && python3 ../main.py --brats --data /data/private_data/20_3d --results /results/gbm_results/gbm_3a_atlas/fold-2 --deep_supervision --depth 6 --filters 64 96 128 192 256 384 512 --min_fmap 2 --scheduler --learning_rate 0.0003 --epochs 100 --nfolds 3 --fold 2 --amp --gpus 1 --task 20 --save_ckpt

echo End Training!

echo Save predicts fold-0!

export CUDA_VISIBLE_DEVICES=0 && python ../main.py --exec_mode predict --task 20_3d --data /data/private_data/20_3d --brats --dim 3 --fold 0 --nfolds 3 --ckpt_path /results/gbm_results/gbm_3a_atlas/fold-0/checkpoints/best*.ckpt --results /results/gbm_infer/gbm_3a_atlas --amp --tta --save_preds

python metrics.py --path_to_pred /results/gbm_infer/gbm_3a_atlas/*fold=0_tta --path_to_target  /data/private_data/gbm_3a_atlas_train/labels --out /results/gbm_infer/gbm_3a_atlas/metrics_gbm_3a_atlas_fold-0.csv


echo Save predicts fold-1!

export CUDA_VISIBLE_DEVICES=0 && python ../main.py --exec_mode predict --task 20_3d --data /data/private_data/20_3d --brats --dim 3 --fold 1 --nfolds 3 --ckpt_path /results/gbm_results/gbm_3a_atlas/fold-1/checkpoints/best*.ckpt --results /results/gbm_infer/gbm_3a_atlas --amp --tta --save_preds

python metrics.py --path_to_pred /results/gbm_infer/gbm_3a_atlas/*fold=1_tta --path_to_target  /data/private_data/gbm_3a_atlas_train/labels --out /results/gbm_infer/gbm_3a_atlas/metrics_gbm_3a_atlas_fold-1.csv

echo Save predicts fold-2!

export CUDA_VISIBLE_DEVICES=0 && python ../main.py --exec_mode predict --task 20_3d --data /data/private_data/20_3d --brats --dim 3 --fold 2 --nfolds 3 --ckpt_path /results/gbm_results/gbm_3a_atlas/fold-2/checkpoints/best*.ckpt --results /results/gbm_infer/gbm_3a_atlas --amp --tta --save_preds

python metrics.py --path_to_pred /results/gbm_infer/gbm_3a_atlas/*fold=2_tta --path_to_target  /data/private_data/gbm_3a_atlas_train/labels --out /results/gbm_infer/gbm_3a_atlas/metrics_gbm_3a_atlas_fold-2.csv