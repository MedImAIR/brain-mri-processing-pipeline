

# python3 glioma2mod.py

# python3 ../preprocess.py --data /data/private_data/ --task 18 --ohe --exec_mode training --results /data/private_data

# cd /results/glioma_results/ && mkdir 2mod
# cd /results/glioma_results/2mod && mkdir fold-0
echo Training fold-0!

export CUDA_VISIBLE_DEVICES=0 && python3 ../main.py --no_back_in_output --data /data/private_data/18_3d --results /results/glioma_results/2mod/fold-0 --deep_supervision --depth 6 --filters 64 96 128 192 256 384 512 --min_fmap 2 --scheduler --learning_rate 0.0003 --epochs 150 --nfolds 3 --fold 0 --amp --gpus 1 --task 18 --save_ckpt


cd /results/glioma_results/2mod && mkdir fold-1
echo Training fold-1!

export CUDA_VISIBLE_DEVICES=0 && python3 ../main.py --no_back_in_output --data /data/private_data/18_3d --results /results/glioma_results/2mod/fold-1 --deep_supervision --depth 6 --filters 64 96 128 192 256 384 512 --min_fmap 2 --scheduler --learning_rate 0.0003 --epochs 150 --nfolds 3 --fold 1 --amp --gpus 1 --task 18 --save_ckpt

cd /results/glioma_results/2mod && mkdir fold-2
echo Training fold-2!

export CUDA_VISIBLE_DEVICES=0 && python3 ../main.py --no_back_in_output --data /data/private_data/18_3d --results /results/glioma_results/2mod/fold-2 --deep_supervision --depth 6 --filters 64 96 128 192 256 384 512 --min_fmap 2 --scheduler --learning_rate 0.0003 --epochs 150 --nfolds 3 --fold 2 --amp --gpus 1 --task 18 --save_ckpt