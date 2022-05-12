
This repository is based on the [NVIDIA nnU-Net For PyTorch](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Segmentation/nnUNet) and includes scripts to train the nnU-Net model to achieve state-of-the-art accuracy and is tested and maintained by NVIDIA. In their repository you will find a detailed description of the model architecture, the changes made, and any information about the pipeline used. We just adapted their implementation for datasets with different preprocessing and a different number of classes.


## Quick Start Guide

The training takes place into the docker prepared by Nvidia.

    
1. Build the nnU-Net PyTorch NGC container from Nvidia.
    
This command will use the Dockerfile to create a Docker image named `nnunet`, downloading all the required components automatically.

```
cd nnUNet
docker build -t nnunet .
```
    
The NGC container contains all the components optimized for usage on NVIDIA hardware.
    
2. Start an interactive session in the NGC container to run preprocessing/training/inference.
    
The following command will launch the container and mount the `./data` directory as a volume to the `/data` directory inside the container, `./results` directory to the `/results` directory in the container, and `./brain-mri-processing-pipeline` directory of this repository to the `./brain-mri-processing-pipeline` directory in the container.
    
```
docker run -it --runtime=nvidia --shm-size=8g --ulimit memlock=-1 --ulimit stack=67108864 --rm -v ${PWD}/data:/data -v ${PWD}/results:/results -v ${PWD}/brain-mri-processing-pipeline:/brain-mri-processing-pipeline  nnunet:latest /bin/bash
```

3. To run script for preparing dataset:

```
python preprocess.py --task 01 --dim 3
```
The task number must be written inside the `data_preprocessing/configs.py`

The preprocessing pipeline consists of the following steps:

1. Cropping to the region of non-zero values.
2. Padding volumes so that dimensions are at least as patch size.
3. Normalizing:
    * For CT modalities the voxel values are clipped to 0.5 and 99.5 percentiles of the foreground voxels and then data is normalized with mean and standard deviation from collected from foreground voxels.
    * For MRI modalities z-score normalization is applied.
    
    
4. To start training:
   
```
python main.py --gpus <gpus> --fold <fold> --dim <dim> [--amp]
```
for example:

```
python main.py --deep_supervision --depth 6 --filters 64 96 128 192 256 384 512 --min_fmap 2 --scheduler --learning_rate 0.0003 --epochs 150 --nfolds 3 --fold 0 --gpus 1 --amp -deep_supervision --task <task> --save_ckpt --data </path_to_dataset/task_3d> --results </path_to_results_folder>
```

To see descriptions of the arguments run `python main.py --help`. You can customize the training process. For details, see the [NVIDIA repo](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Segmentation/nnUNet).

To load the pretrained model provide `--ckpt_path <path/to/checkpoint>`.

5. To save predictions:

```
python main.py --exec_mode predict --task <task> --data <path_to_dataset>  --dim 3 --fold <fold> --nfolds <num_folds> --ckpt_path </path_to.ckpt> --results <path_to_sae_results> --amp --tta --save_preds
```

The script will then:

* Load the checkpoint from the directory specified by the `<path/to/checkpoint>` directory
* Run inference on the preprocessed dataset corresponding to fold number
* Prediction masks in the NumPy format will be saved
                       
6. To calculate metrics run:

```
python metrics/metrics.py --path_to_pred <folder with predictions> --path_to_1_reg <path to folder for resample> --path_to_target <path to labels> --path_to_resamp <path to folder for resample> --dataset <dataset name> --out <path to json file with metrics> [--subjects <list subjects_id>]
```

### Scripts and sample code

In the root directory, the most important files are:

* `main.py`: Entry point to the application. Runs training, evaluation, inference or benchmarking.
* `preprocess.py`: Entry point to data preprocessing.
* `Dockerfile`: Container with the basic set of dependencies to run nnU-Net.
* `requirements.txt:` Set of extra requirements for running nnU-Net.
    
The `data_preprocessing` folder contains information about the data preprocessing used by nnU-Net. Its contents are:
    
* `configs.py`: Defines dataset configuration like patch size or spacing.
* `preprocessor.py`: Implements data preprocessing pipeline.
    
The `data_loading` folder contains information about the data pipeline used by nnU-Net. Its contents are:
    
* `data_module.py`: Defines `LightningDataModule` used by PyTorch Lightning.
* `dali_loader.py`: Implements DALI data loader.
    
The `nnunet` folder contains information about the building blocks of nnU-Net and the way they are assembled. Its contents are:
    
* `metrics.py`: Implements dice metric
* `loss.py`: Implements loss function.
* `nn_unet.py`: Implements training/validation/test logic and dynamic creation of U-Net architecture used by nnU-Net.
    
The `utils` folder includes:

* `args.py`: Defines command line arguments.
* `utils.py`: Defines utility functions.
* `logger.py`: Defines logging callback for performance benchmarking.

Other folders included in the root directory are:

* `images/`: Contains a model diagram.
* `scripts/`: Provides scripts for training, benchmarking and inference of nnU-Net.

To see the full list of available options and their descriptions, use the `-h` or `--help` command-line option, for example:

`python main.py --help`

The following example output is printed when running the model:

```
usage: main.py [-h] [--exec_mode {train,evaluate,predict}] [--data DATA] [--results RESULTS] [--logname LOGNAME] [--task TASK] [--gpus GPUS] [--learning_rate LEARNING_RATE] [--gradient_clip_val GRADIENT_CLIP_VAL] [--negative_slope NEGATIVE_SLOPE] [--tta] [--brats] [--deep_supervision] [--more_chn] [--invert_resampled_y] [--amp] [--benchmark] [--focal] [--sync_batchnorm] [--save_ckpt] [--nfolds NFOLDS] [--seed SEED] [--skip_first_n_eval SKIP_FIRST_N_EVAL] [--ckpt_path CKPT_PATH] [--fold FOLD] [--patience PATIENCE] [--batch_size BATCH_SIZE] [--val_batch_size VAL_BATCH_SIZE] [--profile] [--momentum MOMENTUM] [--weight_decay WEIGHT_DECAY] [--save_preds] [--dim {2,3}] [--resume_training] [--num_workers NUM_WORKERS] [--epochs EPOCHS] [--warmup WARMUP] [--norm {instance,batch,group}] [--nvol NVOL] [--depth DEPTH] [--min_fmap MIN_FMAP] [--deep_supr_num DEEP_SUPR_NUM] [--res_block] [--filters FILTERS [FILTERS ...]] [--data2d_dim {2,3}] [--oversampling OVERSAMPLING] [--overlap OVERLAP] [--affinity {socket,single_single,single_single_unique,socket_unique_interleaved,socket_unique_continuous,disabled}] [--scheduler] [--optimizer {sgd,adam}] [--blend {gaussian,constant}] [--train_batches TRAIN_BATCHES] [--test_batches TEST_BATCHES]
```
For details, see the [NVIDIA repo](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Segmentation/nnUNet).
