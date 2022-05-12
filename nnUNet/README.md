# nnU-Net For PyTorch

This repository is based on the [NVIDIA solution](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Segmentation/nnUNet) and includes scripts to train the nnU-Net model to achieve state-of-the-art accuracy and is tested and maintained by NVIDIA. In their repository you will find a detailed description of the model architecture, the changes made, and any information about the pipeline used. We just adapted their implementation for datasets with different preprocessing and a different number of classes.


## Quick Start Guide

1. Clone the repository.

Executing this command will create your local repository with all the code to run nnU-Net.
```
git clone https://github.com/MedImAIR/brain-mri-processing-pipeline.git
cd brain-mri-processing-pipeline/nnUNet
```
    
2. Build the nnU-Net PyTorch NGC container.
    
This command will use the Dockerfile to create a Docker image named `nnunet`, downloading all the required components automatically.

```
docker build -t nnunet .
```
    
The NGC container contains all the components optimized for usage on NVIDIA hardware.
    
3. Start an interactive session in the NGC container to run preprocessing/training/inference.
    
The following command will launch the container and mount the `./data` directory as a volume to the `/data` directory inside the container, and `./results` directory to the `/results` directory in the container.
    
```
mkdir data results
docker run -it --runtime=nvidia --shm-size=8g --ulimit memlock=-1 --ulimit stack=67108864 --rm -v ${PWD}/data:/data -v ${PWD}/results:/results nnunet:latest /bin/bash
```

4. Prepare BraTS dataset.

To preprocess the data run:
```
python preprocess.py --task 01 --dim 3
python preprocess.py --task 01 --dim 2
```

Then `ls /data` should print:
```
01_3d 01_2d Task01_BrainTumour
```

For the specifics concerning data preprocessing, see the [Getting the data](#getting-the-data) section.
    
5. Start training.
   
Training can be started with:
```
python scripts/train.py --gpus <gpus> --fold <fold> --dim <dim> [--amp]
```

To see descriptions of the train script arguments run `python scripts/train.py --help`. You can customize the training process. For details, see the [Training process](#training-process) section.

6. Start benchmarking.

The training and inference performance can be evaluated by using benchmarking scripts, such as:
 
```
python scripts/benchmark.py --mode {train,predict} --gpus <ngpus> --dim {2,3} --batch_size <bsize> [--amp] 
```

To see descriptions of the benchmark script arguments run `python scripts/benchmark.py --help`.


7. Start inference/predictions.
   
Inference can be started with:
```
python scripts/inference.py --data <path/to/data> --dim <dim> --fold <fold> --ckpt_path <path/to/checkpoint> [--amp] [--tta] [--save_preds]
```

Note: You have to prepare either validation or test dataset to run this script by running `python preprocess.py --task 01 --dim {2,3} --exec_mode {val,test}`. After preprocessing inside given task directory (e.g. `/data/01_3d/` for task 01 and dim 3) it will create `val` or `test` directory with preprocessed data ready for inference. Possible workflow:

```
python preprocess.py --task 01 --dim 3 --exec_mode val
python scripts/inference.py --data /data/01_3d/val --dim 3 --fold 0 --ckpt_path <path/to/checkpoint> --amp --tta --save_preds
```

Then if you have labels for predicted images you can evaluate it with `evaluate.py` script. For example:

```
python evaluate.py --preds /results/preds_task_01_dim_3_fold_0_tta --lbls /data/Task01_BrainTumour/labelsTr
```

To see descriptions of the inference script arguments run `python scripts/inference.py --help`. You can customize the inference process. For details, see the [Inference process](#inference-process) section.

Now that you have your model trained and evaluated, you can choose to compare your training results with our [Training accuracy results](#training-accuracy-results). You can also choose to benchmark yours performance to [Training performance benchmark](#training-performance-results), or [Inference performance benchmark](#inference-performance-results). Following the steps in these sections will ensure that you achieve the same accuracy and performance results as stated in the [Results](#results) section.
    
## Advanced

The following sections provide greater details of the dataset, running training and inference, and the training results.

### Scripts and sample code

In the root directory, the most important files are:

* `main.py`: Entry point to the application. Runs training, evaluation, inference or benchmarking.
* `preprocess.py`: Entry point to data preprocessing.
* `download.py`: Downloads given dataset from [Medical Segmentation Decathlon](http://medicaldecathlon.com/).
* `Dockerfile`: Container with the basic set of dependencies to run nnU-Net.
* `requirements.txt:` Set of extra requirements for running nnU-Net.
* `evaluate.py`: Compare predictions with ground truth and get final score.
    
The `data_preprocessing` folder contains information about the data preprocessing used by nnU-Net. Its contents are:
    
* `configs.py`: Defines dataset configuration like patch size or spacing.
* `preprocessor.py`: Implements data preprocessing pipeline.
* `convert2tfrec.py`: Implements conversion from NumPy files to tfrecords.
    
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

The `notebooks` folder includes:

* `BraTS21.ipynb`: Notebook with our solution for BraTS21 challenge.
* `custom_dataset.ipynb`: Notebook which demonstrates how to use nnU-Net with custom dataset.

Other folders included in the root directory are:

* `images/`: Contains a model diagram.
* `scripts/`: Provides scripts for training, benchmarking and inference of nnU-Net.

### Command-line options

To see the full list of available options and their descriptions, use the `-h` or `--help` command-line option, for example:

`python main.py --help`

The following example output is printed when running the model:

```
usage: main.py [-h] [--exec_mode {train,evaluate,predict}] [--data DATA] [--results RESULTS] [--logname LOGNAME] [--task TASK] [--gpus GPUS] [--learning_rate LEARNING_RATE] [--gradient_clip_val GRADIENT_CLIP_VAL] [--negative_slope NEGATIVE_SLOPE] [--tta] [--brats] [--deep_supervision] [--more_chn] [--invert_resampled_y] [--amp] [--benchmark] [--focal] [--sync_batchnorm] [--save_ckpt] [--nfolds NFOLDS] [--seed SEED] [--skip_first_n_eval SKIP_FIRST_N_EVAL] [--ckpt_path CKPT_PATH] [--fold FOLD] [--patience PATIENCE] [--batch_size BATCH_SIZE] [--val_batch_size VAL_BATCH_SIZE] [--profile] [--momentum MOMENTUM] [--weight_decay WEIGHT_DECAY] [--save_preds] [--dim {2,3}] [--resume_training] [--num_workers NUM_WORKERS] [--epochs EPOCHS] [--warmup WARMUP] [--norm {instance,batch,group}] [--nvol NVOL] [--depth DEPTH] [--min_fmap MIN_FMAP] [--deep_supr_num DEEP_SUPR_NUM] [--res_block] [--filters FILTERS [FILTERS ...]] [--data2d_dim {2,3}] [--oversampling OVERSAMPLING] [--overlap OVERLAP] [--affinity {socket,single_single,single_single_unique,socket_unique_interleaved,socket_unique_continuous,disabled}] [--scheduler] [--optimizer {sgd,adam}] [--blend {gaussian,constant}] [--train_batches TRAIN_BATCHES] [--test_batches TEST_BATCHES]

optional arguments:
  -h, --help            show this help message and exit
  --exec_mode {train,evaluate,predict}
                        Execution mode to run the model (default: train)
  --data DATA           Path to data directory (default: /data)
  --results RESULTS     Path to results directory (default: /results)
  --logname LOGNAME     Name of dlloger output (default: None)
  --task TASK           Task number. MSD uses numbers 01-10 (default: None)
  --gpus GPUS           Number of gpus (default: 1)
  --learning_rate LEARNING_RATE
                        Learning rate (default: 0.0008)
  --gradient_clip_val GRADIENT_CLIP_VAL
                        Gradient clipping norm value (default: 0)
  --negative_slope NEGATIVE_SLOPE
                        Negative slope for LeakyReLU (default: 0.01)
  --tta                 Enable test time augmentation (default: False)
  --brats               Enable BraTS specific training and inference (default: False)
  --deep_supervision    Enable deep supervision (default: False)
  --more_chn            Create encoder with more channels (default: False)
  --invert_resampled_y  Resize predictions to match label size before resampling (default: False)
  --amp                 Enable automatic mixed precision (default: False)
  --benchmark           Run model benchmarking (default: False)
  --focal               Use focal loss instead of cross entropy (default: False)
  --sync_batchnorm      Enable synchronized batchnorm (default: False)
  --save_ckpt           Enable saving checkpoint (default: False)
  --nfolds NFOLDS       Number of cross-validation folds (default: 5)
  --seed SEED           Random seed (default: 1)
  --skip_first_n_eval SKIP_FIRST_N_EVAL
                        Skip the evaluation for the first n epochs. (default: 0)
  --ckpt_path CKPT_PATH
                        Path to checkpoint (default: None)
  --fold FOLD           Fold number (default: 0)
  --patience PATIENCE   Early stopping patience (default: 100)
  --batch_size BATCH_SIZE
                        Batch size (default: 2)
  --val_batch_size VAL_BATCH_SIZE
                        Validation batch size (default: 4)
  --profile             Run dlprof profiling (default: False)
  --momentum MOMENTUM   Momentum factor (default: 0.99)
  --weight_decay WEIGHT_DECAY
                        Weight decay (L2 penalty) (default: 0.0001)
  --save_preds          Enable prediction saving (default: False)
  --dim {2,3}           UNet dimension (default: 3)
  --resume_training     Resume training from the last checkpoint (default: False)
  --num_workers NUM_WORKERS
                        Number of subprocesses to use for data loading (default: 8)
  --epochs EPOCHS       Number of training epochs (default: 1000)
  --warmup WARMUP       Warmup iterations before collecting statistics (default: 5)
  --norm {instance,batch,group}
                        Normalization layer (default: instance)
  --nvol NVOL           Number of volumes which come into single batch size for 2D model (default: 4)
  --depth DEPTH         The depth of the encoder (default: 5)
  --min_fmap MIN_FMAP   Minimal dimension of feature map in the bottleneck (default: 4)
  --deep_supr_num DEEP_SUPR_NUM
                        Number of deep supervision heads (default: 2)
  --res_block           Enable residual blocks (default: False)
  --filters FILTERS [FILTERS ...]
                        [Optional] Set U-Net filters (default: None)
  --data2d_dim {2,3}    Input data dimension for 2d model (default: 3)
  --oversampling OVERSAMPLING
                        Probability of crop to have some region with positive label (default: 0.4)
  --overlap OVERLAP     Amount of overlap between scans during sliding window inference (default: 0.5)
  --affinity {socket,single_single,single_single_unique,socket_unique_interleaved,socket_unique_continuous,disabled}
                        type of CPU affinity (default: socket_unique_contiguous)
  --scheduler           Enable cosine rate scheduler with warmup (default: False)
  --optimizer {sgd,adam}
                        Optimizer (default: adam)
  --blend {gaussian,constant}
                        How to blend output of overlapping windows (default: gaussian)
  --train_batches TRAIN_BATCHES
                        Limit number of batches for training (used for benchmarking mode only) (default: 0)
  --test_batches TEST_BATCHES
                        Limit number of batches for inference (used for benchmarking mode only) (default: 0)
```

#### Dataset guidelines

To train nnU-Net you will need to preprocess your dataset as a first step with `preprocess.py` script. Run `python scripts/preprocess.py --help` to see descriptions of the preprocess script arguments.

For example to preprocess data for 3D U-Net run: `python preprocess.py --task 01 --dim 3`.

In `data_preprocessing/configs.py` for each [Medical Segmentation Decathlon](http://medicaldecathlon.com/) task there are defined: patch size, precomputed spacings and statistics for CT datasets.

The preprocessing pipeline consists of the following steps:

1. Cropping to the region of non-zero values.
2. Resampling to the median voxel spacing of their respective dataset (exception for anisotropic datasets where the lowest resolution axis is selected to be the 10th percentile of the spacings).
3. Padding volumes so that dimensions are at least as patch size.
4. Normalizing:
    * For CT modalities the voxel values are clipped to 0.5 and 99.5 percentiles of the foreground voxels and then data is normalized with mean and standard deviation from collected from foreground voxels.
    * For MRI modalities z-score normalization is applied.

### Training process

The model trains for at least `--min_epochs` and at most `--max_epochs` epochs. After each epoch evaluation, the validation set is done and validation loss is monitored for early stopping (see `--patience` flag). Default training settings are:
* Adam optimizer with learning rate of 0.0008 and weight decay 0.0001.
* Training batch size is set to 2 for 3D U-Net and 16 for 2D U-Net.
    
This default parametrization is applied when running scripts from the `scripts/` directory and when running `main.py` without explicitly overriding these parameters. By default, the training is in full precision. To enable AMP, pass the `--amp` flag. AMP can be enabled for every mode of execution.

The default configuration minimizes a function `L = (1 - dice_coefficient) + cross_entropy` during training and reports achieved convergence as [dice coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient) per class. The training, with a combination of dice and cross entropy has been proven to achieve better convergence than a training using only dice.

The training can be run directly without using the predefined scripts. The name of the training script is `main.py`. For example:

```
python main.py --exec_mode train --task 01 --fold 0 --gpus 1 --amp
```
  
Training artifacts will be saved to `/results` in the container. Some important artifacts are:
* `/results/logs.json`: Collected dice scores and loss values evaluated after each epoch during training on validation set.
* `/results/checkpoints`: Saved checkpoints. By default, two checkpoints are saved - one after each epoch ('last.ckpt') and one with the highest validation dice (e.g 'epoch=5.ckpt' for if highest dice was at 5th epoch).

To load the pretrained model provide `--ckpt_path <path/to/checkpoint>`.

### Inference process

Inference can be launched by passing the `--exec_mode predict` flag. For example:

```
python main.py --exec_mode predict --task 01 --fold 0 --gpus 1 --amp --tta --save_preds --ckpt_path <path/to/checkpoint>
```

The script will then:

* Load the checkpoint from the directory specified by the `<path/to/checkpoint>` directory
* Run inference on the preprocessed validation dataset corresponding to fold 0
* Print achieved score to the console
* If `--save_preds` is provided then resulting masks in the NumPy format will be saved in the `/results` directory
                       
