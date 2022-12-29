# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ctypes
import os

import nvidia_dlprof_pytorch_nvtx
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, early_stopping

from data_loading.data_module import DataModule
from nnunet.nn_unet import NNUnet
from utils.args import get_main_args
from utils.gpu_affinity import set_affinity
from utils.logger import LoggingCallback
from utils.utils import make_empty_dir, set_cuda_devices, verify_ckpt_path

if __name__ == "__main__":
    args = get_main_args()

    if args.profile:
        nvidia_dlprof_pytorch_nvtx.init()
        print("Profiling enabled")

    if args.affinity != "disabled":
        affinity = set_affinity(int(os.getenv("LOCAL_RANK", "0")), args.gpus, mode=args.affinity)

    # Limit number of CPU threads
    os.environ["OMP_NUM_THREADS"] = "1"
    # Set device limit on the current device cudaLimitMaxL2FetchGranularity = 0x05
    _libcudart = ctypes.CDLL("libcudart.so")
    pValue = ctypes.cast((ctypes.c_int * 1)(), ctypes.POINTER(ctypes.c_int))
    _libcudart.cudaDeviceSetLimit(ctypes.c_int(0x05), ctypes.c_int(128))
    _libcudart.cudaDeviceGetLimit(pValue, ctypes.c_int(0x05))
    assert pValue.contents.value == 128

    set_cuda_devices(args)
    seed_everything(args.seed)
    ckpt_path1 = '/home/polina/DeepLearningExamples/PyTorch/Segmentation/nnUNet/results/3_fold_brats_2021/fold-0/checkpoints/epoch=135-dice_mean=90.62.ckpt'
    ckpt_path2 = '/home/polina/DeepLearningExamples/PyTorch/Segmentation/nnUNet/results/3_fold_brats_2021/fold-1/checkpoints/epoch=118-dice_mean=91.05.ckpt'
    ckpt_path3 = '/home/polina/DeepLearningExamples/PyTorch/Segmentation/nnUNet/results/3_fold_brats_2021/fold-2/checkpoints/epoch=121-dice_mean=90.77.ckpt'
    
    callbacks = None
    model_ckpt = None
    
    model1 = NNUnet(args).load_from_checkpoint(ckpt_path1).state_dict()
    model2 = NNUnet(args).load_from_checkpoint(ckpt_path2).state_dict()
    model3 = NNUnet(args).load_from_checkpoint(ckpt_path3).state_dict()
    for key in model1:
        model1[key] = (model1[key] + model2[key] + model3[key]) / 3.
    model =NNUnet(args)
    model.load_state_dict(model1)
   
    trainer.save_checkpoint("average.ckpt")
#     torch.save(model1, '/home/polina/DeepLearningExamples/PyTorch/Segmentation/nnUNet/results/3_fold_brats_2021/avarage_weights.ckpt')

#     check
#     ckpt_path = '/home/polina/DeepLearningExamples/PyTorch/Segmentation/nnUNet/results/3_fold_brats_2021/avarage_weights.ckpt'
#     model = NNUnet(args).load_from_checkpoint(ckpt_path)