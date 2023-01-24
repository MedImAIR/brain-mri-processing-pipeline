# brain-mri-processing-pipeline
Here we report a comprehensive study of multimodal MRI brain cancer image segmentation on open-source datasets. Our results demonstrate that most popular standardization steps add no value to artificial neural network performance; moreover, preprocessing can hamper model performance. We suggest that image intensity normalization approaches do not contribute to model accuracy because of the reduction of signal variance with image standardization. Finally, we show the contribution of scull-stripping in data preprocessing is almost negligible if measured in terms of clinically relevant metrics.

We show that the only essential transformation for accurate analysis is the unification of voxel spacing across the dataset. In contrast, anatomy alignment in form of non-rigid atlas registration is not necessary and most intensity equalization steps do not improve model productiveness. 

#### This repository contains:
1. `datasets` - preparation of the opensourse data for preprocessing and training;
3. `preprocessing` - scripts for data preprocessing;
4. `nnUnet` - training and scoring  the segmentation models;
5. `figures` - code for the paper illustrations;

## Brain MRI preprocessing main pipeline:
![image](/figures/abstract.png)

#### Data preprocessing
The most of data processing is executed with [ANTS](http://stnava.github.io/ANTs/). Intensity normalization N4 and SUSAN are executed inside [CaPTk](https://www.med.upenn.edu/cbica/captk/) container. You need to install [HD-BEt](https://github.com/MIC-DKFZ/HD-BET) for histagram matching and [HD-Bet](https://github.com/MIC-DKFZ/HD-BET) for skull stripping.

#### To reproduce the study:
1. Go to `datasets`, after you download the data from [TCIA](https://www.cancerimagingarchive.net/);
2. Go to `preprocessing` for data preprocessing;
3. Go to `nnUnet` for model training and scoring;


## Training & Prediction

Model realization is based on the [NVIDIA nnU-Net For PyTorch](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Segmentation/nnUNet) and includes scripts to train the nnU-Net and UNETR models to achieve state-of-the-art accuracy and is tested and maintained by NVIDIA. In their repository you will find a detailed description of the model architecture, the changes made, and any information about the pipeline used. We just adapted their implementation for datasets with different preprocessing and a different number of classes.


To run experimrnts, see instructions in `nnUnet` folder.
