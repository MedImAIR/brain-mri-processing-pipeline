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


#### To reproduce the study:
1. Go to `datasets`, after you download the data from [TCIA](https://www.cancerimagingarchive.net/);
2. Go to `preprocessing` for data preprocessing;
3. Go to `nnUnet` for model training and scoring;

In the `Readme.md` of the following repositoritories you will find the instructions to follow.
