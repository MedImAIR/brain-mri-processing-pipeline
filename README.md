# brain-mri-processing-pipeline
Here we report a comprehensive study of multimodal MRI brain cancer image segmentation on open-source datasets. Our results demonstrate that most popular standardization steps add no value to artificial neural network performance; moreover, preprocessing can hamper model performance. We suggest that image intensity normalization approaches do not contribute to model accuracy because of the reduction of signal variance with image standardization. Finally, we show the contribution of scull-stripping in data preprocessing is almost negligible if measured in terms of clinically relevant metrics.

We show that the only essential transformation for accurate analysis is the unification of voxel spacing across the dataset. In contrast, anatomy alignment in form of non-rigid atlas registration is not necessary and most intensity equalization steps do not improve model productiveness. 

#### This repository contains:
1. `datasets_prep` - preparation of the opensourse data for preprocessing and training;
3. `main_pipeline` - scripts for data preprocessing;
4. `nnUnet` - training and scoring  the segmentation models;
5. `illustrations` - code for the paper illustrations;

## Brain MRI preprocessing main pipeline:
![image](illustrations/abstract1.png)


#### To reproduce the study:
1. Go to `datasets_prep`, in the `Readme.md`you will find the instructions to follow after you download the data from [TCIA](https://www.cancerimagingarchive.net/)


`main_pipeline` - folder for ablation study on typical MRI brain data preprocessing.
- `1_z_score.py` - checking influence of Z-score image normalization;
- `2_n4.py` - checking influence of n4 image bias field correction;
- `3_susan.py` - checking influence of SUSAN image denoising;
- `4_upsample.py`
- `5_atlas.py`
- `6_ss.py`
- `7_hist.py`

`ss_pipeline` - folder for ablation study on skull-stripping

`dataset_prep` - folder for preparation of three open sourse datasets
