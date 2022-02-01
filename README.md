# brain-mri-processing-pipeline
Here we explore how data processing affects nnUnet segmentation quality


`main_pipeline` - folder for ablation study on typical MRI brain data preprocessing.
- `1_z_score.py` - checking influence of Z-score image normalization;
- `2_n4.py` - checking influence of n4 image bias field correction;
- `3_susan.py` - checking influence of SUSAN image denoising;
- `4_upsample.py`
- `5_atlas.py`
- `6_ss.py`
- `7_hist.py`

`ss_pipeline` - folder for ablation study on scull-stripping

`dataset_prep` - folder for preparation of three open sourse datasets
