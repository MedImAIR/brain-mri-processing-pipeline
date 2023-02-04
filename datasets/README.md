The nstructions for dataset dowlnload and composition for three open-source datasets from [TCIA](https://www.cancerimagingarchive.net/) archive.
1. [TCGA-GBM images - 102 subjects (DICOM, 32 GB)](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=41517733) from DICOM-SEG Conversions for TCGA-LGG and TCGA-GBM Segmentation Datasets (DICOM-Glioma-SEG). 102 subjects out of 102 availible with segmentation masks in native image space.

2. [TCGA-LGG images - 65 subjects (DICOM, 17 GB)](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=41517733) from DICOM-SEG Conversions for TCGA-LGG and TCGA-GBM Segmentation Datasets (DICOM-Glioma-SEG); 36 subjects out of 65 availible with segmentation masks in native image space.

3. BGPD dataset in under publication on [TCIA](https://www.cancerimagingarchive.net/).

#### Dataset preparation:
1. Compose folders with needes modalities:
1.1. `gbm_paths.csv` with the chosen four modalities;
1.2. `lgg_paths.csv` with the chosen four modalities;

2. Convert `DICOM` to `Nifty` files with [dicom2niix];
2.1. `convert_dcm2niix_lgg_gbm.sh` for `LGG` and `GBM` datasets for 4 modalities;

 

