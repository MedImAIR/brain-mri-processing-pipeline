The `README` contains instructions for data preprocesing for four open-source datasets from [TCIA](https://www.cancerimagingarchive.net/) archive.
1. [TCGA-GBM images - 102 subjects (DICOM, 32 GB)](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=41517733) original dataset;
2. [TCGA-LGG images - 65 subjects (DICOM, 17 GB)](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=41517733) original dataset;
3. [Images and Radiation Therapy Structures - 242 subjects (DICOM, 26 GB)] (https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70229053);
4. BGPD dataset in under publication.

#### Dataset preparation:
1. Compose folders with needes modalities:
1.1. `gbm_paths.csv` with chosen four modalities;
1.2. `lgg_paths.csv` with chosen four modalities;

2. Convert `DICOM` to `Nifty` files with [dicom2niix];
2.1. `convert_dcm2niix_lgg_gbm.sh` for `LGG` and `GBM` datasets for 4 modalities;
2.2. convert_dcm2niix_schw.sh for 2 modalities;
 

