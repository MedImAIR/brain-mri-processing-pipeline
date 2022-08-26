#!/bin/bash
#   Converts DICOM files in specified directory to Nifti (.nii.gz)

mkdir /anvar/public_datasets/preproc_study/lgg/orig1/

cd /anvar/public_datasets/preproc_study/lgg/dicom/
    for sub in */ ; do
        cd $sub
        
            mkdir /anvar/public_datasets/preproc_study/lgg/orig1/$sub/

            dcm2niix -b n -f CT1 -z y -o /anvar/public_datasets/preproc_study/lgg/orig1/$sub/ CT1/
#             dcm2niix -b n -f T1 -z y -o /anvar/public_datasets/preproc_study/lgg/orig/$sub/ T1/
#             dcm2niix -b n -f T2 -z y -o /anvar/public_datasets/preproc_study/lgg/orig/$sub/ T2/
#             dcm2niix -b n -f FLAIR -z y -o /anvar/public_datasets/preproc_study/lgg/orig/$sub/ FLAIR/
            dcm2niix -b n -f CT1_SEG -z y -o /anvar/public_datasets/preproc_study/lgg/orig1/$sub/ RTSTRUCT/CT1/
            
        cd ..
    done
cd ..