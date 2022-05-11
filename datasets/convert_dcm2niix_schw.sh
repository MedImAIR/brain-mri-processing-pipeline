#!/bin/bash
#   Converts DICOM files in specified directory to Nifti (.nii.gz)
#   requires dcm2niix installed, in this case it is running under neurodocker
#   usage:
#   docker run --rm -it -v /your/directory/with/data/:/data -v ${PWD}:/home neurodocker:0.0.1
#   docker run --rm -it -v /anvar/public_datasets/preproc_study/schw/:/data -v ${PWD}:/home neurodocker:0.0.1

# home/projects/brain-mri-processing-pipeline/datasets_prep/convert_dcm2niix_schw.sh

# projects

cd ../data/dicom/
    for sub in */ ; do
        cd $sub
            mkdir ../../../data/orig/$sub/;

#             dcm2niix -b n -f T1 -z y -o ../../../data/orig/$sub/ T1/;
#             dcm2niix -b n -f T2 -z y -o ../../../data/orig/$sub/ T2/;
            dcm2niix -b n -f T1_SEG -z y -o ../../../data/orig/$sub/ RTSTRUCT_DCM/;


        cd ..
        
    done
    
cd ..