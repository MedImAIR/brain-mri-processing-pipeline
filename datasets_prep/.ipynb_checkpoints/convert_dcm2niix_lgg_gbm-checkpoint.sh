#!/bin/bash
#   Converts DICOM files in specified directory to Nifti (.nii.gz)
#   requires dcm2niix installed, in this case it is running under neurodocker
#   usage:
#   docker run --rm -it -v /your/directory/with/data/:/data -v ${PWD}:/home 
#   neurodocker:0.0.1

cd ../data
    for sub in */ ; do
        cd $sub

            dcm2niix -f CT1 y -o ../../home/training_data/$sub/ CT1/
            dcm2niix -f T1 y -o ../../home/training_data/$sub/ T1/
            dcm2niix -f T2 y -o ../../home/training_data/$sub/ T2/
            dcm2niix -f FLAIR y -o ../../home/training_data/$sub/ FLAIR/


        cd ..
        
    done
    
cd ..