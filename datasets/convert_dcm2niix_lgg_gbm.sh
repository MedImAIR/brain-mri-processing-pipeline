#!/bin/bash
#   Converts DICOM files in specified directory to Nifti (.nii.gz)
#   requires dcm2niix installed, in this case it is running under neurodocker
#   usage:
#   docker run --rm -it -v /your/directory/with/data/:/data -v ${PWD}:/home neurodocker:0.0.1
#   docker run --rm -it -v /anvar/public_datasets/preproc_study/gbm/:/data -v ${PWD}:/home neurodocker:0.0.1
#   docker run -it --rm --cpuset-cpus='0-15' -v /anvar/public_datasets/preproc_study/lgg/:/data -v ${PWD}:/home --entrypoint /bin/bash cbica/captk:latest

# /home/projects/brain-mri-processing-pipeline/datasets/convert_dcm2niix_lgg_gbm.sh

# projects

mkdir ../data/orig

cd ../data/dicom/
    for sub in */ ; do
        cd $sub
            mkdir ../../../data/orig/$sub/

            ../../../work/CaPTk/bin/install/appdir/usr/bin/dcm2niix -b n -f CT1 -z y -o ../../../data/orig/$sub/ CT1/
            ../../../work/CaPTk/bin/install/appdir/usr/bin/dcm2niix -b n -f T1 -z y -o ../../../data/orig/$sub/ T1/
            ../../../work/CaPTk/bin/install/appdir/usr/bin/dcm2niix -b n -f T2 -z y -o ../../../data/orig/$sub/ T2/
            ../../../work/CaPTk/bin/install/appdir/usr/bin/dcm2niix -b n -f FLAIR -z y -o ../../../data/orig/$sub/ FLAIR/


        cd ..
        
    done
    
cd ..