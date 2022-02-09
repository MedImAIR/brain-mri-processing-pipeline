"""
    usage:  run under CaPTk environement;
    the file should be located in `home` folder.
    
    1. chmod +x /home/projects/brain-mri-processing-pipeline/main_pipeline/3b_n4.sh
    2. /home/projects/brain-mri-processing-pipeline/main_pipeline/3b_n4.sh
    
    with docker image cbica/captk:latest:
    docker run -it --rm --cpuset-cpus='0-15' -v /anvar/public_datasets/preproc_study/gbm/3a_atlas/:/input -v /anvar/public_datasets/preproc_study/gbm/3b_n4/:/output -v ${PWD}:/home --entrypoint /bin/bash cbica/captk:latest
    
    info: https://cbica.github.io/CaPTk/preprocessing_susan.html
    ./CaPTk/bin/install/appdir/usr/bin/Preprocessing -h

"""
for sub in /input/* ; do
    echo ${sub};

    ./CaPTk/bin/install/appdir/usr/bin/Preprocessing -i ${sub}/T1.nii.gz -o /output/${sub: -12}/T1.nii.gz -n4 -nB 200;
    
    ./CaPTk/bin/install/appdir/usr/bin/Preprocessing -i ${sub}/CT1.nii.gz -o /output/${sub: -12}/CT1.nii.gz -n4 -nB 200;
    
    ./CaPTk/bin/install/appdir/usr/bin/Preprocessing -i ${sub}/T2.nii.gz -o /output/${sub: -12}/T2.nii.gz -n4 -nB 200;
    
    ./CaPTk/bin/install/appdir/usr/bin/Preprocessing -i ${sub}/FLAIR.nii.gz -o /output/${sub: -12}/FLAIR.nii.gz -n4 -nB 200;    
    
    # Copy segmentation file into folder
    cp ${sub}/CT1_SEG.nii.gz /output/${sub: -12}/CT1_SEG.nii.gz;
    
done
