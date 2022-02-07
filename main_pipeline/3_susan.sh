"""
    usage:  run under CaPTk environement;
    the file should be located in `home` folder.
    
    1. chmod +x 3_susan.sh 
    2. /home/projects/brain-mri-processing-pipeline/main_pipeline/3_susan.sh
    
    with docker image cbica/captk:latest:
    docker run -it --rm --cpuset-cpus='0-15' -v /anvar/public_datasets/preproc_study/gbm/orig/:/input -v /anvar/public_datasets/preproc_study/gbm/3_susan_captk/:/output -v ${PWD}:/home --entrypoint /bin/bash cbica/captk:latest
    
    info: https://cbica.github.io/CaPTk/preprocessing_susan.html
    ./CaPTk/bin/install/appdir/usr/bin/Preprocessing -h

"""
for sub in /input/* ; do
    echo ${sub}

    ./CaPTk/bin/install/appdir/usr/bin/Preprocessing -i ${sub}/T1.nii.gz -o /output/${sub: -12}/T1_n4.nii.gz -n4 -nB 200;
    ./CaPTk/bin/install/appdir/usr/bin/Preprocessing -i /output/${sub: -12}/T1_n4.nii.gz -o /output/${sub: -12}/T1.nii.gz -ss;
#     rm /output/${sub: -12}/T1_n4.nii.gz
    
    ./CaPTk/bin/install/appdir/usr/bin/Preprocessing -i ${sub}/CT1.nii.gz -o /output/${sub: -12}/CT1_n4.nii.gz -n4 -nB 200;
    ./CaPTk/bin/install/appdir/usr/bin/Preprocessing -i /output/${sub: -12}/CT1_n4.nii.gz -o /output/${sub: -12}/CT1.nii.gz -ss;
    
    ./CaPTk/bin/install/appdir/usr/bin/Preprocessing -i ${sub}/T2.nii.gz -o /output/${sub: -12}/T2_n4.nii.gz -n4 -nB 200;
    ./CaPTk/bin/install/appdir/usr/bin/Preprocessing -i /output/${sub: -12}/T2_n4.nii.gz -o /output/${sub: -12}/T2.nii.gz -ss;
    
    ./CaPTk/bin/install/appdir/usr/bin/Preprocessing -i ${sub}/FLAIR.nii.gz -o /output/${sub: -12}/FLAIR_n4.nii.gz -n4 -nB 200;    
    ./CaPTk/bin/install/appdir/usr/bin/Preprocessing -i /output/${sub: -12}/FLAIR_n4.nii.gz -o /output/${sub: -12}/FLAIR.nii.gz -ss; 
done
