"""
    usage:  run under CaPTk environement;
    the file should be located in `home` folder.
    
    1. chmod +x /home/projects/brain-mri-processing-pipeline/main_pipeline/3c_n4_susan.sh
    2. /home/projects/brain-mri-processing-pipeline/main_pipeline/3d_susan.sh
    
    with docker image cbica/captk:latest:
    docker run -it --rm --cpuset-cpus='0-15' -v /anvar/public_datasets/preproc_study/schw/3a_atlas/:/input -v /anvar/public_datasets/preproc_study/schw/3d_susan/:/output -v ${PWD}:/home --entrypoint /bin/bash cbica/captk:latest
    
    info: https://cbica.github.io/CaPTk/preprocessing_susan.html
    ./CaPTk/bin/install/appdir/usr/bin/Preprocessing -h

"""
for sub in /input/* ; do
    echo ${sub};

    ./CaPTk/bin/install/appdir/usr/bin/Preprocessing -i ${sub}/T1.nii.gz -o /output/${sub: -10}/T1.nii.gz -ss;

#     ./CaPTk/bin/install/appdir/usr/bin/Preprocessing -i ${sub}/CT1.nii.gz -o /output/${sub: -12}/CT1.nii.gz -ss;
 
    ./CaPTk/bin/install/appdir/usr/bin/Preprocessing -i ${sub}/T2.nii.gz -o /output/${sub: -10}/T2.nii.gz -ss;
 
#     ./CaPTk/bin/install/appdir/usr/bin/Preprocessing -i ${sub}/FLAIR.nii.gz -o /output/${sub: -12}/FLAIR.nii.gz -ss; 

    cp ${sub}/T1_SEG.nii.gz /output/${sub: -10}/T1_SEG.nii.gz;
    
done
