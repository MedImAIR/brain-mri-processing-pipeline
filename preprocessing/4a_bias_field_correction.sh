"""
    usage:  run under CaPTk environement;
    the file should be located in `home` folder.
    
    1. chmod +x /home/your/script/location/4a_bias_field_correction.sh
    2. /home/your/script/location/4a_bias_field_correction.sh
    
    with docker image cbica/captk:latest:
    docker run -it --rm --cpuset-cpus='0-15' -v /home/your/input/folder/location/:/input -v /home/your/output/folder/location/:/output -v ${PWD}:/home --entrypoint /bin/bash cbica/captk:latest
    
    info: https://cbica.github.io/CaPTk/preprocessing_susan.html
    ./CaPTk/bin/install/appdir/usr/bin/Preprocessing -h

"""
mkdir /output/;
for sub in /input/* ; do
    echo ${sub};
    subname="$(echo ${sub} | cut -d'/' -f3)";

    ./CaPTk/bin/install/appdir/usr/bin/Preprocessing -i ${sub}/T1.nii.gz -o /output/${subname}/T1.nii.gz -n4 -nB 200;
    
    ./CaPTk/bin/install/appdir/usr/bin/Preprocessing -i ${sub}/CT1.nii.gz -o /output/${subname}/CT1.nii.gz -n4 -nB 200;
    
    ./CaPTk/bin/install/appdir/usr/bin/Preprocessing -i ${sub}/T2.nii.gz -o /output/${subname}/T2.nii.gz -n4 -nB 200;
    
    ./CaPTk/bin/install/appdir/usr/bin/Preprocessing -i ${sub}/FLAIR.nii.gz -o /output/${subname}/FLAIR.nii.gz -n4 -nB 200;    
    
#     # Copy segmentation file into folder
    cp ${sub}/mask_GTV_FLAIR.nii.gz /output/${subname}/mask_GTV_FLAIR.nii.gz;
    
done
