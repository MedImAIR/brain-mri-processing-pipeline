import os, sys
import numpy as np
import ants
import argparse
import shutil
import logging
import subprocess
from glob import glob

parser = argparse.ArgumentParser()

parser.add_argument('--path', type=str, default='/anvar/public_datasets/preproc_study/bgpd/orig/', 
                    help='root dir for subject sequences data')
parser.add_argument('--fixedfilename', type=list, default=['CT1.nii.gz'], help='name of file to register')
parser.add_argument('--maskfilename', type=list, default=['mask_GTV_FLAIR.nii.gz'], help='name of mask to register')
parser.add_argument('--movingfilenames', type=list, default=['T1.nii.gz','T2.nii.gz','FLAIR.nii.gz'], help='names of files')
parser.add_argument('--resamplingtarget', type=str, default=['./utils/sri24_T1.nii'], 
                    help= 'resampling target for all images')
parser.add_argument('--output', type=str, default='/anvar/public_datasets/preproc_study/bgpd/1_reg/', 
                    help= 'output folder')

args = parser.parse_args()

def check_multiple_channels(path_to_img):
    """check that for Untypical channels, like [ 3.823199  7.646398 11.469597].
    This happens on GBM or LGG datasets, with multichanel target, after registration.
    """
    img = ants.image_read(path_to_img)

    channels = np.unique(img.numpy())[1:]
    if np.shape(channels)[0] > 1:
        if channels[0] != 1:
            print(path_to_img)
            print('Untypical channels', channels)
            result_arr = img.numpy()
            result_arr[result_arr == channels[0]] = int(1)
            result_arr[result_arr == channels[1]] = int(2)
            result_arr[result_arr == channels[2]] = int(3)
            img_new = img.new_image_like(result_arr)
            img = img_new
            channels = [1,2,3]

    return(img , channels)

def rigid_reg(fixed, moving):
    """Rigidly register `moving` image onto `fixed` image and apply resulting transformation on `mask`.
    Returns mask in `fixed` resolution."""
    
    if type(fixed) is str:
        # Read images if input is pathlike
        fixed = ants.image_read(fixed)
        moving = ants.image_read(moving)
    
    # Compute registration if input is ants.image
    res = ants.registration(fixed=fixed, moving=moving,
                            type_of_transform='Rigid')
    
    new_img = ants.apply_transforms(fixed, moving,
                                    transformlist = res['fwdtransforms'][0])
    return new_img
                    
if __name__ == "__main__":
    
    """Pipeline with CT1 Rigid registration and Z-score calculation
       nohup python 1_reg.py > log_gbm/1_reg_check.out &
    """
    os.makedirs(args.output, exist_ok=True)

    logging.basicConfig(filename=args.output + "logging.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    
    logging.info("{} Folder processing".format(args.path))  
    
    subjects_paths = [f.path for f in os.scandir(args.path) if f.is_dir()]
    subjects = [f.split('/')[-1] for f in subjects_paths ]

    for subject in subjects:
        # Creating folder to save subject data
        logging.info("{} Subject processing".format(subject)) 
        os.makedirs(args.output + subject + '/', exist_ok=True)
        
        # if subject is not processed fully, and all files are gathered
        if len(os.listdir(args.output + subject + '/')) < len(os.listdir(args.path + subject + '/')):
            img_fixed = ants.image_read(args.path + subject + '/' + args.fixedfilename[0])
            mask_fixed, channels = check_multiple_channels(args.path + subject + '/' + args.maskfilename[0])
            
            # Reorient fixed
            img_fixed = ants.reorient_image2(img_fixed, orientation = 'RPI')
            mask_fixed = ants.reorient_image2(mask_fixed, orientation = 'RPI')
            # Saving fixed
            if not np.isinf(img_fixed.numpy()).all():
                ants.image_write(img_fixed, args.output + subject + '/' + args.fixedfilename[0], ri=False);
                
            ants.image_write(mask_fixed, args.output + subject + '/' + args.maskfilename[0], ri=False);

            for name in args.movingfilenames:
                # Reorient moving
                img_moving = ants.image_read(args.path + subject + '/' + name)
                img_moving = ants.reorient_image2(img_moving, orientation = 'RPI')
                # Image registration
                logging.info("Rigid registration to {} started.".format(name))
                registered_img = rigid_reg(img_fixed, img_moving)
                logging.info("Rigid registration to {} completed.".format(name))
                # Saving moving
                ants.image_write(registered_img, args.output + subject + '/' + name, ri=False);
        

    logging.info(str(args))                         