import os, sys
import numpy as np
import ants
import argparse
import shutil
import logging
import subprocess
from glob import glob

parser = argparse.ArgumentParser()

parser.add_argument('--path', type=str, default='/anvar/public_datasets/preproc_study/gbm/orig/', 
                    help='root dir for subject sequences data')
parser.add_argument('--fixedfilename', type=list, default=['CT1.nii.gz'], help='name of file to register')
parser.add_argument('--movingfilenames', type=list, default=['T1.nii.gz', 'FLAIR.nii.gz', 'T2.nii.gz'], help='names of files')
parser.add_argument('--output', type=str, default='/anvar/public_datasets/preproc_study/gbm/4_interp/', 
                    help= 'output folder')

args = parser.parse_args()

def calculate_z_score(img):
    """ Calculates Z-score normalisation over ants.img and returns new image"""
    
    if type(img) is str:
        # Read images if input is pathlike
        img = ants.image_read(img)
        
    img_z = (img.numpy() - img.numpy().mean())/img.numpy().std()
    new_img = img.new_image_like(img_z)
    
    return new_img

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
    
    """Pipeline with CT1 Rigid registration and interpolation to template, n4 and Z-score calculation
       nohup python 2_n4.py > 2_n4.out &
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
        img_fixed = ants.image_read(args.path + subject + '/' + args.fixedfilename[0])
        
        logging.info("N4 image correction started {}.".format(subject))
        img_fixed_n4 = ants.utils.bias_correction.abp_n4(img_fixed)
        logging.info("N4 image correction completed {}.".format(subject))

        for name in args.movingfilenames:
            # Searching for filenames
            img_moving = ants.image_read(args.path + subject + '/' + name)
            img_moving_n4 = ants.utils.bias_correction.abp_n4(img_moving)
            # Image registration
            logging.info("Rigid registration to {} started.".format(args.fixedfilename[0]))
            registered_img = rigid_reg(img_fixed_n4, img_moving_n4)
            
            logging.info("Saving {} file".format(name))
            ants.image_write(registered_img, args.output + subject + '/' + name, ri=False);
            
        ants.image_write(img_fixed_n4, args.output + subject + '/' + args.fixedfilename[0], ri=False);

    logging.info(str(args))                         