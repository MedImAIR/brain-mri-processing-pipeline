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
parser.add_argument('--fixedfilename', type=list, default=['FLAIR.nii.gz'], help='name of file to register')
parser.add_argument('--maskfilename', type=list, default=['mask_GTV_FLAIR.nii.gz'], help='name of mask to register')
parser.add_argument('--movingfilenames', type=list, default=['CT1.nii.gz','T2.nii.gz','T1.nii.gz'], help='names of files')
parser.add_argument('--resamplingtarget', type=str, default=['./utils/sri24_T1.nii'], 
                    help= 'resampling target for all images')
parser.add_argument('--output', type=str, default='/anvar/public_datasets/preproc_study/bgpd/3a_atlas/', 
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
       nohup python 3a_atlas.py > log_bgpd/3a_atlas.out &
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
        if len(os.listdir(args.output + subject + '/')) < 4:
            img_fixed = ants.image_read(args.path + subject + '/' + args.fixedfilename[0])
            mask_fixed = ants.image_read(args.path + subject + '/' + args.maskfilename[0])
            img_target = ants.image_read(args.resamplingtarget[0])

            # Reorient fixed
            img_fixed = ants.reorient_image2(img_fixed, orientation = 'RPI')
            mask_fixed = ants.reorient_image2(mask_fixed, orientation = 'RPI')

            # Register fixed image to SRI
            logging.info("Atlas registration started {}.".format(subject))
            img_to_sri = ants.registration(fixed=img_target, moving=img_fixed,
                                type_of_transform='Rigid')
            img_fixed_res = ants.apply_transforms(img_target, img_fixed,
                                        transformlist = img_to_sri['fwdtransforms'][0])
            logging.info("Atlas registration completed {}.".format(subject))        

            # Applying mask transform
            mask_fixed_res = ants.apply_transforms(img_fixed_res, mask_fixed,
                                        transformlist = img_to_sri['fwdtransforms'][0])

            # Reorient to one side
            img_fixed_res = ants.reorient_image2(img_fixed_res, orientation = 'RPI')
            mask_fixed_res = ants.reorient_image2(mask_fixed_res, orientation = 'RPI')

            # Saving fixed
            ants.image_write(img_fixed_res, args.output + subject + '/' + args.fixedfilename[0], ri=False);
            ants.image_write(mask_fixed_res, args.output + subject + '/' + args.maskfilename[0], ri=False);

            for name in args.movingfilenames:
                # Reorient moving
                img_moving = ants.image_read(args.path + subject + '/' + name)
                img_moving = ants.reorient_image2(img_moving, orientation = 'RPI')

                # Image registration
                logging.info("Rigid registration to {} started.".format(name))
                registered_img = rigid_reg(img_fixed, img_moving)
                img_moving_res = ants.apply_transforms(img_fixed_res, registered_img,
                                        transformlist = img_to_sri['fwdtransforms'][0])
                logging.info("Rigid registration to {} completed.".format(name))
                # Saving moving images
                img_moving_res = ants.reorient_image2(img_moving_res, orientation = 'RPI')
                ants.image_write(img_moving_res, args.output + subject + '/' + name, ri=False);


    logging.info(str(args))                         