import os, sys
import numpy as np
import ants
import argparse
import shutil
import logging
import subprocess
from glob import glob

parser = argparse.ArgumentParser()

parser.add_argument('--path', type=str, default='/anvar/public_datasets/preproc_study/lgg/orig/', 
                    help='root dir for subject sequences data')
parser.add_argument('--fixedfilename', type=list, default=['CT1.nii.gz'], help='name of file to register')
parser.add_argument('--maskfilename', type=list, default=['CT1_SEG.nii.gz'], help='name of mask to register')
parser.add_argument('--movingfilenames', type=list, default=['T1.nii.gz','T2.nii.gz','FLAIR.nii.gz'], help='names of files')
parser.add_argument('--resamplingtarget', type=str, default=['./utils/sri24_T1.nii'], 
                    help= 'resampling target for all images')
parser.add_argument('--output', type=str, default='/anvar/public_datasets/preproc_study/lgg/4a_resamp/', 
                    help= 'output folder')
parser.add_argument('--channels', type=str, default=[1], 
                    help= 'channels in mask')



args = parser.parse_args()

def check_multiple_channels(path_to_img):
    """check that for Untypical channels, like [ 3.823199  7.646398 11.469597].
    This happens on GBM or LGG datasets, with multichanel target, after registration.
    """
    img = ants.image_read(path_to_img)

    channels = np.unique(img.numpy())[1:]
    if np.shape(channels)[0] == 3:
        if channels[0] != 1:
            print(path_to_img)
            print('Untypical three channels', channels)
            result_arr = img.numpy()
            result_arr[result_arr == channels[0]] = int(1)
            result_arr[result_arr == channels[1]] = int(2)
            result_arr[result_arr == channels[2]] = int(3)
            img_new = img.new_image_like(result_arr)
            img = img_new
            channels = [1,2,3]
    # if two channels       
    if np.shape(channels)[0] == 2:
        if channels[0] != 1:
            print(path_to_img)
            print('Untypical two channels', channels)
            result_arr = img.numpy()
            result_arr[result_arr == channels[0]] = int(1)
            result_arr[result_arr == channels[1]] = int(2)
            img_new = img.new_image_like(result_arr)
            img = img_new
            channels = [1,2]
    return(img , channels)

def resample_by_channels(img, 
                         channels = [1,2,3], 
                         interpolator = 0):
# empty array
    img_res = ants.resample_image(img, (1, 1, 1), False, interpolator)
    result_arr = np.zeros_like(img_res.numpy())

    for channel in channels:
        # float is needed by ants to save an image
        temp_img = img.new_image_like((img[:,:,:] == int(channel))*float(channel))
        temp_img_res = ants.resample_image(temp_img, (1, 1, 1), False, interpolator)
        temp_arr = np.round(temp_img_res.numpy(), 0)
        result_arr += (temp_arr[:,:,:] == int(channel))*float(channel)

    img_res =  img_res.new_image_like(result_arr)
    return img_res

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
       nohup python 4a_resamp.py > log_bgpd/4a_resamp.out &
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
        
        if len(os.listdir(args.output + subject + '/')) < (len(args.movingfilenames)+2):
            img_fixed = ants.image_read(args.path + subject + '/' + args.fixedfilename[0])
            mask_fixed, channels = check_multiple_channels(args.path + subject + '/' + args.maskfilename[0])

            # Reorient fixed
            img_fixed = ants.reorient_image2(img_fixed, orientation = 'RPI')
            mask_fixed = ants.reorient_image2(mask_fixed, orientation = 'RPI')

            # Resampling fixed
            logging.info("Resampling fixed started {}.".format(subject))
            img_fixed_res = ants.resample_image(img_fixed, (1, 1, 1), False, 0)

            logging.info("Resampling fixed completed {}.".format(subject))
            mask_fixed_res = resample_by_channels(mask_fixed, channels = channels)

            # Saving fixed, and checking, that resampled image has no `inf` artefacts
            if not np.isinf(img_fixed_res.numpy()).all():
                ants.image_write(img_fixed_res, args.output + subject + '/' + args.fixedfilename[0], ri=False);

            ants.image_write(mask_fixed_res, args.output + subject + '/' + args.maskfilename[0], ri=False);

            for name in args.movingfilenames:
                # Reorient moving
                img_moving = ants.image_read(args.path + subject + '/' + name)
                img_moving = ants.reorient_image2(img_moving, orientation = 'RPI')
                # Image registration
                logging.info("Rigid registration to {} started.".format(name))
                registered_img = rigid_reg(img_fixed, img_moving)
                img_moving_res = ants.resample_image(registered_img, (1, 1, 1), False, 0)
                
                logging.info("Rigid registration to {} completed.".format(name))
                # Saving moving images
                if not np.isinf(img_moving_res.numpy()).all():
                    ants.image_write(img_moving_res, args.output + subject + '/' + name, ri=False);
                if (np.shape(mask_fixed_res.numpy()) != np.shape(img_moving_res.numpy())):
                    print('Shape mismatch', subject, np.shape(mask_fixed_res.numpy()), np.shape(img_moving_res.numpy()))
                    break
    logging.info(str(args))                         