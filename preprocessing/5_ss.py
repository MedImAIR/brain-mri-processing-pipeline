import os,sys
import numpy as np
import ants
import argparse
import logging
import subprocess

parser = argparse.ArgumentParser()

parser.add_argument('--path', type=str, default='/anvar/public_datasets/preproc_study/gbm/4a_resamp/', 
                    help='root dir for subject sequences data')
parser.add_argument('--fixedfilename', type=list, default=['CT1.nii.gz'], help='name of file to register')
parser.add_argument('--maskfilename', type=list, default=['CT1_SEG.nii.gz'], help='name of mask to register to RPI')
parser.add_argument('--movingfilenames', type=list, default=['FLAIR.nii.gz','T2.nii.gz','T1.nii.gz'], help='names of files')
parser.add_argument('--output', type=str, default='/mnt/public_data/preproc_study/gbm/5_ss_shared/', 
                    help= 'output folder')
parser.add_argument('--mode', type=str, default='shared', help= 'mode individual or shared ')
parser.add_argument('--device', type=str, default='0', help= 'gpu or cpu, if gpu - should be `int` ')

args = parser.parse_args()


def hdbet(src_path, dst_path):
    command = ["hd-bet", "-i", src_path, "-o", dst_path, "-device", args.device]
    subprocess.call(command)
    return

# saving mask
def hdbet_mask(src_path, dst_path):
    command = ["hd-bet", "-i", src_path, "-o", dst_path, "-device", args.device, "-s", "1"]
    subprocess.call(command)
    return

  
#     logging.info("HD-BET  FL started.")
#     hdbet(args.output + 'FL_to_SRI.nii.gz', args.output + 'FL_to_SRI_bet.nii.gz')
    
#     logging.info("HD-BET T2 started.")
#     hdbet(args.output + 'T2_to_SRI.nii.gz', args.output + 'T2_to_SRIbet.nii.gz')
    
#     logging.info("HD-BET T1CE started.")
#     hdbet(args.output + 'T1CE_to_SRI.nii.gz', args.output + 'T1CE_to_SRI_bet.nii.gz')
    
#     logging.info("HD-BET T1 started.")
#     hdbet(args.output + 'T1_to_SRI.nii.gz', args.output + 'T1_to_SRI_bet.nii.gz')
    

                    
if __name__ == "__main__":
    
    """Pipeline with HD-BET ss calculation
       cpu: nohup python 5_ss.py > 5_ss.out &
       gbm: python main_pipeline/5_ss.py --path /anvar/public_datasets/preproc_study/gbm/3a_atlas/ --output /mnt/public_data/preproc_study/gbm/5_ss_indiv/ --device 0 
       schw: python main_pipeline/5_ss.py --path /anvar/public_datasets/preproc_study/schw/3a_atlas/ --output /mnt/public_data/preproc_study/schw/5_ss_shared/ --device 0 --mode shared
       bgpd: python main_pipeline/5_ss.py --path /anvar/public_datasets/preproc_study/bgpd/3a_atlas/ --output /mnt/public_data/preproc_study/bgpd/5_ss_shared/ --device 0 --mode shared
       
    """
    os.makedirs(args.output, exist_ok=True)
    logging.basicConfig(filename=args.output + "logging.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info("{} Folder processing".format(args.path))  
    logging.info("{} Mode for skull stripping".format(args.mode)) 
    subjects_paths = [f.path for f in os.scandir(args.path) if f.is_dir()]
    subjects = [f.split('/')[-1] for f in subjects_paths]
    
    if isinstance(args.fixedfilename, str):
        args.fixedfilename = [args.fixedfilename]
        args.maskfilename = [args.maskfilename]
        args.movingfilenames = [args.movingfilenames]
    
    for subject in subjects:
        # Creating folder to save subject data
        logging.info("{} Subject processing".format(subject)) 
        os.makedirs(args.output + subject + '/', exist_ok=True)
        # If package is completed
        if len(os.listdir(args.output + subject + '/')) < 4:
        
            img_fixed = ants.image_read(args.path + subject + '/' + args.fixedfilename[0])
            mask_fixed = ants.image_read(args.path + subject + '/' + args.maskfilename[0])
            # Reorient fixed
            img_fixed = ants.reorient_image2(img_fixed, orientation = 'RPI')
            mask_fixed = ants.reorient_image2(mask_fixed, orientation = 'RPI')
            # Saving fixed
            ants.image_write(img_fixed, args.output + subject + '/' + args.fixedfilename[0][:-7] + '_RPI.nii.gz' , ri=False);
            ants.image_write(mask_fixed, args.output + subject + '/' + args.maskfilename[0], ri=False);

            logging.info("HD-BET fixed image started.")
            hdbet_mask(args.output + subject + '/' + args.fixedfilename[0][:-7] + '_RPI.nii.gz', 
                       args.output + subject + '/' + args.fixedfilename[0])
            logging.info("HD-BET fixed image completed.")
            print(os.listdir(args.output + subject + '/'))
            # Removing excessive files
            os.remove(args.output + subject + '/' + args.fixedfilename[0][:-7] + '_RPI.nii.gz')
    #         os.remove(args.output + subject + '/' + args.fixedfilename[0][:-7] + '_mask.nii.gz' 

            # Processing moving images

            for name in args.movingfilenames:

                    # Reorient moving
                    img_moving = ants.image_read(args.path + subject + '/' + name)
                    logging.info("RPI reorientation to {} started.".format(name))
                    img_moving = ants.reorient_image2(img_moving, orientation = 'RPI')
                    # Saved reoriented image
                    ants.image_write(img_moving, args.output + subject + '/' + name[:-7] + '_RPI.nii.gz' , ri=False);

                    if args.mode == 'individual':
                        logging.info("Individual mask for {} started.".format(name))
                        hdbet_mask(args.output + subject + '/' + name[:-7] + '_RPI.nii.gz',
                                   args.output + subject + '/' + name)
                        # Removing excessive files
                        os.remove(args.output + subject + '/' + name[:-7] + '_RPI.nii.gz')

                    else:
                        logging.info("Mask for fixed image applied to {} started.".format(name))
                        mask = ants.image_read(args.output + subject + '/' + args.fixedfilename[0][:-7] + '_mask.nii.gz')
                        # Saving mask multiplication
                        ants.image_write(img_moving.new_image_like(img_moving.numpy()*mask.numpy()),
                                         args.output + subject + '/' + name)
                        # Removing excessive files
                        os.remove(args.output + subject + '/' + name[:-7] + '_RPI.nii.gz')
  
    logging.info(str(args))                         