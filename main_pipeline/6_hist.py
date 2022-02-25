import os, sys
import numpy as np
import ants
import argparse
import shutil
import logging
import subprocess
import tqdm
import matplotlib.pyplot as plt 
import torchio as tio
from glob2 import glob
from torchio.transforms import HistogramStandardization
import mpu.io


parser = argparse.ArgumentParser()

parser.add_argument('--path', type=str, default='/anvar/public_datasets/preproc_study/bgpd/3a_atlas/', 
                    help='root dir for subject sequences data')
parser.add_argument('--fixedfilename', type=list, default=['FLAIR.nii.gz'], help='name of file to register')
parser.add_argument('--maskfilename', type=list, default=['mask_GTV_FLAIR.nii.gz'], help='name of mask to register to RPI')
parser.add_argument('--movingfilenames', type=list, default=['CT1.nii.gz','T2.nii.gz','T1.nii.gz'], help='names of files')
parser.add_argument('--output', type=str, default='/mnt/public_data/preproc_study/bgpd/6_hist/', 
                    help= 'output folder')
parser.add_argument('--seed', type=str, default='utils/example.json', help= 'mode individual or shared ')
parser.add_argument('--device', type=str, default='cpu', help= 'gpu or cpu, if gpu - should be `int` ')

args = parser.parse_args()

base_dir = args.path
save_dir = args.autput
seed = mpu.io.read(args.seed)

if __name__ == "__main__":
    
    os.makedirs(args.output, exist_ok=True)
    logging.basicConfig(filename=args.output + "logging.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    
    # Separate dataset for each fold
    for fold in ['fold_0','fold_1', 'fold_2']:
        logging.info(fold, " started.")
        # Create dataset
        temp_t1_list = []
        temp_t2_list = []
        temp_ct1_list = []
        temp_fl_list = []

        for patient in os.listdir(base_dir):
            if patient not in seed[fold]:
                if os.path.isdir(base_dir + patient):

                    temp_t1_list.append(base_dir + patient + '/T1.nii.gz')
                    temp_t2_list.append(base_dir + patient + '/T2.nii.gz')
                    temp_ct1_list.append(base_dir + patient + '/CT1.nii.gz')
                    temp_fl_list.append(base_dir + patient + '/FLAIR.nii.gz')

        logging.info("Training T1 landmarks started.")
        t1_landmarks = HistogramStandardization.train(temp_t1_list)
        logging.info("Training T2 landmarks started.")
        t2_landmarks = HistogramStandardization.train(temp_t2_list)
        logging.info("Training CT1 landmarks started.")
        ct1_landmarks = HistogramStandardization.train(temp_ct1_list)
        logging.info("Training FLAIR landmarks started.")
        fl_landmarks = HistogramStandardization.train(temp_fl_list)

        # Saving landmarks
        landmarks_dict = {
        't1': t1_landmarks,
        't2': t2_landmarks,
        'ct1': ct1_landmarks,
        'fl': fl_landmarks
        }

        hist_standardize = tio.HistogramStandardization(landmarks_dict)

       # Apply transforms
        for i in range(0, len(os.listdir(base_dir))):
            
            os.makedirs(save_dir + '/6_hist_{}/'.format(fold) + os.listdir(base_dir)[i], exist_ok= True)
            
            if len(os.listdir(save_dir + '/6_hist_{}/'.format(fold) + os.listdir(base_dir)[i])) < 4:
                hist_standard = hist_standardize(subjects_list[i])
                hist_standard['t1'].save(save_dir + '/6_hist_{}/'.format(fold) + os.listdir(base_dir)[i] +'/T1.nii.gz')
                hist_standard['ct1'].save(save_dir + '/6_hist_{}/'.format(fold) + os.listdir(base_dir)[i] +'/CT1.nii.gz')
                hist_standard['fl'].save(save_dir + '/6_hist_{}/'.format(fold) + os.listdir(base_dir)[i] +'/FLAIR.nii.gz')
                hist_standard['t2'].save(save_dir + '/6_hist_{}/'.format(fold) + os.listdir(base_dir)[i] +'/T2.nii.gz')
                shutil.copy(base_dir + os.listdir(base_dir)[i] + '/' + args.maskfilename,
                    save_dir + '/6_hist_{}/'.format(fold) + os.listdir(base_dir)[i] +  '/' + args.maskfilename)