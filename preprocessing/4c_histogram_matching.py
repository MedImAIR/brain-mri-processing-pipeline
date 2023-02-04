import os, sys
import numpy as np
import ants
import argparse
import torch
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

parser.add_argument('--path', type=str, default='lgg/4a/', 
                    help='root dir for subject sequences data')
parser.add_argument('--fixedfilename', type=list, default=['CT1.nii.gz.nii.gz'], help='name of file to register')
parser.add_argument('--maskfilename', type=list, default=['CT1_SEG.nii.gz'], help='name of mask to register to RPI')
parser.add_argument('--movingfilenames', type=list, default=['FLAIR.nii.gz','T1.nii.gz','T2.nii.gz'], help='names of files')
parser.add_argument('--output', type=str, default='lgg/4c/', 
                    help= 'output folder')
parser.add_argument('--seed', type=str, default='./utils/lgg_seed.json', help= 'mode individual or shared ')
parser.add_argument('--device', type=str, default='cpu', help= 'gpu or cpu, if gpu - should be `int` ')

args = parser.parse_args()

base_dir = args.path
save_dir = args.output
seed = mpu.io.read(args.seed)

if __name__ == "__main__":
    
    os.makedirs(save_dir, exist_ok=True)
    logging.basicConfig(filename=args.output + "logging.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    
   
    # Creating a dataset
    subjects_list = []
    for patient in os.listdir(base_dir):
        if os.path.isdir(base_dir + patient):
            subject = tio.Subject(
                t1 = tio.ScalarImage(base_dir + patient + '/T1.nii.gz'),
                t2 = tio.ScalarImage(base_dir + patient + '/T2.nii.gz'),
                # can be commented for other datasets
                ct1 = tio.ScalarImage(base_dir + patient + '/CT1.nii.gz'),
                fl = tio.ScalarImage(base_dir + patient + '/FLAIR.nii.gz')
            )
            subjects_list.append(subject)

    # Separate dataset for each fold
    for fold in ['fold_0', 'fold_1', 'fold_2']:
        print(len(seed[fold]))
        logging.info(fold)
        # Create dataset
        temp_t1_list = []
        temp_t2_list = []
        temp_ct1_list = []
        temp_fl_list = []

        for patient in os.listdir(base_dir):
            if patient not in seed[fold]:
                if os.path.isdir(base_dir + patient):

                    subjects_list.append(subject)
                    temp_t1_list.append(base_dir + patient + '/T1.nii.gz')
                    temp_t2_list.append(base_dir + patient + '/T2.nii.gz')
                    # can be commented for other datasets
                    temp_ct1_list.append(base_dir + patient + '/CT1.nii.gz')
                    temp_fl_list.append(base_dir + patient + '/FLAIR.nii.gz')

        print('For landmarks there are ', len(temp_t1_list))
        logging.info("Training T1 landmarks started.")
        t1_landmarks = HistogramStandardization.train(temp_t1_list)
        logging.info("Training T2 landmarks started.")
        t2_landmarks = HistogramStandardization.train(temp_t2_list)
        # can be commented for other datasets
        logging.info("Training CT1 landmarks started.")
        ct1_landmarks = HistogramStandardization.train(temp_ct1_list)
        logging.info("Training FLAIR landmarks started.")
        fl_landmarks = HistogramStandardization.train(temp_fl_list)

        # Saving landmarks
        landmarks_dict = {
        't1': t1_landmarks,
        't2': t2_landmarks,
        # can be commented for other datasets
        'ct1': ct1_landmarks,
        'fl': fl_landmarks
        }

        # Saving landmarks
        logging.info("Saving landmarks started.")
        torch.save(landmarks_dict, './params/lgg_dict.pth')

        hist_standardize = tio.HistogramStandardization(landmarks_dict)

            # Apply transforms
        for i in range(0, len(os.listdir(base_dir))):
            # Check if it is logging file instead of a folder
            if os.path.isdir(base_dir + os.listdir(base_dir)[i]):
                os.makedirs(save_dir + '/6_hist_{}/'.format(fold) + os.listdir(base_dir)[i], exist_ok= True)

                if len(os.listdir(save_dir + '/6_hist_{}/'.format(fold) + os.listdir(base_dir)[i])) < 4:
                    logging.info(os.listdir(base_dir)[i])
                    
                    # hist standartize for the four landmarks
                    hist_standard = hist_standardize(subjects_list[i])
                    
                    hist_standard['t1'].save(save_dir + '/6_hist_{}/'.format(fold) + os.listdir(base_dir)[i] +'/T1.nii.gz')
                    hist_standard['t2'].save(save_dir + '/6_hist_{}/'.format(fold) + os.listdir(base_dir)[i] +'/T2.nii.gz')
                     # can be commented for other datasets
                    hist_standard['ct1'].save(save_dir + '/6_hist_{}/'.format(fold) + os.listdir(base_dir)[i] +'/CT1.nii.gz')
                    hist_standard['fl'].save(save_dir + '/6_hist_{}/'.format(fold) + os.listdir(base_dir)[i] +'/FLAIR.nii.gz')
                    # saving segmentation file
                    shutil.copy(base_dir + os.listdir(base_dir)[i] + '/' + args.maskfilename[0],
                    save_dir + '/6_hist_{}/'.format(fold) + os.listdir(base_dir)[i] +  '/' + args.maskfilename[0])