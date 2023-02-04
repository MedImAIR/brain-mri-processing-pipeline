""" Performes a quality test between two preprocessing folders, ensuring that the images are different in voxel values, but the same with the shapes and orientations. Checks for NaN values and wrong class labels in the segmentations files, equal shapes, wrong orientation. Does np.allclose to parent file."""

import os,sys
import numpy as np
import ants
import argparse
import logging
import subprocess
import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('--label', type=list, default = [0,1], help = 'labels in the segmentation mask')
parser.add_argument('--modalities', type=list, default = ['FLAIR.nii.gz','T2.nii.gz','T1.nii.gz','mask_GTV_FLAIR.nii.gz'], help='names of files')
parser.add_argument('--current', type=str, default='4b/', help='folder name')
parser.add_argument('--parent', type=str, default='4a/', help='parent folder files')
parser.add_argument('--segmentation_name', type=str, default='mask_GTV_FLAIR.nii.gz', 
                    help= 'segmentation mask')
args = parser.parse_args()



label = args.label
modalities = args.modalities
current = args.current
parent = args.parent
segmentation_name = args.segmentation_name

if __name__ == "__main__":
    
    """
    The function is intended to check, if new dataset has no bugs. Helping to find
    
    """
    print(current)
    for patient in tqdm.tqdm(os.listdir(current)):
        if patient != 'logging.txt':
            img = ants.image_read(current + '/{}/{}'.format(patient, modalities[-1]))
            # reference shape to compare with
            shape = np.shape(img.numpy())
            
            for modality in modalities:
                img = ants.image_read(current + '/{}/{}'.format(patient, modality))

                #checking shapes
                new_shape = np.shape(img.numpy())
                if new_shape != shape:
                    print('Wrong shapes: new ',new_shape, 'old:', shape, patient)

                #checking NaN values
                if not img.sum() > 0:
                    print(patient,modality, 'Amount of nans:', np.shape(np.argwhere(np.isnan(img.numpy()))))

                old_img = ants.image_read(parent + '/{}/{}'.format(patient, modality))

                # check orientation
                old_orient = old_img.get_orientation()    
                img_orient = img.get_orientation()    

                if old_orient != img_orient:
                    print('Wrong orientation from parent: ', old_orient, 'and current: ', img_orient)
                    img_reoriented = ants.reorient_image2(img, orientation = old_orient)
                    # saved reoriented image
                    ants.image_write(img_reoriented, current + '/{}/{}'.format(patient, modality) , ri=False);

                # if the data is changed from the parent, checking for the main modalities
                if modality != segmentation_name:    
                    if np.allclose(img.numpy(), old_img.numpy()):
                        print('Allclose from the parent:', patient)