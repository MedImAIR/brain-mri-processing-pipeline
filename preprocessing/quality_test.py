import os,sys
import numpy as np
import ants
import argparse
import logging
import subprocess
import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('--label', type=list, default = [0,1,2,3], help = 'labels in the segmentation mask')
parser.add_argument('--modalities', type=list, default = ['FLAIR.nii.gz','T1.nii.gz','T2.nii.gz','CT1_SEG.nii.gz'], help='names of files')
parser.add_argument('--current', type=str, default='/anvar/public_datasets/preproc_study/gbm/5_ss_shared/', help='folder name')
parser.add_argument('--parent', type=str, default='/anvar/public_datasets/preproc_study/gbm/4a_resamp/', help='parent folder files')
parser.add_argument('--segmentation_name', type=str, default='CT1_SEG.nii.gz', 
                    help= 'segmentation mask')
args = parser.parse_args()


# nans and wrong labels, equal shapes, wrong orientation, np.allclose to parent
label = args.label
modalities = args.modalities
current = args.current
parent = args.parent
segmentation_name = args.segmentation_name

if __name__ == "__main__":
    
    """
    The function is intended to check, if new dataset has no bugs. Helping to find
    
    """

    for patient in tqdm.tqdm(os.listdir(current)):
        if patient != 'logging.txt':
            img = ants.image_read(current + '/{}/{}'.format(patient, modalities[-1]))
            #reference shape to compare
            shape = np.shape(img.numpy())
            
            for modality in modalities:
                img = ants.image_read(current + '/{}/{}'.format(patient, modality))

                #checking shapes
                new_shape = np.shape(img.numpy())
                if new_shape != shape:
                    print('Wrong shapes: new ',new_shape, 'old:', shape, patient)

                #checking nans
                if not img.sum() > 0:
                    print(patient,modality, 'Amount of nans:', np.shape(np.argwhere(np.isnan(img.numpy()))))

                #checking labels for segmentation
                if modality == segmentation_name:
                    if not (np.unique(img.numpy()) == np.array(label)).sum() == len(label):
                            print('Assert label', patient, np.unique(img.numpy()) )

                old_img = ants.image_read(parent + '/{}/{}'.format(patient, modality))

                # check orientation
                old_orient = old_img.get_orientation()    
                img_orient = img.get_orientation()    

                if old_orient != img_orient:
                    print('Wrong orientation from parent: ', old_orient, 'and current: ', img_orient)
                    img_reoriented = ants.reorient_image2(img, orientation = old_orient)
                    # saved reoriented image
                    ants.image_write(img_reoriented, current + '/{}/{}'.format(patient, modality) , ri=False);

                #if the data is changed from the parent, checking for the main modalities
                if modality != segmentation_name:    
                    if np.allclose(img.numpy(), old_img.numpy()):
                        print('Allclose from the parent:', patient)