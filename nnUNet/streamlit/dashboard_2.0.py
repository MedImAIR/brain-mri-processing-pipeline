# -*- coding: utf-8 -*-
import json
from io import BytesIO
from PIL import Image
import os

import streamlit as st
import pandas as pd
import numpy as np
import ants
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage.measure import find_contours
from glob import glob
import scipy
from scipy.ndimage import measurements
from skimage import exposure, img_as_ubyte

SUBJECTS = ['TCGA-02-0085', 'TCGA-12-1098', 'TCGA-08-0512', 'TCGA-02-0106', 'TCGA-12-1598']
PATH_TO_IMG='/data/private_data/gbm/gbm_4a_resamp/images/'
PATH_TO_LABEL='/data/private_data/gbm/gbm_4a_resamp/labels/'

def to_lbl(pred):
    enh = pred[2]
    c1, c2, c3 = pred[0] > 0.5, pred[1] > 0.5, pred[2] > 0.5
    pred = (c1 > 0).astype(np.uint8)
    pred[(c2 == False) * (c1 == True)] = 2
    pred[(c3 == True) * (c1 == True)] = 3

    components, n = measurements.label(pred == 3)
    for et_idx in range(1, n + 1):
        _, counts = np.unique(pred[components == et_idx], return_counts=True)
        if 1 < counts[0] and counts[0] < 8 and np.mean(enh[components == et_idx]) < 0.9:
            pred[components == et_idx] = 1

    et = pred == 3
    if 0 < et.sum() and et.sum() < 73 and np.mean(enh[et]) < 0.9:
        pred[et] = 1
    pred = np.transpose(pred, (2, 1, 0)).astype(np.uint8)
    return pred

def draw_contours(image, mask):
    """Draw 3 x 4 matplotlib axes with imshow 3D slice and GTV contour overlay."""
    
    assert(image.shape == mask.shape)
    contours = find_contours(mask)
    fig = plt.figure(figsize=(16,12))
    plt.axis('off')
    plt.imshow(image, cmap='gray')
    for cont in range(len(contours)):
            plt.plot(contours[cont][:, 1], contours[cont][:, 0], c='r');
    return fig   

if __name__ == '__main__':
    st.set_page_config(page_title='Glioma Segmentation', page_icon="🧠")
    st.title('Welcome To Project 3D brain glioma segmentation in MRI')
    instructions = """
        Choose any image and get the prediction which will be displayed to the screen.
        """
    st.write(instructions)
    
    available_images = SUBJECTS
    st.sidebar.title("What to do")
    image_name = st.sidebar.selectbox("Choose Subject Name", available_images)
    slice_img = st.sidebar.slider("Slice", min_value=0, max_value=143, value = 72)
    file_img = os.path.join(PATH_TO_IMG, image_name+ '.nii.gz')
    file_label =  os.path.join(PATH_TO_LABEL, image_name+ '_seg.nii.gz')
        
    if (file_img is not None) and (file_label is not None):
        img_orig = nib.load(file_img).get_fdata()
        img_orig *= (255.0/img_orig.max())
#         print(img_orig.shape)
        label = nib.load(file_label).get_fdata().astype(np.uint8)
        lab = img_as_ubyte(exposure.rescale_intensity(label))
#         st.title("Here is the image you've selected")
#         st.write('MRI input images in different modalities')
        col1, col2 = st.columns(2)
        with col1:
#             flair = img_orig[:, :, slice_img, 0,np.newaxis].astype(np.uint8)
            contour = st.checkbox('Contour',key='flair')
            if contour:
                img = draw_contours(img_orig[:, :, slice_img, 0].astype(np.uint8),  lab[:,:,slice_img] )
                st.pyplot(img,clear_figure=False)
                st.write('FLAIR')
            else:
                 st.image(img_orig[:, :, slice_img, 0].astype(np.uint8),caption=["FLAIR"],clamp=True)
            
            contour = st.checkbox('Contour',key='t1')
            if contour:
                img = draw_contours(img_orig[:, :, slice_img, 1].astype(np.uint8),  lab[:,:,slice_img] )
                st.pyplot(img,clear_figure=False)
                st.write('T1')
            else:
                st.image(img_orig[:, :, slice_img, 1].astype(np.uint8),caption=["T1"],clamp=True)

        with col2:
            contour = st.checkbox('Contour',key='t1ce')
            if contour:
                img = draw_contours(img_orig[:, :, slice_img, 2].astype(np.uint8),  lab[:,:,slice_img] )
                st.pyplot(img,clear_figure=False)
                st.write('T1CE')
            else:
                st.image(img_orig[:, :, slice_img, 2].astype(np.uint8),caption=["T1CE"],clamp=True)
            
            contour = st.checkbox('Contour',key='t2')
            if contour:
                img = draw_contours(img_orig[:, :, slice_img, 3].astype(np.uint8),  lab[:,:,slice_img] )
                st.pyplot(img,clear_figure=False)
                st.write('T2')
            else:
                st.image(img_orig[:, :, slice_img, 3].astype(np.uint8),caption=["T2"],clamp=True)
            
