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
from glob import glob
import scipy
from scipy.ndimage import measurements
from skimage import exposure, img_as_ubyte

SUBJECTS = ['TCGA-02-0085', 'TCGA-12-1098', 'TCGA-08-0512', 'TCGA-02-0106', 'TCGA-12-1598']
PATH_TO_IMG='/data/private_data/gbm/gbm_4a_resamp/images/'
PATH_TO_PREDICT='/results/gbm_infer/gbm_4a_resamp/predictions_best_epoch=197-dice_mean=78_14_task=38_fold=0_tta'

# from nnunet.nn_unet import NNUnet
# @st.cache()
# def mri_img_load(img_path, label_path): 
#     img_orig = np.load(img_path).transpose(3,2,1,0)
#     label_orig = np.load(label_path).transpose(0,3,2,1).squeeze()
# #     pred = np.load(pred_path)['arr_0']
   
#     l = np.argsort(label_orig.sum(axis=(0,1)))[-1:]
#      st.write(label_orig.shape)
#     st.write(label_orig.shape)
#     y_wt, y_tc, y_et = label_orig > 0, ((label_orig == 1) + (label_orig == 3)) > 0, label_orig == 3
#     label_orig = np.stack([y_wt, y_tc, y_et], axis=0).astype(int).transpose(1,2,3,0)
# #     p = np.round(pred, 0).transpose(3,2,1,0).astype(np.uint8)
#     imgs = [img_orig[:, :, l, i] for i in [0,3]] 
#     mask_pred = [label_orig[:,:,l,0]] + [label_orig[:,:,l,1]] +[label_orig[:,:,l,2]] 
# #     +[p[:,:,l,0]] +[label_orig[:,:,l,1]] +  [p[:,:,l,1]]+ [label_orig[:,:,l,2]] +[p[:,:,l,2]]
# #     dice_1 = metrics.compute_dice_coefficient((label_orig[:,:,:,0]==1), (p[:,:,:,0]==1))
# #     dice_2 = metrics.compute_dice_coefficient((label_orig[:,:,:,1]==1), (p[:,:,:,1]==1))
# #     dice_3 = metrics.compute_dice_coefficient((label_orig[:,:,:,2]==1), (p[:,:,:,2]==1))
#     return imgs, mask_pred
# , dice_1, dice_2, dice_3

# @st.cache()
# def load_model(path: str = 'models/trained_model_resnet50.pt') -> ResnetModel:
#     """Retrieves the trained model and maps it to the CPU by default,
#     can also specify GPU here."""
#     model = ResnetModel(path_to_pretrained_model=path)
#     return model



# @st.cache()
# def predict(
#         img: Image.Image,
#         index_to_label_dict: dict,
#         model,
#         k: int
#         ) -> list:
#     """Transforming input image according to ImageNet paper
#     The Resnet was initially trained on ImageNet dataset
#     and because of the use of transfer learning, I froze all
#     weights and only learned weights on the final layer.
#     The weights of the first layer are still what was
#     used in the ImageNet paper and we need to process
#     the new images just like they did.
#     This function transforms the image accordingly,
#     puts it to the necessary device (cpu by default here),
#     feeds the image through the model getting the output tensor,
#     converts that output tensor to probabilities using Softmax,
#     and then extracts and formats the top k predictions."""
#     formatted_predictions = model.predict_proba(img, k, index_to_label_dict)
#     return formatted_predictions
#     img = batch["image"]
#         pred = model(img).squeeze(0).cpu().detach().numpy()
#         if self.args.save_preds:
#             meta = batch["meta"][0].cpu().detach().numpy()
#             min_d, max_d = meta[0, 0], meta[1, 0]
#             min_h, max_h = meta[0, 1], meta[1, 1]
#             min_w, max_w = meta[0, 2], meta[1, 2]
#             n_class, original_shape, cropped_shape = pred.shape[0], meta[2], meta[3]
#             if not all(cropped_shape == pred.shape[1:]):
#                 resized_pred = np.zeros((n_class, *cropped_shape))
#                 for i in range(n_class):
#                     resized_pred[i] = resize(
#                         pred[i], cropped_shape, order=3, mode="edge", cval=0, clip=True, anti_aliasing=False
#                     )
#                 pred = resized_pred
#             final_pred = np.zeros((n_class, *original_shape))
#             final_pred[:, min_d:max_d, min_h:max_h, min_w:max_w] = pred
#             if self.args.brats:
#                 final_pred = expit(final_pred)
#             elif self.args.no_back_in_output:
#                 final_pred = expit(final_pred)
#             else:
#                 final_pred = softmax(final_pred, axis=0)
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



if __name__ == '__main__':
    st.set_page_config(page_title='Glioma Segmentation', page_icon="🧠")
#     model = load_model()
#     index_to_class_label_dict = load_index_to_label_dict()
#     model = NNUnet.load_from_checkpoint('/mnt/experiments/gbm_results/gbm_4a_resamp/fold-0/checkpoints/best_epoch=197-dice_mean=78.14.ckpt')
    st.title('Welcome To Project 3D brain glioma segmentation in MRI')
    instructions = """
        Choose any image and get the prediction which will be displayed to the screen.
        """
    st.write(instructions)
    
    available_images = SUBJECTS
    st.sidebar.title("What to do")
    image_name = st.sidebar.selectbox("Choose Subject Name", available_images)
    file_img = os.path.join(PATH_TO_IMG, image_name+ '.nii.gz')
    file_pred =  os.path.join(PATH_TO_PREDICT, image_name+ '.npy.npz')

        
    if (file_img is not None) and (file_pred is not None):
        img_orig = nib.load(file_img).get_fdata()
        img_orig *= (255.0/img_orig.max())
#         label_orig = np.load(file_label).transpose(0,3,2,1).squeeze()
        print(img_orig.shape)
        pred = np.load(file_pred)['arr_0']
#         p = np.round(pred, 0).transpose(3,2,1,0).astype(np.uint8)
        p = to_lbl(pred)
#         st.title("Here is the image you've selected")
#         st.write('MRI input images in different modalities')
        col1, col2 = st.columns(2)
        with col1:
#             st.write("FLAIR")
            slice_flair = st.slider('Slice', min_value=0, max_value=143, value = 72)
            flair = img_orig[:, :, slice_flair, 0,np.newaxis].astype(np.uint8)
            st.image(flair,caption=["FLAIR"],clamp=True)
#             st.write("T1")
            slice_t1 = st.slider('Slice', min_value=0, max_value=143, value = 71)
            st.image(img_orig[:, :, slice_t1, 1].astype(np.uint8),caption=["T1"],clamp=True)
#             st.write('Prediction mask')
            slice_predict = st.slider('Slice', min_value=0, max_value=143, value = 69)
            p = img_as_ubyte(exposure.rescale_intensity(p))
            predictions = p[:,:,slice_predict]
            st.image(predictions, caption=['Whole Tumor, Tumor Core, Enhancing Tumor'])

        with col2:
#             st.write("T1CE")
            slice_t1ce = st.slider('Slice', min_value=0, max_value=143, value = 70)
            st.image(img_orig[:, :, slice_t1ce, 2].astype(np.uint8),caption=["T1CE"],clamp=True)
#             st.write("T2")
            slice_t2 = st.slider('Slice', min_value=0, max_value=143, value = 73)
            st.image(img_orig[:, :, slice_t2, 3].astype(np.uint8),caption=["T2"],clamp=True)

        
#         slice_img = st.slider('Slice', min_value=0, max_value=143, value = 72)
#         imgs = [img_orig[:, :, slice_img, i] for i in [0,1,2,3]] 
#         st.image(imgs, clamp=True, caption=["FLAIR", "T1", "T1CE", "T2"], use_column_width = 'auto')
#         st.write('Prediction mask')
#         slice_predict = st.slider('Slice', min_value=0, max_value=143, value = 70)
#         predictions = p[:,:,slice_predict]
#         st.image(predictions, caption=['Whole Tumor, Tumor Core, Enhancing Tumor'])
#         predictions[0][predictions[0]!=0]=255
#         predictions[1][predictions[1]!=0]=255
#         predictions[2][predictions[2]!=0]=255
#         st.write(np.max(mask_pred[0]))
#         img,label = mri_img_load(file_img,file_label)
    #     resized_image = img.resize((336, 336))
#         st.image(imgs[0], clamp=True, caption=["FLAIR", "T1", "T1CE", "T2"], use_column_width = 'auto')
    #     st.title("Here are the five most likely bird species")
    #     df = pd.DataFrame(data=np.zeros((5, 2)),
    #                       columns=['Species', 'Confidence Level'],
    #                       index=np.linspace(1, 5, 5, dtype=int))

    #     for idx, p in enumerate(prediction):
    #         link = 'https://en.wikipedia.org/wiki/' + \
    #             p[0].lower().replace(' ', '_')
    #         df.iloc[idx,
    #                 0] = f'<a href="{link}" target="_blank">{p[0].title()}</a>'
    #         df.iloc[idx, 1] = p[1]
    #     st.write(df.to_html(escape=False), unsafe_allow_html=True)
    #     st.title(f"Here are three other images of the {prediction[0][0]}")

    #     st.image(images_from_s3)
        # st.title('How it works:')