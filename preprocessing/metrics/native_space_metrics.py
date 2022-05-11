import os
import sys
import argparse
import numpy as np
import torch
from glob import glob
import pandas as pd
import nibabel as nib    
from pathlib import Path
from surface_distance import metrics
from tqdm import tqdm
import torch.nn as nn
import ants


parser = argparse.ArgumentParser()
parser.add_argument('--path_to_pred', default='/anvar/public_datasets/preproc_study/gbm/inference/2a_interp/', help='path to prediction', type=str)
parser.add_argument('--path_to_target',  default= '/anvar/public_datasets/preproc_study/gbm/inference/labels/', help='path to labels', type=str)
parser.add_argument('--path_to_current_space',  default= '/anvar/public_datasets/preproc_study/gbm/2a_interp/', help='path to labels', type=str)
parser.add_argument('--path_to_native_space',  default= '/anvar/public_datasets/preproc_study/gbm/1_reg/', help='path to native space images', type=str)
parser.add_argument('--savedir', default='/anvar/public_datasets/preproc_study/gbm/inference/native_space/', help='path to csv file with metrics', type=str)
parser.add_argument('--path_to_csv_file', default='/anvar/public_datasets/preproc_study/gbm/inference/metrcis/', help='path to of csv file with metrics', type=str)
parser.add_argument('--namefile', default='2a_interp', help='name of csv file with metrics', type=str)
parser.add_argument('--img_modality', default='CT1.nii.gz', help='native image modality', type=str)

args = parser.parse_args()


def sensitivity_and_specificity(mask_gt, mask_pred):
    """ Computes sensitivity and specificity
     sensitivity  = TP/(TP+FN)
     specificity  = TN/(TN+FP) """
    volume_sum = mask_gt.sum() + mask_pred.sum()
    tp = (mask_gt & mask_pred).sum()
    tn = (~mask_gt & ~mask_pred).sum()
    fp = (~mask_gt & mask_pred).sum()
    fn = (mask_gt & ~mask_pred).sum()
#     TP/(TP+FP) - precision; TP/(TP+FN) - recall
    return tp/(tp+fn), tp/(tp+fp), tn/(tn+fp)


def calculate_metrics_brats(true_mask, pred_mask, ids, spaces):
    """ Takes two file locations as input and validates surface distances.
    Be careful with dimensions of saved `pred` it should be 3D.
    
    """
    
    _columns = ['Ids','Dice_1', 'Dice_2', 'Dice_3',
                'Hausdorff95_1', 'Hausdorff95_2', 'Hausdorff95_3',
                'Sensitivity_1', 'Sensitivity_2', 'Sensitivity_3',
               'Specificity_1', 'Specificity_2', 'Specificity_3',
               'Surface_dice_1', 'Surface_dice_2', 'Surface_dice_3',
               'Precision_1', 'Precision_2', 'Precision_3']
    
    df = pd.DataFrame(columns = _columns)
    df.at[0,'Ids'] = ids
    #class 1
    distances = metrics.compute_surface_distances((true_mask[0,:,:,:]==1), (pred_mask[0,:,:,:]==1), spaces)
    df.at[0,'Dice_1'] = metrics.compute_dice_coefficient((true_mask[0,:,:,:]==1), (pred_mask[0,:,:,:]==1))
    df.at[0,'Surface_dice_1'] = metrics.compute_surface_dice_at_tolerance(distances,1)
    df.at[0,'Hausdorff95_1'] = metrics.compute_robust_hausdorff(distances, 95)
    sens, precision, spec = sensitivity_and_specificity((true_mask[0,:,:,:]==1), (pred_mask[0,:,:,:]==1))
    df.at[0,'Sensitivity_1'] = sens
    df.at[0,'Precision_1'] = precision
    df.at[0,'Specificity_1'] = spec
    #class 2
    distances = metrics.compute_surface_distances((true_mask[1,:,:,:]==1), (pred_mask[1,:,:,:]==1), spaces)
    df.at[0,'Dice_2'] = metrics.compute_dice_coefficient((true_mask[1,:,:,:]==1), (pred_mask[1,:,:,:]==1))
    df.at[0,'Surface_dice_2'] = metrics.compute_surface_dice_at_tolerance(distances,1)
    df.at[0,'Hausdorff95_2'] = metrics.compute_robust_hausdorff(distances, 95)
    sens,precision, spec= sensitivity_and_specificity((true_mask[1,:,:,:]==1), (pred_mask[1,:,:,:]==1))
    df.at[0,'Sensitivity_2'] = sens
    df.at[0,'Precision_2'] = precision
    df.at[0,'Specificity_2'] = spec
    #class 3
    distances = metrics.compute_surface_distances((true_mask[2,:,:,:]==1), (pred_mask[2,:,:,:]==1), spaces)
    df.at[0,'Dice_3'] = metrics.compute_dice_coefficient((true_mask[2,:,:,:]==1), (pred_mask[2,:,:,:]==1))
    df.at[0,'Surface_dice_3'] = metrics.compute_surface_dice_at_tolerance(distances,1)
    df.at[0,'Hausdorff95_3'] = metrics.compute_robust_hausdorff(distances, 95)
    sens, precision, spec= sensitivity_and_specificity((true_mask[2,:,:,:]==1), (pred_mask[2,:,:,:]==1))
    df.at[0,'Sensitivity_3'] = sens
    df.at[0,'Precision_3'] = precision
    df.at[0,'Specificity_3'] = spec
    return df

    
def calculate_metrics(path_to_pred, path_to_target, spaces = [1,1,1], out = '/home/polina/glioma/all_dice_metrics.csv'  ):
    
    """ 
    - path_to_pred - path to folder with predict subjects
    - path_to_target - path to folder with target subjects
    - name_pred - name for prediction, ex -brainTumorMask_SRI.nii.gz
    - name_target - name for targets, ex -GTV_to_SRI.nii.gz
    - spaces - if false - [1,1,1]
    - name_csv - name files for each subjects
    - path_csv_all - path to the main file with metrics for each subjects
    """
    _columns = ['Ids','Dice_1', 'Dice_2', 'Dice_3',
                'Hausdorff95_1', 'Hausdorff95_2', 'Hausdorff95_3',
                'Sensitivity_1', 'Sensitivity_2', 'Sensitivity_3',
               'Specificity_1', 'Specificity_2', 'Specificity_3',
               'Surface_dice_1', 'Surface_dice_2', 'Surface_dice_3',
               'Precision_1', 'Precision_2', 'Precision_3']
    
    af_all = pd.DataFrame(columns = _columns)
    pred_folder = Path(path_to_pred)
    target_folder = Path(path_to_target)
    for ids in tqdm(os.listdir(pred_folder)):
        sub = ids[:-4]
        targets = nib.load(target_folder /  f'{sub}_seg.nii.gz')
        spaces = targets.header.get_zooms()
#         print(spaces)
        targets = targets.get_fdata()
        y_wt, y_tc, y_et = targets > 0, ((targets == 1) + (targets == 3)) > 0, targets == 3
        targets = np.stack([y_wt, y_tc, y_et], axis=0).astype(int)
        predictions = np.load((os.path.join(path_to_pred, ids)))
#         pred = nn.functional.interpolate(torch.from_numpy(predictions), size=tuple([23,  0,  0]), mode="trilinear", align_corners=True)
        pred = np.round(predictions['arr_0'], 0)
        pred = np.transpose(pred, (0, 3, 2, 1))
#         print(targets.shape), print(np.unique(targets))
#         print(pred.shape), print(np.unique(pred))
        df=calculate_metrics_brats(targets.astype('int'), pred.astype('int'), sub, spaces)
#         print(df)
        af_all = af_all.append(df)
    af_all.to_csv(out)  
    print(af_all.mean())
    
    
        
if __name__ == "__main__":
    
    '''
    Usage 
    cd projects/brain-mri-processing-pipeline/main_pipeline/stats
    nohup python /home/kate/projects/brain-mri-processing-pipeline/main_pipeline/stats/native_space_metrics.py --path_to_pred /anvar/public_datasets/preproc_study/gbm/inference/7a_resample_aug_300/ --path_to_current_space /anvar/public_datasets/preproc_study/gbm/7a_resamp/ --namefile 4a_resamp_aug > 4a_resamp_aug.out &
    
    nohup python native_space_metrics.py --path_to_pred /anvar/public_datasets/preproc_study/gbm/inference/3b_n4/ --path_to_current_space /anvar/public_datasets/preproc_study/gbm/3b_n4/ --namefile 3b_n4 > 3b_n4.out &
    
    nohup python native_space_metrics.py --path_to_pred /anvar/public_datasets/preproc_study/gbm/inference/3a_susan/ --path_to_current_space /anvar/public_datasets/preproc_study/gbm/3a_atlas/ --namefile 3d_susan > 3d_susan.out &
    
    nohup python native_space_metrics.py --path_to_pred /anvar/public_datasets/preproc_study/gbm/inference/3a_susan/ --path_to_current_space /anvar/public_datasets/preproc_study/gbm/3a_atlas/ --namefile 3d_susan > 3d_susan.out &
    
    nohup python /home/kate/projects/brain-mri-processing-pipeline/main_pipeline/stats/native_space_metrics.py --path_to_pred /anvar/public_datasets/preproc_study/gbm/inference/3a_atlas_aug_300/ --path_to_current_space /anvar/public_datasets/preproc_study/gbm/3a_atlas/ --namefile 3a_atlas_aug > 3a_atlas_aug.out &
    
    nohup python /home/kate/projects/brain-mri-processing-pipeline/main_pipeline/stats/native_space_metrics.py --path_to_pred /anvar/public_datasets/preproc_study/gbm/inference/6_hist/ --path_to_current_space /anvar/public_datasets/preproc_study/gbm/3a_atlas/ --namefile 6_hist > /home/kate/projects/brain-mri-processing-pipeline/main_pipeline/stats/6_hist.out &
    
    nohup python /home/kate/projects/brain-mri-processing-pipeline/main_pipeline/stats/native_space_metrics.py --path_to_pred /anvar/public_datasets/preproc_study/gbm/inference/5_ss_shared/ --path_to_current_space /anvar/public_datasets/preproc_study/gbm/3a_atlas/ --namefile 5_ss > /home/kate/projects/brain-mri-processing-pipeline/main_pipeline/stats/5_ss.out &
    
    nohup python /home/kate/projects/brain-mri-processing-pipeline/main_pipeline/stats/native_space_metrics.py --path_to_pred /anvar/public_datasets/preproc_study/gbm/inference/5_ss_shared/ --path_to_current_space /anvar/public_datasets/preproc_study/gbm/3a_atlas/ --namefile 5_ss > /home/kate/projects/brain-mri-processing-pipeline/main_pipeline/stats/5_ss.out &
    
    nohup python /home/kate/projects/brain-mri-processing-pipeline/main_pipeline/stats/native_space_metrics.py --path_to_pred /anvar/public_datasets/preproc_study/gbm/inference/3a_susan/ --path_to_current_space /anvar/public_datasets/preproc_study/gbm/3a_atlas/ --namefile 3d_susan > /home/kate/projects/brain-mri-processing-pipeline/main_pipeline/stats/3d_susan.out &
    
    
    '''
    
    
    args = parser.parse_args()
    # Save filename for fold
    infer_path = args.path_to_pred
    old_orig = args.path_to_current_space
    new_orig = args.path_to_native_space
    save_dir = args.savedir 
    labels_native = args.path_to_target
    path_to_csv = args.path_to_csv_file
    save_file = args.namefile
    native_img_modality = args.img_modality
    save_dir = save_dir + '/' + save_file
    
    
    for fold in [0,1,2]:
        print('Fold' , fold, 'experiment ',  save_file)
        cur_folder = save_dir + '/' + 'fold={}'.format(fold)
        cur_csv_file = path_to_csv + '/' + save_file + '_fold-{}'.format(fold) + '.csv'
        for fold_path in os.listdir(infer_path):
            os.makedirs(cur_folder, exist_ok=True)

            if 'fold={}'.format(fold) in fold_path:
                print(fold_path)
                infer_dir = fold_path

                for i in os.listdir(old_orig):
                    if os.path.isdir(old_orig + i):
                            print(old_orig + i)
                            try:
                                # *.npz archives sometimes can ve recognised wrong
                                data = np.load(glob(
                                    infer_path + infer_dir + '/' + i + '*.npz')[0], allow_pickle=True)
                                data = data['arr_0'].transpose(0,3,2,1).astype('float32')

                                old_orig_ct1 = ants.image_read(old_orig + i + '/' + native_img_modality)
                                new_orig_ct1 = ants.image_read(new_orig + i + '/' + native_img_modality)

                                old_like_ch_0 = old_orig_ct1.new_image_like(data[0])
                                old_like_ch_1 = old_orig_ct1.new_image_like(data[1])
                                old_like_ch_2 = old_orig_ct1.new_image_like(data[2])

                                res = ants.registration(fixed=new_orig_ct1, moving=old_orig_ct1,
                                                type_of_transform='Rigid')

                                new_img_0 = ants.apply_transforms(new_orig_ct1, old_like_ch_0,
                                                        transformlist = res['fwdtransforms'][0])
                                new_img_1 = ants.apply_transforms(new_orig_ct1, old_like_ch_1,
                                                        transformlist = res['fwdtransforms'][0])
                                new_img_2 = ants.apply_transforms(new_orig_ct1, old_like_ch_2,
                                                        transformlist = res['fwdtransforms'][0])
                                new_img_shape =  new_img_2.numpy().shape

                                new_array = np.zeros(tuple([3] + list(new_img_shape)), dtype='float16')
                                new_array[0] = new_img_0.numpy()
                                new_array[1] = new_img_1.numpy()
                                new_array[2] = new_img_2.numpy()
                                np.savez(cur_folder + '/' + i + '.npz',new_array.transpose(0,3,2,1).astype('float16'))
                            
                            except Exception as e:
                                print(e)
                                pass
                            
        calculate_metrics(cur_folder, labels_native,  spaces = [1,1,1],  out = cur_csv_file)