import os
import sys
import argparse
import numpy as np
import pandas as pd
import ants
import nibabel as nib    
from pathlib import Path
from surface_distance import metrics
from tqdm import tqdm
import SimpleITK as sitk



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
    
    _columns = ['Ids', 'Dice_all','Dice_0', 'Dice_1', 'Dice_2', 'Dice_3',
               'Hausdorff95_all', 'Hausdorff95_0', 'Hausdorff95_1', 'Hausdorff95_2', 'Hausdorff95_4',
               'Sensitivity_all','Sensitivity_0', 'Sensitivity_1', 'Sensitivity_2', 'Sensitivity_4',
               'Specificity_all','Specificity_0', 'Specificity_1', 'Specificity_2', 'Specificity_4',
               'Surface_dice_all','Surface_dice_0', 'Surface_dice_1', 'Surface_dice_2', 'Surface_dice_4',
               'Precision_all', 'Precision_0', 'Precision_1', 'Precision_2', 'Precision_4']
    
    df = pd.DataFrame(columns = _columns)
    df.at[0,'Ids'] = ids
    #all classes
    distances = metrics.compute_surface_distances((true_mask != 0), (pred_mask !=0), spaces)
    df.at[0,'Dice_all'] = metrics.compute_dice_coefficient((true_mask != 0), (pred_mask !=0))
    df.at[0,'Surface_dice_all'] = metrics.compute_surface_dice_at_tolerance(distances, 1)
    df.at[0,'Hausdorff95_all'] = metrics.compute_robust_hausdorff(distances, 95)
    sens, precision, spec = sensitivity_and_specificity((true_mask != 0), (pred_mask !=0))
    df.at[0,'Sensitivity_all'] = sens 
    df.at[0,'Precision_all'] = precision
    df.at[0,'Specificity_all'] = spec
    # class 0
    distances = metrics.compute_surface_distances((true_mask == 0), (pred_mask == 0), spaces)
    df.at[0,'Dice_0'] = metrics.compute_dice_coefficient((true_mask == 0), (pred_mask == 0))
    df.at[0,'Surface_dice_0'] = metrics.compute_surface_dice_at_tolerance(distances, 1)
    df.at[0,'Hausdorff95_0'] = metrics.compute_robust_hausdorff(distances, 95)
    sens, precision, spec = sensitivity_and_specificity((true_mask == 0), (pred_mask == 0))
    df.at[0,'Sensitivity_0'] = sens 
    df.at[0,'Precision_0'] = precision 
    df.at[0,'Specificity_0'] = spec
    #class 1
    distances = metrics.compute_surface_distances((true_mask == 1), (pred_mask == 1), spaces)
    df.at[0,'Dice_1'] = metrics.compute_dice_coefficient((true_mask == 1), (pred_mask == 1))
    df.at[0,'Surface_dice_1'] = metrics.compute_surface_dice_at_tolerance(distances,1)
    df.at[0,'Hausdorff95_1'] = metrics.compute_robust_hausdorff(distances, 95)
    sens, precision, spec = sensitivity_and_specificity((true_mask == 1), (pred_mask == 1))
    df.at[0,'Sensitivity_1'] = sens
    df.at[0,'Precision_1'] = precision
    df.at[0,'Specificity_1'] = spec
    #class 2
    distances = metrics.compute_surface_distances((true_mask == 1), (pred_mask == 2), spaces)
    df.at[0,'Dice_2'] = metrics.compute_dice_coefficient((true_mask == 1), (pred_mask == 2))
    df.at[0,'Surface_dice_2'] = metrics.compute_surface_dice_at_tolerance(distances,1)
    df.at[0,'Hausdorff95_2'] = metrics.compute_robust_hausdorff(distances, 95)
    sens,precision, spec= sensitivity_and_specificity((true_mask == 1), (pred_mask == 2))
    df.at[0,'Sensitivity_2'] = sens
    df.at[0,'Precision_2'] = precision
    df.at[0,'Specificity_2'] = spec
    #class 3
    distances = metrics.compute_surface_distances((true_mask == 1), (pred_mask == 4), spaces)
    df.at[0,'Dice_4'] = metrics.compute_dice_coefficient((true_mask == 1), (pred_mask == 4))
    df.at[0,'Surface_dice_4'] = metrics.compute_surface_dice_at_tolerance(distances,1)
    df.at[0,'Hausdorff95_4'] = metrics.compute_robust_hausdorff(distances, 95)
    sens, precision, spec= sensitivity_and_specificity((true_mask == 1), (pred_mask == 4))
    df.at[0,'Sensitivity_4'] = sens
    df.at[0,'Precision_4'] = precision
    df.at[0,'Specificity_4'] = spec
    return df


def calculate_metrics(path_to_pred, path_to_target, name_pred, name_target, spaces = True, name_csv='dice_metrics.csv', path_csv_all = '/home/polina/glioma/all_dice_metrics.csv'  ):
    
    """ 
    - path_to_pred - path to folder with predict subjects
    - path_to_target - path to folder with target subjects
    - name_pred - name for prediction, ex -brainTumorMask_SRI.nii.gz
    - name_target - name for targets, ex -GTV_to_SRI.nii.gz
    - spaces - if false - [1,1,1]
    - name_csv - name files for each subjects
    - path_csv_all - path to the main file with metrics for each subjects
    """
    df=pd.DataFrame()
    _columns = ['Ids', 'Dice_all','Dice_0', 'Dice_1', 'Dice_2', 'Dice_3',
               'Hausdorff95_all', 'Hausdorff95_0', 'Hausdorff95_1', 'Hausdorff95_2', 'Hausdorff95_4',
               'Sensitivity_all','Sensitivity_0', 'Sensitivity_1', 'Sensitivity_2', 'Sensitivity_4',
               'Specificity_all','Specificity_0', 'Specificity_1', 'Specificity_2', 'Specificity_4',
               'Surface_dice_all','Surface_dice_0', 'Surface_dice_1', 'Surface_dice_2', 'Surface_dice_4',
               'Precision_all', 'Precision_0', 'Precision_1', 'Precision_2', 'Precision_4']
    
    af_all = pd.DataFrame(columns = _columns)
    pred_folder = Path(path_to_pred)
    target_folder = Path(path_to_lab)
    num = 0
    not_ref = 0
    not_mask = []
    for ids in tqdm(os.listdir(pred_folder)):
        if not os.path.isfile(Path(pred_folder / ids / name_pred)):
            not_ref+=1
            continue
            
        if not os.path.isfile(pred_folder / ids / name_csv):
                if not os.path.isfile(target_folder / ids / name_target):
                    not_mask.append(ids)
                    continue
                targets = nib.load(target_folder / ids / name_target).get_fdata()
                predictions = nib.load(pred_folder / ids / name_pred)
                if spaces:
                    spaces = predictions.header.get_zooms()
                else:
                    spaces = [1,1,1]
                predictions = predictions.get_fdata()
                assert(targets.shape == predictions.shape)
                pred = np.round(predictions, 0)
                df=calculate_metrics_brats(targets.astype('int'), pred.astype('int'), ids, spaces)
                df.to_csv(pred_folder / ids / name_csv)
                af_all = af_all.append(df)
                num+=1
        else:
            df= pd.read_csv(pred_folder / ids / name_csv)
            af_all = af_all.append(df)
            num+=1

    af_all.to_csv(path_csv_all)  
    print(num)
    print(not_ref)
    print(not_mask)
    