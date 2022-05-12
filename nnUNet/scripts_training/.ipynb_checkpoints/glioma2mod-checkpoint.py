import json
import os
from glob import glob
from subprocess import call
import time

import nibabel
import numpy as np
from joblib import Parallel, delayed


def load_nifty(directory, example_id, suffix):
    return nibabel.load(os.path.join(directory, suffix + ".nii.gz"))


def load_channels(d, example_id):
    return [load_nifty(d, example_id, suffix) for suffix in ["FL_to_SRI_brain", "T1CE_to_SRI_brain"]]


def get_data(nifty, dtype="int16"):
    if dtype == "int16":
        data = np.abs(nifty.get_fdata().astype(np.int16))
        data[data == -32768] = 0
        return data
    return nifty.get_fdata().astype(np.uint8)


def prepare_nifty(d):
    d_out = os.path.join( '/', d.split("/")[1], d.split("/")[2], d.split("/")[3]  + '_train_GTV_2mod')
    if not os.path.exists(d_out):
        call(f"mkdir {d_out}", shell=True)
    example_id = d.split("/")[-1]
    flair, t1ce = load_channels(d, example_id)
    affine, header = flair.affine, flair.header
    vol = np.stack([get_data(flair), get_data(t1ce)], axis=-1)
    vol = nibabel.nifti1.Nifti1Image(vol, affine, header=header)
#     print('l')
#     print( os.path.join(d_out, example_id + ".nii.gz"))
    nibabel.save(vol, os.path.join(d_out, example_id + ".nii.gz"))

    if os.path.exists(os.path.join(d, "check_gost_gtv_to_ref.nii.gz")):
#         print(os.path.join(d, "check_gost_gtv_to_ref.nii.gz"))
        seg = load_nifty(d, example_id, "check_gost_gtv_to_ref")
        affine, header = seg.affine, seg.header
        vol = get_data(seg, "unit8")
        vol[vol == 4] = 3
        seg = nibabel.nifti1.Nifti1Image(vol, affine, header=header)
#         print(os.path.join(d_out, example_id + "_seg.nii.gz"))
        nibabel.save(seg, os.path.join(d_out, example_id + "_seg.nii.gz"))


def prepare_dirs(data, train):
    d_out = os.path.join( '/', data.split("/")[1], data.split("/")[2], data.split("/")[3]  + '_train_GTV_2mod')
    if not os.path.exists(d_out):
        call(f"mkdir {d_out}", shell=True)
    img_path, lbl_path = os.path.join(d_out, "images"), os.path.join(d_out, "labels")
    call(f"mkdir {img_path}", shell=True)
    if train:
        call(f"mkdir {lbl_path}", shell=True)
    dirs = glob(os.path.join(d_out, "*"))
    for d in dirs:
        if '.nii.gz' in d:
                if "FL_to_SRI_brain" in d or "T1CE_to_SRI_brain" in d:
                    continue
                if "_seg" in d:
                    call(f"mv {d} {lbl_path}", shell=True)
                else:
                    call(f"mv {d} {img_path}", shell=True)
                
#         call(f"rm -rf {d}", shell=True)
         

def prepare_dataset_json(data, train):
    d_out = os.path.join( '/', data.split("/")[1], data.split("/")[2], data.split("/")[3]  + '_train_GTV_2mod')
    images, labels = glob(os.path.join(d_out, "images", "*")), glob(os.path.join(d_out, "labels", "*"))
    images = sorted([img.replace(d_out + "/", "") for img in images])
    labels = sorted([lbl.replace(d_out + "/", "") for lbl in labels])
    
    modality = {"0": "FLAIR", "1": "T1CE"}
    labels_dict = {"1": "gtv"}
    if train:
        key = "training"
        data_pairs = [{"image": img, "label": lbl} for (img, lbl) in zip(images, labels)]
    else:
        key = "test"
        data_pairs = [{"image": img} for img in images]

    dataset = {
        "labels": labels_dict,
        "modality": modality,
        key: data_pairs,
    }

    with open(os.path.join(d_out, "dataset.json"), "w") as outfile:
        json.dump(dataset, outfile)


def run_parallel(func, args):
    return Parallel(n_jobs=os.cpu_count())(delayed(func)(arg) for arg in args)


def prepare_dataset(data, train):
    print(f"Preparing BraTS21 dataset from: {data}")
    start = time.time()
#     run_parallel(prepare_nifty, sorted(glob(os.path.join(data, "*"))))
    for each in sorted(glob(os.path.join(data, "*"))):
        if each.split('/')[-1] in ['604_17', '1332_17', '660_19', '789_15', '117_18_4', '79_18_4',
       '1360_18_4', '1440_17', '20_18_4', '1096_18', '1157_17', '601_17',
       '451_18_4', '171_18_4', '1646_18', '252_18_4', '1177_17',
       '1354_18_4', '1765_18_4', '17p_860', '59_18_4', '1467_17',
       '1575_17', '1028_18_4', '644_19_4', '1555_17', '423_19_4',
       '1391_17', '371_18_4', '135_19_4', '1484_18_4', '1734_18', '269_19_4', '281_19_4', 'Patient_1000114', 'Patient_1001616', 'Patient_110313', 'Patient_59817']:
            continue
        if os.path.exists(os.path.join(each, "check_gost_gtv_to_ref.nii.gz")):
            prepare_nifty(each)
    prepare_dirs(data, train)
    prepare_dataset_json(data, train)
    end = time.time()
    print(f"Preparing time: {(end - start):.2f}")

    
    
if __name__ == "__main__":
    prepare_dataset('/data/private_data/brats_pipeline_out',True)