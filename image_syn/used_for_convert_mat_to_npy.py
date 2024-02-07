import os
import shutil
from tqdm import tqdm
import sys
sys.path.append('/home/jiyang/hover_net/')
import re
import glob
import os
import tqdm
import pathlib
import cv2
import numpy as np
import scipy.io as sio


# from dataset import get_dataset

def load_img(path):
    return cv2.imread(path) 

def load_ann(path):
    # assumes that ann is HxW
    ann_inst = sio.loadmat(path)["inst_map"]
    ann_type = sio.loadmat(path)["type_map"]
    print(ann_inst.dtype)
    print(ann_type.dtype)
    ann = np.dstack([ann_inst, ann_type])
    ann = ann.astype("int32")

    return ann

if __name__ == "__main__":
    image_path = '/data/jiyang/1-hvtps_fixtp1-large17-noaug/2-sdm_halfrepaint_samples_s2_1350/'  # 文件夹1的路径
    label_path = '/data/jiyang/1-hvtps_fixtp1-large17-noaug/1-proc_Mat/'  # 文件夹2的路径


    win_size = [256, 256]
    step_size = [164, 164]#[164, 164]
    # extract_type = "mirror"  # Choose 'mirror' or 'valid'. 'mirror'- use padding at borders. 'valid'- only extract from valid regions.

    # Name of dataset - use Kumar, CPM17 or CoNSeP.
    # This used to get the specific dataset img and ann loading scheme from dataset.py
    dataset_name = "consep"
    save_root = "/data/jiyang/1-hvtps_fixtp1-large17-noaug/3-sdmnpy_forhovernet/"

    os.makedirs(save_root, exist_ok=True)

    # file_list = glob.glob(image_path)
    # file_list.sort()  # ensure same ordering across platform
    file_list = sorted(glob.glob(f'{image_path}/*.png'))

    for file_path in file_list:
        print('file_path:',file_path)
        base_name = pathlib.Path(file_path).stem
        label_path = file_path.replace('2-sdm_halfrepaint_samples_s2_1350','1-proc_Mat')
        label_path = label_path.replace('.png','.mat')
        print('label_path:',label_path)

        img = load_img(file_path)
        ann = load_ann(label_path)
        img = np.concatenate([img, ann], axis=-1)


        np.save("{0}/{1}.npy".format(save_root, base_name), img)

