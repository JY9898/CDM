import os
import shutil
from tqdm import tqdm
import sys
sys.path.append('/home/jiyang/formonusec/hover_net/')
import re
import glob
import os
import tqdm
import pathlib
import cv2
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# from dataset import get_dataset

def load_img(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def load_ann(path):
    # assumes that ann is HxW
    data = sio.loadmat(path)
    ann_inst = data["inst_map"]
    ann_type = data["type_map"]
    ann_num = data["nc_num"]
    # print(ann_inst.dtype)
    # print(ann_type.dtype)
    ann = np.dstack([ann_inst, ann_type])
    ann = ann.astype("int32")

    return ann, ann_num

if __name__ == "__main__":

    # data = np.load('/data4/jiyang/MONUSEC/logs/uncondiff_hvt1_rare3/sample_hvt3/3-sdmnpy_forhovernet/00000_2.npy')
    # plt.figure()
    # plt.subplot(1,5,1)
    # plt.imshow(data[...,:3])
    # plt.subplot(1,5,2)
    # plt.imshow(data[...,3])
    # plt.subplot(1,5,3)
    # plt.imshow(data[...,4])
    # plt.savefig(f'rare3_forhovernet_00000_2_test.png')
    # assert False
    # image_path = '/data4/jiyang/MONUSEC/logs/uncondiff_hvt1_rare4/sample_hvt4/2-sdm_halfrepaint_samples_s1.5_200_100/'  # 文件夹1的路径
    # label_path = '/data4/jiyang/MONUSEC/logs/uncondiff_hvt1_rare4/sample_hvt4/1-proc_Mat/'  # 文件夹2的路径
    # image_path = '/data4/jiyang/MONUSEC/logs/uncondiff_hvt1_rare3/sample_hvt3/2-sdm_halfrepaint_samples_s1.5_200_100/'  # 文件夹1的路径
    # label_path = '/data4/jiyang/MONUSEC/logs/uncondiff_hvt1_rare3/sample_hvt3/1-proc_Mat/'  # 文件夹2的路径
    image_path = '/data4/jiyang/MONUSEC/logs/uncondiff_hvt1_rare3/sample_hvt3_large8/1-hvtps_fixtp3_dila50/2-sdm_halfrepaint_samples_s1.5_200_100/'  # 文件夹1的路径
    label_path = '/data4/jiyang/MONUSEC/logs/uncondiff_hvt1_rare3/sample_hvt3_large8/1-hvtps_fixtp3_dila50/1-proc_Mat/'  # 文件夹2的路径


    win_size = [256, 256]
    step_size = [164, 164]#[164, 164]

    dataset_name = "monusec"
    save_root = "/data4/jiyang/MONUSEC/logs/uncondiff_hvt1_rare3/sample_hvt3_large8/1-hvtps_fixtp3_dila50/3-sdmnpy_forhovernet/"

    os.makedirs(save_root, exist_ok=True)

    # file_list = glob.glob(image_path)
    # file_list.sort()  # ensure same ordering across platform
    file_list = sorted(glob.glob(f'{image_path}/*.png'))
    # class_num = 4 # exclude 0
    nuclei_nums = [0]*4
    all_patch = 0
    flag = False
    for file_path in file_list:
        print('file_path:',file_path)
        base_name = pathlib.Path(file_path).stem
        # if "sdm" in base_name:
        #     extracted_name = re.search(r"(.+?)_sdm", base_name).group(1)
        #     flag = True
        # else:
        extracted_name = base_name
        label_path = file_path.replace('2-sdm_halfrepaint_samples_s1.5_200_100','1-proc_Mat')
        label_path = label_path.replace(f'{base_name}.png',f'{extracted_name}.mat')
        print('label_path:',label_path)

        img = load_img(file_path)
        ann, ann_num = load_ann(label_path)
        # print('ann.shape:',ann.shape)  #  (256, 256, 2)
        nuclei_nums+=ann_num[0]
        # print('ann_num[0]:',ann_num[0])
        img = np.concatenate([img, ann], axis=-1)
        img = img.astype("int32")
        # print('img.shape:',img.shape)  # (256, 256, 5)
        # print('nuclei_nums:',nuclei_nums)  # [ 3 10  0 47]
        # plt.figure()
        # plt.subplot(1,5,1)
        # plt.imshow(img[...,:3])
        # plt.subplot(1,5,2)
        # plt.imshow(img[...,3])
        # plt.subplot(1,5,3)
        # plt.imshow(img[...,4])
        # plt.savefig(f'rare3_forhovernet_{base_name}.png')
        
        # if flag:
        #     assert False

        np.save(f"{save_root}/{base_name}.npy", img)
        # assert False
        all_patch += 1 
    print('patch num:',all_patch,'nuclei_nums:',nuclei_nums)

