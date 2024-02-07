"""extract_patches.py
用于monusac的切分

Patch extraction script.
"""
import sys
import re
import glob
import os
import tqdm
import pathlib
import cv2
import csv
import numpy as np
import scipy.io as sio
from PIL import Image
from dataset import get_dataset
from misc.patch_extractor import PatchExtractor
from misc.utils import rm_n_mkdir
from scipy import ndimage
from scipy.ndimage import center_of_mass, measurements
from skimage import morphology as morph

def get_bounding_box(img):
    """Get bounding box coordinate information."""
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    # print(rows,cols)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]

def get_hv(anninst, anntype):
    """Input annotation must be of original shape.

    The map is calculated only for instances within the crop portion
    but based on the original shape in original image.

    Perform following operation:
    Obtain the horizontal and vertical distance maps for each
    nuclear instance.

    """
    orig_ann = crop_ann = anninst.copy()  # instance ID map
    crop_ann = morph.remove_small_objects(crop_ann, min_size=20)

    x_map = np.zeros(orig_ann.shape[:2], dtype=np.float32)
    y_map = np.zeros(orig_ann.shape[:2], dtype=np.float32)
    tp_map = np.zeros(orig_ann.shape[:2], dtype=np.float32)

    inst_list = list(np.unique(crop_ann))
    inst_list.remove(0)  # 0 is background

    for inst_id in inst_list:
        inst_map_bool = orig_ann == inst_id
        inst_map = np.array(inst_map_bool).astype(np.uint8)
        inst_box = get_bounding_box(inst_map)

        inst_map = inst_map[inst_box[0]: inst_box[1], inst_box[2]: inst_box[3]]

        if inst_map.shape[0] < 2 or inst_map.shape[1] < 2:
            continue
        
        inst_contour = cv2.findContours(
            inst_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        inst_contour = np.squeeze(inst_contour[0][0].astype("int32"))
        if inst_contour.shape[0] < 3:
            continue
        if len(inst_contour.shape) != 2:
            continue # ! check for trickery shape
        
        tp_map[inst_map_bool] = anntype[inst_map_bool]
        # nuclei_num[corresponding_tp[k]-1] += 1
        # instance center of mass, rounded to nearest pixel
        inst_com = list(center_of_mass(inst_map))

        inst_com[0] = int(inst_com[0] + 0.5)  # Y
        inst_com[1] = int(inst_com[1] + 0.5)  # X

        inst_x_range = np.arange(1, inst_map.shape[1] + 1)
        inst_y_range = np.arange(1, inst_map.shape[0] + 1)
        # shifting center of pixels grid to instance center of mass
        inst_x_range -= inst_com[1]
        inst_y_range -= inst_com[0]

        inst_x, inst_y = np.meshgrid(inst_x_range, inst_y_range)

        # remove coord outside of instance
        inst_x[inst_map == 0] = 0
        inst_y[inst_map == 0] = 0
        inst_x = inst_x.astype("float32")
        inst_y = inst_y.astype("float32")

        # normalize min into -1 scale
        if np.min(inst_x) < 0:
            inst_x[inst_x < 0] /= -np.amin(inst_x[inst_x < 0])
        if np.min(inst_y) < 0:
            inst_y[inst_y < 0] /= -np.amin(inst_y[inst_y < 0])
        # normalize max into +1 scale
        if np.max(inst_x) > 0:
            inst_x[inst_x > 0] /= np.amax(inst_x[inst_x > 0])
        if np.max(inst_y) > 0:
            inst_y[inst_y > 0] /= np.amax(inst_y[inst_y > 0])

        ####
        x_map_box = x_map[inst_box[0]: inst_box[1], inst_box[2]: inst_box[3]]
        x_map_box[inst_map > 0] = inst_x[inst_map > 0]

        y_map_box = y_map[inst_box[0]: inst_box[1], inst_box[2]: inst_box[3]]
        y_map_box[inst_map > 0] = inst_y[inst_map > 0]

    # hv_map = np.dstack([x_map, y_map]) # (h, w, 2)

    return x_map, y_map, tp_map
# -------------------------------------------------------------------------------------
if __name__ == "__main__":

    model_mode = 'fast'

    win_size = [256, 256]
    step_size = [164, 164]   # [76, 76]
    extract_type = "valid"  # Choose 'mirror' or 'valid'. 'mirror'- use padding at borders. 'valid'- only extract from valid regions.

    # Name of dataset - use CoNSeP, MoNuSAC or PanNuke.
    # This used to get the specific dataset img and ann loading scheme from dataset.py
    dataset_name = "consep"
    save_root = f"/mntcephfs/lab_data/jiyang/datasets/CoNSeP/patch_{win_size[0]}x{win_size[1]}_{step_size[0]}x{step_size[1]}_{extract_type}/"
    os.makedirs(save_root, exist_ok=False)
    rare_class = 1

    # a dictionary to specify where the dataset path should be
    dataset_info = {
        "trainall_patch": {
            "img": (".png", "/mntcephfs/lab_data/jiyang/datasets/CoNSeP/Train_all/Images/"),
            "ann": (".mat", "/mntcephfs/lab_data/jiyang/datasets/CoNSeP/Train_all/Labels/"),
            "tag": "train",
        },
        "test": {
            "img": (".png", "/mntcephfs/lab_data/jiyang/datasets/CoNSeP/Test/Images/"),
            "ann": (".mat", "/mntcephfs/lab_data/jiyang/datasets/CoNSeP/Test/Labels/"),
            "tag": "test",
        },
        
    }

    rare_path = []
    multi_path = []

    patterning = lambda x: re.sub("([\[\]])", "[\\1]", x)
    parser = get_dataset(dataset_name)
    xtractor = PatchExtractor(win_size, step_size)
    for split_name, split_desc in dataset_info.items():
        img_ext, img_dir = split_desc["img"]
        ann_ext, ann_dir = split_desc["ann"]
        tag = split_desc["tag"]

        out_dir_npy = f"{save_root}/{split_name}_npy/"
        file_list = glob.glob(patterning("%s/*%s" % (ann_dir, ann_ext)))
        file_list.sort()  # ensure same ordering across platform
        rm_n_mkdir(out_dir_npy)

        pbar_format = "Process File: |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
        pbarx = tqdm.tqdm(
            total=len(file_list), bar_format=pbar_format, ascii=True, position=0
        )

        for file_idx, file_path in enumerate(file_list):
            base_name = pathlib.Path(file_path).stem

            img = parser.load_img("%s/%s%s" % (img_dir, base_name, img_ext))
            ann = parser.load_ann(
                "%s/%s%s" % (ann_dir, base_name, ann_ext), True
            )
            if img.shape[0]<win_size[0] or img.shape[1]<win_size[1]:
                continue
            # *
            img = np.concatenate([img, ann], axis=-1)
            sub_patches = xtractor.extract(img, extract_type)

            pbar_format = "Extracting  : |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
            pbar = tqdm.tqdm(
                total=len(sub_patches),
                leave=False,
                bar_format=pbar_format,
                ascii=True,
                position=1,
            )

            for idx, patch in enumerate(sub_patches):
                svpath = "{0}/{1}_{2:03d}.npy".format(out_dir_npy, base_name, idx)
                np.save(svpath, patch)
                if tag == "train":
                    _, _, tp_map = get_hv(patch[:,:,3], patch[:,:,4])
                    tp_uq = np.unique(tp_map)
                    if len(tp_uq)>1:
                        # print("multi_save")
                        multi_path.append(svpath)
                        if np.in1d(rare_class, tp_uq):
                            # print("rare_save")
                            rare_path.append(svpath)
                

                pbar.update()
            pbar.close()
            pbarx.update()
        pbarx.close()

    with open(f"{save_root}/rare_{rare_class}.csv", mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['path'])
        writer.writeheader()
        for path in rare_path:
            writer.writerow({'path': path})


    with open(f"{save_root}/multi.csv", mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['path'])
        writer.writeheader()
        for path in multi_path:
            writer.writerow({'path': path})