# from ...hovernet_utils.post_proc import get_instance_map
import csv, os, glob, argparse, cv2, pathlib
import shutil
import random
import blobfile as bf

import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
import seaborn as sns
import scipy.io as sio
import pandas as pd

from scipy import ndimage
from scipy.ndimage import center_of_mass,measurements,binary_fill_holes,label
from skimage.segmentation import watershed
from torch.optim import AdamW
from tqdm import tqdm
from skimage import morphology as morph

from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from matplotlib import pyplot as plt
import warnings
# colors = {
#     # "0" : ["nolabe", [0  ,   0,   0]],  #bgr版
#     "0" : ["no-neo", [128, 128,   128]],
#     "1" : ["epithelial", [255  ,   165, 0]],
#     "2" : ["lymphocytes", [0,   0,   255]], 
#     "3" : ["macrophages", [0, 255,   255]], 
#     "4" : ["neutrophils", [0, 165,   255]], 
# }
colors = {
    "0" : ["no-neo", [128, 128,   128]],
    "1" : ["miscellaneous", [0, 255,   255]], #yellow bgr版
    "2" : ["inflammatory", [0,   0,   255]], #red
    "3" : ["epithelial", [255  ,   165, 0]], #blue
    "4" : ["spindle-shaped", [0  , 255,   0]], #green
}
type_info_dict = {
    int(k): (v[0], tuple(v[1])) for k, v in colors.items()
}

def noop(*args, **kargs):
    pass


warnings.warn = noop


def remove_small_objects(pred, min_size=64, connectivity=1):
    """Remove connected components smaller than the specified size.

    This function is taken from skimage.morphology.remove_small_objects, but the warning
    is removed when a single label is provided. 

    Args:
        pred: input labelled array
        min_size: minimum size of instance in output array
        connectivity: The connectivity defining the neighborhood of a pixel. 
    
    Returns:
        out: output array with instances removed under min_size

    """
    out = pred

    if min_size == 0:  # shortcut for efficiency
        return out

    if out.dtype == bool:
        selem = ndimage.generate_binary_structure(pred.ndim, connectivity)
        ccs = np.zeros_like(pred, dtype=np.int32)
        ndimage.label(pred, selem, output=ccs)
    else:
        ccs = out

    try:
        component_sizes = np.bincount(ccs.ravel())
    except ValueError:
        raise ValueError(
            "Negative value labels are not supported. Try "
            "relabeling the input with `scipy.ndimage.label` or "
            "`skimage.morphology.label`."
        )

    too_small = component_sizes < min_size
    too_small_mask = too_small[ccs]
    out[too_small_mask] = 0

    return out


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

                        
def proc_np_hv_(blb_raw, h_dir_raw, v_dir_raw):

    ksize = 21 # or 7
    ksize2 = 5
    ksize3 = 3
    blb = np.array(blb_raw >= 0.5, dtype=np.int32)
    # blb = blb_raw.astype(np.int32)

    blb = label(blb)[0]
    blb = remove_small_objects(blb, min_size=20)
    blb[blb > 0] = 1  # background is 0 already

    h_dir = cv2.normalize(
        h_dir_raw, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )
    v_dir = cv2.normalize(
        v_dir_raw, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )

    sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=ksize)
    sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=ksize)

    sobelh = 1 - (
        cv2.normalize(
            sobelh, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
        )
    )
    sobelv = 1 - (
        cv2.normalize(
            sobelv, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
        )
    )
    # h_dir = h_dir_raw
    # v_dir = v_dir_raw

    # sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=ksize)
    # sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=ksize)
    # sobelh = 1 - sobelh
    # sobelv = 1 - sobelv

    overall = np.maximum(sobelh, sobelv)
    # plt.figure(figsize=(6, 6))
    # plt.imshow(overall)
    # plt.savefig('overall00.png')
    overall = overall - (1 - blb)
    overall[overall < 0] = 0
    # plt.figure(figsize=(6, 6))
    # plt.imshow(overall)
    # plt.savefig('overall01.png')
    

    dist = (1.0 - overall) * blb
    ## nuclei values form mountains so inverse to get basins
    dist = -cv2.GaussianBlur(dist, (ksize3, ksize3), 0)
    # plt.figure(figsize=(6, 6))
    # plt.imshow(dist)
    # plt.savefig('dist.png')
    overall = np.array(overall >= 0.4, dtype=np.int32)
    # plt.figure(figsize=(6, 6))
    # plt.imshow(overall)
    # plt.savefig('overall02.png')
    marker = blb - overall
    marker[marker < 0] = 0
    marker = binary_fill_holes(marker).astype("uint8")
    # plt.figure(figsize=(6, 6))
    # plt.imshow(marker)
    # plt.savefig('marker00.png')
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize2, ksize2))
    marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
    # plt.figure(figsize=(6, 6))
    # plt.imshow(marker)
    # plt.savefig('marker01.png')
    marker = label(marker)[0]
    marker = remove_small_objects(marker, min_size=10)
    proced_pred = watershed(dist, markers=marker, mask=blb)
    # plt.figure(figsize=(6, 6))
    # plt.imshow(proced_pred)
    # plt.savefig('proced_pred.png')
    # assert False
    return proced_pred

def get_hv_0(anninst):
    """Input annotation must be of original shape.

    The map is calculated only for instances within the crop portion
    but based on the original shape in original image.

    Perform following operation:
    Obtain the horizontal and vertical distance maps for each
    nuclear instance.

    """
    tp_map = np.zeros(anninst.shape)

    orig_ann = crop_ann = anninst.copy()  # instance ID map
    crop_ann = morph.remove_small_objects(crop_ann, min_size=20)

    x_map = np.zeros(orig_ann.shape[:2], dtype=np.float32)
    y_map = np.zeros(orig_ann.shape[:2], dtype=np.float32)

    nuclei_num = 0

    inst_id_list = np.unique(crop_ann.flatten())[1:]
    for k,inst_id in enumerate(inst_id_list):
        inst_map_bool = orig_ann == inst_id
        inst_map = inst_map_bool.copy().astype(np.uint8)
        inst_box = get_bounding_box(inst_map)  # rmin, rmax, cmin, cmax

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
        
        tp_map[inst_map_bool] = 1
        nuclei_num += 1

        # instance center of mass, rounded to nearest pixel
        inst_com = list(center_of_mass(inst_map))  # 行，列  Y，X

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

    return x_map, y_map, tp_map, nuclei_num

def img_norm1_u8(img):
    img = cv2.normalize(img.astype('float32'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    img = (img * 255).astype('uint8')
    return img

def process(pred_inst):
    """Post processing script for image tiles.

    Args:
        pred_map: commbined output of tp, np and hv branches, in the same order
        nr_types: number of types considered at output of nc branch
        overlaid_img: img to overlay the predicted instances upon, `None` means no
        type_colour (dict) : `None` to use random, else overlay instances of a type to colour in the dict
        output_dtype: data type of output
    
    Returns:
        pred_inst:     pixel-wise nuclear instance segmentation prediction
        pred_type_out: pixel-wise nuclear type prediction 

    """
    inst_info_dict = None
    # if return_centroids or nr_types is not None:
    inst_id_list = np.unique(pred_inst)[1:]  # exlcude background
    # print('inst_id_list in process:',inst_id_list)
    inst_info_dict = {}
    k=0
    for inst_id in inst_id_list:
        inst_map = pred_inst == inst_id
        # TODO: chane format of bbox output
        rmin, rmax, cmin, cmax = get_bounding_box(inst_map)
        inst_bbox = np.array([[rmin, cmin], [rmax, cmax]])
        inst_map = inst_map[
            inst_bbox[0][0] : inst_bbox[1][0], inst_bbox[0][1] : inst_bbox[1][1]
        ]
        inst_map = inst_map.astype(np.uint8)
        inst_moment = cv2.moments(inst_map)
        inst_contour = cv2.findContours(
            inst_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        # * opencv protocol format may break
        inst_contour = np.squeeze(inst_contour[0][0].astype("int32"))
        # < 3 points dont make a contour, so skip, likely artifact too
        # as the contours obtained via approximation => too small or sthg
        if inst_contour.shape[0] < 3:
            # print('continue 1:',inst_id)
            continue
        if len(inst_contour.shape) != 2:
            # print('continue 2:',inst_id)
            continue # ! check for trickery shape.
        k = k+1
        inst_centroid = [
            (inst_moment["m10"] / inst_moment["m00"]),
            (inst_moment["m01"] / inst_moment["m00"]),
        ]
        inst_centroid = np.array(inst_centroid)
        inst_contour[:, 0] += inst_bbox[0][1]  # X
        inst_contour[:, 1] += inst_bbox[0][0]  # Y
        inst_centroid[0] += inst_bbox[0][1]  # X
        inst_centroid[1] += inst_bbox[0][0]  # Y
        inst_info_dict[inst_id] = {  # inst_id should start at 1
            "bbox": inst_bbox,
            "centroid": inst_centroid,
            "contour": inst_contour,
        }
    # print('k=',k)

    return inst_info_dict

def get_hvt1_ori_instmap(path):

    data = np.load(path)
    m = data[:,:,2]
    pred = np.concatenate([m[:,:,None], data[:,:,0:2].copy()], axis=-1)
    instmap = proc_np_hv(pred).astype(np.int32)
    return instmap
def find_first_non_consecutive(arr):
    for i in range(1, len(arr)):
        if arr[i] != arr[i-1] + 1:
            return arr[i-1] + 1
    return arr[-1] + 1  # 如果数组是连续的情况

def get_instmap_based_instori(instori,insthvtps):
    new_instmap = insthvtps.copy()
    new_instmap[instori>0] = 0
    # unique_insthvtps_remove_tp1 = insthvtps_remove_tp1.copy()
    instoriuq = np.unique(instori)[1:]
    for i in instoriuq:
        unique_insthvtps_remove_tp1 = np.unique(new_instmap)
        instmask1 = instori==i
        new_instmap[instmask1] = find_first_non_consecutive(unique_insthvtps_remove_tp1)

    return new_instmap

def get_available_filename(base_filename, save_path, extension):
    filename = os.path.join(save_path, f"{base_filename}.{extension}")
    index = 1
    
    while os.path.exists(filename):
        filename = os.path.join(save_path, f"{base_filename}_proc{index}.{extension}")
        index += 1
    
    return pathlib.Path(filename).stem

def get_available_name(base_name, namedict):
    # filename = os.path.join(save_path, f"{base_filename}.{extension}")
    index = 1
    filename = base_name
    while filename in namedict:
        filename = f"{base_name}_s{index}"
        index += 1
    
    return filename

def get_c1fill_img_mask(inst_map, type_map, allnctemps,rareclass):# allnctemps为rgb版
    inst_info_dict_after = process(inst_map)
    c1_inst = inst_map.copy()
    c1_inst[type_map!=rareclass] = 0
    unique_c1 = np.unique(c1_inst)[1:]
    random_select_cur = random.choice(unique_c1)
    instance_mask = (c1_inst == random_select_cur)

    selected_img, selected_mask, _ = random.choice(allnctemps)

    sele_inst_center = inst_info_dict_after[random_select_cur]["centroid"]
    img_temp = selected_img
    mask_temp = selected_mask

    fill_img = np.zeros((type_map.shape[0],type_map.shape[1],3))
    fill_mask = np.zeros((type_map.shape[0],type_map.shape[1]))
    x = int(np.round(sele_inst_center[0]))
    y = int(np.round(sele_inst_center[1]))
    smallimg_height, smallimg_width, _ = img_temp.shape
    smallimg_center_x = smallimg_width // 2
    smallimg_center_y = smallimg_height // 2
    start_x = x - smallimg_center_x
    start_y = y - smallimg_center_y
    end_x = start_x + smallimg_width
    end_y = start_y + smallimg_height
    if start_x < 0:
        start_x = 0
    if start_y < 0:
        start_y = 0
    if end_x > type_map.shape[1]:
        end_x = type_map.shape[1]
    if end_y > type_map.shape[0]:
        end_y = type_map.shape[0]
    # print(start_y,y,smallimg_center_y, end_y)
    # print(start_x,x,smallimg_center_x, end_x)
    fill_img[start_y:end_y, start_x:end_x,:] = img_temp[
        start_y - (y - smallimg_center_y) : end_y - (y - smallimg_center_y),
        start_x - (x - smallimg_center_x) : end_x - (x - smallimg_center_x),:
    ]
    fill_mask[start_y:end_y, start_x:end_x] = mask_temp[
        start_y - (y - smallimg_center_y) : end_y - (y - smallimg_center_y),
        start_x - (x - smallimg_center_x) : end_x - (x - smallimg_center_x)
    ]
    # fill_mask = np.logical_and(fill_mask!=0, instance_mask).astype(np.uint8) * 255
    # print(np.amin(fill_img),np.amax(fill_img))
    fill_mask[~instance_mask] = 0
    # cv2.imwrite(f'{fillmask_path}/{name}.png',fill_mask) #0-255
    fill_img[~(fill_mask.astype(np.bool_))] = 0
    # cv2.imwrite(f'{fillimg_path}/{name}.png',cv2.cvtColor(fill_img.astype(np.uint8),cv2.COLOR_RGB2BGR))

    return fill_img, fill_mask

def load_img(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
# 
def load_ann(path, rareclass):
    # assumes that ann is HxW
    ann = np.load(path)
    ann = ann.astype("int32")
    anninst = ann[:,:,0].copy()
    anntype = ann[:,:,1].copy()
    mask = (anntype==rareclass).astype(np.uint8)
    anninst = anninst*mask
    minor_class_idx = np.unique(anninst)[1:]

    return ann, minor_class_idx

def load_ann_consep(path, rareclass):
    # assumes that ann is HxW
    annload = sio.loadmat(path)
    ann_inst = annload["inst_map"]
    ann_type = annload["type_map"]
    ann_type[(ann_type == 3) | (ann_type == 4)] = 3
    ann_type[(ann_type == 5) | (ann_type == 6) | (ann_type == 7)] = 4

    mask = (ann_type==rareclass).astype(np.uint8)
    ann_inst = ann_inst*mask
    minor_class_idx = np.unique(ann_inst)[1:]
    ann = np.dstack([ann_inst, ann_type]).astype("int32")
    return ann, minor_class_idx

    


if __name__ == "__main__":
    rareclass = 1
    data_dir = '/mntcephfs/lab_data/jiyang/datasets/CoNSeP/Train/'
    imgs_dir = os.path.join(data_dir,'Images')
    labels_dir = os.path.join(data_dir,'Labels')
    allnctemps = []

    img_list = glob.glob(imgs_dir + '/*.png')
    for imgpath in tqdm(img_list):
        basename = pathlib.Path(imgpath).stem
        img = load_img(imgpath)
        if img.shape[0]<256 or img.shape[1]<256:
            continue
        annpath = labels_dir + '/' + basename + '.mat'
        ann, minor_class_idx = load_ann_consep(annpath, rareclass)
        print('minor_class_idx:',minor_class_idx)
        ann_inst = ann[:,:,0]
        ann_type = ann[:,:,1]
        for i in minor_class_idx:
            inst_id = i# + 1
            inst_map_i = (ann_inst == inst_id).astype(np.uint8)
            rmin, rmax, cmin, cmax = get_bounding_box(inst_map_i)
            inst_bbox = np.array([[rmin, cmin], [rmax, cmax]])
            subimg = img[
                inst_bbox[0][0] : inst_bbox[1][0], inst_bbox[0][1] : inst_bbox[1][1]
            ].copy()
            submask = inst_map_i[
                inst_bbox[0][0] : inst_bbox[1][0], inst_bbox[0][1] : inst_bbox[1][1]
            ].copy()*255
            if subimg.shape[0]<10 or subimg.shape[1]<10:
                continue
            allnctemps.append((subimg,submask,np.sum(inst_map_i)))
            
    dir_path = '/mntcephfs/lab_data/jiyang/logs/DiffProj01/uncondiff_hvt1/test_sample_hvt1/1-hvtps_fixtp1_dila5/1-hvtpsnpy/'
    file_list = glob.glob("%s/*.npy" % dir_path)
    file_list.sort()

    num_class = 5
    thres = 5
    
    svdir = f'/mntcephfs/lab_data/jiyang/logs/DiffProj01/uncondiff_hvt1/test_sample_hvt1/1-hvtpsnpy_proc_large{thres}/'

    show_path = svdir + '1-hvtps_procshow/'
    type_path = svdir + '1-hvtpsnpy_proc/' 
    # figshow_path = svdir + '1-figshow/'
    os.makedirs(show_path, exist_ok=True)
    os.makedirs(type_path, exist_ok=True)
    # os.makedirs(figshow_path, exist_ok=True)

    csv_file_path = f"{svdir}/1-proc_info_with_sample_nucleinum.csv"
    csvfile = open(f"{csv_file_path}", mode='w', newline='')
    writer = csv.writer(csvfile)
    writer.writerow(["basename", "num01", "num02", "num03", "num04"])
    allnum_dict = {}
    samples = []
    sample_num = 0
    rare_num = 0
    
    for filepath in tqdm(file_list):
        # loaded_image_dict = np.load(filepath)  #np.load(filepath, allow_pickle=True).item()
        name = pathlib.Path(filepath).stem
        
        hvtpssamp = np.load(filepath)
        hv = hvtpssamp[:,:,0:2].copy()
        
        plt.figure()
        chd = hvtpssamp.shape[2]
        for channelid in range(chd):
            plt.subplot(1,chd,channelid+1)
            plt.imshow(hvtpssamp[...,channelid])
        plt.savefig(f'{figshow_path}/{name}.png')
        plt.close()
        
        mask = np.zeros((hvtpssamp.shape[0],hvtpssamp.shape[1]))
        for i in range(num_class,1,-1):
            tp = np.round(hvtpssamp[:,:,i])
            tp[tp<0]=0
            tp_mask = tp>0
            mask[tp_mask] = tp[tp_mask]*(i-1)

        inst_map = proc_np_hv_(mask, hv[...,0], hv[...,1]).astype(np.int32)

        type_map = (inst_map>0)*mask
        type_map = type_map.astype(np.uint8)
        nuclei_nums = [len(np.unique((type_map==i)*inst_map))-1 for i in range(1,num_class)]

        if nuclei_nums[rareclass-1] > thres:
            rare_num += nuclei_nums[rareclass-1]
            writer.writerow([name, nuclei_nums[0], nuclei_nums[1], nuclei_nums[2], nuclei_nums[3]])
            
            # newname = get_available_filename(name, show_path, 'png')

            hvtps_show_after = np.zeros((type_map.shape[0], type_map.shape[1], 3), dtype=np.uint8)
            for i in range(num_class):
                predictedmask = type_map==i
                hvtps_show_after[predictedmask] = type_info_dict[i][1]

            inst_info_dict_after = process(inst_map)
            for idx, [inst_id, inst_info] in enumerate(inst_info_dict_after.items()):
                inst_contour = inst_info["contour"]
                cv2.drawContours(hvtps_show_after, [inst_contour], -1, [0, 0, 0], 1)

            cv2.imwrite(f'{show_path}/{name}.png', hvtps_show_after)  # bgr

            fill_img, fill_mask = get_c1fill_img_mask(inst_map, type_map, allnctemps, rareclass)
            svsample = np.concatenate([fill_img, fill_mask[...,None], inst_map[...,None], type_map[...,None]], axis=-1)  #img[0-255],mask[0,255],inst_map[0-id],type_map[0,1,2,3,4]
            np.save(f'{type_path}/{name}', svsample)

            sample_num += 1
            
        print(f'{sample_num} samples, {rare_num} rare nucleis')
    csvfile.close()
