"""
Preliminary screening and ranking of hvt1
"""
import csv
import os,glob,argparse, cv2
import pathlib
import blobfile as bf
from matplotlib import pyplot as plt
import numpy as np
import scipy
from scipy import ndimage
from scipy.ndimage import center_of_mass,measurements,binary_fill_holes,label
from skimage.segmentation import watershed
from skimage import morphology as morph
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from tqdm import tqdm

from guided_diffusion import dist_util, logger
from guided_diffusion.fp16_util import MixedPrecisionTrainer
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    create_gaussian_diffusion,
    diffusion_defaults,
)
from guided_diffusion.train_util import parse_resume_step_from_filename, log_loss_dict
from tensorboardX import SummaryWriter  # 导入TensorBoard的SummaryWriter
from torch.utils.data import DataLoader, Dataset
import seaborn as sns

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

    ksize = 7
    ksize2 = 5
    ksize3 = 3
    # blb = np.array(blb_raw >= 0.5, dtype=np.int32)
    blb = blb_raw.astype(np.int32)

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

    overall = np.maximum(sobelh, sobelv)
    overall = overall - (1 - blb)
    overall[overall < 0] = 0
    

    dist = (1.0 - overall) * blb
    ## nuclei values form mountains so inverse to get basins
    dist = -cv2.GaussianBlur(dist, (ksize3, ksize3), 0)
    overall = np.array(overall >= 0.5, dtype=np.int32)
    marker = blb - overall
    marker[marker < 0] = 0
    marker = binary_fill_holes(marker).astype("uint8")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize2, ksize2))
    marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
    marker = label(marker)[0]
    marker = remove_small_objects(marker, min_size=10)
    proced_pred = watershed(dist, markers=marker, mask=blb)
    
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

# -------------------------------------------------------------------------------------
if __name__ == "__main__":
    rootpath = '/mntcephfs/lab_data/jiyang/logs/DiffProj01/uncondiff_hvt1/test_sample_hvt1/'
    data_dir = rootpath + '0-hvt1npy/'
    img_list = sorted(glob.glob(data_dir + '/*.npy'))
    
    thres = 5
    svpath = rootpath + f'0-hvt1npy_proc_large{thres}/'
    os.makedirs(svpath, exist_ok=True)

    n = 0
    for filel in tqdm(img_list):
        basename = pathlib.Path(filel).stem
        data = np.load(filel).copy()
        m = data[:,:,2].copy()
        m[m<0] = 0
        m = np.round(m)
        
        instmap = proc_np_hv_(m, data[:,:,0], data[:,:,1])
        
        h, v, tp, c1_number = get_hv_0(instmap)
        if c1_number <= thres:#class3 set 1
            continue
        n+=1

        # plt.figure(figsize=(8, 4))
        # plt.subplot(1, 3, 1)
        # plt.imshow(h)
        # plt.title(f'h [{np.amin(h):.2f}, {np.amax(h):.2f}]')
        # plt.axis('off')

        # plt.subplot(1, 3, 2)
        # plt.imshow(v)
        # plt.title(f'v [{np.amin(v):.2f}, {np.amax(v):.2f}]')
        # plt.axis('off')

        # plt.subplot(1, 3, 3)
        # plt.imshow(tp)
        # plt.title(f'tp1 [{np.amin(tp):.2f}, {np.amax(tp):.2f}]')
        # plt.axis('off')
        # plt.savefig(f'{basename}.png')
        # plt.close()

        newhvt1 = np.concatenate([h[:,:,None], v[:,:,None], tp[:,:,None]], axis=-1)
        np.save(f'{svpath}/{basename}', newhvt1)

    print(f'saved {n} samples.')

        
