import os
import math
import random
import skimage
import pathlib
import cv2
import glob
from PIL import Image
import blobfile as bf

from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from scipy import ndimage
from scipy.ndimage import center_of_mass, measurements
from skimage import morphology as morph
import matplotlib.pyplot as plt

def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["npy"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


def load_data_rm(
    *,
    data_dir,
    batch_size,
    rare_class,
    deterministic=False,
    random_rotate=True,
    random_flip=True,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data path")
    
    dataframe = pd.read_csv(data_dir)
    all_files = sorted(dataframe["path"].values) 

    dataset = RMaskDataset(
        all_files,
        rare_class=rare_class,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_rotate=random_rotate,
        random_flip=random_flip,
    )
    print("Len of Dataset:", len(all_files))

    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader

class RMaskDataset(Dataset):
    def __init__(
        self,
        ann_dir,
        rare_class=1,
        shard=0,
        num_shards=1,
        random_flip=True,
        random_rotate=True,
    ):
        super().__init__()
        self.local_labels = ann_dir[shard:][::num_shards]
        self.random_flip = random_flip
        self.random_rotate = random_rotate
        self.rare_class = rare_class

    def __len__(self):
        return len(self.local_labels)

    def __getitem__(self, idx):

        ann_path = self.local_labels[idx]
        
        data = np.load(ann_path)
        ann_inst = (data[..., 3]).astype("int32")
        ann_type = (data[..., 4]).astype("int32")
        ann_type = np.where(ann_type != self.rare_class, 0, 1)
        ann_inst = ann_inst*ann_type

        if self.random_flip and random.random() < 0.5:
            flip_direction = np.random.choice([-1, 0, 1])
            ann_inst = cv2.flip(ann_inst, flip_direction)
            ann_type = cv2.flip(ann_type, flip_direction)

        if self.random_rotate and random.random() < 0.5:
            rot_k = random.choice(range(1, 4))
            ann_inst = np.rot90(ann_inst, k=rot_k).copy()
            ann_type = np.rot90(ann_type, k=rot_k).copy()

        h, v, tp = get_hv(ann_inst, ann_type)
        # tp = (tp == self.rare_class).astype(np.uint8)
        mask = np.dstack((h, v, tp)).astype(np.float32)
        
        out_dict = {}
        return np.transpose(mask, [2, 0, 1]), out_dict



def load_data_mm(
    *,
    data_dir,
    batch_size,
    num_classes,
    deterministic=False,
    random_rotate=True,
    random_flip=True,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    :param data_dir: a dataset directory(already extracted patches, .npy).
    :param batch_size: the batch size of each returned pair.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")

    dataframe = pd.read_csv(data_dir)
    all_files = sorted(dataframe["path"].values) 

    dataset = MtMaskDataset(
        all_files,
        num_classes=num_classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_rotate=random_rotate,
        random_flip=random_flip,
    )
    print("Len of Dataset:", len(all_files))

    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader

class MtMaskDataset(Dataset):
    def __init__(
        self,
        ann_dir,
        num_classes=0,
        shard=0,
        num_shards=1,
        random_flip=True,
        random_rotate=True,
    ):
        super().__init__()
        self.local_labels = ann_dir[shard:][::num_shards]
        self.random_flip = random_flip
        self.random_rotate = random_rotate
        self.num_classes = num_classes

    def __len__(self):
        return len(self.local_labels)

    def __getitem__(self, idx):

        ann_path = self.local_labels[idx]
        # base_name = pathlib.Path(ann_path).stem

        data = np.load(ann_path)
        ann_inst = (data[..., 3]).astype("int32")
        ann_type = (data[..., 4]).astype("int32")

        if self.random_flip and random.random() < 0.5:
            flip_direction = np.random.choice([-1, 0, 1])
            ann_inst = cv2.flip(ann_inst, flip_direction)
            ann_type = cv2.flip(ann_type, flip_direction)

        if self.random_rotate and random.random() < 0.5:
            rot_k = random.choice(range(1, 4))
            ann_inst = np.rot90(ann_inst, k=rot_k).copy()
            ann_type = np.rot90(ann_type, k=rot_k).copy()

        h, v, tp = get_hv(ann_inst, ann_type)
        tp = (np.eye(self.num_classes)[tp.astype(np.uint8)]).astype(np.float32)
        mask = np.concatenate([h[...,None], v[...,None], tp[...,1:]], axis=-1).astype(np.float32)
        
        # tp = (np.eye(self.num_classes)[ann_type]).astype(np.float32)  # float (256,256,5)
        # mask = np.concatenate([hv, tp[:,:,1:]], axis=-1)
        # print(mask.shape)
        # plt.figure(figsize=(14, 6))
        # for j in range(6):
        #     plt.subplot(1, 6, j+1)
        #     plt.imshow(mask[:,:,j])
        #     plt.title(f'[{np.amin(mask[:,:,j]):.2f}, {np.amax(mask[:,:,j]):.2f}]')
        #     plt.axis('off')

        # plt.savefig(f'{base_name}.png')
        # plt.close()
        # assert False
        out_dict = {}
        return np.transpose(mask, [2, 0, 1]), out_dict

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


def load_data_smp(
    *,
    ann_dir,
    batch_size,
    num_classes=5,
    dila_size=5,
    deterministic=False,
    random_flip=False,
    random_rotate=False,
    seed=10,
    rare_class=0,
):
    if not ann_dir:
        raise ValueError("unspecified data directory")
    
    ann_dir = sorted(glob.glob(f'{ann_dir}/*.npy'))
    dataset = MtMaskDatasetSmp(
        ann_dir,
        num_classes=num_classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_flip=random_flip,
        random_rotate=random_rotate,
        rare_class=rare_class,
        dila_size=dila_size,
    )
    print(f'Number of samples: {len(dataset)}')

    if deterministic:
        setup_seed(seed)
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=False
        )
    while True:
        yield from loader

class MtMaskDatasetSmp(Dataset):
    def __init__(
        self,
        ann_dir,
        num_classes=5,
        shard=0,
        num_shards=1,
        random_flip=False,
        random_rotate=False,
        rare_class=0,
        dila_size=5,
    ):
        super().__init__()
        self.local_labels = ann_dir[shard:][::num_shards]
        self.random_flip = random_flip
        self.random_rotate = random_rotate
        self.num_classes = num_classes
        self.dila_size = dila_size
        self.rare_class = rare_class

    def __len__(self):
        return len(self.local_labels)

    def __getitem__(self, idx):

        ann_path = self.local_labels[idx]
        basename = pathlib.Path(ann_path).stem
        data = np.load(ann_path)
        m = data[:,:,2].copy()

        hvtps = np.zeros((data.shape[0], data.shape[1], self.num_classes+1), dtype=np.float32)
        hvtps[...,0:2] = data[...,0:2].copy()
        hvtps[...,self.rare_class+1] = m.copy()

        sele_mask = m.astype(np.uint8)
        expanded_mask = cv2.dilate(sele_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.dila_size, self.dila_size)))
        mask = np.stack([expanded_mask] * (self.num_classes+1), axis=-1)  # or use m
        # print(hvtps.dtype,mask.dtype)
        # print(hvtps.shape,mask.shape)
        # # assert False
        
        # plt.figure(figsize=(14, 6))
        # for i in range(self.num_classes+1):
        #     plt.subplot(2, 3, i+1)
        #     plt.imshow(hvtps[:,:,i])
        #     plt.title(f'[{np.amin(hvtps[:,:,i]):.03f},{np.amax(hvtps[:,:,i]):.03f}]')
        #     plt.axis('off')

        # plt.tight_layout()
        # plt.savefig(f'{basename}_hvtps.png')
        # plt.figure(figsize=(14, 6))
        # for i in range(self.num_classes+1):
        #     plt.subplot(2, 3, i+1)
        #     plt.imshow(mask[:,:,i])
        #     plt.title(f'[{np.amin(mask[:,:,i]):.03f},{np.amax(mask[:,:,i]):.03f}]')
        #     plt.axis('off')

        # plt.tight_layout()
        # plt.savefig(f'{basename}_m.png')

        # assert False
        out_dict = {}
        out_dict['name'] = basename
        out_dict['hvtps'] = np.transpose(hvtps, [2, 0, 1]).astype(np.float32)
        out_dict['mask'] = np.transpose(mask, [2, 0, 1]).astype(np.float32)

        # 
        return out_dict

def setup_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
