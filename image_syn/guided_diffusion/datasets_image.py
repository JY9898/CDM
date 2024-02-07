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

def load_data_img(
    *,
    data_dir,
    batch_size,
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
        raise ValueError("unspecified data directory")

    all_files = _list_image_files_recursively(data_dir)  

    dataset = ImageDataset(
        all_files,
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


class ImageDataset(Dataset):
    def __init__(
        self,
        image_paths,
        shard=0,
        num_shards=1,
        random_flip=True,
        random_rotate=True,
    ):
        super().__init__()       
        self.local_images = image_paths[shard:][::num_shards]
        self.random_flip = random_flip
        self.random_rotate = random_rotate

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        data = np.load(self.local_images[idx])
        
        arr_image = np.array(data[...,:3])
        arr_image = arr_image.astype(np.float32) / 127.5 - 1
        arr_inst = np.array(data[...,3]).astype(np.float32)
        arr_type = np.array(data[...,4]).astype(np.float32)

        if self.random_flip and random.random() < 0.5:
            flip_direction = np.random.choice([-1, 0, 1])
            arr_image = cv2.flip(arr_image, flip_direction)
            arr_inst = cv2.flip(arr_inst, flip_direction)
            arr_type = cv2.flip(arr_type, flip_direction)

        if self.random_rotate and random.random() < 0.5:
            rot_k = random.choice(range(1, 4))
            arr_image = np.rot90(arr_image, k=rot_k).copy()
            arr_inst = np.rot90(arr_inst, k=rot_k).copy()
            arr_type = np.rot90(arr_type, k=rot_k).copy()

        out_dict = {}
        out_dict['label_ori'] = arr_inst[None, ].copy()
        out_dict['label'] = arr_type[None, ]

        # out_dict['label_ori'].shape  # (256,256)
        # out_dict['label'].shape  # (1, 256, 256)
        # arr_image.shape  # (256, 256, 3)
        return np.transpose(arr_image.astype(np.float32), [2, 0, 1]), out_dict



def load_data_smp(
    *,
    data_dir,
    batch_size,
    deterministic=False,
    random_rotate=True,
    random_flip=True,
):
    if not data_dir:
        raise ValueError("unspecified data directory")

    file_list = sorted(glob.glob(f'{data_dir}/*.npy'))
    dataset = ImageDatasetSmp(
        file_list,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_rotate=random_rotate,
        random_flip=random_flip,
    )
    print("Number of samples:", len(dataset))
    
    # train_sampler = DistributedSampler(dataset)
    # train_loader = DataLoader(dataset=dataset, sampler=train_sampler, batch_size=batch_size,shuffle=False, \
    #                           num_workers=2, pin_memory=True,  drop_last=False)
    # while True:
    #     yield from train_loader
    loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=False
        )
    while True:
        yield from loader

class ImageDatasetSmp(Dataset):
    def __init__(
        self,
        image_paths,
        shard=0,
        num_shards=1,
        random_flip=True,
        random_rotate=True,
    ):
        super().__init__()       
        self.local_images = image_paths[shard:][::num_shards]
        self.random_flip = random_flip
        self.random_rotate = random_rotate

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        img_path = self.local_images[idx]
        basename = pathlib.Path(img_path).stem
        data = np.load(img_path)
        arr_image = data[...,:3]#fillimg: (256, 256, 3) 0 205
        arr_mask = data[...,3]  #fillmask (256, 256) 0 255
        arr_inst = data[...,4]  #inst
        arr_type = data[...,5]  #type
        
        out_dict = {}
        arr_image = arr_image / 127.5 - 1
        arr_mask = arr_mask / 255

        if self.random_rotate and random.random() < 0.5:
            rot_k = random.choice(range(1, 4))
            arr_image = np.rot90(arr_image, k=rot_k).copy()
            arr_mask = np.rot90(arr_mask, k=rot_k).copy()
            arr_inst = np.rot90(arr_inst, k=rot_k).copy()
            arr_type = np.rot90(arr_type, k=rot_k).copy()
            
        if self.random_flip and random.random() < 0.5:
            flip_direction = np.random.choice([-1, 0, 1])
            arr_image = cv2.flip(arr_image, flip_direction)
            arr_mask = cv2.flip(arr_mask, flip_direction)
            arr_inst = cv2.flip(arr_inst, flip_direction)
            arr_type = cv2.flip(arr_type, flip_direction)

        out_dict['name'] = basename
        out_dict['label'] = arr_type[None, ...]
        out_dict['label_ori'] = arr_inst[None, ...]
        mask3 = np.repeat(arr_mask[:, :, None], 3, axis=2)
        out_dict['mask'] = np.transpose(mask3, [2, 0, 1])
        return np.transpose(arr_image, [2, 0, 1]), out_dict