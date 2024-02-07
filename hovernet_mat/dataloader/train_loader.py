import csv
import glob
import os
import pathlib
import random
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch.utils.data

import imgaug as ia
from imgaug import augmenters as iaa
from misc.utils import cropping_center

from .augs import (
    add_to_brightness,
    add_to_contrast,
    add_to_hue,
    add_to_saturation,
    gaussian_blur,
    median_blur,
)


####
class FileLoader(torch.utils.data.Dataset):
    """Data Loader. Loads images from a file list and 
    performs augmentation with the albumentation library.
    After augmentation, horizontal and vertical maps are 
    generated.

    Args:
        file_list: list of filenames to load
        input_shape: shape of the input [h,w] - defined in config.py
        mask_shape: shape of the output [h,w] - defined in config.py
        mode: 'train' or 'valid'
        
    """

    # TODO: doc string

    def __init__(
        self,
        file_list,
        with_type=False,  #True
        input_shape=None,  #[270,270]
        mask_shape=None,  #[80,80]
        mode="train",
        setup_augmentor=True,  # False
        target_gen=None,  #
    ):
        assert input_shape is not None and mask_shape is not None
        self.mode = mode
        self.info_list = file_list
        self.with_type = with_type
        self.mask_shape = mask_shape
        self.input_shape = input_shape
        self.id = 0
        self.target_gen_func = target_gen[0]
        self.target_gen_kwargs = target_gen[1]
        if setup_augmentor:
            self.setup_augmentor(0, 0)
        return

    def setup_augmentor(self, worker_id, seed):
        self.augmentor = self.__get_augmentation(self.mode, seed)
        self.shape_augs = iaa.Sequential(self.augmentor[0])
        self.input_augs = iaa.Sequential(self.augmentor[1])
        self.id = self.id + worker_id
        return

    def __len__(self):
        return len(self.info_list)

    def __getitem__(self, idx):
        path = self.info_list[idx]
        data = np.load(path)

        # split stacked channel into image and label
        img = (data[..., :3]).astype("uint8")  # RGB images
        ann = (data[..., 3:]).astype("int32")  # instance ID map and type map

        if self.shape_augs is not None:
            shape_augs = self.shape_augs.to_deterministic()
            img = shape_augs.augment_image(img)
            ann = shape_augs.augment_image(ann)

        if self.input_augs is not None:
            input_augs = self.input_augs.to_deterministic()
            img = input_augs.augment_image(img)

        img = cropping_center(img, self.input_shape)
        feed_dict = {"img": img}
        inst_map = ann[..., 0]  # HW1 -> HW
        # print(ann.shape,inst_map.shape,self.with_type)

        if self.with_type:
            type_map = (ann[..., 1]).copy()
            type_map = cropping_center(type_map, self.mask_shape)
            #type_map[type_map == 5] = 1  # merge neoplastic and non-neoplastic
            feed_dict["tp_map"] = type_map

        # TODO: document hard coded assumption about #input
        target_dict = self.target_gen_func(
            inst_map, self.mask_shape, **self.target_gen_kwargs
        )
        feed_dict.update(target_dict)

        return feed_dict

    def __get_augmentation(self, mode, rng):
        if mode == "train":
            shape_augs = [
                # * order = ``0`` -> ``cv2.INTER_NEAREST``
                # * order = ``1`` -> ``cv2.INTER_LINEAR``
                # * order = ``2`` -> ``cv2.INTER_CUBIC``
                # * order = ``3`` -> ``cv2.INTER_CUBIC``
                # * order = ``4`` -> ``cv2.INTER_CUBIC``
                # ! for pannuke v0, no rotation or translation, just flip to avoid mirror padding
                iaa.Affine(
                    # scale images to 80-120% of their size, individually per axis
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    # translate by -A to +A percent (per axis)
                    translate_percent={"x": (-0.01, 0.01), "y": (-0.01, 0.01)},
                    shear=(-5, 5),  # shear by -5 to +5 degrees
                    rotate=(-179, 179),  # rotate by -179 to +179 degrees
                    order=0,  # use nearest neighbour
                    backend="cv2",  # opencv for fast processing
                    seed=rng,
                ),
                # set position to 'center' for center crop
                # else 'uniform' for random crop
                iaa.CropToFixedSize(
                    self.input_shape[0], self.input_shape[1], position="center"
                ),
                iaa.Fliplr(0.5, seed=rng),
                iaa.Flipud(0.5, seed=rng),
            ]

            input_augs = [
                iaa.OneOf(
                    [
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: gaussian_blur(*args, max_ksize=3),
                        ),
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: median_blur(*args, max_ksize=3),
                        ),
                        iaa.AdditiveGaussianNoise(
                            loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                        ),
                    ]
                ),
                iaa.Sequential(
                    [
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: add_to_hue(*args, range=(-8, 8)),
                        ),
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: add_to_saturation(
                                *args, range=(-0.2, 0.2)
                            ),
                        ),
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: add_to_brightness(
                                *args, range=(-26, 26)
                            ),
                        ),
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: add_to_contrast(
                                *args, range=(0.75, 1.25)
                            ),
                        ),
                    ],
                    random_order=True,
                ),
            ]
        elif mode == "valid":
            shape_augs = [
                # set position to 'center' for center crop
                # else 'uniform' for random crop
                iaa.CropToFixedSize(
                    self.input_shape[0], self.input_shape[1], position="center"
                )
            ]
            input_augs = []

        return shape_augs, input_augs
    
def get_subignore_mask(instmap, positions):
    instmap = instmap.copy()
    positions = positions.copy()
    rmin, rmax, cmin, cmax = positions
    cut_area = instmap[rmin:rmax, cmin:cmax]
    unique_area = np.unique(cut_area)[1:]
    subignore_mask = np.ones((cut_area.shape[0],cut_area.shape[1]), dtype=int)

    for i in unique_area:
        if np.sum(cut_area == i) != np.sum(instmap == i):
            subignore_mask[cut_area == i] = 0
    return subignore_mask
def get_ignore_mask(instmap, positions, ignore_mask):
    instmap = instmap.copy()
    positions = positions.copy()
    ignore_mask = ignore_mask.copy()
    rmin, rmax, cmin, cmax = positions
    cut_area = instmap[rmin:rmax, cmin:cmax]
    unique_area = np.unique(cut_area)[1:]

    for i in unique_area:
        if np.sum(cut_area == i) != np.sum(instmap == i):
            ignore_mask[instmap == i] = 0
    return ignore_mask

def find_and_insert_smallest_missing(arr):
    arr = arr.copy()
    smallest_missing = 1

    for num in arr:
        if num == smallest_missing:
            smallest_missing += 1
        elif num > smallest_missing:
            arr = np.insert(arr, np.searchsorted(arr, num), smallest_missing)
            return smallest_missing, arr

    arr = np.append(arr, smallest_missing)
    return smallest_missing, arr
def get_expand_mask(rgmask, h, w):
    expand_mask = np.zeros((h, w))
    expand_mask[:h // 2, :w // 2] = rgmask[0, 0]  # 左上角
    expand_mask[:h // 2, w // 2:] = rgmask[0, 1]  # 右上角
    expand_mask[h // 2:, :w // 2] = rgmask[1, 0]  # 左下角
    expand_mask[h // 2:, w // 2:] = rgmask[1, 1]  # 右下角
    return expand_mask
class FourGridFileLoader(torch.utils.data.Dataset):
    """Data Loader. Loads images from a file list and 
    performs augmentation with the albumentation library.
    After augmentation, horizontal and vertical maps are 
    generated.

    Args:
        file_list: list of filenames to load
        input_shape: shape of the input [h,w] - defined in config.py
        mask_shape: shape of the output [h,w] - defined in config.py
        mode: 'train' or 'valid'
        
    """

    # TODO: doc string

    def __init__(
        self,
        file_list,
        # file_list_2,
        prob=50,
        with_type=False,  #True
        input_shape=None,  #[270,270]
        mask_shape=None,  #[80,80]
        mode="train",
        setup_augmentor=True,  # False
        target_gen=None,  #
    ):
        assert input_shape is not None and mask_shape is not None
        self.prob = prob
        print('---prob:',prob)
        self.mode = mode
        self.info_list = file_list
        # self.info_list_real = file_list_1
        # self.info_list_generate = file_list_2
        # self.info_list_all = file_list_1 + file_list_2
        self.with_type = with_type
        self.mask_shape = mask_shape
        self.input_shape = input_shape

        height = input_shape[0] // 2 
        width = input_shape[1] // 2 
        self.blocks = [
            [(0, height), (0, width)],
            [(0, height), (width, input_shape[1])],
            [(height, input_shape[0]), (0, width)],
            [(height, input_shape[0]), (width, input_shape[1])]
        ]
        self.height=height
        self.width=width
        self.random_flip = True
        self.random_rotate = True
        self.id = 0
        self.target_gen_func = target_gen[0]
        self.target_gen_kwargs = target_gen[1]
        if setup_augmentor:
            self.setup_augmentor(0, 0)
        return

    def setup_augmentor(self, worker_id, seed):
        self.augmentor = self.__get_augmentation(self.mode, seed)
        self.shape_augs = iaa.Sequential(self.augmentor[0])
        self.input_augs = iaa.Sequential(self.augmentor[1])
        self.id = self.id + worker_id
        return

    def __len__(self):
        return len(self.info_list)

    def __getitem__(self, idx):
        path, label = self.info_list[idx]
        data = np.load(path)
        img = (data[..., :3]).astype("uint8")  # RGB images
        ann = (data[..., 3:]).astype("int32")  # instance ID map and type map
        basename = pathlib.Path(path).stem
        # ignore_mask = np.ones((img.shape[0],img.shape[1]))  # the pixels set to 0:what we want to ignore for the half nucleus
        ignore_mask = np.ones((img.shape[0],img.shape[1]), dtype=int)
        prob = self.prob 
        type_label = [label]*4
        if random.randint(0, 99) < prob:
            rand_index = random.randint(1, 4)  # include 0,4
            
            selected_files = random.sample(self.info_list, rand_index)
            
            rand_pos = np.array([0,1,2,3])
            np.random.shuffle(rand_pos)
            
            for i in range(rand_index):
                rmin = self.blocks[rand_pos[i]][0][0]
                rmax = self.blocks[rand_pos[i]][0][1]
                cmin = self.blocks[rand_pos[i]][1][0]
                cmax = self.blocks[rand_pos[i]][1][1]
                seldata = np.load(selected_files[i][0])
                type_label[rand_pos[i]] = selected_files[i][1]
                
                selimg = (seldata[..., :3]).astype("uint8")  # RGB images
                selann = (seldata[..., 3:]).astype("int32")  # instance ID map and type map
                r = np.random.randint(0, seldata.shape[0] - self.height + 1)
                c = np.random.randint(0, seldata.shape[1] - self.width + 1)

                subignore_mask = get_subignore_mask(selann[...,0], [r, r+self.height, c, c+self.width])
                selimg = selimg[r:r+self.height, c:c+self.width].copy()
                selann = selann[r:r+self.height, c:c+self.width].copy()

                ignore_mask = get_ignore_mask(ann[:,:,0], [rmin, rmax, cmin, cmax], ignore_mask)

                if self.random_flip and random.random() < 0.5:
                    flip_direction = np.random.choice([-1, 0, 1])
                    selimg = cv2.flip(selimg, flip_direction)
                    selann = cv2.flip(selann, flip_direction)
                    subignore_mask = cv2.flip(subignore_mask, flip_direction)

                if self.random_rotate and random.random() < 0.5:
                    rot_k = random.choice(range(1, 4))
                    selimg = np.rot90(selimg, k=rot_k)
                    selann = np.rot90(selann, k=rot_k)
                    subignore_mask = np.rot90(subignore_mask, k=rot_k)
                
                
                ignore_mask[rmin:rmax, cmin:cmax] = subignore_mask

                img[rmin:rmax, cmin:cmax] = selimg
                ann[rmin:rmax, cmin:cmax] = selann
                ann[rmin:rmax, cmin:cmax, 1] = selann[:,:,1]

                uq_idx = np.unique(ann[:,:,0])[1:]
                uq_idx_block = np.unique(selann[:,:,0])[1:]
                blockann_instnew = selann[:,:,0].copy()
                if len(uq_idx_block)>0:
                    for subi in uq_idx_block:
                        newid, uq_idx = find_and_insert_smallest_missing(uq_idx)
                        blockann_instnew[selann[:,:,0]==subi] = newid
                ann[rmin:rmax, cmin:cmax, 0] = blockann_instnew
        rgmask = np.array(type_label).reshape(2, 2)
        expandrgmask = get_expand_mask(rgmask, img.shape[0],img.shape[1])
        
        # plt.figure(figsize=(14, 6))
        # plt.subplot(1, 5, 1)
        # plt.imshow(expandrgmask.astype(np.float32))
        # plt.title(f'{type_label},{np.unique(expandrgmask)}')
        # plt.axis('off')
        # plt.subplot(1, 5, 2)
        # plt.imshow(ann[...,0])
        # plt.title(f'instance map')
        # plt.axis('off')
        # plt.subplot(1, 5, 3)
        # plt.imshow(ann[...,1])
        # plt.title(f'type map')
        # plt.axis('off')
        # plt.subplot(1, 5, 4)
        # plt.imshow(ignore_mask)
        # plt.title(f'ignore mask')
        # plt.axis('off')
        # plt.subplot(1, 5, 5)
        # plt.imshow(img)
        # plt.title(f'img')
        # plt.axis('off')
        # plt.tight_layout()
        # nam = str(basename)
        # plt.savefig(f'{nam}_{rand_index}_{rand_pos}.png')
        # plt.close()
        
        
        # assert False

        if self.random_rotate and random.random() < 0.5:
            rot_k = random.choice(range(1, 4))
            img = np.rot90(img, k=rot_k).copy()
            ann = np.rot90(ann, k=rot_k).copy()
            ignore_mask = np.rot90(ignore_mask, k=rot_k).copy()
            expandrgmask = np.rot90(expandrgmask, k=rot_k).copy()
            
        if self.random_flip and random.random() < 0.5:
            flip_direction = np.random.choice([-1, 0, 1])
            img = cv2.flip(img, flip_direction)
            ann = cv2.flip(ann, flip_direction)
            ignore_mask = cv2.flip(ignore_mask, flip_direction)
            expandrgmask = cv2.flip(expandrgmask, flip_direction)

        # if self.shape_augs is not None:
        #     shape_augs = self.shape_augs.to_deterministic()
        #     img = shape_augs.augment_image(img)
        #     ann = shape_augs.augment_image(ann)

        # if self.input_augs is not None:
        #     input_augs = self.input_augs.to_deterministic()
        #     img = input_augs.augment_image(img)

        img = cropping_center(img, self.input_shape)
        feed_dict = {"img": img}
        inst_map = ann[..., 0]  # HW1 -> HW
        # print(ann.shape,inst_map.shape,self.with_type)

        if self.with_type:
            type_map = (ann[..., 1]).copy()
            type_map = cropping_center(type_map, self.mask_shape)
            #type_map[type_map == 5] = 1  # merge neoplastic and non-neoplastic
            feed_dict["tp_map"] = type_map

        # TODO: document hard coded assumption about #input
        target_dict = self.target_gen_func(
            inst_map, self.mask_shape, **self.target_gen_kwargs
        )
        feed_dict.update(target_dict)
        feed_dict['mask'] = cropping_center(ignore_mask, self.mask_shape)
        feed_dict["rgmask"] = cropping_center(expandrgmask, self.mask_shape)#np.array(type_label).reshape(2, 2)
        return feed_dict

    def __get_augmentation(self, mode, rng):
        if mode == "train":
            shape_augs = [
                # * order = ``0`` -> ``cv2.INTER_NEAREST``
                # * order = ``1`` -> ``cv2.INTER_LINEAR``
                # * order = ``2`` -> ``cv2.INTER_CUBIC``
                # * order = ``3`` -> ``cv2.INTER_CUBIC``
                # * order = ``4`` -> ``cv2.INTER_CUBIC``
                # ! for pannuke v0, no rotation or translation, just flip to avoid mirror padding
                iaa.Affine(
                    # scale images to 80-120% of their size, individually per axis
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    # translate by -A to +A percent (per axis)
                    translate_percent={"x": (-0.01, 0.01), "y": (-0.01, 0.01)},
                    shear=(-5, 5),  # shear by -5 to +5 degrees
                    rotate=(-179, 179),  # rotate by -179 to +179 degrees
                    order=0,  # use nearest neighbour
                    backend="cv2",  # opencv for fast processing
                    seed=rng,
                ),
                # set position to 'center' for center crop
                # else 'uniform' for random crop
                iaa.CropToFixedSize(
                    self.input_shape[0], self.input_shape[1], position="center"
                ),
                iaa.Fliplr(0.5, seed=rng),
                iaa.Flipud(0.5, seed=rng),
            ]

            input_augs = [
                iaa.OneOf(
                    [
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: gaussian_blur(*args, max_ksize=3),
                        ),
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: median_blur(*args, max_ksize=3),
                        ),
                        iaa.AdditiveGaussianNoise(
                            loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                        ),
                    ]
                ),
                iaa.Sequential(
                    [
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: add_to_hue(*args, range=(-8, 8)),
                        ),
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: add_to_saturation(
                                *args, range=(-0.2, 0.2)
                            ),
                        ),
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: add_to_brightness(
                                *args, range=(-26, 26)
                            ),
                        ),
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: add_to_contrast(
                                *args, range=(0.75, 1.25)
                            ),
                        ),
                    ],
                    random_order=True,
                ),
            ]
        elif mode == "valid":
            shape_augs = [
                # set position to 'center' for center crop
                # else 'uniform' for random crop
                iaa.CropToFixedSize(
                    self.input_shape[0], self.input_shape[1], position="center"
                )
            ]
            input_augs = []

        return shape_augs, input_augs
    def __get_overlap_nuclei_ids(self, instmap, positions):
        instmap = instmap.copy()
        positions = positions.copy()
        rmin, rmax, cmin, cmax = positions
        cut_area = instmap[rmin:rmax, cmin:cmax]
        unique_area = np.unique(cut_area)[1:]
        
        # unique_all = np.unique(instmap)[1:]
        disassembled_inst_id = []
        for i in unique_area:
            if np.sum(cut_area == i) != np.sum(instmap == i):
                disassembled_inst_id.append(i)
        return disassembled_inst_id
