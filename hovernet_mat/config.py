import importlib
import random

import cv2
import numpy as np

from dataset import get_dataset


class Config(object):
    """Configuration file."""

    def __init__(self):
        print("Config: /hovernet_mat/config")
        self.seed = 50
        self.prob = 50  # 0-100

        self.logging = True

        # turn on debug flag to trace some parallel processing problems more easily
        self.debug = False

        model_name = "hovernet"
        model_mode = "fast" # choose either `original` or `fast`

        if model_mode not in ["original", "fast"]:
            raise Exception("Must use either `original` or `fast` as model mode")

        nr_type = 5 # number of nuclear types (including background)

        # whether to predict the nuclear type, availability depending on dataset!
        self.type_classification = True

        # shape information - 
        # below config is for original mode. 
        # If original model mode is used, use [270,270] and [80,80] for act_shape and out_shape respectively
        # If fast model mode is used, use [256,256] and [164,164] for act_shape and out_shape respectively
        if model_mode == 'original':
            aug_shape = [540, 540] # patch shape used during augmentation (larger patch may have less border artefacts)
            act_shape = [270, 270] # patch shape used as input to network - central crop performed after augmentation
            out_shape = [80, 80] # patch shape at output of network
        if model_mode == 'fast':
            aug_shape = [512, 512] # patch shape used during augmentation (larger patch may have less border artefacts)
            act_shape = [256, 256] # patch shape used as input to network - central crop performed after augmentation
            out_shape = [164, 164] # patch shape at output of network

        self.dataset_name = "consep" 

        self.taskname = f'hovernet_aug_prob{self.prob}_loss0p1_seed{self.seed}'
        self.log_dir = f'/logs/hovernet_mat/{self.dataset_name}_{self.taskname}_{act_shape[0]:d}x{act_shape[1]:d}_{out_shape[0]:d}x{out_shape[1]:d}/'
        self.train_dir_list = [
            f'/real dataset npy/',  # 1. real dataset
            f'/synthetic dataset-rare class1 npy/',  # 2.1 synthetic dataset
            f'/synthetic dataset-rare class2 npy/',  # 2.2 synthetic dataset
        ]

        self.valid_dir_list = [
            f"/test_npy/"
        ]

        self.shape_info = {
            "train": {"input_shape": act_shape, "mask_shape": out_shape,},
            "valid": {"input_shape": act_shape, "mask_shape": out_shape,},
        }

        # * parsing config to the running state and set up associated variables
        self.dataset = get_dataset(self.dataset_name)

        module = importlib.import_module(
            "models.%s.opt" % model_name
        )
        self.model_config = module.get_config(nr_type, model_mode)
