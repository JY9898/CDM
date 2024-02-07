import os
import torch

from collections import defaultdict
from os.path import isfile, expanduser
from easydict import EasyDict as edict

import yaml
import os
from PIL import Image

from guided_diffusion.script_util_repaint import model_and_diffusion_defaults
import io

import blobfile as bf
# import torch as th
import numpy as np
import random

def load_config(path):
    """
    Loads config file:

    Args:
        path (str): path to the config file

    Returns: "mask_condiff/config/samp_mask_with_nuclei_cond.yaml"
        config (dict): dictionary of the configuration parameters, merge sub_dicts

    """
    with open(path,'r') as f:
        cfg = yaml.safe_load(f)
    print(cfg)

    config = dict()
    config.update(model_and_diffusion_defaults())
    config.update(cfg)
    
    return config


@ staticmethod
def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def load_state_dict(path, backend=None, **kwargs):
    with bf.BlobFile(path, "rb") as f:
        data = f.read()
    return torch.load(io.BytesIO(data), **kwargs)

def setup_seed(seed):
    """
    fix random seed for deterministic training
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True