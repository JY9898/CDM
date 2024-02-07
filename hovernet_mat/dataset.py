import glob
import cv2
import numpy as np
import scipy.io as sio


class __AbstractDataset(object):
    """Abstract class for interface of subsequent classes.
    Main idea is to encapsulate how each dataset should parse
    their images and annotations.
    
    """

    def load_img(self, path):
        raise NotImplementedError

    def load_ann(self, path, with_type=False):
        raise NotImplementedError

####
class __CoNSeP(__AbstractDataset):
    """Defines the CoNSeP dataset as originally introduced in:

    Graham, Simon, Quoc Dang Vu, Shan E. Ahmed Raza, Ayesha Azam, Yee Wah Tsang, Jin Tae Kwak, 
    and Nasir Rajpoot. "Hover-Net: Simultaneous segmentation and classification of nuclei in 
    multi-tissue histology images." Medical Image Analysis 58 (2019): 101563
    
    """

    def load_img(self, path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    def load_ann(self, path, with_type=False):
        # assumes that ann is HxW
        ann_inst = sio.loadmat(path)["inst_map"]
        if with_type:
            ann_type = sio.loadmat(path)["type_map"]

            # merge classes for CoNSeP (in paper we only utilise 3 nuclei classes and background)
            # If own dataset is used, then the below may need to be modified
            ann_type[(ann_type == 3) | (ann_type == 4)] = 3
            ann_type[(ann_type == 5) | (ann_type == 6) | (ann_type == 7)] = 4

            ann = np.dstack([ann_inst, ann_type])
            ann = ann.astype("int32")
        else:
            ann = np.expand_dims(ann_inst, -1)
            ann = ann.astype("int32")

        return ann


class __MoNuSAC(__AbstractDataset):
    """Defines the CoNSeP dataset as originally introduced in:

    Graham, Simon, Quoc Dang Vu, Shan E. Ahmed Raza, Ayesha Azam, Yee Wah Tsang, Jin Tae Kwak, 
    and Nasir Rajpoot. "Hover-Net: Simultaneous segmentation and classification of nuclei in 
    multi-tissue histology images." Medical Image Analysis 58 (2019): 101563
    
    """

    def load_img(self, path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
# 
    def load_ann(self, path, with_type=False):
        # assumes that ann is HxW
        ann_inst = np.load(path)[...,0]
        if with_type:
            ann_type = np.load(path)[...,1]
            ann = np.dstack([ann_inst, ann_type])
            ann = ann.astype("int32")
        else:
            ann = np.expand_dims(ann_inst, -1)
            ann = ann.astype("int32")

        return ann


class __PANUKE(__AbstractDataset):
    """Defines the CoNSeP dataset as originally introduced in:

    Graham, Simon, Quoc Dang Vu, Shan E. Ahmed Raza, Ayesha Azam, Yee Wah Tsang, Jin Tae Kwak, 
    and Nasir Rajpoot. "Hover-Net: Simultaneous segmentation and classification of nuclei in 
    multi-tissue histology images." Medical Image Analysis 58 (2019): 101563
    
    """

    def load_img(self, path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
# 
    def load_ann(self, path, with_type=False):
        # assumes that ann is HxW
        ann_inst = sio.loadmat(path)["inst_map"]
        if with_type:
            ann_type = sio.loadmat(path)["type_map"]
            ann = np.dstack([ann_inst, ann_type])
            ann = ann.astype("int32")
        else:
            ann = np.expand_dims(ann_inst, -1)
            ann = ann.astype("int32")

        return ann

####
def get_dataset(name):
    """Return a pre-defined dataset object associated with `name`."""
    name_dict = {
        "consep": lambda: __CoNSeP(),
        "monusac": lambda: __MoNuSAC(),
        "panuke": lambda: __PANUKE(),
    }
    if name.lower() in name_dict:
        return name_dict[name]()
    else:
        assert False, "Unknown dataset `%s`" % name
