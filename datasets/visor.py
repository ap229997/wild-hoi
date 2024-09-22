# --------------------------------------------------------
# Modified from ho3d.py of https://github.com/JudyYe/ihoi
# --------------------------------------------------------
from __future__ import print_function
from numpy.core.fromnumeric import resize
import pandas as pd
import os
import os.path as osp
import pickle
import torch
import numpy as np
import tqdm
from PIL import Image
from nnutils.hand_utils import cvt_axisang_t_i2o
from datasets.base_data import BaseData, minmax, proj3d

from nnutils import mesh_utils, geom_utils, image_utils


class VISOR(BaseData):
    def __init__(self, cfg, dataset: str, split='val', is_train=True,
                 data_dir='../data/', cache=None):
        data_dir = osp.join(data_dir, 'visor')
        super().__init__(cfg, 'visor', split, is_train, data_dir)
        self.cache = cache if cache is not None else self.cfg.DB.CACHE
        self.anno = {
            'index': [],
            'cad_index': [],
            'hA': [],
            'cTh': [],
            'hTo': [],
            'bbox': [],
            'cam': [],
        }

        self.cache_file = osp.join(osp.dirname(self.data_dir), 'cache', 'visor_train.pkl')
        self.mask_dir = ''
        self.image_dir = osp.join(self.data_dir, '{}', 'rgb', '{}.jpg')
        self.seg_dir = osp.join(self.data_dir, '{}', 'seg', '{}.png')

    def preload_anno(self, load_keys=[]):
        if self.cache and osp.exists(self.cache_file):
            print('!! Load from cache !!', self.cache_file)
            self.anno = pickle.load(open(self.cache_file, 'rb'))
        else:
            print('!! cache file not found, create one !!')

    def get_bbox(self, idx):
        return self.anno['bbox'][idx]
    
    def get_cam(self, idx):
        return self.anno['cam'][idx]

    def get_obj_mask(self, index):
        """fake mask"""
        image = np.array(Image.open(self.image_dir.format(*index)))
        H, W, _= image.shape
        mask = Image.fromarray(np.ones([H, W]).astype(np.uint8) * 255 )
        return mask

    def get_image(self, index):
        return Image.open(self.image_dir.format(*index))

    def get_seg_mask(self, index, size=(640,480)):
        seg = np.array(Image.open(self.seg_dir.format(*index)))
        obj_mask, hand_mask = np.zeros(seg.shape[:2]), np.zeros(seg.shape[:2])
        obj_mask[np.where(seg[:,:,1]==255)] = 255 # green value corresponds to object
        hand_mask[np.where(seg[:,:,2]==255)] = 255 # blue value corresponds to hand
        return obj_mask, hand_mask
