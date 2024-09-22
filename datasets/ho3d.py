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


class HO3D(BaseData):
    def __init__(self, cfg, dataset: str, split='val', is_train=True,
                 data_dir='../data/', cache=None):
        data_dir = osp.join(data_dir, 'ho3d')
        super().__init__(cfg, 'ho3d', split, is_train, data_dir)
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
        self.suf = dataset[len('ho3d'):]
        self.use_gt = self.cfg.DB.GT

        if self.use_gt == 'none':
            if not is_train:
                meta_folder = 'meta_plus'
            else:
                meta_folder = 'meta_gt'
        else:
            meta_folder = 'meta_%s' % self.use_gt

        self.cache_file = osp.join(osp.dirname(self.data_dir), 'cache', '%s_%s_%s.pkl' % (dataset, self.split, self.use_gt))
        self.cache_mesh = osp.join(osp.dirname(self.data_dir), 'cache', '%s_mesh.pkl' % (dataset, self.split))
        self.mask_dir = ''
        self.meta_dir = os.path.join(self.data_dir, '{}', '{}', meta_folder, '{}.pkl')
        self.image_dir = osp.join(self.data_dir, '{}', '{}', 'rgb', '{}.jpg')
        self.seg_dir = osp.join(self.data_dir, '{}', '{}', 'seg', '{}.png')
        self.shape_dir = os.path.join(self.cfg.DB.DIR, 'ho3dobj/models', '{}', 'textured_simple.obj')

    def preload_anno(self, load_keys=[]):
        if self.cache and osp.exists(self.cache_file):
            print('!! Load from cache !!', self.cache_file)
            self.anno = pickle.load(open(self.cache_file, 'rb'))
        else:
            print('creating cahce', self.meta_dir)
            # filter 1e-3
            df = pd.read_csv(osp.join(self.data_dir, '%s%s.csv' % (self.split, self.suf)))
            sub_df = df[df['dist'] < 5]
            sub_df = sub_df[sub_df['vid'] == 'MDF11']
            # sub_df = sub_df[sub_df['frame'] >= 350]
            
            print(len(df), '-->', len(sub_df))
            index_list = sub_df['index']
            folder_list = sub_df['split']

            for i, (index, folder) in enumerate(tqdm.tqdm(zip(index_list, folder_list))):
                index = (folder, index.split('/')[0], index.split('/')[1])
                meta_path = self.meta_dir.format(*index)
                with open(meta_path, "rb") as meta_f:
                    anno = pickle.load(meta_f)

                self.anno['index'].append(index)
                self.anno['cad_index'].append(anno["objName"])
                pose = torch.FloatTensor(anno['handPose'])[None]  # handTrans
                trans = torch.FloatTensor(anno['handTrans'].reshape(3))[None]
                hA = pose[..., 3:]
                rot = pose[..., :3]
                rot, trans = cvt_axisang_t_i2o(rot, trans)
                wTh = geom_utils.axis_angle_t_to_matrix(rot, trans)

                wTo = geom_utils.axis_angle_t_to_matrix(
                    torch.FloatTensor([anno['objRot'].reshape(3)]), 
                    torch.FloatTensor([anno['objTrans'].reshape(3)]))
                hTo = geom_utils.inverse_rt(mat=wTh, return_mat=True) @ wTo

                rot = torch.FloatTensor([[[1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1]]])
                cTw = geom_utils.rt_to_homo(rot, )
                cTh = cTw @ wTh

                cam_intr = torch.FloatTensor(anno['camMat'])
                joint3d = anno['handJoints3D']
                if joint3d.ndim == 1:
                    pad = 0.3
                    joint3d = joint3d[None]
                else:
                    pad = 0.2
                cPoints = np.concatenate([anno['objCorners3D'], joint3d], 0)
                cCorner = mesh_utils.apply_transform(torch.FloatTensor([cPoints]), cTw)
                bbox2d = image_utils.square_bbox(minmax(proj3d(cCorner, cam_intr))[0], pad)

                self.anno['bbox'].append(bbox2d)
                self.anno['cam'].append(cam_intr)
                self.anno['cTh'].append(cTh[0])
                self.anno['hTo'].append(hTo[0])
                self.anno['hA'].append(hA[0])

        self.preload_mesh()

    def preload_mesh(self):
        if self.cache and osp.exists(self.cache_mesh):
            print('!! Load from cache !!')
            self.obj2mesh = pickle.load(open(self.cache_mesh, 'rb'))
        else:
            self.obj2mesh = {}
            print('load mesh')
            for i, cls_id in tqdm.tqdm(enumerate(self.anno['cad_index']), total=len(self.anno['cad_index'])):
                key = cls_id
                if key not in self.obj2mesh:
                    fname = self.shape_dir.format(cls_id)
                    self.obj2mesh[key] = mesh_utils.load_mesh(fname, scale_verts=1)
            print('save cache')
            pickle.dump(self.obj2mesh, open(self.cache_mesh, 'wb'))

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

    def get_inpaint(self, index, size=(640,480)):
        # padding assuming height is preserved while resizing for inpainting
        image = Image.open(self.inpaint_dir.format(*index))
        image = image.resize((min(size), min(size)))
        margin = max(size) - min(size)
        image_padded = Image.new(image.mode, size, (0,0,0))
        image_padded.paste(image, (margin//2, 0))
        return image_padded

    def get_inpaint_small(self, index):
        image = Image.open(self.inpaint_dir.format(*index)) # 256x256 image
        return image

    def get_seg_mask(self, index, size=(640,480)):
        seg = Image.open(self.seg_dir.format(*index))
        seg = np.array(seg.resize(size))
        obj_mask, hand_mask = np.zeros(seg.shape[:2]), np.zeros(seg.shape[:2])
        obj_mask[np.where(seg[:,:,1]==255)] = 255 # green value corresponds to object
        hand_mask[np.where(seg[:,:,2]==255)] = 255 # blue value corresponds to hand
        return obj_mask, hand_mask

    def get_cTo(self, index):
        meta_dir = os.path.join(self.data_dir, '{}', '{}', 'meta', '{}.pkl')
        meta_path = meta_dir.format(*index)
        anno = pickle.load(open(meta_path, 'rb'))
        wTo = geom_utils.axis_angle_t_to_matrix(
                torch.from_numpy(anno['objRot'].reshape(1,3)), 
                torch.from_numpy(anno['objTrans'].reshape(1,3)))
        rot = torch.FloatTensor([[[1, 0, 0],
                [0, -1, 0],
                [0, 0, -1]]])
        cTw = geom_utils.rt_to_homo(rot, )
        cTo = cTw @ wTo
        cTo = geom_utils.matrix_to_se3(cTo)
        return cTo[0]
