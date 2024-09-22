# --------------------------------------------------------
# Modified from sdf_img.py
# --------------------------------------------------------
from __future__ import print_function
import os, glob, random, copy, math, time
from cv2 import THRESH_TOZERO, norm
import numpy as np
from PIL import Image
from numpy.core.fromnumeric import size
from numpy.core.numerictypes import nbytes
from numpy.random.mtrand import sample
import torch
import torch.nn as nn
from torchvision.transforms import transforms
import pytorch3d.ops as op_3d
from pytorch3d.renderer.cameras import PerspectiveCameras
import matplotlib.pyplot as plt

from nnutils.hand_utils import ManopthWrapper, get_nTh
from nnutils import mesh_utils, geom_utils, image_utils


class SdfImg(nn.Module):
    """SDF Wrapper of datasets"""
    def __init__(self, cfg, dataset, is_train, data_dir='../data/', base_idx=0):
        super().__init__()
        self.cfg = cfg
        self.dataset = dataset
        self.train = is_train
        
        self.anno = {
            'index': [],
            'cad_index': [],
            'hA': [],
            'hTo': [],
            'cTh': [],
        }
        if self.cfg.OBJ_GEN and 'obman' not in self.dataset.dataset and 'mow' not in self.dataset.dataset:
            self.anno['n_views'] = []

        self.base_idx = base_idx
        self.data_dir = data_dir

        self.subsample = cfg.DB.NUM_POINTS
        self.hand_wrapper = ManopthWrapper().to('cpu')

        self.transform = transforms.Compose([
            transforms.ColorJitter(.4,.4,.4),
            transforms.ToTensor(),
        ]) if self.train else transforms.ToTensor()

        if self.cfg.XSEC: # cross-sections of synthetic shapes
            file = os.path.join(self.cfg.DB.DIR, 'discriminator/xsecs_center_planes.npz')
            xsecs = np.load(file)
            self.xsec_x, self.xsec_y, self.xsec_z = torch.from_numpy(xsecs['xsec_x']), torch.from_numpy(xsecs['xsec_y']), torch.from_numpy(xsecs['xsec_z'])
            self.xsecs = torch.cat([self.xsec_x, self.xsec_y, self.xsec_z], dim=0)

        self.viz_debug = False

    def preload_anno(self):
        self.dataset.preload_anno(self.anno.keys())
        for key in self.anno:
            self.anno[key] = self.dataset.anno[key]
        self.obj2mesh = self.dataset.obj2mesh
        self.map = self.dataset.map

        if self.cfg.SOLVER.DEBUG:
            self.obj_ids = [75, 3431, 34] # randomly chosen object ids for visualization
            self.viz_done = 0

    def __len__(self):
        if self.cfg.SOLVER.DEBUG: 
            return self.cfg.MODEL.BATCH_SIZE
        else: 
            return len(self.anno['index'])

    def __getitem__(self, idx):
        sample = {}
        if self.cfg.SOLVER.DEBUG and 'visor' not in self.cfg.DB.NAME:
            idx = self.obj_ids[idx%self.cfg.MODEL.BATCH_SIZE]
        idx = self.map[idx] if self.map is not None else idx

        hA = torch.FloatTensor(self.anno['hA'][idx])
        nTh = get_nTh(self.hand_wrapper, hA[None], self.cfg.DB.RADIUS)[0]
        
        # if (self.train and 'visor' not in self.cfg.DB.NAME) or (not self.train and 'visor' not in self.cfg.DB.TESTNAME):
        if 'visor' not in self.dataset.dataset:    
            hTo = torch.FloatTensor(self.anno['hTo'][idx])
            cad_idx = self.anno['cad_index'][idx]
            filename = self.dataset.get_sdf_files(cad_idx)

            oPos_sdf, oNeg_sdf = unpack_sdf_samples(filename, None)
            self.og_pos_sdf, self.og_neg_sdf = copy.deepcopy(oPos_sdf), copy.deepcopy(oNeg_sdf) # cache original points, to be used later

            nPos_sdf = self.norm_points_sdf(oPos_sdf, nTh @ hTo) 
            nNeg_sdf = self.norm_points_sdf(oNeg_sdf, nTh @ hTo) 

            oSdf = torch.cat([
                    self.sample_points(oPos_sdf, self.subsample),
                    self.sample_points(oNeg_sdf, self.subsample),
                ], dim=0)
            sample['oSdf'] = oSdf

            hSdf = self.norm_points_sdf(oSdf, hTo)
            sample['hSdf'] = hSdf

            nPos_sdf, self.pos_indices = self.sample_unit_cube(nPos_sdf, self.subsample, return_ind=True)
            nNeg_sdf, self.neg_indices = self.sample_unit_cube(nNeg_sdf, self.subsample, return_ind=True)
            sample['nSdf_viz'] = torch.cat([nPos_sdf, nNeg_sdf], dim=0) # added only to make visualization script work with visor, remove otherwise
            nSdf = torch.cat([nPos_sdf, nNeg_sdf], dim=0)
            sample['nSdf'] = nSdf

            mesh = self.obj2mesh[cad_idx]
            
            if self.cfg.MODEL.BATCH_SIZE == 1:
                sample['mesh'] = mesh
            xyz, color = op_3d.sample_points_from_meshes(mesh, self.subsample * 2, return_textures=True)
            sample['oObj'] = torch.cat([xyz, color], dim=-1)[0]  # (1, P, 6)
            hObj = torch.cat([
                    mesh_utils.apply_transform(xyz, hTo[None]),
                    color,
                ], dim=-1)[0]
            sample['hObj'] = hObj

            xyz = mesh_utils.apply_transform(xyz, (nTh @ hTo)[None])
            nObj = torch.cat([xyz, color], dim=-1)[0]  # (1, P, 6)
            nObj = self.sample_unit_cube(nObj, self.subsample)
            sample['nObj'] = nObj
            sample['hTo'] = geom_utils.matrix_to_se3(hTo)

        # add dummy values so that dataloading doesn't break, only in case of mixed dataset training
        if self.train and ('ho3d' in self.dataset.dataset or 'visor' in self.dataset.dataset) and 'obman' in self.cfg.DB.NAME:
            sample['oSdf'] = torch.zeros(2*self.subsample,4)
            sample['hSdf'] = torch.zeros(2*self.subsample,4)
            sample['nSdf'] = torch.zeros(2*self.subsample,4)
            sample['nSdf_viz'] = torch.zeros(2*self.subsample,4)
            sample['oObj'] = torch.zeros(2*self.subsample,6)
            sample['hObj'] = torch.zeros(2*self.subsample,6)
            sample['nObj'] = torch.zeros(self.subsample,6)
            sample['hTo'] = torch.zeros(12)

        sample['hA'] = self.rdn_hA(hA)
        sample['nTh'] = geom_utils.matrix_to_se3(nTh)
        sample['indices'] = idx + self.base_idx

        sample['cTh'] = geom_utils.matrix_to_se3(self.anno['cTh'][idx].squeeze(0))
        
        sample['bbox'] = self.get_bbox(idx)
        sample['cam_f'], sample['cam_p'] = self.get_f_p(idx, sample['bbox'])
        sample['image'] = self.get_image(idx, sample['bbox'])
        sample['obj_mask'] = self.get_obj_mask(idx, sample['bbox'])
        
        if 'obman' in self.dataset.dataset or 'mow' in self.dataset.dataset:
            sample['seg_mask'], sample['hand_mask'] = self.get_dummy_mask()
        else:
            sample['seg_mask'], sample['hand_mask'] = self.get_seg_mask(idx, sample['bbox'])

        sample['index'] = self.get_index(idx)

        if self.cfg.OBJ_GEN:
            if self.train and ('obman' in self.dataset.dataset or 'mow' in self.dataset.dataset):
                sample['n_views'] = self.get_dummy_nviews(sample, repeat=4)
            else:    
                sample['n_views'] = self.get_n_view_anno(self.anno['n_views'][idx])

        hand_mesh = self.hand_wrapper(None, sample['hA'][None], mode='inner')[0]
        hand_verts = hand_mesh._verts_padded
        
        if self.train and 'mow' not in self.dataset.dataset and 'obman' not in self.dataset.dataset:
            all_segs = sample['seg_mask'].unsqueeze(0)
            all_hands = sample['hand_mask'].unsqueeze(0)
            all_nTh = sample['nTh'].unsqueeze(0)
            all_cTh = sample['cTh'].unsqueeze(0)
            all_camf, all_camp = sample['cam_f'].unsqueeze(0), sample['cam_p'].unsqueeze(0)
            
            if self.cfg.OBJ_GEN:
                all_segs = torch.cat([all_segs, sample['n_views']['seg']], dim=0)
                all_hands = torch.cat([all_hands, sample['n_views']['hand_seg']], dim=0)
                all_nTh = torch.cat([all_nTh, sample['n_views']['nTh']], dim=0)
                all_cTh = torch.cat([all_cTh, sample['n_views']['cTh']], dim=0)
                nviews_camf, nviews_camp = sample['n_views']['cam_f'], sample['n_views']['cam_p']
                all_camf, all_camp = torch.cat([all_camf, nviews_camf], dim=0), torch.cat([all_camp, nviews_camp], dim=0)

            all_hTx = geom_utils.inverse_rt(all_nTh)
            all_cTx = geom_utils.compose_se3(all_cTh, all_hTx)
            all_cams = PerspectiveCameras(all_camf, all_camp)
            nv = all_segs.shape[0]

            sampled_neg_pts, sampled_hand_pts = [], []
            itx = 0
            max_iter = 50 # run rejection sampling for max 50 times
            while len(sampled_neg_pts) < (self.subsample):
                new_pts = torch.cat([torch.rand((self.subsample, 3))*2-1, torch.zeros((self.subsample, 1))], dim=-1)
                xPoints = torch.stack([new_pts]*nv, dim=0)[...,:3]
                proj_mask = sample_multi_z(xPoints, all_segs, all_cTx, all_cams)
                proj_hand = sample_multi_z(xPoints, all_hands, all_cTx, all_cams)
                mask_weight_alpha = torch.prod(torch.maximum(proj_mask, proj_hand), dim=0)
                mask_weight_alpha = torch.prod(proj_mask, dim=0)
                mask_weight_beta = torch.prod(proj_hand, dim=0)
                inside_mask = mask_weight_alpha*(1-mask_weight_beta)
                
                inside_indices = inside_mask.squeeze(-1)>0.5
                inside_pts = new_pts[inside_indices]
                inside_hand_indices = mask_weight_beta.squeeze(-1)>0.9
                inside_hand_pts = new_pts[inside_hand_indices]
                if len(sampled_neg_pts) == 0: 
                    sampled_neg_pts = inside_pts
                    sampled_hand_pts = inside_hand_pts
                else: 
                    sampled_neg_pts = torch.cat([sampled_neg_pts, inside_pts], dim=0)
                    sampled_hand_pts = torch.cat([sampled_hand_pts, inside_hand_pts], dim=0)
                itx += 1
                if itx == max_iter:
                    break
            occ_labels = torch.ones(sampled_neg_pts.shape[0])
            inside_hand_labels = torch.zeros(sampled_hand_pts.shape[0])
            
            hand_mesh_pts = torch.cat([hand_verts[0], torch.zeros((hand_verts.shape[1], 1))], dim=-1)
            hand_mesh_pts = self.norm_points_sdf(hand_mesh_pts, nTh)
            hand_mesh_pts = self.sample_unit_cube(hand_mesh_pts, len(hand_mesh_pts), replacement=False) # point ordering not preserved
            hand_mesh_labels = torch.zeros(hand_mesh_pts.shape[0])

            remaining_pts = max(2*self.subsample - len(sampled_neg_pts) - len(sampled_hand_pts) - len(hand_mesh_pts), 0)
            if remaining_pts > 0:
                sampled_pts = torch.cat([torch.rand((remaining_pts, 3))*2-1, torch.zeros((remaining_pts, 1))], dim=-1)
                nSdf = torch.cat([sampled_pts, sampled_hand_pts, hand_mesh_pts, sampled_neg_pts], dim=0)
            else:
                nSdf = torch.cat([sampled_hand_pts, hand_mesh_pts, sampled_neg_pts], dim=0)
                nSdf = nSdf[:2*self.subsample]
            sample['nSdf'] = nSdf
            if self.cfg.OBJ_GEN:
                sample['n_views']['nSdf'] = torch.stack([nSdf]*sample['n_views']['cTh'].shape[0], dim=0)
            
            if remaining_pts > 0:
                # repeat the above occupancy label procedure for sampled_pts as well
                xPoints = torch.stack([sampled_pts]*nv, dim=0)[...,:3]
                proj_mask = sample_multi_z(xPoints, all_segs, all_cTx, all_cams)
                proj_hand = sample_multi_z(xPoints, all_hands, all_cTx, all_cams)
                mask_weight_alpha = torch.prod(torch.maximum(proj_mask, proj_hand), dim=0)
                mask_weight_alpha = torch.prod(proj_mask, dim=0)
                mask_weight_beta = torch.prod(proj_hand, dim=0)
                inside_mask = mask_weight_alpha*(1-mask_weight_beta)
                inside_indices = inside_mask.squeeze(-1)>0.5
                occ_labels = torch.cat([inside_indices.float(), inside_hand_labels, hand_mesh_labels, occ_labels], dim=0)
            else:
                occ_labels = torch.cat([inside_hand_labels, hand_mesh_labels, occ_labels], dim=0)
                occ_labels = occ_labels[:2*self.subsample]
            occ_labels = 1-occ_labels # 1 means not occupied (consistent with marching cubes convention used in this code)
            sample['occ_labels'] = occ_labels

        sample['loss_mask'] = 1
        if self.train and ('obman' in self.dataset.dataset or 'mow' in self.dataset.dataset):
            obman_occ = nSdf[...,3]>0
            sample['occ_labels'] = obman_occ.float()
            sample['loss_mask'] = 0

        if self.train and self.cfg.SOLVER.DEBUG and self.viz_done<len(self.obj_ids) and 'mow' not in self.cfg.DB.NAME and 'obman' not in self.cfg.DB.NAME:
            self.viz_done += 1
            save_dir = os.path.join(self.cfg.OUTPUT_DIR, 'viz_%d'%idx)
            os.makedirs(save_dir, exist_ok=True)
            all_images = sample['image'].unsqueeze(0)
            all_segs = sample['seg_mask'].unsqueeze(0)
            all_hands = sample['hand_mask'].unsqueeze(0)
            all_nTh = sample['nTh'].unsqueeze(0)
            all_cTh = sample['cTh'].unsqueeze(0)
            all_camf, all_camp = sample['cam_f'].unsqueeze(0), sample['cam_p'].unsqueeze(0)
            
            if self.cfg.OBJ_GEN:
                all_images = torch.cat([all_images, sample['n_views']['image']], dim=0)
                all_segs = torch.cat([all_segs, sample['n_views']['seg']], dim=0)
                all_hands = torch.cat([all_hands, sample['n_views']['hand_seg']], dim=0)
                all_nTh = torch.cat([all_nTh, sample['n_views']['nTh']], dim=0)
                all_cTh = torch.cat([all_cTh, sample['n_views']['cTh']], dim=0)
                nviews_camf, nviews_camp = sample['n_views']['cam_f'], sample['n_views']['cam_p']
                all_camf, all_camp = torch.cat([all_camf, nviews_camf], dim=0), torch.cat([all_camp, nviews_camp], dim=0)

            nv = all_cTx.shape[0]
            nSdf_exp = torch.stack([nSdf[...,:3]]*nv, dim=0)
            proj_hand = sample_multi_z(nSdf_exp, all_hands, all_cTx, all_cams)
            inside_hand = torch.prod(proj_hand, dim=0)
            hand_indices = inside_hand.squeeze(-1)>0.5
            
            all_images = ((all_images+1)/2).permute(0,2,3,1).cpu().numpy()
            all_segs = all_segs[:,0].cpu().numpy()
            all_hands = all_hands[:,0].cpu().numpy()
            all_hTx = geom_utils.inverse_rt(all_nTh)
            all_cTx = geom_utils.compose_se3(all_cTh, all_hTx)
            all_cams = PerspectiveCameras(all_camf, all_camp)

            save_batch_images(all_images, save_dir, mode='img')
            save_batch_images(all_segs, save_dir, mode='seg')
            save_batch_images(all_hands, save_dir, mode='hand_crop')
            project_points_on_image(nSdf[-len(sampled_neg_pts):], all_cTx, all_cams, all_images, save_dir, name='proj_last')
            project_points_on_image(nSdf[:-len(sampled_neg_pts)], all_cTx, all_cams, all_images, save_dir, name='proj_early')
            project_hand_points_on_image(nSdf_exp[:,hand_indices,:3], all_images, all_cTx, all_cams, save_dir, name='hand_proj')
            project_hand_points_on_image(torch.stack([hand_mesh_pts[...,:3]]*nv, dim=0), all_images, all_cTx, all_cams, save_dir, name='hand_mesh_proj')
            
        if self.cfg.XSEC:
            xsec_id = random.choice(range(len(self.xsecs)))
            sample['xsec'] = self.xsecs[xsec_id]
        
        return sample  
    
    def rdn_hA(self, hA):
        if self.train and not self.cfg.SOLVER.DEBUG:
            hA = hA + (torch.rand([45]) * self.cfg.DB.JIT_ART * 2 - self.cfg.DB.JIT_ART)
        return hA
    
    def norm_points_sdf(self, obj, nTh):
        """
        :param obj: (P, 4)
        :param nTh: (4, 4)
        :return:
        """
        D = 4

        xyz, sdf = obj[None].split([3, D - 3], dim=-1)  # (N, Q, 3)
        nXyz = mesh_utils.apply_transform(xyz, nTh[None])  # (N, Q, 3)
        _, _, scale = geom_utils.homo_to_rt(nTh)  # (N, 3)
        # print(scale)  # only (5 or 1???)
        sdf = sdf * scale[..., 0:1, None]  # (N, Q, 1) -> (N, 3)
        nObj = torch.cat([nXyz, sdf], dim=-1)
        return nObj[0]

    def sample_points(self, points, num_points):
        """

        Args:
            points ([type]): (P, D)
        Returns:
            sampled points: (num_points, D)
        """
        P, D = points.size()
        ones = torch.ones([P])
        inds = torch.multinomial(ones, num_points, replacement=True).unsqueeze(-1)  # (P, 1)
        points = torch.gather(points, 0, inds.repeat(1, D))
        return points

    def sample_unit_cube(self, hObj, num_points, r=1, return_ind=False, replacement=True):
        """
        Args:
            points (P, 4): Description
            num_points ( ): Description
            r (int, optional): Description
        
        Returns:
            sampled points: (num_points, 4)
        """
        D = hObj.size(-1)
        points = hObj[..., :3]
        prob = (torch.sum((torch.abs(points) < r), dim=-1) == 3).float()
        if prob.sum() == 0:
            prob = prob + 1
            # print('oops') # Yufei said to ignore this warning
        inds = torch.multinomial(prob, num_points, replacement=replacement).unsqueeze(-1)  # (P, 1)

        handle = torch.gather(hObj, 0, inds.repeat(1, D))
        if return_ind:
            return handle, inds
        return handle

    def get_index(self, idx):
        index =  self.anno['index'][idx]
        if isinstance(index, tuple) or isinstance(index, list):

            index = '/'.join(index)
        return index

    def get_bbox(self, idx):
        bbox =  self.dataset.get_bbox(idx)  # in scale of pixel torch.floattensor 
        bbox = image_utils.square_bbox(bbox)
        bbox = self.jitter_bbox(bbox)
        return bbox

    def get_f_p(self, idx, bbox):
        cam_intr = self.dataset.get_cam(idx)  # with pixel?? in canvas
        cam_intr = image_utils.crop_cam_intr(cam_intr, bbox, 1)
        f, p = image_utils.screen_intr_to_ndc_fp(cam_intr, 1, 1)
        f, p = self.jitter_fp(f, p) 
        return f, p

    def get_image(self, idx, bbox):
        image = np.array(self.dataset.get_image(self.anno['index'][idx]))
        image = image_utils.crop_resize(image, bbox, return_np=False)
        if self.cfg.SOLVER.DEBUG:
            return transforms.ToTensor()(image)*2-1
        return self.transform(image) * 2 - 1
        
    def get_obj_mask(self, idx, bbox):
        obj_mask = np.array(self.dataset.get_obj_mask(self.anno['index'][idx]))
        # obj_mask = np.array(self.anno['obj_mask'][idx])
        obj_mask = image_utils.crop_resize(obj_mask, bbox,return_np=False)
        return (self.transform(obj_mask) > 0).float()

    def get_seg_mask(self, idx, bbox):
        obj_mask, hand_mask = self.dataset.get_seg_mask(self.anno['index'][idx])
        obj_mask = image_utils.crop_resize(obj_mask, bbox, return_np=False)
        hand_mask = image_utils.crop_resize(hand_mask, bbox, return_np=False)
        return (transforms.ToTensor()(obj_mask) > 0).float(), (transforms.ToTensor()(hand_mask) > 0).float()

    def get_dummy_mask(self, size=(224,224,1)):
        seg_mask, hand_mask = np.zeros(size), np.zeros(size)
        return (transforms.ToTensor()(seg_mask)>0).float(), (transforms.ToTensor()(hand_mask)>0).float()

    def jitter_bbox(self, bbox):
        if self.train and not self.cfg.SOLVER.DEBUG:
            bbox = image_utils.jitter_bbox(bbox, 
                self.cfg.DB.JIT_SCALE, self.cfg.DB.JIT_TRANS)
        return bbox
    
    def jitter_fp(self, f, p):
        if self.train and not self.cfg.SOLVER.DEBUG:
            stddev_p = self.cfg.DB.JIT_P / 224 * 2
            dp = torch.rand_like(p) * stddev_p * 2 - stddev_p
            p += dp
        return f, p

    def get_n_view_anno(self, nview_dict):
        nview_data = {'seg': [], 'hA': [], 'cTh': [], 'nTh': [], 'cam_f': [], 'cam_p': [], 'nSdf': [], 'hand_seg': [], 'image': []}
        for nview in nview_dict:
            seg_mask, hand_mask = self.dataset.get_seg_mask(nview['index'])
            bbox = image_utils.square_bbox(nview['bbox'])
            bbox = self.jitter_bbox(bbox)
            
            seg_mask = image_utils.crop_resize(seg_mask, bbox, return_np=False)
            seg_mask = (transforms.ToTensor()(seg_mask) > 0).float()

            hand_mask = image_utils.crop_resize(hand_mask, bbox, return_np=False)
            hand_mask = (transforms.ToTensor()(hand_mask) > 0).float()

            image = np.array(self.dataset.get_image(nview['index']))
            image = image_utils.crop_resize(image, bbox, return_np=False)
            image = transforms.ToTensor()(image) * 2 - 1

            cam_intr = nview['cam']
            cam_intr = image_utils.crop_cam_intr(cam_intr, bbox, 1)
            f, p = image_utils.screen_intr_to_ndc_fp(cam_intr, 1, 1)
            f, p = self.jitter_fp(f, p) 

            cTh = geom_utils.matrix_to_se3(nview['cTh'].squeeze(0))

            nTh = get_nTh(self.hand_wrapper, nview['hA'][None], self.cfg.DB.RADIUS)[0]
            nSdf = torch.cat([torch.rand((self.subsample, 3))*2-1, torch.zeros((self.subsample, 1))], dim=-1) # dummy value
            
            nTh = geom_utils.matrix_to_se3(nTh)

            hA = self.rdn_hA(nview['hA'])
            nview_data['seg'].append(seg_mask)
            nview_data['hand_seg'].append(hand_mask)
            nview_data['hA'].append(hA)
            nview_data['cTh'].append(cTh) 
            nview_data['nTh'].append(nTh)
            nview_data['image'].append(image)
            nview_data['cam_f'].append(f)
            nview_data['cam_p'].append(p)
            nview_data['nSdf'].append(nSdf)
        
        for k in nview_data: # stack all views into a tensor
            nview_data[k] = torch.stack(nview_data[k], dim=0)

        return nview_data

    def get_dummy_nviews(self, sample, repeat=4):
        nview_data = {'seg': [], 'hA': [], 'cTh': [], 'nTh': [], 'cam_f': [], 'cam_p': [], 'nSdf': [], 'hand_seg': [], 'image': []}
        for _ in range(repeat):
            nview_data['seg'].append(sample['seg_mask'])
            nview_data['hand_seg'].append(sample['hand_mask'])
            nview_data['hA'].append(sample['hA'])
            nview_data['cTh'].append(sample['cTh']) 
            nview_data['nTh'].append(sample['nTh'])
            nview_data['image'].append(sample['image'])
            nview_data['cam_f'].append(sample['cam_f'])
            nview_data['cam_p'].append(sample['cam_p'])
            nview_data['nSdf'].append(sample['nSdf'])
        
        for k in nview_data: # stack all views into a tensor
            nview_data[k] = torch.stack(nview_data[k], dim=0)

        return nview_data

def sample_multi_z(xPoints, z, cTx, cam):
    N1, P, D = xPoints.size()
    N = z.size(0)
    xPoints_exp = xPoints.expand(N, P, D)

    ndcPoints = proj_x_ndc(xPoints_exp, cTx, cam, mode='seg')
    zs = mesh_utils.sample_images_at_mc_locs(z, ndcPoints)  # (N, P, D)
    return zs

def save_batch_images(images, save_dir, mode='img'):
    # numpy image on cpu, range [0,1]
    bz = images.shape[0]
    for j in range(bz):
        Image.fromarray((255*images[j]).astype(np.uint8)).save(os.path.join(save_dir, '%s_%d.png'%(mode, j)))

def plot_pixels_on_image(image, pts, save_path):
    # all matrices are numpy on cpu, range [0,1] for images, pts are in pixels
    plt.imshow(image)
    ix, iy = pts
    plt.scatter(ix, iy, marker='x', color='red', s=2)
    plt.savefig(save_path)
    plt.close()

def project_points_on_image(sdf, cTx, cam, image, save_dir, name='proj'):
    N, P = image.shape[0], sdf.shape[0]
    xPoints = copy.deepcopy(sdf[...,:3])
    xPoints = xPoints.expand(N,P,3)
    pixels = proj_x_ndc(xPoints, cTx, cam)
    pos_indices = sdf[...,3]>0
    pos_pixels = (pixels[0][:,pos_indices], pixels[1][:,pos_indices])
    neg_pixels = (pixels[0][:,~pos_indices], pixels[1][:,~pos_indices])
    for j in range(N):
        plot_pixels_on_image(image[j], (pixels[0][j], pixels[1][j]), os.path.join(save_dir, '%s_%d.png'%(name,j)))
        plot_pixels_on_image(image[j], (pos_pixels[0][j], pos_pixels[1][j]), os.path.join(save_dir, '%s_pos_%d.png'%(name,j)))
        plot_pixels_on_image(image[j], (neg_pixels[0][j], neg_pixels[1][j]), os.path.join(save_dir, '%s_neg_%d.png'%(name,j)))

def project_hand_points_on_image(sdf, image, cTx, cam, save_dir, name='hand_proj'):
    N = image.shape[0]
    pixels = proj_x_ndc(sdf, cTx, cam)
    for j in range(N):
        plot_pixels_on_image(image[j], (pixels[0][j], pixels[1][j]), os.path.join(save_dir, '%s_%d.png'%(name,j)))

def proj_x_ndc(xPoints, cTx, cam, mode='img'):
    # returns pixel values
    cPoints = mesh_utils.apply_transform(xPoints, cTx)
    ndcPoints = mesh_utils.transform_points(cPoints, cam)
    if mode == 'seg':
        return ndcPoints[...,:2]
    proj_pixels = mesh_utils.batch_points_to_pix(ndcPoints[...,:2])
    return proj_pixels

def unpack_sdf_samples(filename, subsample=None):
    npz = np.load(filename)
    pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
    neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))

    if subsample is None:
        return pos_tensor, neg_tensor

    # split the sample into half
    half = int(subsample / 2)

    random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
    random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()

    sample_pos = torch.index_select(pos_tensor, 0, random_pos)
    sample_neg = torch.index_select(neg_tensor, 0, random_neg)

    samples = torch.cat([sample_pos, sample_neg], 0)

    return samples

def remove_nans(tensor):
    tensor_nan = torch.isnan(tensor[:, 3])
    return tensor[~tensor_nan, :]
