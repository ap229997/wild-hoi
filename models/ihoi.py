import os
import os.path as osp
import math
import random
import functools
from typing import Any, List
import sys
sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch_lightning.core.saving import save_hparams_to_yaml

from config.args_config import default_argument_parser, setup_cfg
from nnutils.logger import MyLogger
from datasets import build_dataloader
from models import dec, enc, disc
from nnutils.hand_utils import ManopthWrapper
from nnutils import geom_utils, mesh_utils, slurm_utils

torch.autograd.set_detect_anomaly(True)


class OnCheckpointHparams(Callback):
    def on_save_checkpoint(self, trainer, pl_module):
        # only do this 1 time
        if trainer.current_epoch == 0:
            file_path = f"{trainer.logger.log_dir}/hparams.yaml"
            print(f"Saving hparams to file_path: {file_path}")
            save_hparams_to_yaml(config_yaml=file_path, hparams=pl_module.hparams)


class IHoi(pl.LightningModule):
    def __init__(self, cfg, **kwargs) -> None:
        super().__init__()
        self.hparams.update(cfg)
        self.cfg = cfg

        self.save_hyperparameters() # saves hparams to model checkpoint

        self.dec = dec.build_net(cfg.MODEL, full_cfg=cfg)
        self.enc = enc.build_net(cfg.MODEL.ENC, cfg)
        self.hand_wrapper = ManopthWrapper()
        
        if self.cfg.XSEC:
            self.xsec_disc = disc.build_net()

        self.bce_logits_loss = nn.BCEWithLogitsLoss(reduction='none')
        
        self.minT = -cfg.LOSS.SDF_MINMAX
        self.maxT = cfg.LOSS.SDF_MINMAX
        self.sdf_key = '%sSdf' % cfg.MODEL.FRAME[0]
        self.obj_key = '%sObj' % cfg.MODEL.FRAME[0]
        self.metric = 'val'
        self._train_loader = None

    def soft_argmax(self, heatmaps, num_joints=1):
        depth_dim = heatmaps.shape[1] // num_joints
        H_heatmaps = heatmaps.shape[2]
        W_heatmaps = heatmaps.shape[3]
        heatmaps = heatmaps.reshape((-1, num_joints, depth_dim * H_heatmaps * W_heatmaps))
        heatmaps = F.softmax(heatmaps, 2)
        confidence, _ = torch.max(heatmaps, 2)
        heatmaps = heatmaps.reshape((-1, num_joints, depth_dim, H_heatmaps, W_heatmaps))

        accu_x = heatmaps.sum(dim=(2, 3))
        accu_y = heatmaps.sum(dim=(2, 4))
        accu_z = heatmaps.sum(dim=(3, 4))

        accu_x = accu_x * torch.arange(W_heatmaps).float().to(self.device)[None, None, :]
        accu_y = accu_y * torch.arange(H_heatmaps).float().to(self.device)[None, None, :]
        accu_z = accu_z * torch.arange(depth_dim).float().to(self.device)[None, None, :]

        accu_x = accu_x.sum(dim=2, keepdim=True)
        accu_y = accu_y.sum(dim=2, keepdim=True)
        accu_z = accu_z.sum(dim=2, keepdim=True)

        coord_out = torch.cat((accu_x, accu_y, accu_z), dim=2)

        return coord_out, confidence

    def configure_optimizers(self):
        if self.cfg.XSEC:
            enc_optim = torch.optim.Adam(list(self.enc.parameters())+list(self.dec.parameters()), lr=self.cfg.SOLVER.ENC_LR)
            disc_optim = torch.optim.Adam(self.xsec_disc.parameters(), lr=self.cfg.SOLVER.DISC_LR)
            return [enc_optim, disc_optim]
        else:
            return torch.optim.Adam(self.parameters(), lr=self.cfg.SOLVER.BASE_LR)

    # Alternating schedule for optimizer steps (https://pytorch-lightning.readthedocs.io/en/stable/common/optimization.html)
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu=False, using_native_amp=False, using_lbfgs=False,):
        # update generator every N steps
        if optimizer_idx == 0:
            if batch_idx % self.cfg.SOLVER.ENC_ITER == 0:
                optimizer.step(closure=optimizer_closure)
            else:
                optimizer_closure()
        
        # update discriminator every N steps
        if optimizer_idx == 1:
            if batch_idx % self.cfg.SOLVER.DISC_ITER == 0:
                optimizer.step(closure=optimizer_closure)
            else:
                optimizer_closure()

    def train_dataloader(self):
        if self._train_loader is None:
            loader = build_dataloader(self.cfg, 'train')
            self._train_loader = loader
        return self._train_loader

    def val_dataloader(self):
        val = self.cfg.DB.NAME if self.cfg.DB.TESTNAME == '' else self.cfg.DB.TESTNAME
        if 'ho3d' in val:
            val = 'ho3d_vid'
        val_dataloader = build_dataloader(self.cfg, 'val', is_train=False, shuffle=False, bs=self.cfg.MODEL.BATCH_SIZE, name=val)
        return [val_dataloader, ]
    
    def test_dataloader(self):
        test = self.cfg.DB.NAME if self.cfg.DB.TESTNAME == '' else self.cfg.DB.TESTNAME
        test_dataloader = build_dataloader(self.cfg, self.cfg.TEST.SET, bs=self.cfg.MODEL.BATCH_SIZE, is_train=False, name=test)
        return [test_dataloader, ]

    def get_jsTx(self, hA, hTx):
        hTjs = self.hand_wrapper.pose_to_transform(hA, False)  # (N, J, 4, 4)
        N, num_j, _, _ = hTjs.size()
        jsTh = geom_utils.inverse_rt(mat=hTjs, return_mat=True)
        hTx_exp = geom_utils.se3_to_matrix(hTx
                  ).unsqueeze(1).repeat(1, num_j, 1, 1)
        jsTx = jsTh @ hTx_exp        
        return jsTx

    def get_hTx(self, frame, batch):
        hTn = geom_utils.inverse_rt(batch['nTh'])
        hTx = hTn
        return hTx

    def sdf(self, hA, sdf_hA_jsTx, hTx):
        sdf = functools.partial(sdf_hA_jsTx, hA=hA, jsTx=self.get_jsTx(hA, hTx))
        return sdf

    def forward(self, batch):
        image_feat = self.enc(batch['image'], mask=batch['obj_mask'])  # (N, D, H, W)
        if self.cfg.FACTORED or self.cfg.FACTORED_CAT:
            inpaint_feat = self.enc(batch['inpaint'], mask=batch['obj_mask'])  # (N, D, H, W)
        else:
            inpaint_feat = None
        
        hTx = self.get_hTx(self.cfg.MODEL.FRAME, batch)
        xTh = geom_utils.inverse_rt(hTx)
        cTx = geom_utils.compose_se3(batch['cTh'], hTx)
        cameras = PerspectiveCameras(batch['cam_f'], batch['cam_p'], device=batch['image'].device)

        with torch.enable_grad():
            sdf_hA_jsTx = functools.partial(self.dec, 
                z=image_feat, cTx=cTx, cam=cameras, inpaint_feat=inpaint_feat)
            sdf_hA = functools.partial(self.sdf, sdf_hA_jsTx=sdf_hA_jsTx, hTx=hTx)
            sdf = sdf_hA(batch['hA'])

        out = {
            'sdf': sdf,
            'sdf_hA': sdf_hA,
            'hTx': hTx,
            'xTh': xTh,
        }
        return out
    
    def training_step(self, batch, batch_idx, optimizer_idx=None):
        losses, out = self.step(batch, batch_idx)
        losses = {'train_' + e: v for e,v in losses.items()}
        
        if self.trainer.is_global_zero:
            self.log_dict(losses)

            if self.cfg.SOLVER.DEBUG:
                curr_step = self.global_step
            else: 
                curr_step = self.global_step + 1
            
            if (curr_step % self.cfg.TRAIN.VIS_EVERY == 0) and self.cfg.SOLVER.DEBUG and 'visor' not in self.cfg.DB.NAME: # visor does not have object meshes
                prefix = '%d_%d' % (0, batch_idx)
                try:
                    self.vis_step(out, batch, prefix)
                except:
                    pass
                self.quant_step(out, batch)

            if self.global_step % self.cfg.TRAIN.PRINT_EVERY == 0:
                self.logger.print(self.global_step, self.current_epoch, losses, losses['train_loss'])
        
        if self.cfg.XSEC:
            if optimizer_idx == 1:
                return losses['train_disc']
            else:
                return losses['train_enc']
        else:
            return losses['train_loss']

    def test_step(self, *args):
        if len(args) == 3:
            batch, batch_idx, dataloader_idx = args
            if not isinstance(batch_idx, int):
                batch_idx = batch_idx[0]
                dataloader_idx = dataloader_idx[0]
        elif len(args) == 2:
            batch, batch_idx, = args
            if not isinstance(batch_idx, int):
                batch_idx = batch_idx[0]
            dataloader_idx = 0
        else:
            raise NotImplementedError

        prefix = '%d_%d' % (dataloader_idx, batch_idx)
        losses, out = self.step(batch, 0, mode='test')
        if (batch_idx+1) % self.cfg.TEST.VIS_EVERY == 0:
            try: # rendering breaks down is object geometry is not predicted well
                self.vis_step(out, batch, prefix)
            except:
                pass
        f_res = self.quant_step(out, batch, is_test=True)

        return f_res

    def test_epoch_end(self, outputs: List[Any], save_dir=None) -> None:
        save_dir = self.logger.local_dir if save_dir is None else save_dir
        mean_list = mesh_utils.test_end_fscore(outputs, save_dir)

    def validation_step(self, *args):
        return args

    def validation_step_end(self, batch_parts_outputs):
        args = batch_parts_outputs
        if len(args) == 3:
            batch, batch_idx, dataloader_idx = args
            if not isinstance(batch_idx, int):
                batch_idx = batch_idx[0]
                dataloader_idx = dataloader_idx[0]
        elif len(args) == 2:
            batch, batch_idx, = args
            if not isinstance(batch_idx, int):
                batch_idx = batch_idx[0]
            dataloader_idx = 0
        else:
            raise NotImplementedError
        prefix = '%d_%d' % (dataloader_idx, batch_idx)

        losses, out = self.step(batch, 0, mode='val')
        losses = {'val_' + e: v for e,v in losses.items()}
        self.log_dict(losses, prog_bar=True, sync_dist=True)

        if self.trainer.is_global_zero:
            self.quant_step(out, batch)
        return losses

    def quant_step(self, out, batch, sdf=None, is_test=False):
        device = batch['cam_f'].device
        N = batch['cam_f'].size(0)

        if sdf is None:
            camera = PerspectiveCameras(batch['cam_f'], batch['cam_p'], device=device)
            cTx = geom_utils.compose_se3(batch['cTh'], self.get_hTx(self.cfg.MODEL.FRAME, batch))
            sdf = functools.partial(self.dec, z=out['z'], hA=batch['hA'], jsTx=out['jsTx'], cTx=cTx, cam=camera)
        xObj = mesh_utils.batch_sdf_to_meshes(sdf, N)

        th_list = [.5/100, 1/100,]
        gt_pc = batch[self.obj_key][..., :3]

        hTx = self.get_hTx(self.cfg.MODEL.FRAME, batch)
        hObj = mesh_utils.apply_transform(xObj, hTx) 
        hGt = mesh_utils.apply_transform(gt_pc, hTx)
        f_res = mesh_utils.fscore(hObj, hGt, num_samples=gt_pc.size(1), th=th_list)

        metrics = {}
        for th, th_f in zip(th_list, f_res[:-1]):
            metrics['val_f-%d' % (th*100)] = np.mean(th_f)
        metrics['val_cd'] = np.mean(f_res[-1])
        self.logger.log_metrics(metrics, self.global_step)

        return  [batch['indices'].tolist()] + f_res

    def vis_input(self, out, batch, prefix):
        N = len(batch['hObj'])
        P = batch[self.sdf_key].size(1)
        device = batch['hObj'].device

        self.logger.save_images(self.global_step, batch['image'], '%s/image' % prefix)

        zeros = torch.zeros([N, 3], device=device)
        hHand, _ = self.hand_wrapper(None, batch['hA'], zeros, mode='inner')
        mesh_utils.dump_meshes(osp.join(self.logger.local_dir, '%d_%s/hand' % (self.global_step, prefix)), hHand)
        hHand.textures = mesh_utils.pad_texture(hHand, 'blue')

        hTx = self.get_hTx(self.cfg.MODEL.FRAME, batch)
        batch_nObj = batch['nObj']
        hSdf = mesh_utils.pc_to_cubic_meshes(mesh_utils.apply_transform(batch[self.sdf_key][:, P//2:, :3], hTx))
        hHoi = mesh_utils.join_scene([hHand, hSdf])
        
        cHoi = mesh_utils.apply_transform(hHoi, batch['cTh'])
        cameras = PerspectiveCameras(batch['cam_f'], batch['cam_p'], device=device)
        image_list = mesh_utils.render_geom_rot(cHoi, view_centric=True, cameras=cameras)
        self.logger.save_gif(self.global_step, image_list, '%s/inp' % prefix)
        
        hObj = mesh_utils.pc_to_cubic_meshes(mesh_utils.apply_transform(batch_nObj[...,:3], hTx))
        hHoi_obj = mesh_utils.join_scene([hHand, hObj])
        cHoi_obj = mesh_utils.apply_transform(hHoi_obj, batch['cTh'])
        image_list = mesh_utils.render_geom_rot(cHoi_obj, view_centric=True, cameras=cameras)
        self.logger.save_gif(self.global_step, image_list, '%s/inp_obj' % prefix)
        
        return {'hHand': hHand}
    
    def vis_output(self, out, batch, prefix, cache={}):
        N = len(batch['hObj'])
        device = batch['hObj'].device
        zeros = torch.zeros([N, 3], device=device)
        hHand, hJoints = self.hand_wrapper(None, batch['hA'], zeros, mode='inner')
        hHand.textures = mesh_utils.pad_texture(hHand, 'blue')
        
        cJoints = mesh_utils.apply_transform(hJoints, batch['cTh'])
        cache['hHand'] = hHand

        camera = PerspectiveCameras(batch['cam_f'], batch['cam_p'], device=device)
        hTx = self.get_hTx(self.cfg.MODEL.FRAME, batch)
        cTx = geom_utils.compose_se3(batch['cTh'], hTx)

        sdf = functools.partial(self.dec, z=out['z'], hA=batch['hA'], jsTx=out['jsTx'], cTx=cTx, cam=camera)
            
        xObj = mesh_utils.batch_sdf_to_meshes(sdf, N, bound=True)

        cache['xMesh'] = xObj
        hTx = self.get_hTx(self.cfg.MODEL.FRAME, batch)
        hObj = mesh_utils.apply_transform(xObj, hTx)
        mesh_utils.dump_meshes(osp.join(self.logger.local_dir, '%d_%s/obj' % (self.global_step, prefix)), hObj)

        xHoi = mesh_utils.join_scene([mesh_utils.apply_transform(hHand, geom_utils.inverse_rt(hTx)), xObj])
        image_list = mesh_utils.render_geom_rot(xHoi, scale_geom=True)
        self.logger.save_gif(self.global_step, image_list, '%s/xHoi' % prefix)
        
        cHoi = mesh_utils.apply_transform(xHoi, cTx)
        image = mesh_utils.render_mesh(cHoi, camera)
        self.logger.save_images(self.global_step, image['image'], '%s/cam_mesh' % prefix,
            bg=batch['image'], mask=image['mask'])
        image_list = mesh_utils.render_geom_rot(cHoi, view_centric=True, cameras=camera,
            xyz=cJoints[:, 5], out_size=512)
        self.logger.save_gif(self.global_step, image_list, '%s/cHoi' % prefix)
        
        if self.cfg.OBJ_GEN and 'ho3d' in self.cfg.DB.TESTNAME:
            bz, nv, _ = batch['n_views']['cTh'].shape
            all_cTh = torch.cat([batch['cTh'].unsqueeze(1), batch['n_views']['cTh']], dim=1)
            nviews_hTx = geom_utils.inverse_rt(batch['n_views']['nTh'])
            all_hTx = torch.cat([hTx.unsqueeze(1), nviews_hTx], dim=1)
            all_cTx = geom_utils.compose_se3(all_cTh, all_hTx)
            all_camf = torch.cat([batch['cam_f'].unsqueeze(1), batch['n_views']['cam_f']], dim=1)
            all_camp = torch.cat([batch['cam_p'].unsqueeze(1), batch['n_views']['cam_p']], dim=1)
            all_images = torch.cat([batch['image'].unsqueeze(1), batch['n_views']['image']], dim=1)
            for j in range(bz):
                for k in range(nv+1):
                    nv_cam = PerspectiveCameras(all_camf[j,k].unsqueeze(0), all_camp[j,k].unsqueeze(0)).to(cTx.device)
                    nv_cHoi = mesh_utils.apply_transform(xHoi, all_cTx[j,k].unsqueeze(0))
                    nv_image = mesh_utils.render_mesh(nv_cHoi, nv_cam)
                    self.logger.save_images(self.global_step, nv_image['image'], '%s/cam_mesh_%d_view_%d'%(prefix,j,k),
                        bg=all_images[j,k].unsqueeze(0), mask=nv_image['mask'])

        return cache

    def vis_step(self, out, batch, prefix):
        cache = self.vis_input(out, batch, prefix)
        cache = self.vis_output(out, batch, prefix, cache)
        return cache

    def step(self, batch, batch_idx, mode='train'):
        image_feat = self.enc(batch['image'], mask=batch['obj_mask'])  # (N, D, H, W)
        
        xXyz = batch[self.sdf_key][..., :3]
        hTx = self.get_hTx(self.cfg.MODEL.FRAME, batch)
        cTx = geom_utils.compose_se3(batch['cTh'], hTx)
        cameras = PerspectiveCameras(batch['cam_f'], batch['cam_p'], device=xXyz.device)
        
        hTjs = self.hand_wrapper.pose_to_transform(batch['hA'], False)  # (N, J, 4, 4)
        N, num_j, _, _ = hTjs.size()
        jsTh = geom_utils.inverse_rt(mat=hTjs, return_mat=True)
        hTx_exp = geom_utils.se3_to_matrix(hTx).unsqueeze(1).repeat(1, num_j, 1, 1)
        jsTx = jsTh @ hTx_exp
        
        out = {'z': image_feat, 'jsTx': jsTx, 'cTx': cTx, 'N': N, 'hTx': hTx, 'camera': cameras}
        
        if (mode == 'train' or self.cfg.SOLVER.DEBUG) and self.cfg.OBJ_GEN:
            seg_view = batch['n_views']['seg']
            hand_view = batch['n_views']['hand_seg']
            hA_view = batch['n_views']['hA']
            cTh_view = batch['n_views']['cTh']
            hTx_view = geom_utils.inverse_rt(batch['n_views']['nTh'])
            cTx_view = geom_utils.compose_se3(cTh_view, hTx_view)
            bz, nv, c, h, w = seg_view.shape
            nv_cameras = PerspectiveCameras(batch['n_views']['cam_f'].reshape(bz*nv,-1), 
                        batch['n_views']['cam_p'].reshape(bz*nv,-1), device=xXyz.device)

            hTjs_view = self.hand_wrapper.pose_to_transform(hA_view.reshape(bz*nv,-1), False)  # (N, J, 4, 4)
            _, view_j, _, _ = hTjs_view.size()
            jsTh_view = geom_utils.inverse_rt(mat=hTjs_view, return_mat=True)
            hTx_view_exp = geom_utils.se3_to_matrix(hTx_view.reshape(bz*nv,-1)).unsqueeze(1).repeat(1, view_j, 1, 1)
            jsTx_view = jsTh_view @ hTx_view_exp

            nv_sdf = batch['n_views']['nSdf'].reshape(bz*nv, xXyz.shape[-2], -1)[..., :3]
            nviews = {'seg': seg_view, 'hA': hA_view, 'cTx': cTx_view, 'cam': nv_cameras, 'sdf': nv_sdf, 'hand_seg': hand_view}
        else:
            nviews = None
        
        if mode == 'train': 
            seg = (batch['seg_mask'], batch['hand_mask'])
        else: 
            seg = None
            
        if mode == 'train':
            pred_sdf = self.dec(xXyz, image_feat, batch['hA'], cTx, cameras, jsTx=jsTx, seg=seg, nviews=nviews)
            if isinstance(pred_sdf, tuple):
                outputs = pred_sdf[1]
                pred_sdf = pred_sdf[0]
            out[self.sdf_key] = pred_sdf

        loss, enc_loss, losses = 0., 0., {}

        if mode == 'train' and self.cfg.LOSS.OCCUPANCY > 0.0:
            pred_occ, pred_occ_logits = outputs['pred_occ'], outputs['pred_occ_logits']
            occ_loss = torch.mean(self.bce_logits_loss(pred_occ_logits, batch['occ_labels']))
            losses['occ'] = occ_loss
            loss = loss + self.cfg.LOSS.OCCUPANCY*occ_loss
            enc_loss = enc_loss + self.cfg.LOSS.OCCUPANCY*occ_loss
        
        if mode == 'test':
            return {}, out
        
        if mode == 'train' and self.cfg.LOSS.OCC_CONSISTENCY > 0.0:
            bz, nv, c, h, w = batch['n_views']['image'].shape
            nview_images = batch['n_views']['image'].reshape(bz*nv,c,h,w)
            nview_masks = torch.cat([batch['obj_mask']]*nv, dim=1).reshape(bz*nv,1,h,w)
            nview_image_feat = self.enc(nview_images, mask=nview_masks)
            
            _, views_output = self.dec(nv_sdf, nview_image_feat, hA_view.reshape(bz*nv,-1), 
                                cTx_view.reshape(bz*nv,-1), nv_cameras, jsTx=jsTx_view)
            views_occ, views_occ_logits = views_output['pred_occ'], views_output['pred_occ_logits']
            if self.cfg.LOSS.OCCUPANCY == 0.0:
                pred_occ, pred_occ_logits = outputs['pred_occ'], outputs['pred_occ_logits']
            pred_occ_exp_logits = torch.stack([pred_occ_logits]*nv, dim=1).reshape(bz*nv,-1)
            loss_mask = torch.stack([batch['loss_mask']]*nv, dim=1).reshape(-1,1)
            
            occ_consistency_loss = self.bce_logits_loss(pred_occ_exp_logits, views_occ)
            occ_consistency_loss = torch.mean(loss_mask*occ_consistency_loss)
            losses['occ_consistency'] = occ_consistency_loss
            loss = loss + self.cfg.LOSS.OCC_CONSISTENCY*occ_consistency_loss
            enc_loss = enc_loss + self.cfg.LOSS.OCC_CONSISTENCY*occ_consistency_loss

        if self.cfg.XSEC:
            sdf = functools.partial(self.dec, z=image_feat, hA=batch['hA'], cTx=cTx, cam=cameras, jsTx=jsTx)
            samples, voxel_origin, voxel_size = mesh_utils.grid_xyz_samples(64)  # (P, 4)
            batch_size = hTx.shape[0]
            assert batch_size < 64
            
            ######## generate cross-sections by sampling planes at origin and randomly rotating them
            def rotate_points_randomly(samples):
                theta_x, theta_y, theta_z = torch.rand(1)*2*math.pi, torch.rand(1)*2*math.pi, torch.rand(1)*2*math.pi
                Rx = torch.tensor([[1, 0, 0], [0, torch.cos(theta_x), -torch.sin(theta_x)], [0, torch.sin(theta_x), torch.cos(theta_x)]])
                Ry = torch.tensor([[torch.cos(theta_y), 0, torch.sin(theta_y)], [0, 1, 0], [-torch.sin(theta_y), 0, torch.cos(theta_y)]])
                Rz = torch.tensor([[torch.cos(theta_z), -torch.sin(theta_z), 0], [torch.sin(theta_z), torch.cos(theta_z), 0], [0, 0, 1]])
                # ratate 3D point by Rz*Ry*Rx
                def rotate_point(point, Rz, Ry, Rx):
                    point = torch.matmul(point, Rx.T)
                    point = torch.matmul(point, Ry.T)
                    point = torch.matmul(point, Rz.T)
                    return point
                rot_points = rotate_point(samples[...,:3], Rz, Ry, Rx)
                return rot_points
            
            samples = samples.unsqueeze(0).repeat(batch_size, 1, 1)  # (B, P, 3)
            samples = samples.reshape(batch_size, 64, 64, 64, 4)
            x_plane_pts, y_plane_pts, z_plane_pts = samples[:,32,16:48,16:48], samples[:,16:48,32,16:48], samples[:,16:48,16:48,32]
            all_plane_pts = torch.cat([x_plane_pts, y_plane_pts, z_plane_pts], dim=0) # for passing only one plane points to the network
            all_plane_rot_pts = rotate_points_randomly(all_plane_pts)
            samples = torch.cat([all_plane_rot_pts, all_plane_pts[...,3].unsqueeze(-1)], dim=-1)
            sampled_grid_size = samples.shape[1]
            
            ######## xsec center plane points ########
            samples = samples.reshape(batch_size*3, -1, 4) # (B, N*N, 4)
            samples_subset = torch.gather(samples, 0, torch.randint(0, batch_size*3, (batch_size,)).unsqueeze(1).unsqueeze(2).repeat(1, samples.size(1), 4)) # (B, N*N, 4)
            samples_subset = samples_subset.to(self.device)
            sdf_out = sdf(samples_subset[...,:3])
            samples_subset[:, :, 3:4] = sdf_out[1]['pred_occ'].unsqueeze(2)
            samples_subset_occ = samples_subset.reshape(batch_size, sampled_grid_size, sampled_grid_size, -1)
            
            out['xsec'] = samples_subset_occ[...,3]
            out['xsec_pts'] = samples_subset[..., :3]

            if mode == 'train':
                gen_loss, disc_loss = self.xsec_loss(batch, out)
                loss = loss + self.cfg.LOSS.ENC*gen_loss + self.cfg.LOSS.DISC*disc_loss
                losses['gen'] = gen_loss
                losses['disc'] = disc_loss
                enc_loss = enc_loss + self.cfg.LOSS.ENC*gen_loss
        
        losses['loss'] = loss
        losses['enc'] = enc_loss
        return losses, out

    def xsec_loss(self, batch, out):
        xsec = out['xsec'].unsqueeze(1)
        real_xsec = batch['xsec'].unsqueeze(1)

        d_inp = torch.cat([xsec, real_xsec], dim=0)
        d_out = self.xsec_disc(d_inp).squeeze(-1)
        d_out_fake, d_out_real = torch.split(d_out, xsec.shape[0], dim=0)

        disc_val = {'d_out_real': d_out_real[0], 'd_out_fake': d_out_fake[0]}
        self.logger.log_metrics(disc_val, self.global_step)

        d_loss_real = ((d_out_real-1)**2)
        d_loss_fake = ((d_out_fake)**2)
        e_loss_disc = ((d_out_fake-1)**2)

        return torch.mean(e_loss_disc), torch.mean(d_loss_real) + torch.mean(d_loss_fake)


def main(cfg, args):
    pl.seed_everything(cfg.SEED)
    
    model = IHoi(cfg)
    if args.ckpt is not None:
        print('load from', args.ckpt)
        model = model.load_from_checkpoint(args.ckpt, cfg=cfg, strict=False)

    if args.eval:
        logger = MyLogger(save_dir=cfg.OUTPUT_DIR,
                        name=os.path.dirname(cfg.MODEL_SIG),
                        version=os.path.basename(cfg.MODEL_SIG),
                        subfolder=cfg.TEST.DIR,
                        resume=True,
                        )
        trainer = pl.Trainer(gpus='0,',
                             default_root_dir=cfg.MODEL_PATH,
                             logger=logger,
                            #  resume_from_checkpoint=args.ckpt,
                             )
        print(cfg.MODEL_PATH, trainer.weights_save_path, args.ckpt)

        model.freeze()
        trainer.test(model=model, verbose=False)
    else:
        logger = MyLogger(save_dir=cfg.OUTPUT_DIR,
                        name=os.path.dirname(cfg.MODEL_SIG),
                        version=os.path.basename(cfg.MODEL_SIG),
                        subfolder=cfg.TEST.DIR,
                        resume=args.slurm or args.ckpt is not None,
                        )
        checkpoint_callback = ModelCheckpoint(
            save_top_k=5,
            monitor='val_loss',
            mode='min',
            save_last=True,
        )
        lr_monitor = LearningRateMonitor()

        max_epoch = cfg.TRAIN.EPOCH
        use_sanity_val = 1
        if cfg.SOLVER.DEBUG:
            use_sanity_val = 0
        trainer = pl.Trainer(
                            gpus=-1,
                            accelerator='ddp',
                            num_sanity_val_steps=use_sanity_val,
                            limit_val_batches=1.0,
                            check_val_every_n_epoch=cfg.TRAIN.EVAL_EVERY,
                            default_root_dir=cfg.MODEL_PATH,
                            logger=logger,
                            max_epochs=max_epoch,
                            callbacks=[checkpoint_callback, lr_monitor, OnCheckpointHparams()],
                            progress_bar_refresh_rate=0 if args.slurm else None,
                            plugins=DDPPlugin(find_unused_parameters=True),
                            )
        trainer.fit(model)


if __name__ == '__main__':
    arg_parser = default_argument_parser()
    arg_parser = slurm_utils.add_slurm_args(arg_parser)
    args = arg_parser.parse_args()
    
    cfg = setup_cfg(args)
    save_dir = os.path.dirname(cfg.MODEL_PATH)
    slurm_utils.slurm_wrapper(args, save_dir, main, {'args': args, 'cfg': cfg}, resubmit=False)
