import functools, os
import os.path as osp
from omegaconf.omegaconf import OmegaConf
from pytorch3d.renderer import PerspectiveCameras, mesh
from pytorch3d.renderer.mesh import textures
import torch
from typing import List, Tuple, Union, Callable
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
# from torch import torch_version

from config.defaults import get_cfg_defaults
from nnutils import model_utils

from nnutils import mesh_utils, image_utils, geom_utils
from nnutils.hand_utils import ManopthWrapper, get_nTh



def get_hoi_predictor(args):
    cfg_def = get_cfg_defaults()
    cfg_def = OmegaConf.create(cfg_def.dump())
    cfg = OmegaConf.load(osp.join(args.experiment_directory, 'hparams.yaml'))
    arg_cfg = OmegaConf.from_dotlist(['%s=%s' % (a,b) for a,b in zip(args.opts[::2], args.opts[1::2])])
    cfg = OmegaConf.merge(cfg_def, cfg, arg_cfg)
    cfg.MODEL.BATCH_SIZE = 1
    model = model_utils.load_model(cfg, args.experiment_directory, 'last')

    predictor = Predictor(model)
    return predictor


class Predictor:
    def __init__(self,model,):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.device = device
        self.model = model.to(device)
        self.hand_wrapper = ManopthWrapper().to(device)
    
    def forward_to_mesh(self, batch):
        model = self.model
        cfg = self.model.cfg
        hand_wrapper = self.hand_wrapper

        batch = model_utils.to_cuda(batch, self.device)

        hTx = geom_utils.matrix_to_se3(
            get_nTh(hand_wrapper, batch['hA'].cuda(), cfg.DB.RADIUS, inverse=True)) # what are the hard coded values?

        device = self.device
        hHand, hJoints = hand_wrapper(None, batch['hA'], mode='inner')

        image_feat = model.enc(batch['image'], mask=batch['obj_mask'])  # (N, D, H, W)

        cTx = geom_utils.compose_se3(batch['cTh'], hTx)

        hTjs = hand_wrapper.pose_to_transform(batch['hA'], False)  # (N, J, 4, 4)
        N, num_j, _, _ = hTjs.size()
        jsTh = geom_utils.inverse_rt(mat=hTjs, return_mat=True)
        hTx_exp = geom_utils.se3_to_matrix(hTx
                ).unsqueeze(1).repeat(1, num_j, 1, 1)
        jsTx = jsTh @ hTx_exp

        out = {'z': image_feat, 'jsTx': jsTx}

        camera = PerspectiveCameras(batch['cam_f'], batch['cam_p'], device=device)
        cTx = geom_utils.compose_se3(batch['cTh'], hTx)
        # normal space, joint space jsTn, image space 
        sdf = functools.partial(model.dec, z=out['z'], hA=batch['hA'], 
            jsTx=out['jsTx'].detach(), cTx=cTx.detach(), cam=camera)
        # TODO: handel empty predicdtion
        xObj = mesh_utils.batch_sdf_to_meshes(sdf, N, bound=True)

        hObj = mesh_utils.apply_transform(xObj, hTx)
        out['hObj'] = hObj
        out['hHand'] = hHand
        return out

    def hand_mesh_only(self, batch):
        hHand, hJoints = self.hand_wrapper(None, batch['hA'], mode='inner')
        out['hHand'] = hHand
        return out


def get_sdf_predictor(args):
    cfg_def = get_cfg_defaults()
    cfg_def = OmegaConf.create(cfg_def.dump())
    cfg = OmegaConf.load(osp.join(args.experiment_directory, 'hparams.yaml'))
    arg_cfg = OmegaConf.from_dotlist(['%s=%s' % (a,b) for a,b in zip(args.opts[::2], args.opts[1::2])])
    cfg = OmegaConf.merge(cfg_def, cfg, arg_cfg)
    cfg.MODEL.BATCH_SIZE = 1
    model = model_utils.load_model(cfg, args.experiment_directory, 'last')

    predictor = SDFPredictor(model)
    return predictor


class SDFPredictor:
    def __init__(self,model,):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.device = device
        self.model = model.to(device)
        self.hand_wrapper = ManopthWrapper().to(device)
    
    def forward_to_mesh(self, batch):
        model = self.model
        cfg = self.model.cfg
        hand_wrapper = self.hand_wrapper

        batch = model_utils.to_cuda(batch, self.device)

        hTx = geom_utils.matrix_to_se3(
            get_nTh(hand_wrapper, batch['hA'].cuda(), cfg.DB.RADIUS, inverse=True)) # what are the hard coded values?

        device = self.device
        hHand, hJoints = hand_wrapper(None, batch['hA'], mode='inner')

        image_feat = model.enc(batch['image'], mask=batch['obj_mask'])  # (N, D, H, W)

        cTx = geom_utils.compose_se3(batch['cTh'], hTx)

        hTjs = hand_wrapper.pose_to_transform(batch['hA'], False)  # (N, J, 4, 4)
        N, num_j, _, _ = hTjs.size()
        jsTh = geom_utils.inverse_rt(mat=hTjs, return_mat=True)
        hTx_exp = geom_utils.se3_to_matrix(hTx
                ).unsqueeze(1).repeat(1, num_j, 1, 1)
        jsTx = jsTh @ hTx_exp

        out = {'z': image_feat, 'jsTx': jsTx}

        camera = PerspectiveCameras(batch['cam_f'], batch['cam_p'], device=device)
        cTx = geom_utils.compose_se3(batch['cTh'], hTx)
        # normal space, joint space jsTn, image space 
        sdf = functools.partial(model.dec, z=out['z'], hA=batch['hA'], 
            jsTx=out['jsTx'].detach(), cTx=cTx.detach(), cam=camera)
        
        # TODO: handel empty predicdtion
        xObj = mesh_utils.batch_sdf_to_meshes(sdf, N, bound=True)
        hObj = mesh_utils.apply_transform(xObj, hTx)

        # compute sdf for points on the hand mesh
        xTh = geom_utils.inverse_rt(mat=geom_utils.se3_to_matrix(hTx) ,return_mat=True)
        xHand = mesh_utils.apply_transform(hHand, xTh) # apply_transform() supports both SE(3) & [R T] Mat inputs
        xPoints = xHand.verts_list()[0].unsqueeze(0) # (B, P, 3)
        xPointsSDF = batch_points_to_sdf(sdf, xPoints, N).detach().cpu()
        hPoints = mesh_utils.apply_transform(xPoints, hTx)
        hPointsSDF = torch.cat([hPoints, xPointsSDF.to(hPoints.device)], dim=-1)

        out['hObj'] = hObj
        out['hHand'] = hHand
        out['hPointsSDF'] = hPointsSDF
        return out


def batch_points_to_sdf(sdf: Callable, xPoints, batch_size, total_max_batch=32 ** 3, **kwargs):
    """convert a batched sdf to meshes
    Args:
        sdf (Callable): signature: sdf(points (N, P, 3), **kwargs) where kwargs should be filled 
        xPoints: points to compute sdf
        batch_size ([type]): batch size in **kwargs
        total_max_batch ([type], optional): [description]. Defaults to 32**3.
    Returns:
        Mehses
    """
    samples = xPoints
    num_samples = samples.size(1)

    head = 0
    sdf_values = []
    max_batch = total_max_batch // batch_size
    while head < num_samples:
        sample_subset = samples[:, head: min(head + max_batch, num_samples), 0:3].cuda()
        sdf_out = sdf(sample_subset)
        sdf_values.append(sdf_out)
        head += max_batch

    sdf_values = torch.cat(sdf_values, dim=1)
    return sdf_values


def vis_hand_object(output, data, image, save_dir):
    hHand = output['hHand']
    hObj = output['hObj']
    device = hObj.device

    cam_f, cam_p = data['cam_f'], data['cam_p']
    cTh = data['cTh']

    hHand.textures = mesh_utils.pad_texture(hHand, 'blue')
    hHoi = mesh_utils.join_scene([hObj, hHand]).to(device)
    cHoi = mesh_utils.apply_transform(hHoi, cTh.to(device))
    cameras = PerspectiveCameras(cam_f, cam_p, device=device)
    iHoi = mesh_utils.render_mesh(cHoi, cameras,)
    image_utils.save_images(iHoi['image'], save_dir + '_cHoi', bg=data['image']/2+0.5, mask=iHoi['mask'])
    image_utils.save_images(data['image']/2+0.5, save_dir + '_inp')

    image_list = mesh_utils.render_geom_rot(cHoi, cameras=cameras, view_centric=True)
    image_utils.save_gif(image_list, save_dir + '_cHoi')

    mesh_utils.dump_meshes([save_dir + '_hoi'], hHoi)

def vis_hand(output, data, image, save_dir, mode='right'):
    hHand = output['hHand']
    device = hHand.device

    cam_f, cam_p = data['cam_f'], data['cam_p']
    cTh = data['cTh']

    hHand.textures = mesh_utils.pad_texture(hHand, 'blue')
    cHand = mesh_utils.apply_transform(hHand, cTh.to(device))
    cameras = PerspectiveCameras(cam_f, cam_p, device=device)
    iHand = mesh_utils.render_mesh(cHand, cameras,)
    image_utils.save_images(iHand['image'], save_dir + '_cHand_{}'.format(mode), bg=data['image']/2+0.5, mask=iHand['mask'])
    image_utils.save_images(data['image']/2+0.5, save_dir + '_inp_{}'.format(mode))

    # image_list = mesh_utils.render_geom_rot(cHand, cameras=cameras, view_centric=True)
    # image_utils.save_gif(image_list, save_dir + '_cHand')

    # mesh_utils.dump_meshes([save_dir + '_hoi'], hHand)


def vis_points_hand_object(output, data, image, save_dir):
    hHand = output['hHand']
    hObj = output['hObj']
    device = hObj.device

    cam_f, cam_p = data['cam_f'], data['cam_p']
    cTh = data['cTh']

    # project sdf points into the image and visualize
    hPointsSDF = output['hPointsSDF']
    hPoints = hPointsSDF[..., :3]
    cameras = PerspectiveCameras(cam_f, cam_p, device=device)
    cPoints = mesh_utils.apply_transform(hPoints, cTh.to(device))
    ndcPoints = mesh_utils.transform_points(cPoints, cameras)
    sdf_val = hPointsSDF[..., 3]
    ndcPoints = ndcPoints[sdf_val<0.05].unsqueeze(0)

    # sdf_val = hPointsSDF[..., 3]
    textures = torch.ones_like(hPoints)
    # textures[sdf_val<0.05] = torch.FloatTensor([1,0,0]).cuda()
    for i in range(textures.shape[1]): # currently works only for batch_size=0
        textures[0,i] = torch.FloatTensor(list(colfunc(sdf_val[0,i].item()))).cuda()

    # hHand.textures = mesh_utils.pad_texture(hHand, 'blue')
    hHand.textures = mesh_utils.pad_texture(hHand, textures)
    hHoi = mesh_utils.join_scene([hObj, hHand]).to(device)
    cHoi = mesh_utils.apply_transform(hHoi, cTh.to(device))
    cameras = PerspectiveCameras(cam_f, cam_p, device=device)
    iHoi = mesh_utils.render_mesh(cHoi, cameras,)
    image_utils.save_images(iHoi['image'], os.path.join(save_dir, 'cHoi'), bg=data['image']/2+0.5, mask=iHoi['mask'])
    image_utils.save_images(data['image']/2+0.5, os.path.join(save_dir, 'inp'))

    image_list = mesh_utils.render_geom_rot(cHoi, cameras=cameras, view_centric=True)
    image_utils.save_gif(image_list, os.path.join(save_dir, 'cHoi'))

    mesh_utils.dump_meshes([os.path.join(save_dir, 'hoi')], hHoi)

    points_to_pix(ndcPoints.cpu(), data['image'], save_dir)


def points_to_pix(points, image, save_dir):
    N, C, H, W = image.shape
    ix = points[..., 0]
    iy = points[..., 1]

    ix = ((ix + 1) / 2) * (W-1)
    iy = ((iy + 1) / 2) * (H-1)

    torch.clamp(ix, 0, W-1, out=ix)
    torch.clamp(iy, 0, H-1, out=iy)

    plt_image = np.array(Image.open(os.path.join(save_dir, 'cHoi.png')))
    # plt_image = image[0].permute(1,2,0).cpu().numpy()
    # plt_image = plt_image/2+0.5
    plt.imshow(plt_image)
    plt.scatter(ix[0].cpu().numpy(), iy[0].cpu().numpy(), marker='x', color='red', s=20)
    plt.savefig(os.path.join(save_dir, 'plt_points.png'))

def colfunc(val, minval=0.0, maxval=1.0, startcolor=(1,0,0), stopcolor=(0,0,1)):
    """ Convert value in the range minval...maxval to a color in the range
        startcolor to stopcolor. The colors passed and the one returned are
        composed of a sequence of N component values (e.g. RGB).
    """
    f = float(val-minval) / (maxval-minval)
    return tuple(f*(b-a)+a for (a, b) in zip(startcolor, stopcolor))

def batch_colfunc(val, minval=-0.05, maxval=0.05, startcolor=[1,0,0], stopcolor=[1,1,0]):
    """ Convert value in the range minval...maxval to a color in the range {(0.0, 0.15) used initially}
        startcolor to stopcolor. The colors passed and the one returned are
        composed of a sequence of N component values (e.g. RGB).
    """
    f = torch.clamp(val, minval, maxval)
    f = (f-minval) / (maxval-minval)
    bz, np, _ = val.shape
    red = torch.FloatTensor(startcolor).unsqueeze(0).unsqueeze(0).repeat(bz, np, 1).to(val.device)
    blue = torch.FloatTensor(stopcolor).unsqueeze(0).unsqueeze(0).repeat(bz, np, 1).to(val.device)
    color = f.repeat(1,1,3)*(blue - red) + red
    return color