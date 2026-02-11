import torch
from torch.nn import functional as F
import math
from io import BytesIO
import numpy as np
from typing import Union

from diff_gaussian_rasterization_kiui_orth import GaussianRasterizationSettings as GaussianRasterizationSettingsOrth 
from diff_gaussian_rasterization_kiui_orth import GaussianRasterizer as GaussianRasterizerOrth

from configs.options import Options

import numpy as np 
import pdb

R_fixed = np.array([[ 1., -0., 0.],
                [ 0., -0., 1.],
                [ 0., -1., 0.]])
T_fixed = np.array([0., 0., 0.])

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

world_view_transform = torch.tensor(getWorld2View2(R_fixed, T_fixed, np.array([0.0, 0.0, 0.0]), 1.0)).transpose(0, 1)

def getOrthProjectionMatrix():
    # znear, zfar = 0., 1.0
    znear, zfar = 0., 10.0
    top = 1
    bottom = -top
    right = 1
    left = -right
    # Create an identity matrix
    P = torch.zeros(4, 4)
    z_sign = 1.0  # Adjust this based on handedness (e.g., 1.0 for right-handed, -1.0 for left-handed)
    P[0, 0] = 2.0 / (right - left)
    P[1, 1] = 2.0 / (top - bottom)
    P[2, 2] = -2.0 * z_sign / (zfar - znear)
    P[0, 3] = -(right + left) / (right - left)
    P[1, 3] = -(top + bottom) / (top - bottom)
    P[2, 3] = -(zfar + znear) / (zfar - znear)
    P[3, 3] = 1.0

    return P

projection_matrix = getOrthProjectionMatrix().transpose(0,1)

full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)

camera_center = world_view_transform.inverse()[3, :3] # default is the world coordinate frame origin

tanfovx = math.tan(math.pi / 4.0)
tanfovy = math.tan(math.pi / 4.0)

def render(gaussians: dict, bg_color: torch.Tensor, timestamps: torch.Tensor = None, scaling_modifier=1.0, 
           opt: Options=None, anchor_time: torch.Tensor=None,
           training=True, override_opacity=False,
           ):
    
    # random background color augmentation
    if training:
        bg_color = torch.rand(3).cuda()
    else:
        # bg_color = torch.tensor([1.0, 1.0, 1.0]).cuda()
        bg_color = torch.tensor([0.5, 0.5, 0.5]).cuda()

    # bg_color = torch.tensor([0.5, 0.5, 0.5]).cuda() 
    L = 0
    LP = opt.forder
    
    batch_size, gaussian_num = gaussians['xyz'].shape[0], gaussians['xyz'].shape[1]
    
    screenspace_points = torch.zeros_like(gaussians['xyz'][:, :, 0, :], dtype=gaussians['xyz'].dtype, requires_grad=True, device=gaussians['xyz'].device)
    screenspace_points.retain_grad()

    view_matrix = world_view_transform.float()  # View matrix
    view_proj_matrix = full_proj_transform.float()  # Projection matrix
    campos = camera_center.float()  # Camera position

    if len(opt.down_resolution) > 0:
        render_height, render_width = opt.down_resolution
    else:
        render_height, render_width = opt.image_height, opt.image_width
    raster_settings = GaussianRasterizationSettingsOrth(
        image_height=render_height,
        image_width=render_width,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color if bg_color is not None else bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=view_matrix.cuda(),
        projmatrix=view_proj_matrix.cuda(),
        sh_degree=0,  
        campos=campos.cuda(),
        prefiltered=False,
        debug=False,
    )
    
    rasterizer = GaussianRasterizerOrth(raster_settings=raster_settings)
    render_images = []
    render_depths = []
    final_means = []
    final_opacities = []
    render_alphas = []
    gs_tracks = []

    dummy_time = torch.zeros(1, device=gaussians['xyz'].device)
    output_frames = opt.output_frames
    N = gaussians['xyz'].shape[1]
    if timestamps is None:
        output_frames = 1
        timestamps = dummy_time.repeat(batch_size, output_frames)
    if anchor_time is None:
        anchor_time = torch.zeros((N, 1), device=gaussians['xyz'].device)
    else:
        # normally [0.0, 1.0]
        anchor_time = anchor_time.repeat_interleave(N//2).unsqueeze(-1)  # [N, 1]

    for b in range(batch_size):  # Loop over batch
        means3D_static = gaussians['xyz'][b, :, 0, :].contiguous().float()  # Static position [N, 3]
        dynamic_components = gaussians['xyz'][b, :, 1:, :].contiguous().float()  # Dynamic components [N, 2T, 3]
        opacity = gaussians['opacity'][b, :, :1].contiguous().float()  # Opacity [N, 1]
        if gaussians['opacity'].shape[2] == 1:
            opacity_dynamic = None
        else:
            opacity_dynamic = gaussians['opacity'][b, :, 1:].contiguous().float()  # Dynamic opacity [N, T]
        scales = gaussians['scale'][b, :, :].contiguous().float()  # Scaling [N, 2]
        y_scale = scales.mean(dim=-1, keepdim=True)  # [N, 1]
        scales = torch.cat([scales[..., 0:1], y_scale, scales[..., 1:2]], dim=-1)
        rotations_static = gaussians['rot'][b, :, 0, :].contiguous().float()  # Static rotation [N, 4]
        rotations_dynamic = gaussians['rot'][b, :, 1, :].contiguous().float()  # Dynamic rotation [N, 4]
        colors_precomp = gaussians['rgb'][b, :, :].contiguous().float()  # SH features [N, 3 * (max_sh_degree + 1)**2]

        for tidx in range(output_frames):  # Loop over time slices
            actual_time = timestamps[b, tidx]
            time_basis = actual_time - anchor_time
            if LP > 0:
                polynomial_basis = torch.stack([time_basis ** i for i in range(1, LP+1)], dim=1)  # [N, LP, 1]
                dynamic_poly = (dynamic_components[:, 2 * L:, :] * polynomial_basis).sum(dim=1)
            else:
                dynamic_poly = torch.zeros_like(means3D_static, dtype=means3D_static.dtype, device=means3D_static.device)
            
            dynamic_means = dynamic_poly
            final_means3D = means3D_static + dynamic_means

            if opt.pred_inverse:
                # unnormalize_invdepth
                d_hat = final_means3D[..., 1]
                valid_mask = (d_hat >= 0) & (d_hat <= 1)
                inv_min, inv_max = 0.2, 10.0
                d_inv = d_hat * (inv_max - inv_min) + inv_min
                d = 1 / (d_inv + 1e-6)
                final_means3D[..., 1] = torch.where(valid_mask, d, final_means3D[..., 1] - 10)  # make sure invalid points won't be rendered

            final_rotations = rotations_static + rotations_dynamic * time_basis  # Final rotations [N, 4]

            if opacity_dynamic is not None:
                opacity_dynamic_coef = torch.sigmoid(-opacity_dynamic[:, 0:1] * (time_basis.abs() - opacity_dynamic[:, 1:])) / torch.sigmoid(opacity_dynamic[:, 0:1] * opacity_dynamic[:, 1:])
            else:
                opacity_dynamic_coef = torch.ones_like(opacity)
            if override_opacity:
                opacity_dynamic_coef = (time_basis.abs() <= 0.5).float().to(opacity_dynamic_coef.device)
            final_opacity = opacity * opacity_dynamic_coef  # Final opacity [N, 1]

            # Rasterize visible Gaussians
            rendered_image, radii, depth, alpha = rasterizer(
                means3D=final_means3D,  # Final 3D Gaussian positions
                means2D=torch.zeros_like(final_means3D, dtype=torch.float32), # 2D screen coordinates (unused)
                shs=None,  # Use SH features if available
                colors_precomp=colors_precomp,  # Precomputed colors (SH features)
                opacities=final_opacity,  # Opacity
                scales=scales,  # Scaling factors
                rotations=final_rotations  # Time-dependent rotations
            )
            render_images.append(rendered_image)
            render_depths.append(depth)
            render_alphas.append(alpha)
        
    render_images = torch.stack(render_images, dim=0).view(batch_size, output_frames, 3, render_images[0].shape[-2], render_images[0].shape[-1])
    render_depths = torch.stack(render_depths, dim=0).view(batch_size, output_frames, 1, render_depths[0].shape[-2], render_depths[0].shape[-1])
    render_alphas = torch.stack(render_alphas, dim=0).view(batch_size, output_frames, 1, render_depths[0].shape[-2], render_depths[0].shape[-1])
    
    return {
        "render": render_images,
        "depth": render_depths,
        "alpha": render_alphas,
    }

    