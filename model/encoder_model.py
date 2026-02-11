import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import kiui
import gaussian_renderer_dynamic
# from core.gs import GaussianRenderer
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.amp import autocast
from utils.loss_utils import l1_loss, ssim, msssim
from model.model_utils import GSPredictor
import pdb
from configs.options import Options
from kiui.lpips import LPIPS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from einops import rearrange

from model.midas_loss import ssitrim_loss, ssimse_loss

from model.mixture_model_utils import Truncated_Gaussian_Model

from utils.metrics import compute_psnr, compute_lpips, compute_ssim


class StaticEncoder(nn.Module):
    def __init__(self, opt: Options, **model_kwargs):
        super(StaticEncoder, self).__init__()
        self.opt = opt
        self.model = GSPredictor(opt, **model_kwargs)
        if hasattr(opt, 'compile') and opt.compile:
            self.model = torch.compile(self.model)
        self.gaussian_renderer = gaussian_renderer_dynamic.render
        self.background = torch.tensor(opt.background_color, dtype=torch.float32, device="cuda")

        # LPIPS loss
        self.lpips_loss = LPIPS(net='vgg')
        self.lpips_loss.eval()
        self.lpips_loss.requires_grad_(False)

    def state_dict(self, **kwargs):
        # remove lpips_loss
        state_dict = super().state_dict(**kwargs)
        for k in list(state_dict.keys()):
            if 'lpips_loss' in k:
                del state_dict[k]
        return state_dict
    
    def load_state_dict(self, state_dict, strict=True):
        # Optionally, if the LPIPS keys exist in the state_dict, remove them before loading.
        filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('lpips_loss')}
        return super().load_state_dict(filtered_state_dict, strict=False)
    
    def train(self, mode=True):
        super().train(mode)
        if 'lpips_loss' in self.__dict__:
            self.lpips_loss.eval()
        return self
    
    def forward_gaussians(self, frames, depths, cond_times=None):
        # frames: [B, V, C, H, W]
        # return: gaussians: [B, N, D]
        decoder_out = self.model(frames, depths, cond_times=cond_times)
        return decoder_out

    def forward(self, data, step_ratio=0.0):
        # data: [B, 2, C, H, W]
        input_frames = data['frames'][:, 0:1]  # [B, 1, C, H, W], input features
        input_depths = data['depths'][:, 0:1]  # [B, 1, C, H, W], input features

        results = {}
        
        # predict gaussians
        decoder_out = self.forward_gaussians(input_frames, input_depths)
        with autocast('cuda', enabled=False):
            render_pkg = self.gaussian_renderer(decoder_out["pred_gs"], self.background, opt=self.opt)
        output_frames = render_pkg["render"]
        pred_depths = render_pkg["depth"]
        mse_loss = F.mse_loss(output_frames, input_frames)
        loss = mse_loss

        if self.opt.depth_downsample:
            actual_h = int(self.opt.down_resolution[0] * (2 ** self.opt.decoder_ratio / self.opt.patch_size))
            actual_w = int(self.opt.down_resolution[1] * (2 ** self.opt.decoder_ratio / self.opt.patch_size))
            views = input_depths.shape[1]
            pred_depths = rearrange(pred_depths, 'b v c h w -> (b v) c h w')
            input_depths = rearrange(input_depths, 'b v c h w -> (b v) c h w')
            pred_depths = F.interpolate(pred_depths, (actual_h, actual_w), mode='bilinear', align_corners=True)
            input_depths = F.interpolate(input_depths, (actual_h, actual_w), mode='nearest')
            pred_depths = rearrange(pred_depths, '(b v) c h w -> b v c h w', v=views)
            input_depths = rearrange(input_depths, '(b v) c h w -> b v c h w', v=views)

        depth_loss = torch.zeros(1, device=input_frames.device)
        if self.opt.epoch > self.opt.depth_start_epoch:
            loss_func = ssitrim_loss if "trim" in self.opt.depth_loss_type else ssimse_loss
            depth_loss = loss_func(pred_depths, input_depths, None, self.opt.ignore_large_loss)  # normalized input_depths with no mask

            loss = loss + self.opt.lambda_depth * depth_loss
        
        if self.opt.lambda_lpips > 0 and self.opt.epoch > self.opt.lpips_start_epoch:
            down_res_H, down_res_W = self.opt.down_resolution
            loss_lpips = self.lpips_loss(
                F.interpolate(input_frames.reshape(-1, 3, down_res_H, down_res_W) * 2 - 1, (256, 256), mode='bilinear', align_corners=False), 
                F.interpolate(output_frames.reshape(-1, 3, down_res_H, down_res_W) * 2 - 1, (256, 256), mode='bilinear', align_corners=False),
            ).mean()
            results['loss_lpips'] = loss_lpips
            loss = loss + self.opt.lambda_lpips * loss_lpips
        
        pred_depth = 1.0 / (render_pkg["depth"] + 1e-8)
        B, V, C, H, W = pred_depth.shape
        reshaped_depth = pred_depth.view(B, V * C * H * W) # Shape [B*V, H*W]
        min_vals = reshaped_depth.min(dim=1, keepdim=True)[0] # Shape [B*V, 1]
        max_vals = reshaped_depth.max(dim=1, keepdim=True)[0] # Shape [B*V, 1]
        # Normalize the depth values
        pred_depth = (pred_depth - min_vals.view(B, 1, 1, 1, 1)) / (max_vals.view(B, 1, 1, 1, 1) - min_vals.view(B, 1, 1, 1, 1) + 1e-8)
        pred_depth = pred_depth.clamp(0, 1)  # Ensure values are between 0 and 1

        results['loss'] = loss
        results['mse_loss'] = mse_loss
        results['depth_loss'] = depth_loss
        results['pred_frames'] = output_frames
        results['gaussians'] = decoder_out["pred_gs"]
        results['pred_depths'] = pred_depth
        results['gt_depths'] = input_depths
            
        with torch.no_grad():
            down_res_H, down_res_W = self.opt.down_resolution
            input_frames_256 = F.interpolate(input_frames.reshape(-1, 3, down_res_H, down_res_W), (256, 256), mode='bilinear', align_corners=False)
            output_frames_256 = F.interpolate(output_frames.reshape(-1, 3, down_res_H, down_res_W), (256, 256), mode='bilinear', align_corners=False)

            psnr = compute_psnr(input_frames_256, output_frames_256)
            ssim = compute_ssim(input_frames_256, output_frames_256)
            lpips = compute_lpips(input_frames_256 * 2 - 1, output_frames_256 * 2 - 1)

            results['psnr'] = psnr.mean()
            results['ssim'] = ssim.mean()
            results['loss_lpips'] = lpips.mean()

        return results
    