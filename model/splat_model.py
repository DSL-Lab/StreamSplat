import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import kiui
import gaussian_renderer_dynamic
from torch.amp import autocast
from utils.loss_utils import l1_loss, ssim, msssim
from model.model_utils import GSPredictor, SplatPredictor
from configs.options_decoder import Options
from kiui.lpips import LPIPS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from model.midas_loss import ssitrim_loss, ssimse_loss
from model.encoder_model import StaticEncoder
from utils.metrics import compute_psnr, compute_ssim, compute_lpips


def _count_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

class SplatModel(StaticEncoder):
    def __init__(self, opt: Options, **model_kwargs):
        super().__init__(opt)
        self.opt = opt
        self.model = SplatPredictor(opt, **model_kwargs)
        if hasattr(opt, 'compile') and opt.compile:
            self.model = torch.compile(self.model)
        self.gaussian_renderer = gaussian_renderer_dynamic.render
        self.background = torch.tensor(opt.background_color, dtype=torch.float32, device="cuda")

        # LPIPS loss
        if self.opt.lambda_lpips > 0:
            self.lpips_loss = LPIPS(net='vgg')
            self.lpips_loss.requires_grad_(False)
            self.lpips_loss.eval()
            if hasattr(opt, 'compile') and opt.compile:
                self.lpips_loss = torch.compile(self.lpips_loss)
        
    def train(self, mode=True):
        super().train(mode)
        if 'lpips_loss' in self.__dict__:
            self.lpips_loss.eval()
        return self

    def state_dict(self, **kwargs):
        """Remove non-trainable modules (LPIPS, tracker) from state dict before saving."""
        state_dict = super().state_dict(**kwargs)
        for k in list(state_dict.keys()):
            if 'lpips_loss' in k or 'tracker_prior' in k:
                del state_dict[k]
        return state_dict
    
    def load_state_dict(self, state_dict, strict=True):
        """Load state dict, filtering out non-trainable modules if present."""
        filtered_state_dict = {k: v for k, v in state_dict.items() 
                             if not k.startswith('lpips_loss') and not k.startswith('tracker_prior')}
        return super().load_state_dict(filtered_state_dict, strict=strict)
        
    def forward_gaussians(self, frames, depths, cond_times=None):
        # frames: [B, V, C, H, W]
        # return: gaussians: [B, N, D]
        decoder_out = self.model(frames, depths, cond_times=cond_times)
        return decoder_out
    
    def compute_losses(self, input_frames, target_depth, supv_masks, render_pkg):
        output_frames = render_pkg["render"]  # [B, V, C, H, W]
        pred_depths = render_pkg["depth"]
        depth_mask = render_pkg["alpha"] > 0.1  # [B, V, 1, H, W]
        metrics = {}
        with torch.no_grad():
            B, V, C, H, W = input_frames.shape
            
            # All frames
            input_frames_all_256 = F.interpolate(input_frames.reshape(-1, 3, H, W), (256, 256), mode='bilinear', align_corners=False)
            output_frames_all_256 = F.interpolate(output_frames.reshape(-1, 3, H, W), (256, 256), mode='bilinear', align_corners=False)
            metrics['psnr'] = compute_psnr(input_frames_all_256, output_frames_all_256).mean()
            metrics['ssim'] = compute_ssim(input_frames_all_256, output_frames_all_256).mean()
            metrics['lpips'] = compute_lpips(input_frames_all_256 * 2 - 1, output_frames_all_256 * 2 - 1).mean()

            metrics['input_frames'] = input_frames
            metrics['pred_frames'] = output_frames

            # Middle frames [1:-1]
            if V > 2:
                input_frames_middle = input_frames[:, 1:-1]
                output_frames_middle = output_frames[:, 1:-1]
                
                input_frames_middle_256 = F.interpolate(input_frames_middle.reshape(-1, 3, H, W), (256, 256), mode='bilinear', align_corners=False)
                output_frames_middle_256 = F.interpolate(output_frames_middle.reshape(-1, 3, H, W), (256, 256), mode='bilinear', align_corners=False)

                metrics['psnr_novel'] = compute_psnr(input_frames_middle_256, output_frames_middle_256).mean()
                metrics['ssim_novel'] = compute_ssim(input_frames_middle_256, output_frames_middle_256).mean()
                metrics['lpips_novel'] = compute_lpips(input_frames_middle_256 * 2 - 1, output_frames_middle_256 * 2 - 1).mean()
            else:
                metrics['psnr_novel'] = torch.tensor(0.0, device=input_frames.device)
                metrics['ssim_novel'] = torch.tensor(0.0, device=input_frames.device)
                metrics['lpips_novel'] = torch.tensor(0.0, device=input_frames.device)

            # First and last frames (given views)
            if V >= 1:
                indices = [0]
                if V > 1:
                    indices.append(V - 1)
                
                input_frames_ends = input_frames[:, indices]
                output_frames_ends = output_frames[:, indices]

                input_frames_ends_256 = F.interpolate(input_frames_ends.reshape(-1, 3, H, W), (256, 256), mode='bilinear', align_corners=False)
                output_frames_ends_256 = F.interpolate(output_frames_ends.reshape(-1, 3, H, W), (256, 256), mode='bilinear', align_corners=False)
                
                metrics['psnr_given'] = compute_psnr(input_frames_ends_256, output_frames_ends_256).mean()
                metrics['ssim_given'] = compute_ssim(input_frames_ends_256, output_frames_ends_256).mean()
                metrics['lpips_given'] = compute_lpips(input_frames_ends_256 * 2 - 1, output_frames_ends_256 * 2 - 1).mean()
            else: # Should not happen if V >= 1
                metrics['psnr_given'] = torch.tensor(0.0, device=input_frames.device)
                metrics['ssim_given'] = torch.tensor(0.0, device=input_frames.device)
                metrics['lpips_given'] = torch.tensor(0.0, device=input_frames.device)
            
            psnr = metrics['psnr']

        if self.opt.skip:
            # skip the frame_0
            input_frames = input_frames[:, 1:]
            target_depth = target_depth[:, 1:]
            output_frames = output_frames[:, 1:]
            pred_depths = pred_depths[:, 1:]
            supv_masks = supv_masks[:, 1:]
            depth_mask = depth_mask[:, 1:]

            # skip the last frame
            input_frames = input_frames[:, :-1]
            target_depth = target_depth[:, :-1]
            output_frames = output_frames[:, :-1]
            pred_depths = pred_depths[:, :-1]
            supv_masks = supv_masks[:, :-1]
            depth_mask = depth_mask[:, :-1]

        mse_loss = F.mse_loss(output_frames, input_frames)
        loss = mse_loss 
        supv_mse_loss = torch.zeros(1, device=input_frames.device)
        if self.opt.lambda_mask > 0:
            supv_masks = supv_masks.expand(-1, -1, 3, -1, -1)  # [B, V, C, H, W]
            supv_mse_loss = F.mse_loss(output_frames[supv_masks], input_frames[supv_masks])
            loss += self.opt.lambda_mask * supv_mse_loss

        depth_loss = torch.zeros(1, device=input_frames.device)
        if self.opt.epoch >= self.opt.depth_start_epoch:
            loss_func = ssitrim_loss if "trim" in self.opt.depth_loss_type else ssimse_loss
            depth_loss = loss_func(pred_depths, target_depth, depth_mask, self.opt.ignore_large_loss)

            loss = mse_loss + self.opt.lambda_depth * depth_loss
        
        loss_lpips = torch.tensor(0.0, device=input_frames.device)
        if self.opt.lambda_lpips > 0 and self.opt.epoch >= self.opt.lpips_start_epoch:
            down_res_H, down_res_W = self.opt.down_resolution
            loss_lpips = self.lpips_loss(
                F.interpolate(input_frames.reshape(-1, 3, down_res_H, down_res_W) * 2 - 1, (256, 256), mode='bilinear', align_corners=False), 
                F.interpolate(output_frames.reshape(-1, 3, down_res_H, down_res_W) * 2 - 1, (256, 256), mode='bilinear', align_corners=False),
            ).mean()
            loss = loss + self.opt.lambda_lpips * loss_lpips

        return loss, mse_loss, supv_mse_loss, depth_loss, loss_lpips, psnr, metrics


    def forward(self, data, step_ratio=0.0):
        # data: [B, V, C, H, W]
        input_frames = data['frames']  # [B, V, C, H, W], input features
        input_depths = data['depths']  # [B, V, C, H, W], input features
        timestamps = data['timestamps']  # [B, V], input timestamps
        supv_masks = data['supv_masks']  # [B, V], input timestamps
        if supv_masks is None:
            supv_masks = torch.ones_like(input_depths, device=timestamps.device).bool()
        timestamps = torch.as_tensor(timestamps, dtype=torch.float32, device=input_frames.device)
        timestamps = timestamps / timestamps[..., -1].unsqueeze(-1)
        anchor_time = torch.tensor([0.0, 1.0], device=input_frames.device)
        supv_masks = (supv_masks > 0)
       
        max_depth = input_depths.flatten(2).max(dim=2)[0][:, :, None, None, None]
        min_depth = input_depths.flatten(2).min(dim=2)[0][:, :, None, None, None]
        target_depth = input_depths # without normalization
        input_depths = (input_depths - min_depth) / (max_depth - min_depth)
        
        results = {}
        
        # predict gaussians
        decoder_out = self.forward_gaussians(input_frames, input_depths, timestamps)  # dict
        # pdb.set_trace()

        with autocast('cuda', enabled=False):
            render_pkg = self.gaussian_renderer(decoder_out["pred_gs"], self.background, 
                                                opt=self.opt, timestamps=timestamps, 
                                                anchor_time=anchor_time,
                                                training=self.training,
                                                )
            
            loss, mse_loss, supv_mse_loss, depth_loss, loss_lpips, psnr, metrics = self.compute_losses(input_frames, target_depth, supv_masks, render_pkg)

            if hasattr(self.opt, 'fix_opacity') and self.opt.fix_opacity:
                render_pkg_fix = self.gaussian_renderer(decoder_out["pred_gs"], self.background, 
                                                    opt=self.opt, timestamps=timestamps,
                                                    anchor_time=anchor_time,
                                                    override_opacity=True, training=self.training,
                                                    )
                loss_fix, mse_loss_fix, supv_mse_loss_fix, depth_loss_fix, loss_lpips_fix, psnr_fix, metrics_fix = self.compute_losses(input_frames, target_depth, supv_masks, render_pkg_fix)
                loss = (loss + loss_fix) * 0.5
            else:
                render_pkg_fix = None
                mse_loss_fix = torch.zeros_like(mse_loss)
                depth_loss_fix = torch.zeros_like(depth_loss)  
                loss_lpips_fix = torch.zeros_like(loss_lpips)
                supv_mse_loss_fix = torch.zeros_like(supv_mse_loss)
                psnr_fix = torch.zeros_like(psnr)
        
        pred_depth = render_pkg["depth"]
        depth_mask = pred_depth >= 0.2  # [B, V, 1, H, W]
        pred_depth[~depth_mask] += 10  # set invalid depth to max depth
        pred_depth = 1.0 / (pred_depth + 1e-8)
        B, V, C, H, W = pred_depth.shape
        # Reshape to treat each depth map independently for min/max calculation
        reshaped_depth = pred_depth.view(B, V*C * H * W) # Shape [B*V, H*W]
        min_vals = reshaped_depth.min(dim=1, keepdim=True)[0] # Shape [B*V, 1]
        max_vals = reshaped_depth.max(dim=1, keepdim=True)[0] # Shape [B*V, 1]
        # Normalize the depth values
        pred_depth = (pred_depth - min_vals.view(B, 1, 1, 1, 1)) / (max_vals.view(B, 1, 1, 1, 1) - min_vals.view(B, 1, 1, 1, 1) + 1e-8)
        pred_depth = pred_depth.clamp(0, 1)  # Ensure values are between 0 and 1

        results['pred_depths'] = pred_depth
        results['gt_depths'] = target_depth
        results['pred_frames'] = render_pkg["render"]
        results['input_frames'] = input_frames
        results['timestamps'] = timestamps

        if render_pkg_fix is not None:
            results['pred_depths_fix'] = 1.0 / (render_pkg_fix["depth"] + 1e-8)
            results['pred_frames_fix'] = render_pkg_fix["render"]
        else:
            # dummy results to avoid errors
            results['pred_depths_fix'] = torch.zeros_like(results['pred_depths'])
            results['pred_frames_fix'] = torch.zeros_like(results['pred_frames'])
        
        results['loss'] = loss
        results['mse_loss'] = mse_loss
        results['depth_loss'] = depth_loss
        results['loss_lpips'] = loss_lpips
        results['supv_mse_loss'] = supv_mse_loss
        results['psnr'] = psnr
        results['mse_loss_fix'] = mse_loss_fix
        results['depth_loss_fix'] = depth_loss_fix
        results['loss_lpips_fix'] = loss_lpips_fix
        results['supv_mse_loss_fix'] = supv_mse_loss_fix
        results['psnr_fix'] = psnr_fix
        results['metrics'] = metrics

        return results
    
    