import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import kiui
import gaussian_renderer_dynamic
from torch.amp import autocast
from model.model_utils import GSPredictor, SplatPredictor
from configs.options_decoder import Options
from model.encoder_model import StaticEncoder


class SplatModel(StaticEncoder):
    def __init__(self, opt: Options, **model_kwargs):
        super().__init__(opt)
        self.opt = opt
        self.model = SplatPredictor(opt, **model_kwargs)
        if hasattr(opt, 'compile') and opt.compile:
            self.model = torch.compile(self.model)
        self.gaussian_renderer = gaussian_renderer_dynamic.render
        self.background = torch.tensor(opt.background_color, dtype=torch.float32, device="cuda")
        self.lpips_loss = None
    
    def load_state_dict(self, state_dict, strict=True):
        # if opt.use_dino, remove missing keys related to condition_encoder
        missing_keys, unexpected_keys = super().load_state_dict(state_dict, strict=strict)
        if self.opt.use_dino:
            missing_keys = [k for k in missing_keys if "condition_encoder" not in k]
        return missing_keys, unexpected_keys
    
    def forward_gaussians(self, frames, depths, cond_times=None):
        # frames: [B, V, C, H, W]
        # return: gaussians: [B, N, D]
        decoder_out = self.model(frames, depths, cond_times=cond_times)
        return decoder_out
    
    def forward(self, data, step_ratio=0.0):
        # data: [B, V, C, H, W]
        input_frames = data['frames']  # [B, V, C, H, W], input features
        input_depths = data['depths']  # [B, V, C, H, W], input features
        timestamps = data['timestamps']  # [B, V], input timestamps
        timestamps = torch.as_tensor(timestamps, dtype=torch.float32, device=input_frames.device)
        timestamps = timestamps / timestamps[..., -1].unsqueeze(-1)
        anchor_time = torch.tensor([0.0, 1.0], device=input_frames.device)
        results = {}
        decoder_out = self.forward_gaussians(input_frames, input_depths, timestamps)  # dict
        with autocast('cuda', enabled=False):
            render_pkg = self.gaussian_renderer(decoder_out["pred_gs"], self.background, 
                                                opt=self.opt, timestamps=timestamps, 
                                                anchor_time=anchor_time,
                                                override_opacity=False, training=self.training,
                                                )
            
        results['pred_frames'] = render_pkg["render"]
        results['input_frames'] = input_frames
        results['timestamps'] = timestamps

        return results
    