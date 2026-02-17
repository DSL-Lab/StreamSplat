import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from configs.options_decoder import Options
import math
from einops import rearrange
from collections import OrderedDict
from torch.utils.checkpoint import checkpoint
import numpy as np
import pdb
from encoders.dinov2_wrapper import Dinov2Wrapper
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    NamedTuple,
    NewType,
    Optional,
    Sized,
    Tuple,
    Type,
    TypeVar,
    Union,
)
from torch.amp import autocast

from model.depth_wrapper import DepthAnythingWrapper

from model.mixture_model_utils import Truncated_Gaussian_Model

from model.transformer_utils import TransformerEncoder, GaussianUpsampler, TransformerConditionalDecoder, MLP

        
def _count_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

class GSEncoder(nn.Module):
    def __init__(self, opt: Options, **kwargs):
        super(GSEncoder, self).__init__()
        self.width = opt.hidden_dim 
        self.patch_size = opt.patch_size
        self.num_layers = opt.num_layers

        if len(opt.down_resolution) > 0:
            self.actual_input_res = opt.down_resolution
        else:
            self.actual_input_res = (opt.image_height, opt.image_width)

        self.in_channels = opt.in_channels

        if opt.enable_depth:
            self.in_channels += 1

        self.transformer_encoder = TransformerEncoder(
            in_channels=self.in_channels,
            input_res=self.actual_input_res,
            patch_size=self.patch_size,
            layers=self.num_layers,
            width=self.width,
            heads=self.width // 64,
            window_size=opt.bwindow_size
        )
        self.transformer_encoder.set_grad_checkpointing(opt.checkpointing)

    def forward(self, x, timestamp=None):
        assert x.dim() == 5, f"Input shape should be [b, #views, c, h, w] but {x.shape} is given"
        batch_size, input_views = x.shape[0], x.shape[1]

        features = self.transformer_encoder(x, timestamp)  # [B, V, N, D]
        return features

class SplatDecoder(nn.Module):
    def __init__(self, opt: Options, **kwargs):
        super().__init__()

        self.opt = opt
        self.width = opt.decoder_hidden_dim
        self.patch_size = opt.patch_size
        self.input_res = opt.down_resolution
        self.num_layers = opt.decoder_num_layers

        if len(opt.down_resolution) > 0:
            self.actual_input_res = opt.down_resolution
        else:
            self.actual_input_res = (opt.image_height, opt.image_width)

        self.transformer_decoder = TransformerConditionalDecoder(
            input_res=self.actual_input_res,
            patch_size=self.patch_size,
            layers=self.num_layers,
            width=self.width,
            heads=self.width // 64,
            window_size=opt.bwindow_size,
            condition_dim=opt.hidden_dim,
            condition_len=576 if opt.use_dino else 2304,
            encoder_dim=opt.hidden_dim,
            drop_path_rate=opt.drop_path_rate,
        )
        self.token_len = (self.actual_input_res[0] // self.patch_size) * (self.actual_input_res[1] // self.patch_size)

        self.transformer_decoder.set_grad_checkpointing(opt.checkpointing)
    
    def forward(self, latent, condition=None):
        features = self.transformer_decoder(latent, condition)  # [B, V, N, D]

        return features

inverse_sigmoid = lambda x: np.log(x / (1 - x))
artanh = lambda x: 0.5 * np.log((1 + x) / (1 - x))

class GSPMDecoder(nn.Module):
    def __init__(self,
                 opt: Options, 
                 transformer_dim: int, 
                 mlp_dim=None, 
                 init_density=0.2, 
                 clip_scaling=0.1,
                 bias=True,
                 ):
        super(GSPMDecoder, self).__init__()
        
        self.embed_dim = transformer_dim
        if mlp_dim is not None:
            self.mlp_dim = mlp_dim
        else:
            self.mlp_dim = transformer_dim
        self.clip_scaling = clip_scaling
        self.opt = opt

        token_w = self.opt.down_resolution[1] * (2 ** self.opt.decoder_ratio / self.opt.patch_size) # (512 / 2)
        token_h = self.opt.down_resolution[0] * (2 ** self.opt.decoder_ratio / self.opt.patch_size) # (288 / 2) when decoder_ratio = 3
        w_coords = torch.linspace(-1, 1, int(token_w) + 1, dtype=torch.float32)
        w_coords = (w_coords[1:] + w_coords[:-1]) / 2
        h_coords = torch.linspace(1, -1, int(token_h) + 1, dtype=torch.float32)
        h_coords = (h_coords[1:] + h_coords[:-1]) / 2
        z_map, x_map = torch.meshgrid(h_coords, w_coords, indexing='ij')
        self.register_buffer("z_map", z_map.flatten())
        self.register_buffer("x_map", x_map.flatten())

        self.z_max, self.x_max = 1. / (token_h), 1. / (token_w)

        self.ratio = 1
        self.actual_h = int(self.opt.down_resolution[0] * (2 ** self.opt.decoder_ratio / self.opt.patch_size))
        self.actual_w = int(self.opt.down_resolution[1] * (2 ** self.opt.decoder_ratio / self.opt.patch_size))
        self.pixelshuffle = nn.PixelShuffle(upscale_factor=int(self.ratio**(0.5)))
        
        self.all_keys = ["xyz_static", "xyz_dynamic",
                          "rot_static", "rot_dynamic", 
                          "opacity", "opacity_dynamic",
                          "scale", "rgb"]
        self.key_dims = {"xyz_static": 3, "xyz_dynamic": 3 * (opt.forder),
                         "rot_static": 4, "rot_dynamic": 4,
                         "opacity": 1, "opacity_dynamic": 1,
                         "scale": 2, "rgb": 3,
                         "xz_scale": 2, "y_scale": 1}
        
        self.pred_keys = opt.pred_keys
        self.has_pred = len(self.pred_keys) > 0

        self.sample_keys = opt.sample_keys

        self.opacity_activation = opt.opacity_activation

        self.fix_keys = opt.fix_keys

        self.keep_dynamic = False  # default to False, change in splatpredictor

        self.scale_min = 0.001
        self.scale_max = 0.3

        self.nr_keys = len(self.sample_keys)
        self.nr_mix = 1
        self.n_sample = 1

        self.gs_layer = nn.ModuleDict()
        for key in self.pred_keys:
            if key in self.fix_keys:
                continue
            if key == "xyz_static":
                layer = nn.Linear(self.mlp_dim, 3 * self.ratio, bias=bias)
            elif key == "scale":
                self.scale_min = 0.001
                self.scale_max = 0.3
                layer = nn.Linear(self.mlp_dim, 2 * self.ratio, bias=bias)
            elif key == "rot_static":
                layer = nn.Linear(self.mlp_dim, 4 * self.ratio, bias=bias)
            elif key == "opacity":
                layer = nn.Linear(self.mlp_dim, 1 * self.ratio, bias=bias)
            elif key == "rgb":
                color_dim = 3
                layer = nn.Linear(self.mlp_dim, color_dim * self.ratio, bias=bias)
            else:
                raise NotImplementedError
            self.gs_layer[key] = layer
        
        self.prior = Truncated_Gaussian_Model(n_sample=self.n_sample, nr_mix=self.nr_mix)

        self.mix_layer = nn.ModuleDict()
        for keys in self.sample_keys:
            if keys == "xyz_static":
                self.add_layer(self.mix_layer, keys, 3, bias=bias)
            elif keys == "scale":
                self.add_layer(self.mix_layer, keys, 3, init_val=-0.5, bias=bias)  # tanh
            elif keys == "opacity":
                # self.add_layer(self.mix_layer, keys, 1, init_val=-1.0)  # init to 0.1
                self.add_layer(self.mix_layer, keys, 1, init_val=0., bias=bias)  
            elif keys == "rgb":
                self.add_layer(self.mix_layer, keys, 3, bias=bias)
            elif keys == "xyz_dynamic":
                self.add_layer(self.mix_layer, keys, self.key_dims[keys], bias=bias)
            else:
                raise NotImplementedError
            
        self.sampled_val = {}

    def add_layer(self, layer_dict, key, dim, init_val=0., bias=True):
        if self.nr_mix > 1:
            pred_prob = nn.Linear(self.mlp_dim, self.nr_mix * self.ratio, bias=bias) 
            torch.nn.init.xavier_uniform_(pred_prob.weight)
            layer_dict[f"{key}_prob"] = pred_prob

        pred_mean = nn.Linear(self.mlp_dim, self.nr_mix * dim * self.ratio, bias=bias)
        torch.nn.init.xavier_normal_(pred_mean.weight, 0.01)
        layer_dict[f"{key}_mean"] = pred_mean

        pred_scale = nn.Linear(self.mlp_dim, self.nr_mix * dim * self.ratio, bias=bias)
        torch.nn.init.xavier_normal_(pred_scale.weight, 0.01)
        layer_dict[f"{key}_scale"] = pred_scale

    def key_activation(self, v: torch.Tensor, key=''):
        # [B, N, D]
        B = v.shape[0]
        assert v.isnan().sum() == 0, f"NaN detected in {key}"
        v = v.type(torch.float32)
        if key == "xyz_static":
            # (B, N, 3) v shape
            v = torch.tanh(v)
            x_pred, y_pred, z_pred = v.chunk(3, dim=-1)
            y_val = 0.5 + y_pred * 0.5
            # predict x and z with pixel position
            x_offset = self.x_max * x_pred
            z_offset = self.z_max * z_pred
            x_map = self.x_map.repeat(self.opt.input_frames).reshape(1, -1, 1)
            z_map = self.z_map.repeat(self.opt.input_frames).reshape(1, -1, 1)
            x_val = x_map + x_offset
            z_val = z_map + z_offset
            v = torch.cat([x_val, y_val, z_val], dim=-1)
        elif key == "scale":
            v = 0.1 * F.softplus(v)
        elif key == "rot_static":
            v = F.normalize(v, dim=-1)
        elif key == "opacity":
            if self.opacity_activation == "sigmoid":
                v = torch.sigmoid(v)
                v = 0.05 + 0.95 * v
            else:
                v = F.relu(v + 10)
        elif key == "shs":
            pass 
        elif key == "rgb":
            v = torch.sigmoid(v)
        else:
            raise NotImplementedError
        if v.dim() == 3:
            v = v.repeat(1, 1, self.n_sample).reshape(v.shape[0], -1, v.shape[-1])  # maybe incorrect
        elif v.dim() == 4:
            v = v.repeat(1, 1, 1, self.n_sample).reshape(v.shape[0], -1, v.shape[-1])
        else:
            raise NotImplementedError
        return v

    @autocast('cuda', enabled=False)
    def forward(self, feats, timestamp=None):
        # [B, V, N, D]
        assert feats.shape[-1] == self.embed_dim
        B = feats.shape[0]
        input_views = feats.shape[1]
        N = feats.shape[1] * feats.shape[2] * self.ratio * self.n_sample
        feats = feats.type(torch.float32)
        feats = rearrange(feats, 'b v n d -> (b v) n d') 

        gsparams = {}
        prior_params = {}
        for key in self.fix_keys:
            if key == "scale":
                fix_v = 0.03 * torch.ones(B, N, 3).to(feats.device)
            elif key == "rot_static":
                fix_v = torch.zeros(B, N, 4).to(feats.device)
                fix_v[:, :, 0] = 1.
            elif key == "rot_dynamic":
                fix_v = torch.zeros(B, N, 4).to(feats.device)
            elif key == "xyz_dynamic":
                fix_v = torch.zeros(B, N, self.key_dims[key]//3, 3).to(feats.device)
            elif key == "opacity_dynamic":
                fix_v = torch.ones(B, N, 1).to(feats.device)
            else:
                raise NotImplementedError
            gsparams[key] = fix_v
        
        def reorder(v):
            # [B*V, H*W, d*r*r]
            v = rearrange(v, 'B (h w) D -> B D h w', h=self.actual_h, w=self.actual_w).contiguous()
            v = self.pixelshuffle(v)  # [B*V, d, H*r, W*r]
            v = rearrange(v, '(b v) d h w -> b (v h w) d', v=input_views)
            return v
        
        for key in self.pred_keys:
            v = feats
            v = self.gs_layer[key](v)  
            v = reorder(v)
            
            gsparams[key] = self.key_activation(v, key)
        
        for key in self.sample_keys:
            logits_pred = torch.ones(feats.shape[0], feats.shape[1], 1).to(feats.device).float()
            activation = "tanh"
            means = reorder(self.mix_layer[f"{key}_mean"](feats))  # [B, N, nr_mix * dim]
            log_scales = reorder(self.mix_layer[f"{key}_scale"](feats))
            logits, means, log_scales = self.prior.expand_params(logits_pred, means, log_scales, mean_activation=activation)
            prior_params[key] = {"logits": logits, "means": means, "log_scales": log_scales}  # [B, N*r, dim, nr_mix]
            
            if key == "xyz_static":
                val, probs = self.prior.sample(logits, means, log_scales)
                x_pred, y_pred, z_pred = val.chunk(3, dim=-1)
                y_val = (0.5 + y_pred * 0.5)
                assert y_val.min() >= 0. and y_val.max() <= 1., f"y_val: {y_val.min()}, {y_val.max()}"
                x_offset = self.x_max * x_pred
                z_offset = self.z_max * z_pred
                x_map = self.x_map.repeat(self.opt.input_frames).reshape(1, -1, 1)
                z_map = self.z_map.repeat(self.opt.input_frames).reshape(1, -1, 1)
                x_val = x_map + x_offset
                z_val = z_map + z_offset
                gsparams["xyz_static"] = torch.cat([x_val, y_val, z_val], dim=-1)
            elif key == "scale":
                val, probs = self.prior.sample(logits, means, log_scales, b=None)
                val = 0.2 * (val + 1.)  # [0, 0.4]
                gsparams["scale"] = val
            elif key == "rgb":
                val, probs = self.prior.sample(logits, means, log_scales)
                val = 0.5 + 0.5 * val
                gsparams["rgb"] = val
            elif key == "opacity":
                val, probs = self.prior.sample(logits, means, log_scales)
                val = 0.5 + 0.5 * val
                gsparams["opacity"] = val
            elif key == "xyz_dynamic":
                val, probs = self.prior.sample(logits, means, log_scales)
                val = val * 0.2
                gsparams["xyz_dynamic"] = val
            else:
                raise NotImplementedError
        
        if self.opacity_activation == "exp":
            gsparams["opacity"] = torch.exp(-gsparams["opacity"] * gsparams["scale"].mean(dim=-1, keepdim=True))

        # post process to merge static and dynamic in one tensor,
        # shape (B, N, 1 + 2 * self.opt.forder, 3)
        if not self.keep_dynamic:
            static_xyz = gsparams.pop("xyz_static")
            dynamic_xyz = gsparams.pop("xyz_dynamic")
            dynamic_xyz = dynamic_xyz.reshape(*v.shape[:2], self.key_dims["xyz_dynamic"]//3, 3)  # [B, N, L * forder, 3]
            gsparams["xyz"] = torch.cat([static_xyz[:, :, None], dynamic_xyz], dim=2)
            gsparams["rot"] = torch.cat([gsparams.pop("rot_static")[:, :, None], gsparams.pop("rot_dynamic")[:, :, None]], dim=2)

        return gsparams, prior_params

class GSDynamicDecoder(nn.Module):
    def __init__(self, opt: Options, transformer_dim: int, mlp_dim=None, bias=True):
        super(GSDynamicDecoder, self).__init__()
        self.opt = opt
        self.embed_dim = transformer_dim
        self.mlp_dim = mlp_dim if mlp_dim is not None else transformer_dim
        self.key_dims = {"xyz_dynamic": 3 * (opt.forder), "opacity_dynamic": 2}
        self.gs_layer = nn.ModuleDict()
        self.prior = Truncated_Gaussian_Model(n_sample=1, nr_mix=1)
        self.pm = opt.pm_dynamic
        self.register_buffer("dynamic_scalar", torch.tensor([0.5, 0.1, 0.5]))

        for key in ["xyz_dynamic", "opacity_dynamic"]:
            if key == "xyz_dynamic":
                layer = MLP(self.mlp_dim*2, self.key_dims[key], n_neurons=self.mlp_dim, n_hidden_layers=2, activation="silu", output_activation=None, bias=bias)
                if self.pm:
                    pred_scale = nn.Linear(self.mlp_dim*2, self.key_dims[key], bias=False)
                    torch.nn.init.xavier_normal_(pred_scale.weight, 0.01)
                    self.gs_layer[f'{key}_scale'] = pred_scale
            elif key == "opacity_dynamic":
                layer = MLP(self.mlp_dim*2, self.key_dims[key], n_neurons=self.mlp_dim, n_hidden_layers=2, activation="silu", output_activation=None, bias=bias)
            else:
                raise NotImplementedError
            self.gs_layer[key] = layer
    
    @autocast('cuda', enabled=False)
    def forward(self, feats, timestamp=None):
        """
        Perform dynamic predictions.
        """
        feats = feats.type(torch.float32)
        feats = rearrange(feats, 'b v n d -> (b v) n d')
        gsparams = {}
        prior_params = {}
        for key in ["xyz_dynamic", "opacity_dynamic"]:
            v = feats
            if f'{key}_scale' in self.gs_layer and key == "xyz_dynamic":
                logits_pred = torch.ones(feats.shape[0], feats.shape[1], 1).to(feats.device).float()
                means = self.gs_layer[key](v) 
                log_scales = self.gs_layer[f'{key}_scale'](v)
                logits, means, log_scales = self.prior.expand_params(logits_pred, means, log_scales, mean_activation='tanh')
                prior_params[key] = {"logits": logits, "means": means, "log_scales": log_scales}  # [B, N*r, dim, nr_mix]
                val, probs = self.prior.sample(logits, means, log_scales)
                val = val.reshape(*val.shape[:2], -1, 3)  # [B, N, L * forder, 3]
                val = val * self.dynamic_scalar
                gsparams[key] = val
            elif f'{key}_scale' in self.gs_layer and key == "opacity_dynamic":
                logits_pred = torch.ones(feats.shape[0], feats.shape[1], 1).to(feats.device).float()
                val = self.gs_layer[key](v) 
                log_scales = self.gs_layer[f'{key}_scale'](v)
                scalar = torch.exp(val[..., 0:1])
                means = val[..., 1:2]
                logits, means, log_scales = self.prior.expand_params(logits_pred, means, log_scales, mean_activation='tanh')
                prior_params[key] = {"logits": logits, "means": means, "log_scales": log_scales}  # [B, N*r, dim, nr_mix]
                val, probs = self.prior.sample(logits, means, log_scales)
                val = 0.5 + 0.5 * val  # t1 in [0, 1]
                gsparams[key] = torch.cat([scalar, val], dim=-1)
            else:
                v = self.gs_layer[key](v)
                gsparams[key] = self.key_activation(v, key)
        return gsparams, prior_params

    @autocast('cuda', enabled=False)
    def key_activation(self, v: torch.Tensor, key=''):
        """
        Apply activation functions for dynamic keys.
        """
        v = v.type(torch.float32)
        if key == "xyz_dynamic":
            v = F.tanh(v).reshape(*v.shape[:2], -1, 3)
            v = v * torch.tensor([0.5, 0.1, 0.5]).to(v.device)
            v = v * self.dynamic_scalar
        elif key == "opacity_dynamic":
            v[..., 0] = torch.relu(v[..., 0])  # scalar
            v[..., 1] = torch.sigmoid(v[..., 1])  # t
        else:
            raise NotImplementedError
        return v

class GSPredictor(nn.Module):
    def __init__(self, opt: Options, **model_kwargs):
        super(GSPredictor, self).__init__()
        self.opt = opt
        self.encoder = GSEncoder(opt, **model_kwargs)
        self.patch_size = opt.patch_size

        if len(opt.down_resolution) > 0:
            self.actual_input_res = opt.down_resolution
        else:
            self.actual_input_res = (opt.image_height, opt.image_width)

        # Upsampler
        self.decoder_ratio = opt.decoder_ratio
        if self.decoder_ratio > 0:
            self.gaussian_upsampler = GaussianUpsampler(width=opt.hidden_dim,
                                                        ch_decay=2,
                                                        up_ratio=(2**self.decoder_ratio),
                                                        low_channels=opt.decoder_dim, 
                                                        window_size=opt.window_size, 
                                                        opt=opt)
            transformer_dim = self.gaussian_upsampler.out_channels
        else:
            transformer_dim = opt.hidden_dim
        # check if exist gaussian upsampler
        # if hasattr(self.encoder, "gaussian_upsampler"):
        #     transformer_dim = self.encoder.gaussian_upsampler.out_channels
        # else:
        #     transformer_dim = self.opt.hidden_dim
        if self.opt.use_pm:
            self.predictor = GSPMDecoder(opt, transformer_dim=transformer_dim)
        else:
            raise NotImplementedError("Currently only support probabilistic predictor")
    
    def _freeze(self):
        for name, param in self.named_parameters():
            if "xyz_static" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    
    def forward_encoder(self, frames, depths, cond_times=None):
        mask = torch.ones_like(depths)  # dummy mask
        max_depth = depths.flatten(1).max(dim=1)[0][:, None, None, None, None]
        min_depth = depths.flatten(1).min(dim=1)[0][:, None, None, None, None]
        target_depth = depths # without normalization
        input_depth = (depths - min_depth) / (max_depth - min_depth)
       
        frames = torch.cat([frames, input_depth], dim=2)
        encoder_output = self.encoder(frames, cond_times)

        return encoder_output, target_depth, mask
    
    def upsampling(self, output):
        input_views = output.shape[1]
        if self.decoder_ratio > 0:
            output = output.reshape(output.shape[0], -1, output.shape[-1])  # [B, V, N, D] -> [B, V*N, D]
            output = self.gaussian_upsampler(output)  # [B, V*N, D]
            output = rearrange(output, 'b (v n) d -> b v n d', v=input_views)
        return output

    def forward(self, frames, depths, cond_times=None):
        batch_size, input_views = frames.shape[0], frames.shape[1]
        encoder_output, target_depth, mask = self.forward_encoder(frames, depths, cond_times)  # [B, V, N, D]
        output = encoder_output
        num_features = output.shape[2]

        output = self.upsampling(output)  # [B, V, N, D]

        # Decoding
        pred_gs, prior_params = self.predictor(output)
        
        return {'pred_gs': pred_gs, 'gt_depth': target_depth, 'gt_depth_mask': mask}, prior_params

class SplatPredictor(nn.Module):
    def __init__(self, opt: Options, **model_kwargs):
        super().__init__()
        self.opt = opt
        self.decoder = SplatDecoder(opt, **model_kwargs)
        self.patch_size = opt.patch_size

        # # Upsampler
        self.decoder_ratio = opt.decoder_ratio
        if self.decoder_ratio > 0:
            self.gaussian_upsampler = GaussianUpsampler(width=opt.hidden_dim,
                                                        ch_decay=2,
                                                        up_ratio=(2**self.decoder_ratio),
                                                        low_channels=opt.decoder_dim, 
                                                        window_size=opt.window_size, 
                                                        opt=opt)
            transformer_dim = self.gaussian_upsampler.out_channels
        else:
            transformer_dim = opt.hidden_dim
    
        self.gs_predictor = GSPredictor(opt, **model_kwargs)
        self.gs_predictor.predictor.keep_dynamic = True  # keep dynamic for post combination

        if opt.use_dino:
            self.condition_encoder = Dinov2Wrapper(model_name="dinov2_vitb14_reg")
            self.condition_dim = 768
        
        self.gs_dynamic_predictor = GSDynamicDecoder(opt, transformer_dim=self.gs_predictor.predictor.embed_dim, bias=True) 
        self.encoder_proj = nn.Linear(192, 192, bias=False)  # hard code for now
    
    def state_dict(self, **kwargs):
        # remove the condition encoder from the state dict
        state_dict = super().state_dict(**kwargs)
        for k in list(state_dict.keys()):
            if "condition_encoder" in k:
                del state_dict[k]
        return state_dict
    
    def train(self, mode=True):
        super().train(mode)
        if self.opt.use_dino:
            self.condition_encoder.eval()
        self.gs_predictor.eval()
        return self
    
    def _freeze_predictor(self):
        self.gs_predictor.eval()
        for name, param in self.gs_predictor.named_parameters():
            if "dynamic" in name:
                param.requires_grad = True
            elif "encoder" in name:
                param.requires_grad = False
            elif "upsampler" in name:
                param.requires_grad = False
            elif "gs_layer" in name:
                param.requires_grad = False
            else:
                param.requires_grad = False
        
    def forward_encoder(self, frames, depths, cond_times=None):
        frames = torch.cat([frames, depths], dim=2)
        encoder_output = self.gs_predictor.encoder(frames, timestamp=None)

        return encoder_output
    
    def upsampling(self, output):
        input_views = output.shape[1]
        if self.decoder_ratio > 0:
            output = output.reshape(output.shape[0], -1, output.shape[-1])  # [B, V, N, D] -> [B, V*N, D]
            output = self.gaussian_upsampler(output)  # [B, V*N, D]
            output = rearrange(output, 'b (v n) d -> b v n d', v=input_views)
        return output

    def forward_condition(self, frames, depths):
        # [B, V, C, H, W]
        input_views = frames.shape[1]
        frames = rearrange(frames, 'b v c h w -> (b v) c h w')
        depths = rearrange(depths, 'b v c h w -> (b v) c h w')

        frames = F.interpolate(frames, size=(252, 448), mode='bilinear', align_corners=False)
        depths = F.interpolate(depths, size=(252, 448), mode='bilinear', align_corners=False)

        cls_condition, frames_condition = self.condition_encoder(frames)
        
        condition = frames_condition
        condition = rearrange(condition, '(b v) n d -> b v n d', v=input_views)

        return condition

    def combine(self, pred_gs):
        static_xyz = pred_gs.pop("xyz_static")
        dynamic_xyz = pred_gs.pop("xyz_dynamic")
        dynamic_xyz = dynamic_xyz.reshape(*static_xyz.shape[:2], self.opt.forder, 3)
        pred_gs["xyz"] = torch.cat([static_xyz[:, :, None], dynamic_xyz], dim=2)
        pred_gs["rot"] = torch.cat([pred_gs.pop("rot_static")[:, :, None], pred_gs.pop("rot_dynamic")[:, :, None]], dim=2)
        if "opacity_dynamic" in pred_gs.keys():
            pred_gs["opacity"] = torch.cat([pred_gs["opacity"], pred_gs.pop("opacity_dynamic")], dim=-1)
        return pred_gs

    def forward(self, frames, depths, cond_times=None):
        frames = torch.cat([frames[:, 0:1], frames[:, -1:]], dim=1)  # [B, 2, C, H, W]
        depths = torch.cat([depths[:, 0:1], depths[:, -1:]], dim=1)  # [B, 2, C, H, W]
        input_views = frames.shape[1]
        frames = rearrange(frames, 'b v c h w -> (b v) 1 c h w')
        depths = rearrange(depths, 'b v c h w -> (b v) 1 c h w')
        encoder_output = self.forward_encoder(frames, depths)  # [B*V, 1, N, D]
        encoder_output = rearrange(encoder_output, '(b v) 1 n d -> b v n d', v=input_views)
        encoder_0 = encoder_output[:, 0:1]  # [B, V, N, D]
        encoder_1 = encoder_output[:, -1:]
        upsampled_encoder_0 = self.gs_predictor.upsampling(encoder_0)
        upsampled_encoder_1 = self.gs_predictor.upsampling(encoder_1)
        gs_0, _ = self.gs_predictor.predictor(upsampled_encoder_0)
        gs_1, _ = self.gs_predictor.predictor(upsampled_encoder_1)

        if self.opt.use_dino:
            frames = rearrange(frames, '(b v) 1 c h w -> b v c h w', v=input_views)
            depths = rearrange(depths, '(b v) 1 c h w -> b v c h w', v=input_views)
            condition_output = self.forward_condition(frames, depths)  # [B, V, N, D]
            condition_0 = condition_output[:, 0:1]  # [B, V, N, D]
            condition_1 = condition_output[:, -1:]
        else:
            condition_0 = encoder_0
            condition_1 = encoder_1

        decoder_input = torch.cat([encoder_0, encoder_1], dim=2)  # [B, V, 2*N, D]
        condition_input = torch.cat([condition_1, condition_0], dim=2)  # [B, V, 2*N, D]
        decoder_output = self.decoder(decoder_input, condition_input)
        decoder_output_0 = decoder_output[:, :, :self.decoder.token_len]  # [B, V, N, D]
        decoder_output_1 = decoder_output[:, :, self.decoder.token_len:]

        upsampled_dynamic_0 = self.upsampling(decoder_output_0)
        upsampled_dynamic_1 = self.upsampling(decoder_output_1)
        upsampled_0 = torch.cat([self.encoder_proj(upsampled_encoder_0), upsampled_dynamic_0], dim=-1)
        upsampled_1 = torch.cat([self.encoder_proj(upsampled_encoder_1), upsampled_dynamic_1], dim=-1)
        gs_0_dynamic, gs_0_prior = self.gs_dynamic_predictor(upsampled_0)
        gs_1_dynamic, gs_1_prior = self.gs_dynamic_predictor(upsampled_1)

        for key in ["xyz_dynamic", "opacity_dynamic"]:
            gs_0[key] = gs_0_dynamic[key]
            gs_1[key] = gs_1_dynamic[key]

        gs_0 = self.combine(gs_0)
        gs_1 = self.combine(gs_1)

        # concate gs_0 and gs_1
        pred_gs = {}
        for key in gs_0.keys():
            pred_gs[key] = torch.cat([gs_0[key], gs_1[key]], dim=1)

        return {'pred_gs': pred_gs, 'gs_0': gs_0, 'gs_1': gs_1}

