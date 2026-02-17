import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import kiui
# import gaussian_renderer
# from torch.cuda.amp import custom_bwd, custom_fwd, autocast
import sys
from configs.options import Options
import math
# from model_utils import *
from einops import rearrange
from collections import OrderedDict
from torch.utils.checkpoint import checkpoint
import numpy as np
import pdb
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


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    output = x * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class GELU_(nn.Module):
    """
    Fast gelu implementation.
    """
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)
    
class LayerNorm(nn.RMSNorm):
    """
    RMSNorm layer.
    """
    def __init__(self, dim, eps=1e-5):
        super().__init__(dim, eps=eps)
    def forward(self, x):
        type_ = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(type_)

class MLP(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        n_neurons: int,
        n_hidden_layers: int,
        activation: str = "silu",
        output_activation: Optional[str] = "silu",
        bias: bool = True,
        dropout: float = 0.0,
        use_residual: bool = False,
        use_rmsnorm: bool = False,
    ):
        super().__init__()
        self.use_residual = use_residual
        self.use_rmsnorm = use_rmsnorm
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        input_norm = LayerNorm(dim_in) if use_rmsnorm else nn.Identity()
        output_norm = nn.Identity()  # no normalization for the output

        layers = [
            input_norm,
            self.make_linear(
                dim_in, n_neurons, is_first=True, is_last=False, bias=bias
            ),
            self.make_activation(activation),
            self.dropout,
        ]
        for i in range(n_hidden_layers - 1):
            layers += [
                self.make_linear(
                    n_neurons, n_neurons, is_first=False, is_last=False, bias=bias
                ),
                self.make_activation(activation),
                self.dropout,
            ]
        layers += [
            self.make_linear(
                n_neurons, dim_out, is_first=False, is_last=True, bias=bias
            ),
            output_norm,
        ]
        self.layers = nn.Sequential(*layers)
        self.output_activation = self.make_activation(output_activation)

    def forward(self, x):
        if self.use_residual:
            residual = x.type(torch.float32)
        x = self.layers(x)
        if self.use_residual:
            x = x + residual
        x = self.output_activation(x)
        return x

    def make_linear(self, dim_in, dim_out, is_first, is_last, bias=True):
        layer = nn.Linear(dim_in, dim_out, bias=bias)
        nn.init.xavier_uniform_(layer.weight)
        if bias:
            nn.init.zeros_(layer.bias)
        return layer

    def make_activation(self, activation):
        if activation is None:
            return nn.Identity()
        if activation == "relu":
            return nn.ReLU(inplace=True)
        elif activation == "silu":
            return nn.SiLU(inplace=True)
        elif activation == "gelu":
            return GELU_()
        elif activation == "tanh":
            return nn.Tanh()
        else:
            raise NotImplementedError

class MultiHeadAttention(nn.Module):
    """
    Computes multi-head attention. Supports nested or padded tensors.

    Args:
        E_q (int): Size of embedding dim for query
        E_k (int): Size of embedding dim for key
        E_v (int): Size of embedding dim for value
        E_total (int): Total embedding dim of combined heads post input projection. Each head
            has dim E_total // nheads
        nheads (int): Number of heads
        dropout (float, optional): Dropout probability. Default: 0.0
        bias (bool, optional): Whether to add bias to input projection. Default: True
    """

    def __init__(
        self,
        E_q: int,
        E_k: int,
        E_v: int,
        E_total: int,
        nheads: int,
        dropout: float = 0.0,
        bias=True,
        device=None,
        dtype=None,
        batch_first=False,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.nheads = nheads
        self.dropout = dropout
        self._qkv_same_embed_dim = E_q == E_k and E_q == E_v
        if self._qkv_same_embed_dim:
            self.packed_proj = nn.Linear(E_q, E_total * 3, bias=bias, **factory_kwargs)
        else:
            self.q_proj = nn.Linear(E_q, E_total, bias=bias, **factory_kwargs)
            self.k_proj = nn.Linear(E_k, E_total, bias=bias, **factory_kwargs)
            self.v_proj = nn.Linear(E_v, E_total, bias=bias, **factory_kwargs)
        E_out = E_q
        self.out_proj = nn.Linear(E_total, E_out, bias=bias, **factory_kwargs)
        assert E_total % nheads == 0, "Embedding dim is not divisible by nheads"
        self.E_head = E_total // nheads
        self.bias = bias
        self.batch_first = batch_first

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask=None,
        is_causal=False,
        need_weights=False,  # for compatibility with nn.MultiheadAttention
    ) -> torch.Tensor:
        """
        Args:
            query (torch.Tensor): query of shape (``N``, ``L_q``, ``E_qk``)
            key (torch.Tensor): key of shape (``N``, ``L_kv``, ``E_qk``)
            value (torch.Tensor): value of shape (``N``, ``L_kv``, ``E_v``)
            attn_mask (torch.Tensor, optional): attention mask of shape (``N``, ``L_q``, ``L_kv``) to pass to SDPA. Default: None
            is_causal (bool, optional): Whether to apply causal mask. Default: False

        Returns:
            attn_output (torch.Tensor): output of shape (N, L_t, E_q)
        """
        if self._qkv_same_embed_dim:
            if query is key and key is value:
                result = self.packed_proj(query)
                query, key, value = torch.chunk(result, 3, dim=-1)
            else:
                q_weight, k_weight, v_weight = torch.chunk(
                    self.packed_proj.weight, 3, dim=0
                )
                if self.bias:
                    q_bias, k_bias, v_bias = torch.chunk(
                        self.packed_proj.bias, 3, dim=0
                    )
                else:
                    q_bias, k_bias, v_bias = None, None, None
                query, key, value = (
                    F.linear(query, q_weight, q_bias),
                    F.linear(key, k_weight, k_bias),
                    F.linear(value, v_weight, v_bias),
                )

        else:
            query = self.q_proj(query)
            key = self.k_proj(key)
            value = self.v_proj(value)

        query = query.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        key = key.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        value = value.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(
            query, key, value, dropout_p=self.dropout, is_causal=is_causal
        )
        attn_output = attn_output.transpose(1, 2).flatten(-2)

        attn_output = self.out_proj(attn_output)

        return attn_output, None

class ResAttBlock(nn.Module):
    """
    Attention block.
    """
    def __init__(self, d_model, n_head, window_size=None, drop_path_rate=0.0):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, d_model, d_model, d_model, n_head)
        self.layernorm1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4, bias=False)),
            ("silu", nn.SiLU(inplace=True)),
            ("c_proj", nn.Linear(d_model * 4, d_model, bias=False))
        ]))
        self.layernorm2 = LayerNorm(d_model)
        self.window_size = window_size

    def attention(self, x, index):
        attn_mask = None
        if self.window_size is not None:
            l = x.shape[1]
            assert l % self.window_size == 0
            if index % 2 == 0:
                x = rearrange(x, 'b (p w) c -> (b p) w c', w=self.window_size)
                x = self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0] 
                x = rearrange(x, '(b l) w c -> b (l w) c', l=l//self.window_size, w=self.window_size)
            else:
                x = torch.roll(x, shifts=self.window_size//2, dims=1)
                x = rearrange(x, 'b (p w) c -> (b p) w c', w=self.window_size)
                x = self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0] 
                x = rearrange(x, '(b l) w c -> b (l w) c', l=l//self.window_size, w=self.window_size)
                x = torch.roll(x, shifts=-self.window_size//2, dims=1)
        else:
            x = self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]
        return x

    def forward(self, x, index, condition=None):
        # no condition in encoder, its a dummy argument
        y = self.layernorm1(x)
        y = self.attention(y, index)
        x = x.type(torch.float32) + y  # residual in fp32
        y = self.layernorm2(x)
        y = self.mlp(y)
        x = x.type(torch.float32) + y  # residual in fp32
        return x

class ConditionalResAttBlock(nn.Module):
    def __init__(self, d_model, n_head, window_size=None, drop_path_rate=0.0):
        super().__init__()
        self.window_size = window_size

        self.self_attn = MultiHeadAttention(d_model, d_model, d_model, d_model, n_head)
        self.self_attn_ln = LayerNorm(d_model)

        self.cross_attn = MultiHeadAttention(d_model, d_model, d_model, d_model, n_head)
        self.cross_attn_ln = LayerNorm(d_model)

        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4, bias=False)),
            ("silu", nn.SiLU(inplace=True)),
            ("c_proj", nn.Linear(d_model * 4, d_model, bias=False))
        ]))
        self.mlp_ln = LayerNorm(d_model)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def window_attention(self, x, attn_layer, index):
        attn_mask = None
        if self.window_size is not None:
            l = x.shape[1]
            assert l % self.window_size == 0, "Sequence length must be divisible by window size"
            if index % 2 == 0:
                # Even index: split into windows without shifting.
                x = rearrange(x, 'b (p w) c -> (b p) w c', w=self.window_size)
                x = attn_layer(x, x, x, need_weights=False, attn_mask=attn_mask)[0]
                x = rearrange(x, '(b p) w c -> b (p w) c', p=l // self.window_size)
            else:
                # Odd index: roll by half a window, then split.
                x = torch.roll(x, shifts=self.window_size // 2, dims=1)
                x = rearrange(x, 'b (p w) c -> (b p) w c', w=self.window_size)
                x = attn_layer(x, x, x, need_weights=False, attn_mask=attn_mask)[0]
                x = rearrange(x, '(b p) w c -> b (p w) c', p=l // self.window_size)
                x = torch.roll(x, shifts=-self.window_size // 2, dims=1)
        else:
            x = attn_layer(x, x, x, need_weights=False, attn_mask=attn_mask)[0]
        return x

    def forward(self, x, index, condition):
        residual = x.type(torch.float32)
        x_ln = self.self_attn_ln(x)
        x2 = self.window_attention(x_ln, self.self_attn, index)
        x = residual + self.drop_path(x2)

        x = rearrange(x, 'b (p n) d -> (b p) n d', p=2)  # split back to frame_0 and frame_1
        residual = x.type(torch.float32)
        x_ln = self.cross_attn_ln(x)
        x2 = self.cross_attn(x_ln, condition, condition, need_weights=False)[0]
        x = residual + self.drop_path(x2)
        x = rearrange(x, '(b p) n d -> b (p n) d', p=2)  # combine frame_0 and frame_1

        residual = x.type(torch.float32)
        x_ln = self.mlp_ln(x)
        x2 = self.mlp(x_ln)
        x = residual + self.drop_path(x2)

        return x

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        seq_len = x.size(1)
        return self.pe[:seq_len, :].unsqueeze(0)

class Transformer(nn.Module):
    def __init__(self, width, layers, heads, window_size=None, block_cls=ResAttBlock, drop_path_rate=0.0):
        super().__init__()
        self.width = width
        self.layers = layers
        blocks = []
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, layers)]  # stochastic depth decay rule
        inter_dpr = [0.0] + dpr
        if drop_path_rate > 0.0:
            print(f"inter_dpr: {inter_dpr}")
        for _ in range(layers):
            layer = block_cls(width, heads, window_size=window_size, drop_path_rate=inter_dpr[_]) 
            blocks.append(layer)

        self.resblocks = nn.Sequential(*blocks)
        self.grad_checkpointing = False

    def set_grad_checkpointing(self, flag=True):
        self.grad_checkpointing = flag

    def forward(self, x, condition=None):
        for res_i, module in enumerate(self.resblocks):
            if self.grad_checkpointing:
                x = checkpoint(module, x, res_i, condition, use_reentrant=False)
            else:
                x = module(x, res_i, condition)

        return x

class TransformerBase(nn.Module):
    def __init__(self, width, layers, heads, window_size, token_len, block_cls, drop_path_rate=0.0):
        super().__init__()
        self.layernorm1 = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads, window_size=window_size, block_cls=block_cls, drop_path_rate=drop_path_rate)
        self.layernorm2 = LayerNorm(width)
    
    def set_grad_checkpointing(self, set_checkpointing=True):
        self.transformer.set_grad_checkpointing(set_checkpointing)
    
    def forward(self, x, condition=None):
        # x [B, V*N, D]
        x = self.layernorm1(x)
        x = self.transformer(x, condition)
        x = self.layernorm2(x)
        return x

class TransformerEncoder(TransformerBase):
    def __init__(self, input_res, in_channels, patch_size, width, layers, heads, window_size):
        self.input_res = input_res
        self.patch_size = patch_size
        token_len = (self.input_res[0] // patch_size) * (self.input_res[1] // patch_size)
        super().__init__(width, layers, heads, window_size, token_len, ResAttBlock)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        self.positional_encoding = SinusoidalPositionalEncoding(max_len=token_len, d_model=width)

    def forward(self, x, condition=None):
        _, v = x.shape[:2]
        x = rearrange(x, 'b v c h w -> (b v) c h w')
        x = self.conv(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        x = x + self.positional_encoding(x).to(x.dtype)
        x = super().forward(x, condition)

        x = rearrange(x, 'b (v n) d -> b v n d', v=v)
        return x

class TransformerConditionalDecoder(TransformerBase):
    def __init__(self, input_res, patch_size, width, layers, heads, window_size, encoder_dim=None, condition_len=576, condition_dim=None, drop_path_rate=0.1):
        self.input_res = input_res
        self.patch_size = patch_size
        self.width = width
        token_len = (input_res[0] // patch_size) * (input_res[1] // patch_size)
        super().__init__(width, layers, heads, window_size, token_len, ConditionalResAttBlock, drop_path_rate=drop_path_rate)
        self.positional_embedding = nn.Parameter(torch.zeros(1, token_len*2, width))
        nn.init.trunc_normal_(self.positional_embedding, std=0.02)
        self.cls_embedding = nn.Parameter(torch.zeros(1, 2, width))
        nn.init.trunc_normal_(self.cls_embedding, std=0.02)
        self.positional_encoding = SinusoidalPositionalEncoding(max_len=condition_len, d_model=width)

        if condition_dim is not None:
            self.condition_proj = nn.Linear(condition_dim, width, bias=False)
        else:
            self.condition_proj = nn.Identity()
        
        self.out_proj = nn.Identity()

        self.dropout = nn.Dropout(drop_path_rate)
        self.condition_ln = LayerNorm(width)

    def forward(self, latent, condition):
        b, v = latent.shape[:2]
        latent = rearrange(latent, 'b v n d -> (b v) n d')  # [B, 2*N, D]
        condition = rearrange(condition, 'b v n d -> (b v) n d')  # [B, 2*N, D]
        condition = rearrange(condition, 'b (p n) d -> (b p) n d', p=2)  # [B*2, N, D]

        condition = self.condition_proj(condition)
        latent = latent + self.positional_embedding
        cls_embedding = self.cls_embedding.repeat_interleave(latent.shape[1]//2, dim=1).contiguous()  # [1, N, D]
        latent = latent + cls_embedding

        condition = condition + self.positional_encoding(condition).to(condition.dtype)  # [B*2, N, D]
        condition = self.condition_ln(condition)
        condition = self.dropout(condition)

        x = super().forward(latent, condition)
        x = self.out_proj(x)

        x = rearrange(x, '(b v) n d -> b v n d', v=v)
        return x

class TransformerDecoder(TransformerBase):
    def __init__(self, token_len, width, layers, heads, window_size, encoder_dim=None):
        self.width = width
        super().__init__(width, layers, heads, window_size, token_len, ResAttBlock)
        self.positional_embedding = nn.Parameter(torch.zeros(1, token_len, width))
        nn.init.trunc_normal_(self.positional_embedding, std=0.02)

        if encoder_dim is not None and encoder_dim != width:
            self.encoder_proj = nn.Linear(encoder_dim, width)
            self.out_proj = nn.Linear(width, encoder_dim)
        else:
            self.encoder_proj = nn.Identity()
            self.out_proj = nn.Identity()


    def forward(self, latent, condition=None, reverse=False):
        _, v = latent.shape[:2]
        latent = rearrange(latent, 'b v n d -> (b v) n d')

        latent = self.encoder_proj(latent)

        latent = latent + self.positional_embedding.to(latent.dtype)
        x = super().forward(latent, condition)
        x = self.out_proj(x)

        x = rearrange(x, '(b v) n d -> b v n d', v=v)
        return x

class PSUpsamplerBlock(nn.Module):
    """
    Upsampling block.
    """
    def __init__(self, d_model, d_model_out, scale_factor, resolution=None, view_num=1):
        super().__init__()

        self.scale_factor = scale_factor
        self.d_model_out = d_model_out
        self.residual_fc = nn.Linear(d_model, d_model_out * (scale_factor**2), bias=True)
        self.pixelshuffle = nn.PixelShuffle(scale_factor)

        self.resolution = resolution

        self.view_num = view_num

    def forward(self, x):
        x = self.residual_fc(x)
        bs, l, c = x.shape
        x = x.reshape(bs * self.view_num, self.resolution[0], self.resolution[1], c).permute(0, 3, 1, 2)
        x = self.pixelshuffle(x)
        x = x.permute(0, 2, 3, 1).reshape(bs, self.view_num, self.resolution[0]*self.scale_factor, self.resolution[1]*self.scale_factor, self.d_model_out)
        x = x.reshape(bs, -1, self.d_model_out)
        return x

class GaussianUpsampler(nn.Module):
    """
    Upsampler.
    """
    def __init__(self, width, up_ratio, ch_decay=1, low_channels=64, window_size=None, opt: Options=None):
        super().__init__()
        self.up_ratio = up_ratio
        self.low_channels = low_channels
        self.window_size = window_size
        
        self.base_width = width

        if len(opt.down_resolution) > 0:
            self.input_res = (opt.down_resolution[0] // opt.patch_size, opt.down_resolution[1] // opt.patch_size)
        else:
            self.input_res = (opt.image_height // opt.patch_size, opt.image_width // opt.patch_size)
        
        resolution = [self.input_res[0], self.input_res[1]]
        for res_log2 in range(int(np.log2(up_ratio))):
            _width = width
            width = max(width // ch_decay, self.low_channels)
            heads = int(width / 64)
            width = heads * 64
            self.add_module(f'upsampler_{res_log2}', PSUpsamplerBlock(_width, width, scale_factor=2, resolution=resolution, view_num=opt.input_frames))
            resolution = [resolution[0]*2, resolution[1]*2]
            encoder = Transformer(width, 2, heads, window_size=window_size)
            self.add_module(f'attention_{res_log2}', encoder)
        self.out_channels = width
        self.layernorm2 = LayerNorm(width)

    def forward(self, x):
        for res_log2 in range(int(np.log2(self.up_ratio))):
            x = getattr(self, f'upsampler_{res_log2}')(x)
            x = getattr(self, f'attention_{res_log2}')(x)
        x = self.layernorm2(x)
        return x
        


