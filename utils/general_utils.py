#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import sys
from datetime import datetime
import numpy as np
import random
# from pointops2.functions.pointops import furthestsampling, knnquery
from torch.optim.lr_scheduler import LRScheduler
import math

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[...,0]*r[...,0] + r[...,1]*r[...,1] + r[...,2]*r[...,2] + r[...,3]*r[...,3])
    
    q = r / norm[..., None]
    R = torch.zeros((q.size(0), q.size(1), 3, 3), device='cuda')
    r = q[..., 0]
    x = q[..., 1]
    y = q[..., 2]
    z = q[..., 3]

    R[..., 0, 0] = 1 - 2 * (y*y + z*z)
    R[..., 0, 1] = 2 * (x*y - r*z)
    R[..., 0, 2] = 2 * (x*z + r*y)
    R[..., 1, 0] = 2 * (x*y + r*z)
    R[..., 1, 1] = 1 - 2 * (x*x + z*z)
    R[..., 1, 2] = 2 * (y*z - r*x)
    R[..., 2, 0] = 2 * (x*z - r*y)
    R[..., 2, 1] = 2 * (y*z + r*x)
    R[..., 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L


def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))
    
def knn(x, src, k, transpose=False):
    if transpose:
        x = x.transpose(1, 2).contiguous()
        src = src.transpose(1, 2).contiguous()
    b, n, _ = x.shape
    m = src.shape[1]
    x = x.view(-1, 3)
    src = src.view(-1, 3)
    x_offset = torch.full((b,), n, dtype=torch.long, device=x.device)
    src_offset = torch.full((b,), m, dtype=torch.long, device=x.device)
    x_offset = torch.cumsum(x_offset, dim=0).int()
    src_offset = torch.cumsum(src_offset, dim=0).int()
    idx, dists = knnquery(k, src, x, src_offset, x_offset)
    idx = idx.view(b, n, k) - (src_offset - m)[:, None, None]
    return idx.long(), dists.view(b, n, k)
    
def fps(x, k):
    b, n, _ = x.shape
    x = x.view(-1, 3).contiguous()
    offset = torch.full((b,), n, dtype=torch.long, device=x.device)
    new_offset = torch.full((b,), k, dtype=torch.long, device=x.device)
    offset = torch.cumsum(offset, dim=0).int()
    new_offset = torch.cumsum(new_offset, dim=0).int()
    idx = furthestsampling(x, offset, new_offset).long()
    return idx

class CosineWarmupScheduler(LRScheduler):
    def __init__(self, optimizer, warmup_iters: int, max_iters: int, initial_lr: float = 1e-10, last_iter: int = -1, min_lr: float = 0.0, decay: bool = True):
        self.warmup_iters = warmup_iters
        self.max_iters = max_iters
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.decay = decay
        self._epoch = 0
        super().__init__(optimizer, last_iter)

    def get_lr(self):
        # logger.debug(f"step count: {self._step_count} | warmup iters: {self.warmup_iters} | max iters: {self.max_iters}")
        if self._step_count <= self.warmup_iters:
            return [
                self.initial_lr + (base_lr - self.initial_lr) * self._step_count / self.warmup_iters
                for base_lr in self.base_lrs]
        elif not self.decay:
            return [base_lr for base_lr in self.base_lrs]
        elif self._step_count >= self.max_iters:
            return [min(self.min_lr, base_lr) for base_lr in self.base_lrs]
        else:
            cos_iter = self._step_count - self.warmup_iters
            cos_max_iter = self.max_iters - self.warmup_iters
            cos_theta = cos_iter / cos_max_iter * math.pi
            cos_lr = [base_lr * (1. + math.cos(cos_theta)) / 2 for base_lr in self.base_lrs]
            return [max(self.min_lr, lr) for lr in cos_lr]
        
    def step(self, epoch=None):
        super().step()
        self._epoch = epoch


class CosineWeightDecayScheduler(LRScheduler):
    def __init__(
        self,
        optimizer,
        max_iters: int,
        initial_wd: float = 0.05,
        final_wd: float = 0.20,
        last_epoch: int = -1,
    ):
        """
        对 weight_decay 进行余弦“增加”调度，从 initial_wd -> final_wd。
        
        参数:
        - optimizer: 任何带有 'weight_decay' param_group 的 optimizer
        - max_iters: 总的调度步数
        - initial_wd: 第 0 步时的 weight_decay
        - final_wd: 第 max_iters 步时的 weight_decay
        - last_epoch: 如需从中途恢复训练，传入上次迭代 idx
        """
        self.max_iters = max_iters
        self.initial_wd = initial_wd
        self.final_wd = final_wd
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self._step_count
        # 限制在 [0, max_iters]
        if step <= 0:
            factor = 0.0
        elif step >= self.max_iters:
            factor = 1.0
        else:
            # factor 从 0 -> 1，按 1 - cos(pi * t / T) / 2
            factor = 0.5 * (1 - math.cos(math.pi * step / self.max_iters))

        wd = self.initial_wd + factor * (self.final_wd - self.initial_wd)
        return [wd for _ in self.optimizer.param_groups]

    def step(self, epoch=None):
        # 先让父类更新 last_epoch
        self._step_count += 1
        # super().step(epoch)
        # 再把新的 weight_decay 写到 optimizer
        new_wd = self.get_lr()
        for group, wd in zip(self.optimizer.param_groups, new_wd):
            group['weight_decay'] = wd