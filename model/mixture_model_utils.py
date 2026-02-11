import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

eps = 1e-3

class Truncated_Gaussian_Model(nn.Module):
    def __init__(self, n_sample=1, nr_mix=1):
        super(Truncated_Gaussian_Model, self).__init__()
        self.n_sample = n_sample
        self.nr_mix = nr_mix
        self.log_scales_min = -5.0
        self.log_scales_max = 2.0
        self.log_scales_a = (self.log_scales_min + self.log_scales_max) / 2
        self.log_scales_b = (self.log_scales_max - self.log_scales_min) / 2
        self.perform_sampling = True
    
    def activate_mean(self, means, mean_activation='tanh'):
        if mean_activation == 'tanh':
            return torch.tanh(means) * (1.0 - eps)  # (-1, 1)
        else:
            raise ValueError(f"Unknown activation function: {self.mean_activation}")
    
    def expand_params(self, logits, means, log_scales, mean_activation='tanh'):
        """
        Expand the parameters to n_samples.
        """
        B, N, _ = means.shape  # [B, N, dim*nr_mix]
        dim = int(means.shape[-1] / self.nr_mix)
        logits = logits.repeat(1, 1, self.n_sample).reshape(B, -1, 1, self.nr_mix)  # [B, N*n_sample, 1, nr_mix]

        means = means.reshape(B, -1, dim, self.nr_mix)  # [B, N, dim, nr_mix]
        log_scales = log_scales.reshape(B, -1, dim, self.nr_mix)  # [B, N, dim, nr_mix]

        means = means.repeat(1, 1, self.n_sample, 1).reshape(means.shape[0], -1, dim, self.nr_mix)  # [B, N*n_sample, dim, nr_mix]
        log_scales = log_scales.repeat(1, 1, self.n_sample, 1).reshape(means.shape[0], -1, dim, self.nr_mix)  # [B, N*n_sample, dim, nr_mix]

        means = self.activate_mean(means.type(torch.float32), mean_activation)
        log_scales = log_scales.type(torch.float32)

        logits = F.softmax(logits.type(torch.float32), dim=-1) 

        return logits, means, log_scales

    def get_mix_params(self, logits, means, log_scales):
        return means.squeeze(-1), log_scales.squeeze(-1), 1.0
    
    def cdf_fn(self, x, means, log_scales):
        """
        Cumulative distribution function of the Gaussian distribution.
        """
        inv_std = torch.exp(-log_scales)
        return 0.5 * (
            1 + torch.erf((x - means) * inv_std / math.sqrt(2))
        )
    
    def log_pdf_fn(self, x, means, log_scales):
        """
        Log probability density function of the Gaussian distribution.
        """
        scales = torch.exp(log_scales)
        var = scales**2
        return (
            -((x - means) ** 2) / (2 * var)
            - log_scales
            - math.log(math.sqrt(2 * math.pi))
        )
    
    def icdf_fn(self, p, means, log_scales):
        """
        Inverse cumulative distribution function of the Gaussian distribution.
        """
        scales = torch.exp(log_scales)
        return means + scales * torch.erfinv(2 * p - 1) * math.sqrt(2)

    def sample(self, logits_input, means_input, log_scales_input, a=torch.as_tensor(-1.0 + eps, dtype=torch.float32), b=torch.as_tensor(1.0 - eps, dtype=torch.float32), interval=0.05):

        means, log_scales, probs = self.get_mix_params(logits_input, means_input, log_scales_input)  # [B, N*n_sample, dim], [B, N*n_sample, 1]

        if not self.training or not self.perform_sampling:
            probs_samples = self.log_pdf_fn(means, means, log_scales).exp().mean(dim=-1, keepdim=True)
            probs = probs * probs_samples.tanh()
            return means, probs

        if a is None:
            cdf_lower = torch.zeros_like(means, dtype=torch.float32)
        else:
            a = a.to(means.device)
            a = a.expand_as(means)
            cdf_lower = self.cdf_fn(a, means, log_scales)

        if b is None:
            cdf_upper = torch.ones_like(means, dtype=torch.float32)
        else:
            b = b.to(means.device)
            b = b.expand_as(means)
            cdf_upper = self.cdf_fn(b, means, log_scales)
            
        # sampling
        cdf_delta = cdf_upper - cdf_lower
        cdf_mid = (cdf_upper + cdf_lower) / 2

        u_normal = torch.rand_like(means).clamp(eps, 1.0 - eps)
        u_normal = u_normal * cdf_delta + cdf_lower # normal condition
        u_edge = 0.5  # dummy value
        u = torch.where(cdf_delta > eps, u_normal, u_edge)
        u = torch.where(log_scales > self.log_scales_min, u, u_edge)
        val = self.icdf_fn(u, means, log_scales)
        val = torch.where(cdf_delta > eps, val, a)

        select_val = means
        if b is not None:
            select_val = torch.where(select_val < b, select_val, b)
        if a is not None:
            select_val = torch.where(select_val > a, select_val, a)
        val = torch.where(log_scales > self.log_scales_min, val, select_val)

        # cdf interval
        left = val - interval
        if a is not None:
            left = torch.max(left, a)
        right = val + interval
        if b is not None:
            right = torch.min(right, b)
        cdf_left = self.cdf_fn(left, means, log_scales)
        cdf_right = self.cdf_fn(right, means, log_scales)
        cdf = (cdf_right - cdf_left) / (cdf_delta + 1e-5)
        probs = probs * cdf.min(dim=-1).values.unsqueeze(-1)

        return val, probs
    