# source: https://github.com/manman1995/Mutual-Information-driven-Pan-sharpening/blob/main/models/GPPNN.py
from torch.distributions import Normal, Independent, kl
import torch
import torch.nn as nn

from typing import Optional, Tuple, Union

import numpy as np

from mmengine.model import BaseModule
from torch import Tensor

from mmdet.registry import MODELS
from .utils import weighted_loss

@weighted_loss
def kl_divergence_loss( posterior_latent_space, prior_latent_space):
    kl_div = kl.kl_divergence(posterior_latent_space, prior_latent_space)
    return kl_div

@MODELS.register_module()
class MutualInfoLoss(nn.Module):
    def __init__(self, input_channels=1024, channels=64, latent_size = 64):
        super(MutualInfoLoss, self).__init__()
        self.input_channels = input_channels
        self.latent_size = latent_size
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.input_norm = nn.LayerNorm(input_channels)
        self.mu_head_rgb = self._build_head(input_channels, latent_size)
        self.logvar_head_rgb = self._build_head(input_channels, latent_size)
        self.mu_head_lwir = self._build_head(input_channels, latent_size)
        self.logvar_head_lwir = self._build_head(input_channels, latent_size)
        self.bce = nn.BCELoss(reduction='none')
        self.tanh = torch.nn.Tanh()

    @staticmethod
    def _build_head(in_channels, latent_size):
        return nn.Sequential(
            nn.Linear(in_channels, latent_size),
            nn.LayerNorm(latent_size),
            nn.ReLU(inplace=True),
            nn.Linear(latent_size, latent_size),
        )

    def _ensure_finite(self, tensor, name):
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f'MutualInfoLoss produced non-finite values in {name}')

    def _latent_stats(self, feat, mu_head, logvar_head, prefix):
        self._ensure_finite(feat, f'{prefix}_feat')
        feat = self.input_norm(feat)
        self._ensure_finite(feat, f'{prefix}_feat_norm')
        mu = self.tanh(mu_head(feat))
        logvar = self.tanh(logvar_head(feat))
        self._ensure_finite(mu, f'{prefix}_mu')
        self._ensure_finite(logvar, f'{prefix}_logvar')
        std = torch.exp(0.5 * logvar).clamp(min=1e-6, max=1e2)
        self._ensure_finite(std, f'{prefix}_std')
        return mu, logvar, std

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar).clamp(min=1e-6, max=1e2)
        self._ensure_finite(std, 'reparam_std')
        eps = torch.randn_like(std)
        self._ensure_finite(eps, 'reparam_eps')
        return eps.mul(std).add_(mu)

    def forward(self, rgb_feat, lwir_feat):
        rgb_feat = self.pool(rgb_feat).flatten(1)
        lwir_feat = self.pool(lwir_feat).flatten(1)
        mu_rgb, logvar_rgb, std_rgb = self._latent_stats(
            rgb_feat, self.mu_head_rgb, self.logvar_head_rgb, 'rgb')
        mu_lwir, logvar_lwir, std_lwir = self._latent_stats(
            lwir_feat, self.mu_head_lwir, self.logvar_head_lwir, 'lwir')
        z_rgb = self.reparametrize(mu_rgb, logvar_rgb)
        self._ensure_finite(z_rgb, 'z_rgb')
        dist_rgb = Independent(Normal(loc=mu_rgb, scale=std_rgb), 1)
        z_lwir = self.reparametrize(mu_lwir, logvar_lwir)
        self._ensure_finite(z_lwir, 'z_lwir')
        dist_lwir = Independent(Normal(loc=mu_lwir, scale=std_lwir), 1)
        bi_di_kld = (
            kl_divergence_loss(dist_rgb, dist_lwir) +
            kl_divergence_loss(dist_lwir, dist_rgb)
        ) / self.latent_size
        z_rgb_norm = torch.sigmoid(z_rgb)
        z_lwir_norm = torch.sigmoid(z_lwir)
        ce_rgb_lwir = self.bce(z_rgb_norm, z_lwir_norm.detach()).mean(dim=1)
        ce_lwir_rgb = self.bce(z_lwir_norm, z_rgb_norm.detach()).mean(dim=1)
        latent_loss = (ce_rgb_lwir + ce_lwir_rgb - bi_di_kld).mean()
        self._ensure_finite(latent_loss, 'latent_loss')

        return latent_loss
