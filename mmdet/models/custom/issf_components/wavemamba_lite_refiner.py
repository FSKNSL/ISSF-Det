"""WaveMamba-lite style FPN frequency refiner for ISS-MoE."""

import math
from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from mmdet.registry import MODELS

from .wavelet_frequency_prior_refiner import WaveletFrequencyPriorRefiner


@MODELS.register_module()
class WaveMambaLiteFrequencyRefiner(WaveletFrequencyPriorRefiner):
    """Adaptive LL fusion and gated high-frequency preservation.

    This keeps the existing shared-FPN latent interface.  On selected FPN
    levels, LL is softly pulled toward an adaptive VIS/LWIR low-frequency
    mixture, while LH/HL/HH are softly pulled toward the signed cross-modal
    absolute maximum through a learned content gate.
    """

    def __init__(self,
                 levels: Sequence[int] = (1, 2),
                 channels: int = 256,
                 hidden_channels: int = 64,
                 low_strength: float = 0.04,
                 high_strength: float = 0.06,
                 low_lwir_prior: float = 0.65,
                 high_mask_prior: float = 0.5,
                 detach_absmax: bool = True,
                 residual_scale: float = 0.5,
                 eps: float = 1e-6,
                 init_cfg=None) -> None:
        super().__init__(
            levels=levels,
            beta=1.0,
            high_enhance_weight=0.0,
            low_lwir_blend_weight=0.0,
            detach_masks=detach_absmax,
            residual_scale=residual_scale,
            eps=eps,
            init_cfg=init_cfg)
        self.channels = int(channels)
        self.hidden_channels = int(hidden_channels)
        self.low_strength = float(low_strength)
        self.high_strength = float(high_strength)
        self.low_lwir_prior = float(low_lwir_prior)
        self.high_mask_prior = float(high_mask_prior)
        self._validate_lite_args()

        self.low_gate = self._build_gate(self.channels * 3)
        self.high_gate = self._build_gate(self.channels * 4)
        self._init_gate(self.low_gate, self._logit(self.low_lwir_prior))
        self._init_gate(self.high_gate, self._logit(self.high_mask_prior))
        self.latest_aux_logs: Dict[str, Tensor] = {}

    def _validate_lite_args(self) -> None:
        if len(self.levels) == 0:
            raise ValueError('levels must not be empty')
        if self.channels <= 0:
            raise ValueError(f'channels must be positive, got {self.channels}')
        if self.hidden_channels <= 0:
            raise ValueError(
                f'hidden_channels must be positive, got '
                f'{self.hidden_channels}')
        if not 0.0 <= self.low_strength <= 1.0:
            raise ValueError(
                f'low_strength must be in [0, 1], got {self.low_strength}')
        if not 0.0 <= self.high_strength <= 1.0:
            raise ValueError(
                f'high_strength must be in [0, 1], got {self.high_strength}')
        if not 0.0 < self.low_lwir_prior < 1.0:
            raise ValueError(
                'low_lwir_prior must be in (0, 1), got '
                f'{self.low_lwir_prior}')
        if not 0.0 < self.high_mask_prior < 1.0:
            raise ValueError(
                'high_mask_prior must be in (0, 1), got '
                f'{self.high_mask_prior}')

    def _build_gate(self, in_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, self.hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(self.hidden_channels, self.channels, kernel_size=1),
        )

    @staticmethod
    def _logit(value: float) -> float:
        return math.log(value / (1.0 - value))

    @staticmethod
    def _init_gate(gate: nn.Sequential, bias: float) -> None:
        final_conv = gate[-1]
        if not isinstance(final_conv, nn.Conv2d):
            raise TypeError('gate must end with nn.Conv2d')
        nn.init.zeros_(final_conv.weight)
        nn.init.constant_(final_conv.bias, bias)

    def get_aux_logs(self):
        return dict(self.latest_aux_logs)

    def _check_level_inputs(self, vis: Tensor, lwir: Tensor,
                            level: int) -> None:
        if vis.shape != lwir.shape:
            raise ValueError(
                'VIS/LWIR FPN features must have identical shapes, got '
                f'{tuple(vis.shape)} and {tuple(lwir.shape)}')
        if vis.shape[1] != self.channels:
            raise ValueError(
                f'Expected {self.channels} channels at level {level}, got '
                f'{vis.shape[1]}')

    @staticmethod
    def _mean_abs(*tensors: Tensor) -> Tensor:
        return torch.stack([tensor.detach().abs().mean()
                            for tensor in tensors]).mean()

    def _refine_level_lite(self, vis: Tensor, lwir: Tensor, level: int
                           ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        self._check_level_inputs(vis, lwir, level)
        height, width = vis.shape[-2:]
        vis_pad = self._pad_even(vis)
        lwir_pad = self._pad_even(lwir)

        ll_vis, lh_vis, hl_vis, hh_vis = self._haar_dwt(vis_pad)
        ll_lwir, lh_lwir, hl_lwir, hh_lwir = self._haar_dwt(lwir_pad)

        low_input = torch.cat(
            [ll_vis, ll_lwir, (ll_vis - ll_lwir).abs()], dim=1)
        low_gate = torch.sigmoid(self.low_gate(low_input))
        ll_shared = low_gate * ll_lwir + (1.0 - low_gate) * ll_vis
        ll_vis_ref = ll_vis + self.low_strength * (ll_shared - ll_vis)
        ll_lwir_ref = ll_lwir + self.low_strength * (
            ll_shared - ll_lwir)

        high_vis_mag = (lh_vis.abs() + hl_vis.abs() + hh_vis.abs()) / 3.0
        high_lwir_mag = (
            lh_lwir.abs() + hl_lwir.abs() + hh_lwir.abs()) / 3.0
        high_input = torch.cat([
            ll_shared,
            high_vis_mag,
            high_lwir_mag,
            (high_vis_mag - high_lwir_mag).abs(),
        ], dim=1)
        high_mask = torch.sigmoid(self.high_gate(high_input))
        lh_vis_ref, lh_lwir_ref = self._refine_high_pair_lite(
            lh_vis, lh_lwir, high_mask)
        hl_vis_ref, hl_lwir_ref = self._refine_high_pair_lite(
            hl_vis, hl_lwir, high_mask)
        hh_vis_ref, hh_lwir_ref = self._refine_high_pair_lite(
            hh_vis, hh_lwir, high_mask)

        vis_out = self._haar_idwt(
            ll_vis_ref, lh_vis_ref, hl_vis_ref, hh_vis_ref)
        lwir_out = self._haar_idwt(
            ll_lwir_ref, lh_lwir_ref, hl_lwir_ref, hh_lwir_ref)
        vis_out = self._crop_to_shape(vis_out, height, width)
        lwir_out = self._crop_to_shape(lwir_out, height, width)
        if self.residual_scale != 1.0:
            vis_out = vis + self.residual_scale * (vis_out - vis)
            lwir_out = lwir + self.residual_scale * (lwir_out - lwir)

        low_delta = self._mean_abs(
            ll_vis_ref - ll_vis, ll_lwir_ref - ll_lwir)
        low_energy = self._mean_abs(ll_vis, ll_lwir).clamp(min=self.eps)
        high_delta = self._mean_abs(
            lh_vis_ref - lh_vis, hl_vis_ref - hl_vis, hh_vis_ref - hh_vis,
            lh_lwir_ref - lh_lwir, hl_lwir_ref - hl_lwir,
            hh_lwir_ref - hh_lwir)
        high_energy = self._mean_abs(
            lh_vis, hl_vis, hh_vis, lh_lwir, hl_lwir, hh_lwir).clamp(
                min=self.eps)
        logs = dict(
            low_gate=low_gate.detach().mean(),
            high_mask=high_mask.detach().mean(),
            low_delta_ratio=low_delta / low_energy,
            high_delta_ratio=high_delta / high_energy,
            delta_vis=(vis_out.detach() - vis.detach()).abs().mean(),
            delta_lwir=(lwir_out.detach() - lwir.detach()).abs().mean())
        return vis_out, lwir_out, logs

    def _refine_high_pair_lite(self, high_vis: Tensor, high_lwir: Tensor,
                               high_mask: Tensor
                               ) -> Tuple[Tensor, Tensor]:
        high_max = self._signed_absmax(high_vis, high_lwir)
        high_vis = high_vis + self.high_strength * high_mask * (
            high_max - high_vis)
        high_lwir = high_lwir + self.high_strength * high_mask * (
            high_max - high_lwir)
        return high_vis, high_lwir

    def forward(self, p_vis, p_lwir):
        if len(p_vis) != len(p_lwir):
            raise ValueError(
                f'Expected same number of VIS/LWIR levels, got {len(p_vis)} '
                f'and {len(p_lwir)}')
        if any(level >= len(p_vis) for level in self.levels):
            raise ValueError(
                f'levels {self.levels} exceed available FPN levels '
                f'{len(p_vis)}')

        selected = set(self.levels)
        refined_vis = []
        refined_lwir = []
        logs: Dict[str, Tensor] = {}
        for level, (vis, lwir) in enumerate(zip(p_vis, p_lwir)):
            if level in selected:
                vis, lwir, level_logs = self._refine_level_lite(
                    vis, lwir, level)
                for key, value in level_logs.items():
                    logs[f'wavemamba_lite_{key}_l{level}'] = value
            refined_vis.append(vis)
            refined_lwir.append(lwir)
        self.latest_aux_logs = logs
        return tuple(refined_vis), tuple(refined_lwir)
