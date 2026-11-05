"""FPN-level wavelet frequency prior refinement for ISS-MoE."""

from numbers import Real
from typing import Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from mmengine.model import BaseModule
from torch import Tensor

from mmdet.registry import MODELS


@MODELS.register_module()
class WaveletFrequencyPriorRefiner(BaseModule):
    """Haar frequency prior refiner for VIS/LWIR FPN latents.

    The refiner keeps the two latent streams separate. On selected FPN levels,
    LL is biased slightly toward LWIR structure, while LH/HL/HH are enhanced
    toward the signed cross-modal absolute maximum instead of being attenuated.
    """

    def __init__(self,
                 levels: Sequence[int] = (1, 2),
                 beta: float = 8.0,
                 high_enhance_weight: float = 0.05,
                 low_lwir_blend_weight: Union[float, Sequence[float]] = 0.03,
                 detach_masks: bool = True,
                 residual_scale: float = 1.0,
                 eps: float = 1e-6,
                 init_cfg=None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.levels = tuple(int(level) for level in levels)
        self.beta = float(beta)
        self.high_enhance_weight = float(high_enhance_weight)
        self.low_lwir_blend_weight = low_lwir_blend_weight
        self.low_lwir_blend_weights = self._parse_low_lwir_blend_weights(
            low_lwir_blend_weight)
        self.detach_masks = bool(detach_masks)
        self.residual_scale = float(residual_scale)
        self.eps = float(eps)
        self._validate_args()

    def _parse_low_lwir_blend_weights(
            self, low_lwir_blend_weight: Union[float, Sequence[float]]
    ) -> Tuple[float, ...]:
        if isinstance(low_lwir_blend_weight, Real):
            return tuple(float(low_lwir_blend_weight) for _ in self.levels)
        weights = tuple(float(weight) for weight in low_lwir_blend_weight)
        if len(weights) != len(self.levels):
            raise ValueError(
                'low_lwir_blend_weight sequence length must match levels, '
                f'got {len(weights)} weights for levels {self.levels}')
        return weights

    def _validate_args(self) -> None:
        if len(set(self.levels)) != len(self.levels):
            raise ValueError(f'levels must be unique, got {self.levels}')
        if any(level < 0 for level in self.levels):
            raise ValueError(f'levels must be non-negative, got {self.levels}')
        if self.beta <= 0:
            raise ValueError(f'beta must be positive, got {self.beta}')
        if not 0.0 <= self.high_enhance_weight <= 1.0:
            raise ValueError(
                'high_enhance_weight must be in [0, 1], got '
                f'{self.high_enhance_weight}')
        for weight in self.low_lwir_blend_weights:
            if not 0.0 <= weight <= 1.0:
                raise ValueError(
                    'low_lwir_blend_weight values must be in [0, 1], got '
                    f'{self.low_lwir_blend_weights}')
        if not 0.0 <= self.residual_scale <= 1.0:
            raise ValueError(
                'residual_scale must be in [0, 1], got '
                f'{self.residual_scale}')
        if self.eps <= 0:
            raise ValueError(f'eps must be positive, got {self.eps}')

    @staticmethod
    def _pad_even(x: Tensor) -> Tensor:
        pad_h = x.shape[-2] % 2
        pad_w = x.shape[-1] % 2
        if pad_h == 0 and pad_w == 0:
            return x
        return F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')

    @staticmethod
    def _crop_to_shape(x: Tensor, height: int, width: int) -> Tensor:
        return x[..., :height, :width]

    @staticmethod
    def _haar_dwt(x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        x00 = x[..., 0::2, 0::2]
        x01 = x[..., 0::2, 1::2]
        x10 = x[..., 1::2, 0::2]
        x11 = x[..., 1::2, 1::2]

        ll = (x00 + x01 + x10 + x11) * 0.5
        lh = (x00 - x01 + x10 - x11) * 0.5
        hl = (x00 + x01 - x10 - x11) * 0.5
        hh = (x00 - x01 - x10 + x11) * 0.5
        return ll, lh, hl, hh

    @staticmethod
    def _haar_idwt(ll: Tensor, lh: Tensor, hl: Tensor, hh: Tensor) -> Tensor:
        x00 = (ll + lh + hl + hh) * 0.5
        x01 = (ll - lh + hl - hh) * 0.5
        x10 = (ll + lh - hl - hh) * 0.5
        x11 = (ll - lh - hl + hh) * 0.5

        x = ll.new_empty(
            (ll.shape[0], ll.shape[1], ll.shape[2] * 2, ll.shape[3] * 2))
        x[..., 0::2, 0::2] = x00
        x[..., 0::2, 1::2] = x01
        x[..., 1::2, 0::2] = x10
        x[..., 1::2, 1::2] = x11
        return x

    def _lwir_low_confidence(self, ll_vis: Tensor, ll_lwir: Tensor) -> Tensor:
        abs_vis = ll_vis.abs()
        abs_lwir = ll_lwir.abs()
        score = (abs_lwir - abs_vis) / (abs_vis + abs_lwir + self.eps)
        conf = torch.sigmoid(self.beta * score)
        if self.detach_masks:
            conf = conf.detach()
        return conf

    def _refine_low_pair(self, ll_vis: Tensor, ll_lwir: Tensor,
                         low_lwir_blend_weight: float
                         ) -> Tuple[Tensor, Tensor]:
        conf_lwir = self._lwir_low_confidence(ll_vis, ll_lwir)
        ll_vis = ll_vis + low_lwir_blend_weight * conf_lwir * (
            ll_lwir - ll_vis)
        return ll_vis, ll_lwir

    def _signed_absmax(self, high_vis: Tensor, high_lwir: Tensor) -> Tensor:
        mask = high_vis.abs() >= high_lwir.abs()
        if self.detach_masks:
            mask = mask.detach()
        return torch.where(mask, high_vis, high_lwir)

    def _refine_high_pair(self, high_vis: Tensor,
                          high_lwir: Tensor) -> Tuple[Tensor, Tensor]:
        high_max = self._signed_absmax(high_vis, high_lwir)
        high_vis = high_vis + self.high_enhance_weight * (
            high_max - high_vis)
        high_lwir = high_lwir + self.high_enhance_weight * (
            high_max - high_lwir)
        return high_vis, high_lwir

    def _refine_level(self, vis: Tensor, lwir: Tensor,
                      low_lwir_blend_weight: float
                      ) -> Tuple[Tensor, Tensor]:
        if vis.shape != lwir.shape:
            raise ValueError(
                'VIS/LWIR FPN features must have identical shapes, got '
                f'{tuple(vis.shape)} and {tuple(lwir.shape)}')
        height, width = vis.shape[-2:]
        vis_pad = self._pad_even(vis)
        lwir_pad = self._pad_even(lwir)

        ll_vis, lh_vis, hl_vis, hh_vis = self._haar_dwt(vis_pad)
        ll_lwir, lh_lwir, hl_lwir, hh_lwir = self._haar_dwt(lwir_pad)
        ll_vis, ll_lwir = self._refine_low_pair(
            ll_vis, ll_lwir, low_lwir_blend_weight)
        lh_vis, lh_lwir = self._refine_high_pair(lh_vis, lh_lwir)
        hl_vis, hl_lwir = self._refine_high_pair(hl_vis, hl_lwir)
        hh_vis, hh_lwir = self._refine_high_pair(hh_vis, hh_lwir)

        vis_out = self._haar_idwt(ll_vis, lh_vis, hl_vis, hh_vis)
        lwir_out = self._haar_idwt(ll_lwir, lh_lwir, hl_lwir, hh_lwir)
        vis_out = self._crop_to_shape(vis_out, height, width)
        lwir_out = self._crop_to_shape(lwir_out, height, width)
        if self.residual_scale != 1.0:
            vis_out = vis + self.residual_scale * (vis_out - vis)
            lwir_out = lwir + self.residual_scale * (lwir_out - lwir)
        return vis_out, lwir_out

    def forward(self, p_vis, p_lwir):
        if len(p_vis) != len(p_lwir):
            raise ValueError(
                f'Expected same number of VIS/LWIR levels, got {len(p_vis)} '
                f'and {len(p_lwir)}')
        if any(level >= len(p_vis) for level in self.levels):
            raise ValueError(
                f'levels {self.levels} exceed available FPN levels '
                f'{len(p_vis)}')

        level_to_low_weight = dict(zip(self.levels,
                                       self.low_lwir_blend_weights))
        refined_vis = []
        refined_lwir = []
        for level, (vis, lwir) in enumerate(zip(p_vis, p_lwir)):
            if level in level_to_low_weight:
                vis, lwir = self._refine_level(
                    vis, lwir, level_to_low_weight[level])
            refined_vis.append(vis)
            refined_lwir.append(lwir)
        return tuple(refined_vis), tuple(refined_lwir)
