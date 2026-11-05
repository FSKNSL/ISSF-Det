"""Spectral mask and residual spectral conditioning front ends."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from mmdet.registry import MODELS


class SpectralPatchMaskNet(nn.Module):
    def __init__(self, imgshape, patch_num=20):
        super().__init__()
        height, width = imgshape
        self.patch_num = patch_num
        self.conv1 = nn.Conv2d(3, 16, 7, 4, 3)
        self.conv2 = nn.Conv2d(16, 32, 7, 4, 3)
        self.conv3 = nn.Conv2d(32, 64, 7, 4, 3)
        self.flatten = nn.Flatten()
        out_height = self._conv_out_size(
            self._conv_out_size(self._conv_out_size(height)))
        out_width = self._conv_out_size(
            self._conv_out_size(self._conv_out_size(width)))

        self.trans2list = nn.Sequential(
            nn.Linear(64 * out_height * out_width, 1000),
            nn.Linear(
                in_features=1000,
                out_features=np.power(self.patch_num, 2),
                bias=True),
            nn.Sigmoid())

    @staticmethod
    def _conv_out_size(size):
        return (int(size) + 2 * 3 - 7) // 4 + 1

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x_final = self.trans2list(self.flatten(x))
        return x_final


class SpectralMaskSelector(nn.Module):
    def __init__(self, imgshape):
        super().__init__()

        self.g_mask_binary_list = SpectralPatchMaskNet(imgshape)

    def forward(self, x):
        mask_list = self.g_mask_binary_list(x)
        return mask_list


class MaskFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, mask_list_, topk):
        topk = int(topk)
        mask_list_topk, _ = mask_list_.topk(topk)
        mask_list_min = torch.min(mask_list_topk, dim=1).values
        mask_list_min_ = mask_list_min.unsqueeze(-1)
        ge = torch.ge(mask_list_, mask_list_min_)
        zero = torch.zeros_like(mask_list_)
        one = torch.ones_like(mask_list_)
        mask_list = torch.where(ge, one, zero)
        return mask_list

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


@MODELS.register_module()
class SpectralMaskBaseline(BaseModule):
    """Spectral mask front end with hard Top-K selection."""
    def __init__(self,
                 imgshape,
                 keep_low,
                 patch_num=20,
                 topk=320,
                 topk_vis=None,
                 topk_lwir=None,
                 low_keep_size=2,
                 low_keep_size_vis=None,
                 low_keep_size_lwir=None):
        super().__init__()
        self.patch_num = patch_num
        self.keep_low = keep_low
        self.imgshape = imgshape
        self.topk_vis = topk if topk_vis is None else topk_vis
        self.topk_lwir = topk if topk_lwir is None else topk_lwir
        self.low_keep_size_vis = (
            low_keep_size if low_keep_size_vis is None else low_keep_size_vis)
        self.low_keep_size_lwir = (
            low_keep_size if low_keep_size_lwir is None else low_keep_size_lwir)
        self._validate_mask_args()
        self.Gmaskbinarylist_vis = SpectralMaskSelector(imgshape)
        self.Gmaskbinarylist_lwir = SpectralMaskSelector(imgshape)
        self.MaskFun_vis = MaskFunction()
        self.MaskFun_lwir = MaskFunction()

    def _validate_mask_args(self):
        num_bins = self.patch_num * self.patch_num
        for name, value in (('topk_vis', self.topk_vis),
                            ('topk_lwir', self.topk_lwir)):
            if not 1 <= int(value) <= num_bins:
                raise ValueError(
                    f'{name} must be in [1, {num_bins}], got {value}')
        for name, value in (('low_keep_size_vis', self.low_keep_size_vis),
                            ('low_keep_size_lwir', self.low_keep_size_lwir)):
            if not 0 <= int(value) <= self.patch_num:
                raise ValueError(
                    f'{name} must be in [0, {self.patch_num}], got {value}')

    def _apply_low_frequency_rule(self, mask_list, low_keep_size):
        low_keep_size = int(low_keep_size)
        if low_keep_size == 0:
            return mask_list
        center = int(self.patch_num / 2)
        half = int(low_keep_size / 2)
        start = center - half
        end = start + low_keep_size
        value = 1 if self.keep_low else 0
        mask_list[:, :, start:end, start:end] = value
        return mask_list

    def forward(self, img_vis, img_lwir):
        """Forward function."""

        vis_fre = torch.fft.fft2(img_vis)
        fre_m_vis = torch.abs(vis_fre)
        fre_m_vis = torch.fft.fftshift(fre_m_vis)

        lwir_fre = torch.fft.fft2(img_lwir)
        fre_m_lwir = torch.abs(lwir_fre)
        fre_m_lwir = torch.fft.fftshift(fre_m_lwir)
        mask_vis_list_ = self.Gmaskbinarylist_vis(fre_m_vis)
        mask_lwir_list_ = self.Gmaskbinarylist_lwir(fre_m_lwir)
        mask_vis_list = self.MaskFun_vis.apply(
            mask_vis_list_, self.topk_vis).reshape(
                (-1, 1, self.patch_num, self.patch_num))
        mask_lwir_list = self.MaskFun_lwir.apply(
            mask_lwir_list_, self.topk_lwir).reshape(
                (-1, 1, self.patch_num, self.patch_num))
        mask_vis_list = self._apply_low_frequency_rule(
            mask_vis_list, self.low_keep_size_vis)
        mask_lwir_list = self._apply_low_frequency_rule(
            mask_lwir_list, self.low_keep_size_lwir)

        scale_factor = [
            self.imgshape[0] / self.patch_num,
            self.imgshape[1] / self.patch_num,
        ]
        mask_vis = F.interpolate(
            mask_vis_list, scale_factor=scale_factor, mode='nearest')
        mask_lwir = F.interpolate(
            mask_lwir_list, scale_factor=scale_factor, mode='nearest')

        return mask_vis, mask_lwir


@MODELS.register_module()
class MRSCResidual(SpectralMaskBaseline):
    """Bounded residual conditioning applied to a spectral mask.

    For each modality, the spectral mask ``M`` is mapped to
    ``H = 1 - lambda_mrsc * (1 - M)``.
    """

    def __init__(self, *args, lambda_mrsc=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_mrsc = float(lambda_mrsc)
        if not 0.0 <= self.lambda_mrsc < 1.0:
            raise ValueError(
                'lambda_mrsc must be in [0, 1), got '
                f'{self.lambda_mrsc}')

    def forward(self, img_vis, img_lwir):
        mask_vis, mask_lwir = super().forward(img_vis, img_lwir)
        mask_vis = 1.0 - self.lambda_mrsc * (1.0 - mask_vis)
        mask_lwir = 1.0 - self.lambda_mrsc * (1.0 - mask_lwir)
        return mask_vis, mask_lwir


@MODELS.register_module()
class MRSC(SpectralMaskBaseline):
    """Continuous, bounded spectral conditioning for both modalities.

    Each modality keeps its own response estimator.  Log-normalized spectral
    magnitudes are mapped to a 20x20 response grid, restricted by exact
    Top-K support, protected at low frequencies and interpolated into the
    bounded residual response ``H = 1 - lambda_mrsc * (1 - Q)``.  The final
    response is conjugate-symmetric so the detector's phase-preserving inverse
    FFT remains real-valued up to numerical precision.
    """

    def __init__(self, *args, lambda_mrsc=0.1, eps=1e-6, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_mrsc = float(lambda_mrsc)
        self.eps = float(eps)
        if not 0.0 <= self.lambda_mrsc < 1.0:
            raise ValueError(
                'lambda_mrsc must be in [0, 1), got '
                f'{self.lambda_mrsc}')
        if self.eps <= 0.0:
            raise ValueError(f'eps must be positive, got {self.eps}')
        if not self.keep_low:
            raise ValueError('MRSC requires keep_low=True')

    def _normalize_magnitude(self, magnitude):
        magnitude = torch.log1p(magnitude)
        mean = magnitude.mean(dim=(2, 3), keepdim=True)
        std = magnitude.std(
            dim=(2, 3), keepdim=True, unbiased=False).clamp_min(self.eps)
        return (magnitude - mean) / std

    @staticmethod
    def _exact_topk_support(response, topk):
        _, indices = response.topk(int(topk), dim=1)
        support = torch.zeros_like(response)
        return support.scatter(1, indices, 1.0)

    @staticmethod
    def _enforce_conjugate_symmetry(shifted_response):
        response = torch.fft.ifftshift(
            shifted_response, dim=(-2, -1))
        conjugate_partner = torch.roll(
            torch.flip(response, dims=(-2, -1)),
            shifts=(1, 1), dims=(-2, -1))
        response = 0.5 * (response + conjugate_partner)
        return torch.fft.fftshift(response, dim=(-2, -1))

    def _build_response(self, image, scorer, topk, low_keep_size,
                        lambda_mrsc=None):
        magnitude = torch.fft.fftshift(
            torch.abs(torch.fft.fft2(image)), dim=(-2, -1))
        normalized = self._normalize_magnitude(magnitude)
        response = scorer(normalized)
        support = self._exact_topk_support(response, topk)
        response = (support * response).reshape(
            (-1, 1, self.patch_num, self.patch_num))
        response = self._apply_low_frequency_rule(
            response, low_keep_size)
        response = F.interpolate(
            response, size=self.imgshape, mode='bilinear',
            align_corners=False)
        if lambda_mrsc is None:
            lambda_mrsc = self.lambda_mrsc
        response = 1.0 - lambda_mrsc * (1.0 - response)
        return self._enforce_conjugate_symmetry(response)

    def forward(self, img_vis, img_lwir):
        mask_vis = self._build_response(
            img_vis, self.Gmaskbinarylist_vis, self.topk_vis,
            self.low_keep_size_vis)
        mask_lwir = self._build_response(
            img_lwir, self.Gmaskbinarylist_lwir, self.topk_lwir,
            self.low_keep_size_lwir)
        return mask_vis, mask_lwir
