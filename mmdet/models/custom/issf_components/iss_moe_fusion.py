"""Interaction-Structured Spatial Mixture-of-Experts fusion for ISSF-Det.

The experts and spatial router receive separable visible and infrared FPN
latents ``(p_vis, p_lwir)``. Perturbation is applied at the latent level for
interaction supervision.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.model import BaseModule
from mmdet.registry import MODELS
from mmdet.utils import OptConfigType


EXPERT_ROLES = ('uni_vis', 'uni_lwir', 'syn', 'red')


def _make_projection(channels: int, num_groups: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
        nn.GroupNorm(num_groups, channels),
        nn.ReLU(inplace=True),
    )


def _make_expert_block(channels: int, num_groups: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(
            channels, channels, kernel_size=3, padding=1,
            groups=channels, bias=False),
        nn.GroupNorm(num_groups, channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(channels, channels, kernel_size=1, bias=False),
        nn.GroupNorm(num_groups, channels),
    )


class DualInputExpertHead(nn.Module):
    """Light expert head that receives separable VIS/LWIR FPN latents."""

    def __init__(self, channels: int = 256, num_levels: int = 5,
                 num_groups: int = 32):
        super().__init__()
        self.num_levels = num_levels
        self.projections = nn.ModuleList([
            _make_projection(channels, num_groups)
            for _ in range(num_levels)
        ])
        self.blocks = nn.ModuleList([
            _make_expert_block(channels, num_groups)
            for _ in range(num_levels)
        ])

    def forward(self, p_vis, p_lwir):
        if len(p_vis) != self.num_levels or len(p_lwir) != self.num_levels:
            raise ValueError(
                f'Expected {self.num_levels} FPN levels, got '
                f'{len(p_vis)} VIS and {len(p_lwir)} LWIR levels')

        outs = []
        for level, (vis, lwir) in enumerate(zip(p_vis, p_lwir)):
            base = self.projections[level](torch.cat([vis, lwir], dim=1))
            outs.append(F.relu(
                base + self.blocks[level](base), inplace=False))
        return tuple(outs)


class DualSpatialReweightModel(nn.Module):
    """Per-level spatial gate conditioned on separable VIS/LWIR latents."""

    def __init__(self, channels: int = 256, num_experts: int = 4,
                 num_scales: int = 5, hidden_dim: int = 256,
                 num_groups: int = 32):
        super().__init__()
        self.num_experts = num_experts
        self.num_scales = num_scales
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    channels * 2, hidden_dim, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, num_experts, kernel_size=1),
            )
            for _ in range(num_scales)
        ])

    def forward(self, p_vis, p_lwir):
        if len(p_vis) != self.num_scales or len(p_lwir) != self.num_scales:
            raise ValueError(
                f'Expected {self.num_scales} FPN levels, got '
                f'{len(p_vis)} VIS and {len(p_lwir)} LWIR levels')

        weights = []
        for level, (vis, lwir) in enumerate(zip(p_vis, p_lwir)):
            gate_input = torch.cat([vis, lwir], dim=1)
            weights.append(F.softmax(self.heads[level](gate_input), dim=1))
        return tuple(weights)


@MODELS.register_module()
class ISSMoEFusionDualLatent(BaseModule):
    """Dual-latent, spatial-gated ISS-MoE fusion.

    Normal path:
        ``x_vis/x_lwir -> FPN latent(s) -> p_vis/p_lwir -> experts/gate``.

    Perturbed path:
        ``mask(p_vis)`` or ``mask(p_lwir)`` is fed to the same experts, so the
        interaction loss observes direct latent replacement instead of a
        replacement hidden behind the reducer/FPN.
    """

    def __init__(
        self,
        neck: OptConfigType,
        backbone_channels=(256, 512, 1024, 2048),
        num_scales: int = 5,
        reweight_feat_level: int = 2,
        reweight_hidden_dim: int = 256,
        num_groups: int = 32,
        loss_MI: OptConfigType = None,
        perturb_mode: str = 'channel_mean',
        latent_refiner: OptConfigType = None,
        latent_refiner_gate_source: str = 'refined',
        gate_input_adapter: OptConfigType = None,
        separate_fpn: bool = False,
        expert_roles=EXPERT_ROLES,
        gate_mode: str = 'learned',
    ) -> None:
        super().__init__()

        self.backbone_channels = tuple(backbone_channels)
        self.num_scales = num_scales
        self.expert_roles = tuple(expert_roles)
        if not self.expert_roles:
            raise ValueError('expert_roles must contain at least one role.')
        if len(set(self.expert_roles)) != len(self.expert_roles):
            raise ValueError(
                f'expert_roles must not contain duplicates, got '
                f'{self.expert_roles}.')
        if any(role not in EXPERT_ROLES for role in self.expert_roles):
            raise ValueError(
                f'expert_roles must be selected from {EXPERT_ROLES}, got '
                f'{self.expert_roles}.')
        canonical_order = tuple(
            role for role in EXPERT_ROLES if role in self.expert_roles)
        if self.expert_roles != canonical_order:
            raise ValueError(
                f'expert_roles must follow canonical order {EXPERT_ROLES}, '
                f'got {self.expert_roles}.')
        self.num_experts = len(self.expert_roles)
        self.gate_mode = gate_mode
        if self.gate_mode not in ('learned', 'uniform'):
            raise ValueError(
                'gate_mode must be learned or uniform, got '
                f'{self.gate_mode!r}')
        self.reweight_feat_level = reweight_feat_level
        self.perturb_mode = perturb_mode
        self.latent_refiner_gate_source = latent_refiner_gate_source
        self.separate_fpn = separate_fpn
        if self.perturb_mode not in ('matched_noise', 'channel_mean', 'zero'):
            raise ValueError(
                'perturb_mode must be one of matched_noise, channel_mean, '
                f'zero, got {self.perturb_mode!r}')
        if self.latent_refiner_gate_source not in ('refined', 'raw'):
            raise ValueError(
                'latent_refiner_gate_source must be refined or raw, got '
                f'{self.latent_refiner_gate_source!r}')
        if loss_MI is not None:
            raise ValueError(
                'ISSMoEFusionDualLatent does not support MI loss; set '
                'loss_MI=None.')
        self.use_mi = False

        if self.separate_fpn:
            self.fpn_vis = MODELS.build(copy.deepcopy(neck))
            self.fpn_lwir = MODELS.build(copy.deepcopy(neck))
            self.shared_fpn = None
        else:
            self.shared_fpn = MODELS.build(neck)
        self.fpn_out_channels = neck.get('out_channels', 256)
        self.latent_refiner = (
            MODELS.build(latent_refiner)
            if latent_refiner is not None else None)
        if gate_input_adapter is not None:
            # Isolate adapter initialization from subsequent expert and gate RNG.
            with torch.random.fork_rng(devices=[]):
                self.gate_input_adapter = MODELS.build(gate_input_adapter)
        else:
            self.gate_input_adapter = None

        self.experts = nn.ModuleList([
            DualInputExpertHead(
                channels=self.fpn_out_channels,
                num_levels=num_scales,
                num_groups=num_groups,
            )
            for _ in range(self.num_experts)
        ])
        if self.gate_mode == 'uniform':
            self.reweight = None
        else:
            self.reweight = DualSpatialReweightModel(
                channels=self.fpn_out_channels,
                num_experts=self.num_experts,
                num_scales=num_scales,
                hidden_dim=reweight_hidden_dim,
                num_groups=num_groups,
            )

    @staticmethod
    def _ensure_finite(tensor, name):
        if not torch.isfinite(tensor).all():
            raise RuntimeError(
                f'ISSMoEFusionDualLatent produced non-finite values in '
                f'{name}')

    def _check_feature_tuple(self, feat_tuple, name):
        if len(feat_tuple) != self.num_scales:
            raise ValueError(
                f'{name} expected {self.num_scales} levels, got '
                f'{len(feat_tuple)}')
        for level, feat in enumerate(feat_tuple):
            self._ensure_finite(feat, f'{name}[{level}]')

    @staticmethod
    def _matched_noise(feat_tuple):
        noisy = []
        for feat in feat_tuple:
            mu = feat.mean(dim=(2, 3), keepdim=True)
            sigma = feat.std(
                dim=(2, 3), keepdim=True, unbiased=False).clamp(min=1e-4)
            noisy.append(mu + sigma * torch.randn_like(feat))
        return tuple(noisy)

    @staticmethod
    def _channel_mean(feat_tuple):
        return tuple(
            feat.mean(dim=(2, 3), keepdim=True).expand_as(feat)
            for feat in feat_tuple)

    @staticmethod
    def _zero_features(feat_tuple):
        return tuple(torch.zeros_like(feat) for feat in feat_tuple)

    def _mask_features(self, feat_tuple):
        if self.perturb_mode == 'matched_noise':
            return self._matched_noise(feat_tuple)
        if self.perturb_mode == 'channel_mean':
            return self._channel_mean(feat_tuple)
        if self.perturb_mode == 'zero':
            return self._zero_features(feat_tuple)
        raise AssertionError(f'Unhandled perturb_mode: {self.perturb_mode}')

    @staticmethod
    def _concat_feature_tuples(*feat_tuples):
        return tuple(
            torch.cat(level_feats, dim=0)
            for level_feats in zip(*feat_tuples))

    @staticmethod
    def _split_expert_outputs(expert_outputs, batch_size: int):
        first = []
        second = []
        for expert_out in expert_outputs:
            first.append(tuple(feat[:batch_size] for feat in expert_out))
            second.append(tuple(feat[batch_size:] for feat in expert_out))
        return tuple(first), tuple(second)

    def _fpn_vis(self, x_vis):
        if self.separate_fpn:
            return self.fpn_vis(x_vis)
        return self.shared_fpn(x_vis)

    def _fpn_lwir(self, x_lwir):
        if self.separate_fpn:
            return self.fpn_lwir(x_lwir)
        return self.shared_fpn(x_lwir)

    def _compute_latents(self, x_vis, x_lwir, x_vis_aux=None,
                         return_raw: bool = False):
        p_vis = self._fpn_vis(x_vis)
        p_lwir = self._fpn_lwir(x_lwir)
        self._check_feature_tuple(p_vis, 'p_vis')
        self._check_feature_tuple(p_lwir, 'p_lwir')
        p_vis_raw = p_vis
        p_lwir_raw = p_lwir
        p_vis_aux = None
        if x_vis_aux is not None:
            p_vis_aux = self._fpn_vis(x_vis_aux)
            self._check_feature_tuple(p_vis_aux, 'p_vis_aux')
        if self.latent_refiner is not None:
            if p_vis_aux is None:
                p_vis, p_lwir = self.latent_refiner(p_vis, p_lwir)
            else:
                p_vis, p_lwir = self.latent_refiner(
                    p_vis, p_lwir, p_vis_aux)
            self._check_feature_tuple(p_vis, 'p_vis_refined')
            self._check_feature_tuple(p_lwir, 'p_lwir_refined')
        if return_raw:
            return p_vis, p_lwir, p_vis_raw, p_lwir_raw
        return p_vis, p_lwir

    def get_aux_refiner_logs(self):
        logs = {}
        if (self.latent_refiner is not None
                and hasattr(self.latent_refiner, 'get_aux_logs')):
            logs.update(self.latent_refiner.get_aux_logs())
        if (self.gate_input_adapter is not None
                and hasattr(self.gate_input_adapter, 'get_aux_logs')):
            logs.update(self.gate_input_adapter.get_aux_logs())
        if self.reweight is not None and hasattr(self.reweight, 'get_logs'):
            logs.update(self.reweight.get_logs())
        return logs

    def _apply_experts(self, p_vis, p_lwir):
        return tuple(expert(p_vis, p_lwir) for expert in self.experts)

    def _check_expert_outputs(self, expert_outputs, prefix):
        if len(expert_outputs) != self.num_experts:
            raise ValueError(
                f'{prefix} expected {self.num_experts} experts, got '
                f'{len(expert_outputs)}')
        for expert_idx, expert_out in enumerate(expert_outputs):
            self._check_feature_tuple(
                expert_out, f'{prefix}[{expert_idx}]')

    def _compute_weights(self, p_vis, p_lwir):
        if self.gate_mode == 'uniform':
            weights = tuple(
                feat.new_full(
                    (feat.shape[0], self.num_experts,
                     feat.shape[2], feat.shape[3]),
                    1.0 / float(self.num_experts))
                for feat in p_vis)
        else:
            weights = self.reweight(p_vis, p_lwir)
        for level, weight in enumerate(weights):
            self._ensure_finite(weight, f'expert_weights[{level}]')
        return weights

    def _weighted_fuse(self, expert_outputs, weights):
        num_levels = len(expert_outputs[0])
        num_experts = len(expert_outputs)
        if len(weights) != num_levels:
            raise ValueError(
                f'Expected {num_levels} spatial weight levels, got '
                f'{len(weights)}')

        outs = []
        for level in range(num_levels):
            weight = weights[level]
            if weight.shape[1] != num_experts:
                raise ValueError(
                    f'Expected {num_experts} experts at level {level}, got '
                    f'{weight.shape[1]}')
            acc = None
            for expert_idx in range(num_experts):
                term = (
                    weight[:, expert_idx:expert_idx + 1, :, :]
                    * expert_outputs[expert_idx][level])
                acc = term if acc is None else acc + term
            outs.append(acc)
        return tuple(outs)

    @staticmethod
    def _zero_mi(outs):
        device = outs[0].device
        return torch.zeros((), device=device), torch.zeros((), device=device)

    def _forward_from_latents(self, p_vis, p_lwir, gate_p_vis=None,
                              gate_p_lwir=None):
        expert_outputs = self._apply_experts(p_vis, p_lwir)
        self._check_expert_outputs(expert_outputs, 'expert')
        if gate_p_vis is None:
            gate_p_vis = p_vis
        if gate_p_lwir is None:
            gate_p_lwir = p_lwir
        weights = self._compute_weights(gate_p_vis, gate_p_lwir)
        outs = self._weighted_fuse(expert_outputs, weights)
        mi_loss_vis, mi_loss_lwir = self._zero_mi(outs)
        return outs, mi_loss_vis, mi_loss_lwir, weights, expert_outputs

    def _select_gate_latents(self, p_vis, p_lwir, p_vis_raw, p_lwir_raw):
        if self.latent_refiner_gate_source == 'raw':
            return p_vis_raw, p_lwir_raw
        return None, None

    def _apply_gate_input_adapter(self, p_vis, p_lwir, gate_c1_vis=None,
                                  gate_c1_lwir=None):
        if self.gate_input_adapter is None:
            return p_vis, p_lwir
        if gate_c1_vis is None or gate_c1_lwir is None:
            raise ValueError(
                'gate_input_adapter requires gate_c1_vis and gate_c1_lwir.')
        return self.gate_input_adapter(
            p_vis, p_lwir, gate_c1_vis, gate_c1_lwir)

    def _paired_perturbed_from_latents(self, p_vis, p_lwir):
        batch_size = p_vis[0].shape[0]
        p_vis_masked = self._mask_features(p_vis)
        p_lwir_masked = self._mask_features(p_lwir)

        p_vis_pair = self._concat_feature_tuples(p_vis_masked, p_vis)
        p_lwir_pair = self._concat_feature_tuples(p_lwir, p_lwir_masked)

        expert_outputs_pair = self._apply_experts(p_vis_pair, p_lwir_pair)
        self._check_expert_outputs(expert_outputs_pair, 'expert_pair')

        vis_outputs, lwir_outputs = self._split_expert_outputs(
            expert_outputs_pair, batch_size)
        return {'vis': vis_outputs, 'lwir': lwir_outputs}

    def forward(self, x_vis, x_lwir, x_vis_aux=None, gate_c1_vis=None,
                gate_c1_lwir=None):
        p_vis, p_lwir, p_vis_raw, p_lwir_raw = self._compute_latents(
            x_vis, x_lwir, x_vis_aux=x_vis_aux, return_raw=True)
        gate_p_vis, gate_p_lwir = self._select_gate_latents(
            p_vis, p_lwir, p_vis_raw, p_lwir_raw)
        if self.gate_input_adapter is not None:
            if gate_p_vis is None:
                gate_p_vis = p_vis
            if gate_p_lwir is None:
                gate_p_lwir = p_lwir
            gate_p_vis, gate_p_lwir = self._apply_gate_input_adapter(
                gate_p_vis, gate_p_lwir, gate_c1_vis, gate_c1_lwir)
        return self._forward_from_latents(
            p_vis, p_lwir, gate_p_vis=gate_p_vis, gate_p_lwir=gate_p_lwir)

    def forward_perturbed(self, x_vis, x_lwir, mask_modality: str):
        p_vis, p_lwir = self._compute_latents(x_vis, x_lwir)
        if mask_modality == 'vis':
            p_vis = self._mask_features(p_vis)
        elif mask_modality == 'lwir':
            p_lwir = self._mask_features(p_lwir)
        else:
            raise ValueError(
                f"mask_modality must be 'vis' or 'lwir', got "
                f'{mask_modality!r}')

        expert_outputs = self._apply_experts(p_vis, p_lwir)
        self._check_expert_outputs(expert_outputs, 'expert_p')
        return expert_outputs

    def forward_perturbed_pair(self, x_vis, x_lwir):
        p_vis, p_lwir = self._compute_latents(x_vis, x_lwir)
        return self._paired_perturbed_from_latents(p_vis, p_lwir)

    def forward_with_perturbations(self, x_vis, x_lwir, x_vis_aux=None,
                                   gate_c1_vis=None, gate_c1_lwir=None):
        """Return full fusion outputs plus paired latent-perturbation outputs.

        This keeps the detector loss path from recomputing the shared FPN when
        both full and VIS/LWIR-masked expert outputs are needed.
        """
        p_vis, p_lwir, p_vis_raw, p_lwir_raw = self._compute_latents(
            x_vis, x_lwir, x_vis_aux=x_vis_aux, return_raw=True)
        gate_p_vis, gate_p_lwir = self._select_gate_latents(
            p_vis, p_lwir, p_vis_raw, p_lwir_raw)
        if self.gate_input_adapter is not None:
            if gate_p_vis is None:
                gate_p_vis = p_vis
            if gate_p_lwir is None:
                gate_p_lwir = p_lwir
            gate_p_vis, gate_p_lwir = self._apply_gate_input_adapter(
                gate_p_vis, gate_p_lwir, gate_c1_vis, gate_c1_lwir)
        normal = self._forward_from_latents(
            p_vis, p_lwir, gate_p_vis=gate_p_vis, gate_p_lwir=gate_p_lwir)
        perturbed = self._paired_perturbed_from_latents(p_vis, p_lwir)
        return normal, perturbed
