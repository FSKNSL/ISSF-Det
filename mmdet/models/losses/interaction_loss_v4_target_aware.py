"""Expert interaction loss with GT-region supervision.

The class keeps :class:`InteractionLossV4Soft` feature/gate diagnostics and
adds a weak target-region version of the same four-expert interaction loss.
The target-region terms reuse the global uni/syn/red formulas, with GT-region
weighted pooling replacing global average pooling.
"""

from typing import Tuple

import torch
import torch.nn.functional as F

from mmdet.registry import MODELS
from .interaction_loss_v4_soft import InteractionLossV4Soft


@MODELS.register_module()
class InteractionLossV4TargetAware(InteractionLossV4Soft):
    """Target-region interaction supervision for spatial experts."""

    accepts_expert_weights = True
    accepts_region_masks = True
    _EXPERT_NAMES = ('uni_vis', 'uni_lwir', 'syn', 'red')

    def __init__(self,
                 lambda_region_interaction: float = 0.0,
                 lambda_region_uni_active: float = 1.0,
                 lambda_region_uni_stable: float = 1.0,
                 lambda_region_syn: float = 1.0,
                 lambda_region_red: float = 1.0,
                 region_interaction_feat_levels: Tuple[int, ...] = None,
                 region_syn_loss_type: str = 'cos_margin',
                 region_syn_dir_margin: float = 0.35,
                 region_syn_raw_margin: float = 0.10,
                 region_syn_residual_margin: float = 0.10,
                 region_syn_pair_dir_weight: float = 0.5,
                 region_syn_pair_raw_weight: float = 0.25,
                 region_syn_residual_weight: float = 1.0,
                 lambda_region_syn_residual: float = 1.0,
                 region_syn_target_margin: float = 0.05,
                 region_syn_gap_margin: float = 0.05,
                 region_syn_target_weight: float = 0.5,
                 region_syn_gap_weight: float = 1.0,
                 region_syn_full_margin: float = 0.02,
                 region_syn_delta_margin: float = 0.05,
                 region_syn_full_weight: float = 0.5,
                 region_syn_delta_weight: float = 1.0,
                 region_syn_core_kernel: int = 3,
                 region_syn_ring_kernel: int = 5,
                 region_syn_residual_branches: Tuple[str, ...] = (
                     'vismask', 'lwirmask'),
                 **kwargs) -> None:
        super().__init__(**kwargs)
        if region_syn_loss_type not in (
                'cos_margin', 'anchor_residual', 'target_residual_only',
                'target_contrast_gap', 'target_delta_contrast',
                'target_local_delta_contrast', 'target_local_delta_gap'):
            raise ValueError(
                'region_syn_loss_type must be "cos_margin", '
                '"anchor_residual", "target_residual_only", or '
                '"target_contrast_gap", "target_delta_contrast", or '
                '"target_local_delta_contrast", or '
                '"target_local_delta_gap", '
                f'got {region_syn_loss_type!r}')
        if region_syn_core_kernel < 1 or region_syn_core_kernel % 2 == 0:
            raise ValueError(
                'region_syn_core_kernel must be a positive odd integer, got '
                f'{region_syn_core_kernel}')
        if region_syn_ring_kernel < 1 or region_syn_ring_kernel % 2 == 0:
            raise ValueError(
                'region_syn_ring_kernel must be a positive odd integer, got '
                f'{region_syn_ring_kernel}')
        region_syn_residual_branches = tuple(region_syn_residual_branches)
        valid_residual_branches = ('vismask', 'lwirmask')
        if not region_syn_residual_branches or any(
                branch not in valid_residual_branches
                for branch in region_syn_residual_branches):
            raise ValueError(
                'region_syn_residual_branches must be a non-empty tuple '
                f'using {valid_residual_branches}, got '
                f'{region_syn_residual_branches!r}')
        self.lambda_region_interaction = float(lambda_region_interaction)
        self.lambda_region_uni_active = float(lambda_region_uni_active)
        self.lambda_region_uni_stable = float(lambda_region_uni_stable)
        self.lambda_region_syn = float(lambda_region_syn)
        self.lambda_region_red = float(lambda_region_red)
        self.region_interaction_feat_levels = (
            tuple(region_interaction_feat_levels)
            if region_interaction_feat_levels is not None
            else self.response_delta_feat_levels)
        self.region_syn_loss_type = region_syn_loss_type
        self.region_syn_dir_margin = float(region_syn_dir_margin)
        self.region_syn_raw_margin = float(region_syn_raw_margin)
        self.region_syn_residual_margin = float(region_syn_residual_margin)
        self.region_syn_pair_dir_weight = float(region_syn_pair_dir_weight)
        self.region_syn_pair_raw_weight = float(region_syn_pair_raw_weight)
        self.region_syn_residual_weight = float(region_syn_residual_weight)
        self.lambda_region_syn_residual = float(lambda_region_syn_residual)
        self.region_syn_target_margin = float(region_syn_target_margin)
        self.region_syn_gap_margin = float(region_syn_gap_margin)
        self.region_syn_target_weight = float(region_syn_target_weight)
        self.region_syn_gap_weight = float(region_syn_gap_weight)
        self.region_syn_full_margin = float(region_syn_full_margin)
        self.region_syn_delta_margin = float(region_syn_delta_margin)
        self.region_syn_full_weight = float(region_syn_full_weight)
        self.region_syn_delta_weight = float(region_syn_delta_weight)
        self.region_syn_core_kernel = int(region_syn_core_kernel)
        self.region_syn_ring_kernel = int(region_syn_ring_kernel)
        self.region_syn_residual_branches = region_syn_residual_branches

    def _check_region_masks(self, target_region_masks):
        if not isinstance(target_region_masks, (tuple, list)):
            raise TypeError(
                'InteractionLossV4TargetAware requires target_region_masks '
                f'as a tuple/list, got {type(target_region_masks)}')
        if len(target_region_masks) <= max(
                self.region_interaction_feat_levels):
            raise ValueError(
                'Expected masks for region_interaction_feat_levels '
                f'{self.region_interaction_feat_levels}, '
                f'got {len(target_region_masks)} levels')
        for level in self.region_interaction_feat_levels:
            mask = target_region_masks[level]
            if not isinstance(mask, torch.Tensor) or mask.dim() != 4:
                raise TypeError(
                    f'target_region_masks[{level}] must be [B,1,H,W], got '
                    f'{type(mask)} with shape {getattr(mask, "shape", None)}')
            if mask.shape[1] != 1:
                raise ValueError(
                    f'target_region_masks[{level}] channel must be 1, got '
                    f'{mask.shape[1]}')

    def _weighted_region_pool_raw(self, feat, spatial_weight, region_mask):
        if feat.shape[0] != spatial_weight.shape[0]:
            raise ValueError(
                f'Batch mismatch: feat B={feat.shape[0]}, '
                f'weight B={spatial_weight.shape[0]}')
        if feat.shape[-2:] != spatial_weight.shape[-2:]:
            raise ValueError(
                f'Spatial mismatch: feat {tuple(feat.shape[-2:])}, '
                f'weight {tuple(spatial_weight.shape[-2:])}')
        if feat.shape[-2:] != region_mask.shape[-2:]:
            raise ValueError(
                f'Mask mismatch: feat {tuple(feat.shape[-2:])}, '
                f'mask {tuple(region_mask.shape[-2:])}')

        weight = spatial_weight.detach() * region_mask.detach()
        denom = weight.sum(dim=(2, 3))
        valid = denom.squeeze(1) > self.eps
        pooled = (feat * weight).sum(dim=(2, 3)) / denom.clamp(min=self.eps)
        return pooled, valid

    def _weighted_region_pool(self, feat, spatial_weight, region_mask):
        pooled, valid = self._weighted_region_pool_raw(
            feat, spatial_weight, region_mask)
        return self._standardize_vector(pooled), valid

    def _response_delta_region(self, normal_outputs, perturbed_outputs,
                               expert_weights, target_region_masks,
                               expert_idx: int):
        total = normal_outputs[0][0].new_zeros(())
        valid_levels = normal_outputs[0][0].new_zeros(())

        for level in self.region_interaction_feat_levels:
            gate = expert_weights[level][:, expert_idx:expert_idx + 1]
            mask = target_region_masks[level]
            normal, valid = self._weighted_region_pool(
                normal_outputs[expert_idx][level], gate, mask)
            perturbed, _ = self._weighted_region_pool(
                perturbed_outputs[expert_idx][level], gate, mask)
            if not torch.any(valid):
                continue
            delta = (normal - perturbed).abs().mean(dim=1)
            total = total + delta[valid].mean()
            valid_levels = valid_levels + 1.0

        if valid_levels.item() <= 0:
            return total, valid_levels
        return total / valid_levels.clamp(min=1.0), valid_levels

    def _region_delta_logs(self, normal_outputs, perturbed_outputs,
                           expert_weights, target_region_masks, suffix: str):
        logs = {}
        for expert_idx, name in enumerate(self._EXPERT_NAMES):
            delta, _ = self._response_delta_region(
                normal_outputs, perturbed_outputs, expert_weights,
                target_region_masks, expert_idx=expert_idx)
            logs[f'{name}_region_delta_{suffix}'] = delta.detach()
        return logs

    def _region_pool_expert(self, outputs, expert_weights, target_region_masks,
                            expert_idx: int, level: int):
        gate = expert_weights[level][:, expert_idx:expert_idx + 1]
        mask = target_region_masks[level]
        return self._weighted_region_pool(outputs[expert_idx][level], gate, mask)

    def _region_pool_expert_raw(self, outputs, expert_weights,
                                target_region_masks, expert_idx: int,
                                level: int):
        gate = expert_weights[level][:, expert_idx:expert_idx + 1]
        mask = target_region_masks[level]
        return self._weighted_region_pool_raw(
            outputs[expert_idx][level], gate, mask)

    def _region_stable_loss(self, normal_vec, perturbed_vec, valid):
        if not torch.any(valid):
            return normal_vec.new_zeros(())
        normal = F.normalize(normal_vec[valid], p=2, dim=1, eps=self.eps)
        perturbed = F.normalize(
            perturbed_vec[valid], p=2, dim=1, eps=self.eps)
        return (1 - self.cos_sim(normal, perturbed)).mean()

    def _region_syn_loss(self, normal_vec, perturbed_vec, valid):
        if not torch.any(valid):
            return normal_vec.new_zeros(())
        normal = F.normalize(normal_vec[valid], p=2, dim=1, eps=self.eps)
        perturbed = F.normalize(
            perturbed_vec[valid], p=2, dim=1, eps=self.eps)
        syn_cos = self.cos_sim(normal, perturbed)
        return torch.clamp(syn_cos - self.max_sim_syn, min=0).mean()

    def _zero_region_syn_anchor_logs(self, ref):
        return {
            'region_syn_dir_dist_vis': ref.new_zeros(()),
            'region_syn_dir_dist_lwir': ref.new_zeros(()),
            'region_syn_raw_dist_vis': ref.new_zeros(()),
            'region_syn_raw_dist_lwir': ref.new_zeros(()),
            'region_syn_residual_raw_dist': ref.new_zeros(()),
            'region_syn_pair_dir_term': ref.new_zeros(()),
            'region_syn_pair_raw_term': ref.new_zeros(()),
            'region_syn_residual_raw_term': ref.new_zeros(()),
        }

    def _zero_region_syn_target_residual_logs(self, ref):
        return {
            'region_syn_residual_term': ref.new_zeros(()),
            'region_syn_residual_cos_vismask': ref.new_zeros(()),
            'region_syn_residual_cos_lwirmask': ref.new_zeros(()),
            'region_syn_residual_norm_vismask': ref.new_zeros(()),
            'region_syn_residual_norm_lwirmask': ref.new_zeros(()),
        }

    def _zero_region_syn_target_contrast_gap_logs(self, ref):
        return {
            'region_syn_score_full': ref.new_zeros(()),
            'region_syn_score_vismask': ref.new_zeros(()),
            'region_syn_score_lwirmask': ref.new_zeros(()),
            'region_syn_gap_vismask': ref.new_zeros(()),
            'region_syn_gap_lwirmask': ref.new_zeros(()),
            'region_syn_target_contrast': ref.new_zeros(()),
        }

    def _zero_region_syn_target_delta_contrast_logs(self, ref):
        return {
            'region_syn_full_contrast': ref.new_zeros(()),
            'region_syn_delta_contrast_vismask': ref.new_zeros(()),
            'region_syn_delta_contrast_lwirmask': ref.new_zeros(()),
            'region_syn_full_term': ref.new_zeros(()),
            'region_syn_delta_term': ref.new_zeros(()),
        }

    def _zero_region_syn_target_local_delta_contrast_logs(self, ref):
        return {
            'region_syn_local_full_contrast': ref.new_zeros(()),
            'region_syn_local_delta_contrast_vismask': ref.new_zeros(()),
            'region_syn_local_delta_contrast_lwirmask': ref.new_zeros(()),
            'region_syn_local_full_term': ref.new_zeros(()),
            'region_syn_local_delta_term': ref.new_zeros(()),
        }

    def _zero_region_syn_target_local_delta_gap_logs(self, ref):
        return {
            'region_syn_local_gap_score_full': ref.new_zeros(()),
            'region_syn_local_gap_score_vismask': ref.new_zeros(()),
            'region_syn_local_gap_score_lwirmask': ref.new_zeros(()),
            'region_syn_local_gap_vismask': ref.new_zeros(()),
            'region_syn_local_gap_lwirmask': ref.new_zeros(()),
            'region_syn_local_delta_gap_vismask': ref.new_zeros(()),
            'region_syn_local_delta_gap_lwirmask': ref.new_zeros(()),
            'region_syn_local_full_term': ref.new_zeros(()),
            'region_syn_local_gap_term': ref.new_zeros(()),
            'region_syn_local_delta_term': ref.new_zeros(()),
        }

    def _rms_distance(self, lhs, rhs):
        return (lhs - rhs).pow(2).mean(dim=1).clamp(min=self.eps).sqrt()

    def _region_syn_anchor_residual_loss(self, full_raw, no_vis_raw,
                                         no_lwir_raw, valid):
        if not torch.any(valid):
            zero = full_raw.new_zeros(())
            return zero, self._zero_region_syn_anchor_logs(full_raw)

        full_anchor = full_raw[valid].detach()
        no_vis = no_vis_raw[valid]
        no_lwir = no_lwir_raw[valid]

        full_dir = F.normalize(full_anchor, p=2, dim=1, eps=self.eps)
        no_vis_dir = F.normalize(no_vis, p=2, dim=1, eps=self.eps)
        no_lwir_dir = F.normalize(no_lwir, p=2, dim=1, eps=self.eps)
        dir_dist_vis = torch.linalg.vector_norm(
            full_dir - no_vis_dir, ord=2, dim=1)
        dir_dist_lwir = torch.linalg.vector_norm(
            full_dir - no_lwir_dir, ord=2, dim=1)
        dir_margin = full_raw.new_tensor(self.region_syn_dir_margin)
        pair_dir_loss = 0.5 * (
            torch.relu(dir_margin - dir_dist_vis).mean()
            + torch.relu(dir_margin - dir_dist_lwir).mean())

        raw_dist_vis = self._rms_distance(full_anchor, no_vis)
        raw_dist_lwir = self._rms_distance(full_anchor, no_lwir)
        raw_margin = full_raw.new_tensor(self.region_syn_raw_margin)
        pair_raw_loss = 0.5 * (
            torch.relu(raw_margin - raw_dist_vis).mean()
            + torch.relu(raw_margin - raw_dist_lwir).mean())

        residual = full_anchor - 0.5 * (no_vis + no_lwir)
        residual_dist = residual.pow(2).mean(dim=1).clamp(
            min=self.eps).sqrt()
        residual_margin = full_raw.new_tensor(
            self.region_syn_residual_margin)
        residual_loss = torch.relu(
            residual_margin - residual_dist).mean()

        loss = (
            full_raw.new_tensor(self.region_syn_pair_dir_weight)
            * pair_dir_loss
            + full_raw.new_tensor(self.region_syn_pair_raw_weight)
            * pair_raw_loss
            + full_raw.new_tensor(self.region_syn_residual_weight)
            * residual_loss)
        logs = {
            'region_syn_dir_dist_vis': dir_dist_vis.mean().detach(),
            'region_syn_dir_dist_lwir': dir_dist_lwir.mean().detach(),
            'region_syn_raw_dist_vis': raw_dist_vis.mean().detach(),
            'region_syn_raw_dist_lwir': raw_dist_lwir.mean().detach(),
            'region_syn_residual_raw_dist': residual_dist.mean().detach(),
            'region_syn_pair_dir_term': pair_dir_loss.detach(),
            'region_syn_pair_raw_term': pair_raw_loss.detach(),
            'region_syn_residual_raw_term': residual_loss.detach(),
        }
        return loss, logs

    def _region_syn_target_residual_loss(self, full_raw, no_vis_raw,
                                         no_lwir_raw, valid):
        if not torch.any(valid):
            zero = full_raw.new_zeros(())
            return zero, self._zero_region_syn_target_residual_logs(full_raw)

        pred = full_raw[valid]
        residual_vis = (full_raw[valid] - no_vis_raw[valid]).detach()
        residual_lwir = (full_raw[valid] - no_lwir_raw[valid]).detach()

        pred_norm = F.normalize(pred, p=2, dim=1, eps=self.eps)
        residual_vis_normed = F.normalize(
            residual_vis, p=2, dim=1, eps=self.eps)
        residual_lwir_normed = F.normalize(
            residual_lwir, p=2, dim=1, eps=self.eps)
        cos_vis = self.cos_sim(pred_norm, residual_vis_normed)
        cos_lwir = self.cos_sim(pred_norm, residual_lwir_normed)
        residual_terms = []
        if 'vismask' in self.region_syn_residual_branches:
            residual_terms.append((1 - cos_vis).mean())
        if 'lwirmask' in self.region_syn_residual_branches:
            residual_terms.append((1 - cos_lwir).mean())
        residual_loss = torch.stack(residual_terms).mean()
        loss = residual_loss * full_raw.new_tensor(
            self.lambda_region_syn_residual)
        logs = {
            'region_syn_residual_term': residual_loss.detach(),
            'region_syn_residual_cos_vismask': cos_vis.mean().detach(),
            'region_syn_residual_cos_lwirmask': cos_lwir.mean().detach(),
            'region_syn_residual_norm_vismask': self._rms_distance(
                full_raw[valid], no_vis_raw[valid]).mean().detach(),
            'region_syn_residual_norm_lwirmask': self._rms_distance(
                full_raw[valid], no_lwir_raw[valid]).mean().detach(),
        }
        return loss, logs

    def _region_contrast_score(self, feat, target_mask, context_mask):
        if feat.shape[-2:] != target_mask.shape[-2:]:
            raise ValueError(
                f'Mask mismatch: feat {tuple(feat.shape[-2:])}, '
                f'target {tuple(target_mask.shape[-2:])}')
        if feat.shape[-2:] != context_mask.shape[-2:]:
            raise ValueError(
                f'Mask mismatch: feat {tuple(feat.shape[-2:])}, '
                f'context {tuple(context_mask.shape[-2:])}')

        target_mask = target_mask.detach().clamp(min=0.0, max=1.0)
        context_mask = context_mask.detach().clamp(min=0.0, max=1.0)
        activation = feat.pow(2).mean(dim=1, keepdim=True).clamp(
            min=self.eps).sqrt()
        target_denom = target_mask.sum(dim=(2, 3))
        context_denom = context_mask.sum(dim=(2, 3))
        valid = (
            (target_denom.squeeze(1) > self.eps)
            & (context_denom.squeeze(1) > self.eps))

        target = (
            (activation * target_mask).sum(dim=(2, 3))
            / target_denom.clamp(min=self.eps))
        context = (
            (activation * context_mask).sum(dim=(2, 3))
            / context_denom.clamp(min=self.eps))
        score = (
            torch.log(target.squeeze(1).clamp(min=self.eps))
            - torch.log(context.squeeze(1).clamp(min=self.eps)))
        return score, valid

    def _region_target_contrast_score(self, feat, region_mask):
        if feat.shape[-2:] != region_mask.shape[-2:]:
            raise ValueError(
                f'Mask mismatch: feat {tuple(feat.shape[-2:])}, '
                f'mask {tuple(region_mask.shape[-2:])}')
        mask = region_mask.detach().clamp(min=0.0, max=1.0)
        return self._region_contrast_score(feat, mask, 1.0 - mask)

    @staticmethod
    def _morph_max_pool(mask, kernel: int):
        if kernel <= 1:
            return mask
        return F.max_pool2d(
            mask, kernel_size=kernel, stride=1, padding=kernel // 2)

    def _region_core_ring_masks(self, region_mask):
        mask = region_mask.detach().clamp(min=0.0, max=1.0)
        inv_mask = 1.0 - mask
        eroded = 1.0 - self._morph_max_pool(
            inv_mask, self.region_syn_core_kernel)
        core = eroded.clamp(min=0.0, max=1.0)
        core_valid = core.sum(dim=(2, 3), keepdim=True) > self.eps
        core = torch.where(core_valid, core, mask)

        dilated = self._morph_max_pool(
            mask, self.region_syn_ring_kernel)
        ring = (dilated - mask).clamp(min=0.0, max=1.0)
        ring_valid = ring.sum(dim=(2, 3), keepdim=True) > self.eps
        ring = torch.where(ring_valid, ring, inv_mask)
        return core, ring

    def _region_local_contrast_score(self, feat, region_mask):
        if feat.shape[-2:] != region_mask.shape[-2:]:
            raise ValueError(
                f'Mask mismatch: feat {tuple(feat.shape[-2:])}, '
                f'mask {tuple(region_mask.shape[-2:])}')
        core, ring = self._region_core_ring_masks(region_mask)
        return self._region_contrast_score(feat, core, ring)

    def _region_syn_target_contrast_gap_loss(self, full_feat, vis_feat,
                                             lwir_feat, region_mask):
        full_score, valid = self._region_target_contrast_score(
            full_feat, region_mask)
        if not torch.any(valid):
            zero = full_feat.new_zeros(())
            return zero, self._zero_region_syn_target_contrast_gap_logs(
                full_feat)

        with torch.no_grad():
            vis_score, valid_vis = self._region_target_contrast_score(
                vis_feat, region_mask)
            lwir_score, valid_lwir = self._region_target_contrast_score(
                lwir_feat, region_mask)

        valid = valid & valid_vis & valid_lwir
        if not torch.any(valid):
            zero = full_feat.new_zeros(())
            return zero, self._zero_region_syn_target_contrast_gap_logs(
                full_feat)

        full = full_score[valid]
        vis = vis_score[valid].detach()
        lwir = lwir_score[valid].detach()
        target_margin = full.new_tensor(self.region_syn_target_margin)
        gap_margin = full.new_tensor(self.region_syn_gap_margin)
        target_term = torch.relu(target_margin - full).mean()
        gap_vis = full - vis
        gap_lwir = full - lwir
        gap_term = 0.5 * (
            torch.relu(gap_margin - gap_vis).mean()
            + torch.relu(gap_margin - gap_lwir).mean())
        loss = (
            full.new_tensor(self.region_syn_target_weight) * target_term
            + full.new_tensor(self.region_syn_gap_weight) * gap_term)
        logs = {
            'region_syn_score_full': full.mean().detach(),
            'region_syn_score_vismask': vis.mean().detach(),
            'region_syn_score_lwirmask': lwir.mean().detach(),
            'region_syn_gap_vismask': gap_vis.mean().detach(),
            'region_syn_gap_lwirmask': gap_lwir.mean().detach(),
            'region_syn_target_contrast': target_term.detach(),
        }
        return loss, logs

    def _region_syn_target_delta_contrast_loss(self, full_feat, vis_feat,
                                               lwir_feat, region_mask):
        full_score, valid = self._region_target_contrast_score(
            full_feat, region_mask)
        if not torch.any(valid):
            zero = full_feat.new_zeros(())
            return zero, self._zero_region_syn_target_delta_contrast_logs(
                full_feat)

        delta_vis_score, valid_vis = self._region_target_contrast_score(
            full_feat - vis_feat.detach(), region_mask)
        delta_lwir_score, valid_lwir = self._region_target_contrast_score(
            full_feat - lwir_feat.detach(), region_mask)

        valid = valid & valid_vis & valid_lwir
        if not torch.any(valid):
            zero = full_feat.new_zeros(())
            return zero, self._zero_region_syn_target_delta_contrast_logs(
                full_feat)

        full = full_score[valid]
        delta_vis = delta_vis_score[valid]
        delta_lwir = delta_lwir_score[valid]
        full_margin = full.new_tensor(self.region_syn_full_margin)
        delta_margin = full.new_tensor(self.region_syn_delta_margin)
        full_term = torch.relu(full_margin - full).mean()
        delta_term = 0.5 * (
            torch.relu(delta_margin - delta_vis).mean()
            + torch.relu(delta_margin - delta_lwir).mean())
        loss = (
            full.new_tensor(self.region_syn_full_weight) * full_term
            + full.new_tensor(self.region_syn_delta_weight) * delta_term)
        logs = {
            'region_syn_full_contrast': full.mean().detach(),
            'region_syn_delta_contrast_vismask': delta_vis.mean().detach(),
            'region_syn_delta_contrast_lwirmask': delta_lwir.mean().detach(),
            'region_syn_full_term': full_term.detach(),
            'region_syn_delta_term': delta_term.detach(),
        }
        return loss, logs

    def _region_syn_target_local_delta_contrast_loss(self, full_feat,
                                                     vis_feat, lwir_feat,
                                                     region_mask):
        full_score, valid = self._region_local_contrast_score(
            full_feat, region_mask)
        if not torch.any(valid):
            zero = full_feat.new_zeros(())
            return zero, (
                self._zero_region_syn_target_local_delta_contrast_logs(
                    full_feat))

        delta_vis_score, valid_vis = self._region_local_contrast_score(
            full_feat - vis_feat.detach(), region_mask)
        delta_lwir_score, valid_lwir = self._region_local_contrast_score(
            full_feat - lwir_feat.detach(), region_mask)

        valid = valid & valid_vis & valid_lwir
        if not torch.any(valid):
            zero = full_feat.new_zeros(())
            return zero, (
                self._zero_region_syn_target_local_delta_contrast_logs(
                    full_feat))

        full = full_score[valid]
        delta_vis = delta_vis_score[valid]
        delta_lwir = delta_lwir_score[valid]
        full_margin = full.new_tensor(self.region_syn_full_margin)
        delta_margin = full.new_tensor(self.region_syn_delta_margin)
        full_term = torch.relu(full_margin - full).mean()
        delta_term = 0.5 * (
            torch.relu(delta_margin - delta_vis).mean()
            + torch.relu(delta_margin - delta_lwir).mean())
        loss = (
            full.new_tensor(self.region_syn_full_weight) * full_term
            + full.new_tensor(self.region_syn_delta_weight) * delta_term)
        logs = {
            'region_syn_local_full_contrast': full.mean().detach(),
            'region_syn_local_delta_contrast_vismask':
            delta_vis.mean().detach(),
            'region_syn_local_delta_contrast_lwirmask':
            delta_lwir.mean().detach(),
            'region_syn_local_full_term': full_term.detach(),
            'region_syn_local_delta_term': delta_term.detach(),
        }
        return loss, logs

    def _region_syn_target_local_delta_gap_loss(self, full_feat, vis_feat,
                                                lwir_feat, region_mask):
        full_score, valid = self._region_local_contrast_score(
            full_feat, region_mask)
        if not torch.any(valid):
            zero = full_feat.new_zeros(())
            return zero, self._zero_region_syn_target_local_delta_gap_logs(
                full_feat)

        with torch.no_grad():
            vis_score, valid_vis = self._region_local_contrast_score(
                vis_feat, region_mask)
            lwir_score, valid_lwir = self._region_local_contrast_score(
                lwir_feat, region_mask)
        delta_vis_score, valid_delta_vis = self._region_local_contrast_score(
            full_feat - vis_feat.detach(), region_mask)
        delta_lwir_score, valid_delta_lwir = self._region_local_contrast_score(
            full_feat - lwir_feat.detach(), region_mask)

        valid = valid & valid_vis & valid_lwir
        valid = valid & valid_delta_vis & valid_delta_lwir
        if not torch.any(valid):
            zero = full_feat.new_zeros(())
            return zero, self._zero_region_syn_target_local_delta_gap_logs(
                full_feat)

        full = full_score[valid]
        vis = vis_score[valid].detach()
        lwir = lwir_score[valid].detach()
        delta_vis = delta_vis_score[valid]
        delta_lwir = delta_lwir_score[valid]

        full_margin = full.new_tensor(self.region_syn_full_margin)
        gap_margin = full.new_tensor(self.region_syn_gap_margin)
        delta_margin = full.new_tensor(self.region_syn_delta_margin)

        full_term = torch.relu(full_margin - full).mean()
        gap_vis = full - vis
        gap_lwir = full - lwir
        gap_term = 0.5 * (
            torch.relu(gap_margin - gap_vis).mean()
            + torch.relu(gap_margin - gap_lwir).mean())
        delta_term = 0.5 * (
            torch.relu(delta_margin - delta_vis).mean()
            + torch.relu(delta_margin - delta_lwir).mean())
        loss = (
            full.new_tensor(self.region_syn_full_weight) * full_term
            + full.new_tensor(self.region_syn_gap_weight) * gap_term
            + full.new_tensor(self.region_syn_delta_weight) * delta_term)
        logs = {
            'region_syn_local_gap_score_full': full.mean().detach(),
            'region_syn_local_gap_score_vismask': vis.mean().detach(),
            'region_syn_local_gap_score_lwirmask': lwir.mean().detach(),
            'region_syn_local_gap_vismask': gap_vis.mean().detach(),
            'region_syn_local_gap_lwirmask': gap_lwir.mean().detach(),
            'region_syn_local_delta_gap_vismask': delta_vis.mean().detach(),
            'region_syn_local_delta_gap_lwirmask': delta_lwir.mean().detach(),
            'region_syn_local_full_term': full_term.detach(),
            'region_syn_local_gap_term': gap_term.detach(),
            'region_syn_local_delta_term': delta_term.detach(),
        }
        return loss, logs

    def _region_triplet_loss(self, anchor, positive, negative, valid):
        if not torch.any(valid):
            return anchor.new_zeros(())
        return self.triplet_loss(anchor[valid], positive[valid],
                                 negative[valid])

    def _region_weighted_total(self, uni_active, uni_stable, syn, red,
                               uni_vis=None, uni_lwir=None):
        if self.loss_aggregation == 'iss_moe_role_avg':
            if uni_vis is None or uni_lwir is None:
                raise ValueError(
                    'iss_moe_role_avg requires paired perturbation so '
                    'region_uni_vis and region_uni_lwir losses are both '
                    'available.')
            return 0.25 * (uni_vis + uni_lwir + syn + red)
        return (
            self.lambda_region_uni_active * uni_active
            + self.lambda_region_uni_stable * uni_stable
            + self.lambda_region_syn * syn
            + self.lambda_region_red * red)

    def _paired_region_interaction_terms(self, normal_outputs, vis_outputs,
                                         lwir_outputs, expert_weights,
                                         target_region_masks):
        self._check_spatial_weights(expert_weights)
        self._check_region_masks(target_region_masks)

        ref = normal_outputs[0][0]
        total_uni_vis = ref.new_zeros(())
        total_uni_lwir = ref.new_zeros(())
        total_stable = ref.new_zeros(())
        total_syn = ref.new_zeros(())
        total_red = ref.new_zeros(())
        valid_levels = ref.new_zeros(())
        syn_anchor_logs = {}
        if self.region_syn_loss_type == 'anchor_residual':
            syn_anchor_logs = self._zero_region_syn_anchor_logs(ref)
        elif self.region_syn_loss_type == 'target_residual_only':
            syn_anchor_logs = self._zero_region_syn_target_residual_logs(ref)
        elif self.region_syn_loss_type == 'target_contrast_gap':
            syn_anchor_logs = (
                self._zero_region_syn_target_contrast_gap_logs(ref))
        elif self.region_syn_loss_type == 'target_delta_contrast':
            syn_anchor_logs = (
                self._zero_region_syn_target_delta_contrast_logs(ref))
        elif self.region_syn_loss_type == 'target_local_delta_contrast':
            syn_anchor_logs = (
                self._zero_region_syn_target_local_delta_contrast_logs(ref))
        elif self.region_syn_loss_type == 'target_local_delta_gap':
            syn_anchor_logs = (
                self._zero_region_syn_target_local_delta_gap_logs(ref))

        for level in self.region_interaction_feat_levels:
            uni_vis_anchor, valid = self._region_pool_expert(
                normal_outputs, expert_weights, target_region_masks, 0, level)
            if not torch.any(valid):
                continue
            uni_vis_pos, _ = self._region_pool_expert(
                lwir_outputs, expert_weights, target_region_masks, 0, level)
            uni_vis_neg, _ = self._region_pool_expert(
                vis_outputs, expert_weights, target_region_masks, 0, level)
            uni_vis_triplet = self._region_triplet_loss(
                uni_vis_anchor, uni_vis_pos, uni_vis_neg, valid)
            uni_vis_stable = self._region_stable_loss(
                uni_vis_anchor, uni_vis_pos, valid)

            uni_lwir_anchor, valid_lwir = self._region_pool_expert(
                normal_outputs, expert_weights, target_region_masks, 1, level)
            uni_lwir_pos, _ = self._region_pool_expert(
                vis_outputs, expert_weights, target_region_masks, 1, level)
            uni_lwir_neg, _ = self._region_pool_expert(
                lwir_outputs, expert_weights, target_region_masks, 1, level)
            uni_lwir_triplet = self._region_triplet_loss(
                uni_lwir_anchor, uni_lwir_pos, uni_lwir_neg, valid_lwir)
            uni_lwir_stable = self._region_stable_loss(
                uni_lwir_anchor, uni_lwir_pos, valid_lwir)

            if self.region_syn_loss_type == 'anchor_residual':
                syn_anchor_raw, valid_syn = self._region_pool_expert_raw(
                    normal_outputs, expert_weights, target_region_masks, 2,
                    level)
                syn_vis_raw, _ = self._region_pool_expert_raw(
                    vis_outputs, expert_weights, target_region_masks, 2,
                    level)
                syn_lwir_raw, _ = self._region_pool_expert_raw(
                    lwir_outputs, expert_weights, target_region_masks, 2,
                    level)
                syn_loss, syn_logs = (
                    self._region_syn_anchor_residual_loss(
                        syn_anchor_raw, syn_vis_raw, syn_lwir_raw,
                        valid_syn))
                for key, value in syn_logs.items():
                    syn_anchor_logs[key] = syn_anchor_logs[key] + value
            elif self.region_syn_loss_type == 'target_residual_only':
                syn_anchor_raw, valid_syn = self._region_pool_expert_raw(
                    normal_outputs, expert_weights, target_region_masks, 2,
                    level)
                syn_vis_raw, _ = self._region_pool_expert_raw(
                    vis_outputs, expert_weights, target_region_masks, 2,
                    level)
                syn_lwir_raw, _ = self._region_pool_expert_raw(
                    lwir_outputs, expert_weights, target_region_masks, 2,
                    level)
                syn_loss, syn_logs = (
                    self._region_syn_target_residual_loss(
                        syn_anchor_raw, syn_vis_raw, syn_lwir_raw,
                        valid_syn))
                for key, value in syn_logs.items():
                    syn_anchor_logs[key] = syn_anchor_logs[key] + value
            elif self.region_syn_loss_type == 'target_contrast_gap':
                syn_loss, syn_logs = (
                    self._region_syn_target_contrast_gap_loss(
                        normal_outputs[2][level], vis_outputs[2][level],
                        lwir_outputs[2][level],
                        target_region_masks[level]))
                for key, value in syn_logs.items():
                    syn_anchor_logs[key] = syn_anchor_logs[key] + value
            elif self.region_syn_loss_type == 'target_delta_contrast':
                syn_loss, syn_logs = (
                    self._region_syn_target_delta_contrast_loss(
                        normal_outputs[2][level], vis_outputs[2][level],
                        lwir_outputs[2][level],
                        target_region_masks[level]))
                for key, value in syn_logs.items():
                    syn_anchor_logs[key] = syn_anchor_logs[key] + value
            elif self.region_syn_loss_type == 'target_local_delta_contrast':
                syn_loss, syn_logs = (
                    self._region_syn_target_local_delta_contrast_loss(
                        normal_outputs[2][level], vis_outputs[2][level],
                        lwir_outputs[2][level],
                        target_region_masks[level]))
                for key, value in syn_logs.items():
                    syn_anchor_logs[key] = syn_anchor_logs[key] + value
            elif self.region_syn_loss_type == 'target_local_delta_gap':
                syn_loss, syn_logs = (
                    self._region_syn_target_local_delta_gap_loss(
                        normal_outputs[2][level], vis_outputs[2][level],
                        lwir_outputs[2][level],
                        target_region_masks[level]))
                for key, value in syn_logs.items():
                    syn_anchor_logs[key] = syn_anchor_logs[key] + value
            else:
                syn_anchor, valid_syn = self._region_pool_expert(
                    normal_outputs, expert_weights, target_region_masks, 2,
                    level)
                syn_vis, _ = self._region_pool_expert(
                    vis_outputs, expert_weights, target_region_masks, 2,
                    level)
                syn_lwir, _ = self._region_pool_expert(
                    lwir_outputs, expert_weights, target_region_masks, 2,
                    level)
                syn_loss = 0.5 * (
                    self._region_syn_loss(syn_anchor, syn_vis, valid_syn)
                    + self._region_syn_loss(syn_anchor, syn_lwir, valid_syn))

            red_anchor, valid_red = self._region_pool_expert(
                normal_outputs, expert_weights, target_region_masks, 3, level)
            red_vis, _ = self._region_pool_expert(
                vis_outputs, expert_weights, target_region_masks, 3, level)
            red_lwir, _ = self._region_pool_expert(
                lwir_outputs, expert_weights, target_region_masks, 3, level)
            red_loss = 0.5 * (
                self._region_stable_loss(red_anchor, red_vis, valid_red)
                + self._region_stable_loss(red_anchor, red_lwir, valid_red))

            total_uni_vis = total_uni_vis + uni_vis_triplet
            total_uni_lwir = total_uni_lwir + uni_lwir_triplet
            total_stable = total_stable + 0.5 * (
                uni_vis_stable + uni_lwir_stable)
            total_syn = total_syn + syn_loss
            total_red = total_red + red_loss
            valid_levels = valid_levels + 1.0

        if valid_levels.item() <= 0:
            zero = ref.new_zeros(())
            return {
                'region_uni_vis': zero,
                'region_uni_lwir': zero,
                'region_uni_active': zero,
                'region_uni_stable': zero,
                'region_syn': zero,
                'region_red': zero,
                'region_total_raw': zero,
                'region_valid': valid_levels.detach(),
                'region_interaction': zero,
                **syn_anchor_logs,
            }

        uni_vis = total_uni_vis / valid_levels.clamp(min=1.0)
        uni_lwir = total_uni_lwir / valid_levels.clamp(min=1.0)
        uni_active = 0.5 * (uni_vis + uni_lwir)
        uni_stable = total_stable / valid_levels.clamp(min=1.0)
        syn = total_syn / valid_levels.clamp(min=1.0)
        red = total_red / valid_levels.clamp(min=1.0)
        total = self._region_weighted_total(
            uni_active, uni_stable, syn, red, uni_vis=uni_vis,
            uni_lwir=uni_lwir)
        syn_anchor_logs = {
            key: value / valid_levels.clamp(min=1.0)
            for key, value in syn_anchor_logs.items()
        }

        return {
            'region_uni_vis': uni_vis,
            'region_uni_lwir': uni_lwir,
            'region_uni_active': uni_active,
            'region_uni_stable': uni_stable,
            'region_syn': syn,
            'region_red': red,
            'region_total_raw': total.detach(),
            'region_valid': valid_levels.detach(),
            'region_interaction': (
                total * total.new_tensor(self.lambda_region_interaction)),
            **syn_anchor_logs,
        }

    def _single_region_interaction_terms(self, normal_outputs,
                                         perturbed_outputs,
                                         perturbed_modality: str,
                                         expert_weights,
                                         target_region_masks):
        self._check_spatial_weights(expert_weights)
        self._check_region_masks(target_region_masks)

        if perturbed_modality == 'vis':
            active_idx, stable_idx = 0, 1
        elif perturbed_modality == 'lwir':
            active_idx, stable_idx = 1, 0
        else:
            raise ValueError(
                f"perturbed_modality must be 'vis' or 'lwir', got "
                f'{perturbed_modality!r}')

        ref = normal_outputs[0][0]
        total_active = ref.new_zeros(())
        total_stable = ref.new_zeros(())
        total_syn = ref.new_zeros(())
        total_red = ref.new_zeros(())
        valid_levels = ref.new_zeros(())

        for level in self.region_interaction_feat_levels:
            active_anchor, valid = self._region_pool_expert(
                normal_outputs, expert_weights, target_region_masks,
                active_idx, level)
            if not torch.any(valid):
                continue
            active_neg, _ = self._region_pool_expert(
                perturbed_outputs, expert_weights, target_region_masks,
                active_idx, level)
            active = self._region_triplet_loss(
                active_anchor, active_anchor.detach(), active_neg, valid)

            stable_anchor, valid_stable = self._region_pool_expert(
                normal_outputs, expert_weights, target_region_masks,
                stable_idx, level)
            stable_pair, _ = self._region_pool_expert(
                perturbed_outputs, expert_weights, target_region_masks,
                stable_idx, level)
            stable = self._region_stable_loss(
                stable_anchor, stable_pair, valid_stable)

            syn_anchor, valid_syn = self._region_pool_expert(
                normal_outputs, expert_weights, target_region_masks, 2, level)
            syn_pair, _ = self._region_pool_expert(
                perturbed_outputs, expert_weights, target_region_masks, 2,
                level)
            syn = self._region_syn_loss(syn_anchor, syn_pair, valid_syn)

            red_anchor, valid_red = self._region_pool_expert(
                normal_outputs, expert_weights, target_region_masks, 3, level)
            red_pair, _ = self._region_pool_expert(
                perturbed_outputs, expert_weights, target_region_masks, 3,
                level)
            red = self._region_stable_loss(red_anchor, red_pair, valid_red)

            total_active = total_active + active
            total_stable = total_stable + stable
            total_syn = total_syn + syn
            total_red = total_red + red
            valid_levels = valid_levels + 1.0

        if valid_levels.item() <= 0:
            zero = ref.new_zeros(())
            return {
                'region_uni_vis': zero,
                'region_uni_lwir': zero,
                'region_uni_active': zero,
                'region_uni_stable': zero,
                'region_syn': zero,
                'region_red': zero,
                'region_total_raw': zero,
                'region_valid': valid_levels.detach(),
                'region_interaction': zero,
            }

        uni_active = total_active / valid_levels.clamp(min=1.0)
        uni_stable = total_stable / valid_levels.clamp(min=1.0)
        syn = total_syn / valid_levels.clamp(min=1.0)
        red = total_red / valid_levels.clamp(min=1.0)
        total = self._region_weighted_total(
            uni_active, uni_stable, syn, red)

        return {
            'region_uni_active': uni_active,
            'region_uni_stable': uni_stable,
            'region_syn': syn,
            'region_red': red,
            'region_total_raw': total.detach(),
            'region_valid': valid_levels.detach(),
            'region_interaction': (
                total * total.new_tensor(self.lambda_region_interaction)),
        }

    def forward(self,
                normal_outputs,
                perturbed_outputs,
                perturbed_modality: str,
                expert_weights=None,
                target_region_masks=None):
        if target_region_masks is None:
            raise ValueError(
                'InteractionLossV4TargetAware requires target_region_masks '
                'from detector.loss().')

        if self._is_paired_outputs(perturbed_outputs):
            out = super().forward(
                normal_outputs,
                perturbed_outputs,
                perturbed_modality,
                expert_weights=expert_weights)
            vis_outputs, lwir_outputs = self._get_paired_outputs(
                perturbed_outputs)

            for modality, outputs in (
                    ('vis', vis_outputs), ('lwir', lwir_outputs)):
                out.update(self._region_delta_logs(
                    normal_outputs, outputs, expert_weights,
                    target_region_masks, f'{modality}mask'))

            region_terms = self._paired_region_interaction_terms(
                normal_outputs, vis_outputs, lwir_outputs, expert_weights,
                target_region_masks)
            out['total_main'] = (
                out['total_main'] + region_terms.pop('region_interaction'))
            out.update(region_terms)
            return out

        out = super().forward(
            normal_outputs,
            perturbed_outputs,
            perturbed_modality,
            expert_weights=expert_weights)

        out.update(self._region_delta_logs(
            normal_outputs, perturbed_outputs, expert_weights,
            target_region_masks, 'singlemask'))
        region_terms = self._single_region_interaction_terms(
            normal_outputs, perturbed_outputs, perturbed_modality,
            expert_weights, target_region_masks)
        out['total_main'] = (
            out['total_main'] + region_terms.pop('region_interaction'))
        out.update(region_terms)
        return out


@MODELS.register_module()
class InteractionLossV4TargetAwareSubset(InteractionLossV4TargetAware):
    """Apply the existing role loss to a configured expert subset.

    Fusion and gate tensors only contain the active experts. This adapter
    restores their canonical four-role positions for the unchanged loss
    formulas; missing roles are represented by zero tensors and must have
    zero coefficients in the experiment config.
    """

    _CANONICAL_ROLES = ('uni_vis', 'uni_lwir', 'syn', 'red')

    def __init__(self, expert_roles, **kwargs) -> None:
        self.expert_roles = tuple(expert_roles)
        if not self.expert_roles:
            raise ValueError('expert_roles must contain at least one role.')
        if len(set(self.expert_roles)) != len(self.expert_roles):
            raise ValueError(
                f'expert_roles must not contain duplicates, got '
                f'{self.expert_roles}.')
        if any(
                role not in self._CANONICAL_ROLES
                for role in self.expert_roles):
            raise ValueError(
                f'expert_roles must be selected from '
                f'{self._CANONICAL_ROLES}, got {self.expert_roles}.')
        canonical_order = tuple(
            role for role in self._CANONICAL_ROLES
            if role in self.expert_roles)
        if self.expert_roles != canonical_order:
            raise ValueError(
                'expert_roles must follow canonical order '
                f'{self._CANONICAL_ROLES}, got {self.expert_roles}.')

        super().__init__(**kwargs)
        active = set(self.expert_roles)
        self._DECORR_PAIRS = tuple(
            pair for pair in self._DECORR_PAIRS
            if self._CANONICAL_ROLES[pair[0]] in active
            and self._CANONICAL_ROLES[pair[1]] in active)
        self._RED_MONITOR_PAIRS = tuple(
            pair for pair in self._RED_MONITOR_PAIRS
            if self._CANONICAL_ROLES[pair[0]] in active
            and self._CANONICAL_ROLES[pair[1]] in active)

    def _expand_outputs(self, outputs):
        if len(outputs) != len(self.expert_roles):
            raise ValueError(
                f'Expected {len(self.expert_roles)} expert outputs for '
                f'{self.expert_roles}, got {len(outputs)}.')
        reference = outputs[0]
        zeros = tuple(torch.zeros_like(feat) for feat in reference)
        by_role = dict(zip(self.expert_roles, outputs))
        return tuple(
            by_role.get(role, zeros) for role in self._CANONICAL_ROLES)

    def _expand_perturbed_outputs(self, outputs):
        if self._is_paired_outputs(outputs):
            return {
                key: self._expand_outputs(value)
                for key, value in outputs.items()
            }
        return self._expand_outputs(outputs)

    def _expand_weights(self, weights):
        if not isinstance(weights, (tuple, list)):
            raise TypeError(
                f'expert_weights must be a tuple/list, got {type(weights)}.')
        role_to_index = {
            role: index for index, role in enumerate(self.expert_roles)
        }
        expanded = []
        for level, weight in enumerate(weights):
            if weight.shape[1] != len(self.expert_roles):
                raise ValueError(
                    f'expert_weights[{level}] expected '
                    f'{len(self.expert_roles)} channels, got '
                    f'{weight.shape[1]}.')
            # The parent region loss uses canonical role 0 to decide whether
            # a GT region is valid before evaluating every role.  When that
            # role is absent, use the sum of active softmax weights (one) as
            # a validity-only proxy; the missing feature itself stays zero
            # and its loss coefficients are zero in the subset config.
            validity_proxy = weight.sum(dim=1, keepdim=True)
            channels = []
            for role in self._CANONICAL_ROLES:
                if role in role_to_index:
                    index = role_to_index[role]
                    channels.append(weight[:, index:index + 1])
                else:
                    channels.append(validity_proxy)
            expanded.append(torch.cat(channels, dim=1))
        return tuple(expanded)

    def _decorr_terms(self, normal_outputs):
        if self._DECORR_PAIRS:
            return super()._decorr_terms(normal_outputs)
        return {'decorr': normal_outputs[0][0].new_zeros(())}

    def forward(self,
                normal_outputs,
                perturbed_outputs,
                perturbed_modality: str,
                expert_weights=None,
                target_region_masks=None):
        return super().forward(
            self._expand_outputs(normal_outputs),
            self._expand_perturbed_outputs(perturbed_outputs),
            perturbed_modality,
            expert_weights=self._expand_weights(expert_weights),
            target_region_masks=target_region_masks)


@MODELS.register_module()
class InteractionLossV4ScaleAwareTarget(InteractionLossV4TargetAware):
    """Target-aware interaction loss with scale-assigned GT masks.

    The detector reads the attributes below when rasterising GT regions.
    Existing target-aware losses remain unchanged because only this subclass
    enables scale-aware mask construction.
    """

    region_scale_aware = True

    def __init__(
            self,
            region_scale_thresholds: Tuple[float, ...] = (32.0, 96.0),
            region_scale_level_groups: Tuple[Tuple[int, ...], ...] = (
                (0, 1), (1, 2), (2, 3)),
            **kwargs) -> None:
        super().__init__(**kwargs)
        thresholds = tuple(float(value) for value in region_scale_thresholds)
        level_groups = tuple(
            tuple(int(level) for level in levels)
            for levels in region_scale_level_groups)
        if any(value <= 0.0 for value in thresholds):
            raise ValueError(
                'region_scale_thresholds must contain positive values, got '
                f'{thresholds}')
        if any(left >= right for left, right in zip(
                thresholds, thresholds[1:])):
            raise ValueError(
                'region_scale_thresholds must be strictly increasing, got '
                f'{thresholds}')
        if len(level_groups) != len(thresholds) + 1:
            raise ValueError(
                'region_scale_level_groups must contain one more group than '
                f'thresholds, got {level_groups} and {thresholds}')
        if any(not levels or any(level < 0 for level in levels)
               for levels in level_groups):
            raise ValueError(
                'region_scale_level_groups must contain non-empty groups of '
                f'non-negative levels, got {level_groups}')

        configured_levels = set(self.region_interaction_feat_levels)
        assigned_levels = {
            level for levels in level_groups for level in levels
        }
        if not assigned_levels.issubset(configured_levels):
            raise ValueError(
                'Every scale-assigned level must be enabled by '
                'region_interaction_feat_levels, got assigned '
                f'{sorted(assigned_levels)} and configured '
                f'{sorted(configured_levels)}')

        self.region_scale_thresholds = thresholds
        self.region_scale_level_groups = level_groups
