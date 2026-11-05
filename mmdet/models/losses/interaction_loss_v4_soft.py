"""Expert interaction loss with paired modality perturbations.

Gate-weighted response deltas are logged as diagnostics only; they are not
added to ``total_main``.
"""

from typing import Tuple

import torch
import torch.nn.functional as F

from mmdet.registry import MODELS
from .interaction_loss_v3 import InteractionLossV3


@MODELS.register_module()
class InteractionLossV4Soft(InteractionLossV3):
    """Paired perturbation interaction loss for spatially gated experts.

    Args:
        response_delta_feat_levels: FPN levels used by response-delta
            diagnostics.
    """

    accepts_expert_weights = True

    def __init__(self,
                 response_delta_feat_levels: Tuple[int, ...] = None,
                 loss_aggregation: str = 'weighted_sum',
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.response_delta_feat_levels = (
            tuple(response_delta_feat_levels)
            if response_delta_feat_levels is not None else self.feat_levels)
        if loss_aggregation not in ('weighted_sum', 'iss_moe_role_avg'):
            raise ValueError(
                'loss_aggregation must be "weighted_sum" or '
                '"iss_moe_role_avg", '
                f'got {loss_aggregation!r}')
        self.loss_aggregation = loss_aggregation

    def _check_spatial_weights(self, expert_weights):
        if not isinstance(expert_weights, (tuple, list)):
            raise TypeError(
                'InteractionLossV4Soft requires spatial expert_weights as a '
                f'5-tuple/list, got {type(expert_weights)}')
        if len(expert_weights) <= max(self.response_delta_feat_levels):
            raise ValueError(
                'Expected weights for response_delta_feat_levels '
                f'{self.response_delta_feat_levels}, '
                f'got {len(expert_weights)} levels')
        for level in self.response_delta_feat_levels:
            w = expert_weights[level]
            if not isinstance(w, torch.Tensor) or w.dim() != 4:
                raise TypeError(
                    f'expert_weights[{level}] must be [B,K,H,W], got '
                    f'{type(w)} with shape {getattr(w, "shape", None)}')
            if w.shape[1] < 4:
                raise ValueError(
                    f'expert_weights[{level}] needs at least 4 experts, got '
                    f'{w.shape[1]}')

    def _standardize_vector(self, vec):
        mean = vec.mean(dim=1, keepdim=True)
        std = vec.std(dim=1, keepdim=True, unbiased=False)
        return (vec - mean) / std.clamp(min=self.eps)

    @staticmethod
    def _is_paired_outputs(perturbed_outputs) -> bool:
        return isinstance(perturbed_outputs, dict)

    def _get_paired_outputs(self, perturbed_outputs):
        if not self._is_paired_outputs(perturbed_outputs):
            raise TypeError(
                'paired perturbation outputs must be a dict with vis/lwir '
                f'keys, got {type(perturbed_outputs)}')
        missing = {'vis', 'lwir'} - set(perturbed_outputs.keys())
        if missing:
            raise KeyError(f'Missing paired perturbation keys: {missing}')
        return perturbed_outputs['vis'], perturbed_outputs['lwir']

    def _weighted_pool(self, feat, spatial_weight):
        if feat.shape[0] != spatial_weight.shape[0]:
            raise ValueError(
                f'Batch mismatch: feat B={feat.shape[0]}, '
                f'weight B={spatial_weight.shape[0]}')
        if feat.shape[-2:] != spatial_weight.shape[-2:]:
            raise ValueError(
                f'Spatial mismatch: feat {tuple(feat.shape[-2:])}, '
                f'weight {tuple(spatial_weight.shape[-2:])}')

        weight = spatial_weight.detach()
        denom = weight.sum(dim=(2, 3)).clamp(min=self.eps)
        pooled = (feat * weight).sum(dim=(2, 3)) / denom
        return self._standardize_vector(pooled)

    def _response_delta(self, normal_outputs, perturbed_outputs,
                        expert_weights, expert_idx: int):
        total = normal_outputs[0][0].new_zeros(())
        for level in self.response_delta_feat_levels:
            gate = expert_weights[level][:, expert_idx:expert_idx + 1]
            normal = self._weighted_pool(normal_outputs[expert_idx][level],
                                         gate)
            perturbed = self._weighted_pool(
                perturbed_outputs[expert_idx][level], gate)
            total = total + (normal - perturbed).abs().mean(dim=1).mean()
        return total / len(self.response_delta_feat_levels)

    def _syn_cos_raw(self, normal_outputs, perturbed_outputs):
        total = normal_outputs[0][0].new_zeros(())
        for level in self.feat_levels:
            syn_a = self._project(normal_outputs[2], level, 'syn')
            syn_b = self._project(perturbed_outputs[2], level, 'syn')
            syn_cos = self.cos_sim(
                F.normalize(syn_a, p=2, dim=1),
                F.normalize(syn_b, p=2, dim=1))
            total = total + syn_cos.mean()
        return total / len(self.feat_levels)

    def _syn_feature_norms(self, outputs):
        """Return loss-free norms around the synergy projection path."""
        gap_norm = outputs[2][0].new_zeros(())
        projected_norm = outputs[2][0].new_zeros(())
        with torch.no_grad():
            for level in self.feat_levels:
                pooled = self.gap(outputs[2][level]).flatten(1)
                projected = self.projectors['syn'](pooled)
                gap_norm = gap_norm + torch.linalg.vector_norm(
                    pooled, ord=2, dim=1).mean()
                projected_norm = projected_norm + torch.linalg.vector_norm(
                    projected, ord=2, dim=1).mean()
        n_levels = len(self.feat_levels)
        return gap_norm / n_levels, projected_norm / n_levels

    def _syn_red_losses(self, normal_outputs, perturbed_outputs):
        total_syn = normal_outputs[0][0].new_zeros(())
        total_red = normal_outputs[0][0].new_zeros(())
        for level in self.feat_levels:
            syn_a = self._project(normal_outputs[2], level, 'syn')
            syn_b = self._project(perturbed_outputs[2], level, 'syn')
            syn_cos = self.cos_sim(
                F.normalize(syn_a, p=2, dim=1),
                F.normalize(syn_b, p=2, dim=1))
            total_syn = total_syn + torch.clamp(
                syn_cos - self.max_sim_syn, min=0).mean()

            red_a = self._project(normal_outputs[3], level, 'red')
            red_b = self._project(perturbed_outputs[3], level, 'red')
            red_cos = self.cos_sim(
                F.normalize(red_a, p=2, dim=1),
                F.normalize(red_b, p=2, dim=1))
            total_red = total_red + (1 - red_cos).mean()

        n_levels = len(self.feat_levels)
        return total_syn / n_levels, total_red / n_levels

    def _decorr_terms(self, normal_outputs):
        n_levels = len(self.decorr_feat_levels)
        pair_abs_means = {name: 0.0 for _, _, name in self._DECORR_PAIRS}
        pair_raw_means = {name: 0.0 for _, _, name in self._DECORR_PAIRS}
        red_abs_means = {
            name: 0.0 for _, _, name in self._RED_MONITOR_PAIRS}
        red_raw_means = {
            name: 0.0 for _, _, name in self._RED_MONITOR_PAIRS}

        decorr_sum = normal_outputs[0][0].new_zeros(())
        for level in self.decorr_feat_levels:
            vectors = self._pooled_channel_vectors(normal_outputs, level)
            vectors_norm = [
                F.normalize(v, p=2, dim=1, eps=self.eps) for v in vectors
            ]
            for i, j, name in self._DECORR_PAIRS:
                cos = (vectors_norm[i] * vectors_norm[j]).sum(dim=1)
                abs_cos = cos.abs()
                decorr_sum = decorr_sum + abs_cos.mean()
                pair_abs_means[name] = (
                    pair_abs_means[name] + abs_cos.mean().detach())
                pair_raw_means[name] = (
                    pair_raw_means[name] + cos.mean().detach())
            for i, j, name in self._RED_MONITOR_PAIRS:
                cos = (vectors_norm[i] * vectors_norm[j]).sum(dim=1)
                abs_cos = cos.abs()
                red_abs_means[name] = (
                    red_abs_means[name] + abs_cos.mean().detach())
                red_raw_means[name] = (
                    red_raw_means[name] + cos.mean().detach())

        n_pairs = len(self._DECORR_PAIRS)
        decorr = decorr_sum / (n_levels * n_pairs)
        out = {'decorr': decorr}
        for name in pair_abs_means:
            out[f'abs_cos_{name}'] = pair_abs_means[name] / n_levels
            out[f'cos_sim_{name}'] = pair_raw_means[name] / n_levels
        for name in red_abs_means:
            out[f'abs_cos_{name}'] = red_abs_means[name] / n_levels
            out[f'cos_sim_{name}'] = red_raw_means[name] / n_levels
        return out

    def _paired_main_terms(self, normal_outputs, perturbed_outputs):
        vis_outputs, lwir_outputs = self._get_paired_outputs(perturbed_outputs)
        total_uni_vis = normal_outputs[0][0].new_zeros(())
        total_uni_lwir = normal_outputs[0][0].new_zeros(())
        total_stable = normal_outputs[0][0].new_zeros(())

        for level in self.feat_levels:
            uni_vis_anchor = self._project(
                normal_outputs[0], level, 'uni_vis')
            uni_vis_pos = self._project(lwir_outputs[0], level, 'uni_vis')
            uni_vis_neg = self._project(vis_outputs[0], level, 'uni_vis')
            uni_vis_triplet = self.triplet_loss(
                uni_vis_anchor, uni_vis_pos, uni_vis_neg)
            uni_vis_stable = (1 - self.cos_sim(
                F.normalize(uni_vis_anchor, p=2, dim=1),
                F.normalize(uni_vis_pos, p=2, dim=1))).mean()

            uni_lwir_anchor = self._project(
                normal_outputs[1], level, 'uni_lwir')
            uni_lwir_pos = self._project(vis_outputs[1], level, 'uni_lwir')
            uni_lwir_neg = self._project(lwir_outputs[1], level, 'uni_lwir')
            uni_lwir_triplet = self.triplet_loss(
                uni_lwir_anchor, uni_lwir_pos, uni_lwir_neg)
            uni_lwir_stable = (1 - self.cos_sim(
                F.normalize(uni_lwir_anchor, p=2, dim=1),
                F.normalize(uni_lwir_pos, p=2, dim=1))).mean()

            total_uni_vis = total_uni_vis + uni_vis_triplet
            total_uni_lwir = total_uni_lwir + uni_lwir_triplet
            total_stable = total_stable + 0.5 * (
                uni_vis_stable + uni_lwir_stable)

        n_levels = len(self.feat_levels)
        uni_vis = total_uni_vis / n_levels
        uni_lwir = total_uni_lwir / n_levels
        uni_active = 0.5 * (uni_vis + uni_lwir)
        uni_stable = total_stable / n_levels

        syn_vis, red_vis = self._syn_red_losses(normal_outputs, vis_outputs)
        syn_lwir, red_lwir = self._syn_red_losses(normal_outputs, lwir_outputs)
        syn = 0.5 * (syn_vis + syn_lwir)
        red = 0.5 * (red_vis + red_lwir)

        total_main = self._main_weighted_total(
            uni_active, uni_stable, syn, red, uni_vis=uni_vis,
            uni_lwir=uni_lwir)

        out = {
            'uni_vis': uni_vis,
            'uni_lwir': uni_lwir,
            'uni_active': uni_active,
            'uni_stable': uni_stable,
            'syn': syn,
            'red': red,
            'total_main': total_main,
            'syn_vismask': syn_vis.detach(),
            'syn_lwirmask': syn_lwir.detach(),
            'red_vismask': red_vis.detach(),
            'red_lwirmask': red_lwir.detach(),
        }
        out.update(self._decorr_terms(normal_outputs))
        return out

    def _main_weighted_total(self, uni_active, uni_stable, syn, red,
                             uni_vis=None, uni_lwir=None):
        if self.loss_aggregation == 'iss_moe_role_avg':
            if uni_vis is None or uni_lwir is None:
                raise ValueError(
                    'iss_moe_role_avg requires paired perturbation so '
                    'uni_vis and uni_lwir losses are both available.')
            return 0.25 * (uni_vis + uni_lwir + syn + red)
        return (
            self.lambda_uni_active * uni_active
            + self.lambda_uni_stable * uni_stable
            + self.lambda_syn * syn
            + self.lambda_red * red)

    @staticmethod
    def _stable_unique_idx(perturbed_modality: str) -> int:
        if perturbed_modality == 'vis':
            return 1
        if perturbed_modality == 'lwir':
            return 0
        raise ValueError(
            f"perturbed_modality must be 'vis' or 'lwir', got "
            f'{perturbed_modality!r}')

    def _response_delta_terms(self, normal_outputs, perturbed_outputs,
                              perturbed_modality: str, expert_weights):
        self._check_spatial_weights(expert_weights)
        syn_delta = self._response_delta(
            normal_outputs, perturbed_outputs, expert_weights, expert_idx=2)
        red_delta = self._response_delta(
            normal_outputs, perturbed_outputs, expert_weights, expert_idx=3)
        stable_delta = self._response_delta(
            normal_outputs, perturbed_outputs, expert_weights,
            expert_idx=self._stable_unique_idx(perturbed_modality))
        return syn_delta, red_delta, stable_delta

    def forward(self, normal_outputs, perturbed_outputs,
                perturbed_modality: str, expert_weights=None):
        if self._is_paired_outputs(perturbed_outputs):
            out = self._paired_main_terms(normal_outputs, perturbed_outputs)
            vis_outputs, lwir_outputs = self._get_paired_outputs(
                perturbed_outputs)

            full_gap_norm, full_projected_norm = self._syn_feature_norms(
                normal_outputs)
            out['syn_gap_norm_full'] = full_gap_norm.detach()
            out['syn_projected_norm_full'] = full_projected_norm.detach()

            response_terms = {}
            for modality, outputs in (
                    ('vis', vis_outputs), ('lwir', lwir_outputs)):
                syn_delta, red_delta, stable_delta = (
                    self._response_delta_terms(
                        normal_outputs, outputs, modality, expert_weights))
                response_terms[modality] = (
                    syn_delta, red_delta, stable_delta)
                out[f'syn_cos_raw_{modality}mask'] = self._syn_cos_raw(
                    normal_outputs, outputs).detach()
                out[f'syn_delta_{modality}mask'] = syn_delta.detach()
                out[f'red_delta_{modality}mask'] = red_delta.detach()
                out[f'stable_delta_{modality}mask'] = stable_delta.detach()
                gap_norm, projected_norm = self._syn_feature_norms(outputs)
                out[f'syn_gap_norm_{modality}mask'] = gap_norm.detach()
                out[f'syn_projected_norm_{modality}mask'] = (
                    projected_norm.detach())

            syn_delta = 0.5 * (
                response_terms['vis'][0] + response_terms['lwir'][0])
            red_delta = 0.5 * (
                response_terms['vis'][1] + response_terms['lwir'][1])
            stable_delta = 0.5 * (
                response_terms['vis'][2] + response_terms['lwir'][2])

            out['syn_cos_raw'] = 0.5 * (
                out['syn_cos_raw_vismask']
                + out['syn_cos_raw_lwirmask'])
            out['syn_delta'] = syn_delta.detach()
            out['red_delta'] = red_delta.detach()
            out['stable_delta'] = stable_delta.detach()
            return out

        out = super().forward(
            normal_outputs, perturbed_outputs, perturbed_modality)
        if self.loss_aggregation == 'iss_moe_role_avg':
            out['total_main'] = self._main_weighted_total(
                out['uni_active'], out['uni_stable'], out['syn'],
                out['red'])

        syn_delta, red_delta, stable_delta = (
            self._response_delta_terms(
                normal_outputs, perturbed_outputs, perturbed_modality,
                expert_weights))
        out['syn_cos_raw'] = self._syn_cos_raw(
            normal_outputs, perturbed_outputs).detach()
        out['syn_delta'] = syn_delta.detach()
        out['red_delta'] = red_delta.detach()
        out['stable_delta'] = stable_delta.detach()
        return out
