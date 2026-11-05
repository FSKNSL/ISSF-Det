"""Expert interaction loss with channel-vector decorrelation.

The decorrelation term acts on pooled feature maps before projection. It
compares the two unique experts with each other and with the synergistic
expert. The redundant expert is monitored without an orthogonality penalty.
Interaction and decorrelation terms have separate detector-side weights.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.registry import MODELS
from .interaction_loss_v2 import InteractionLossV2


@MODELS.register_module()
class InteractionLossV3(InteractionLossV2):
    """Interaction loss with cross-expert channel-vector decorrelation.

    Extends :class:`InteractionLossV2` with a ``decorr`` term on the 3 non-red
    expert pairs. The ``forward`` return dict is re-keyed so the detector can
    weight ``total_main`` and ``decorr`` independently.

    Args:
        decorr_feat_levels: FPN levels at which to compute the decorrelation
            term. Defaults to ``(1, 2)``.
        eps: numerical eps for ``F.normalize`` when computing cosine
            similarity. Defaults to ``1e-8``.
        Other arguments are inherited from :class:`InteractionLossV2`.

    Returns (from :meth:`forward`):
        dict with keys:
            ``uni_active``, ``uni_stable``, ``syn``, ``red``.
            ``total_main`` - weighted sum of the four interaction terms;
                the detector multiplies it by
                ``interaction_loss_weight``).
            ``decorr`` - mean |cos| over the 3 non-red pairs across
                ``decorr_feat_levels`` (detector multiplies it by
                ``decorr_weight``).
            ``cos_sim_uni_vis_uni_lwir``, ``cos_sim_uni_vis_syn``,
            ``cos_sim_uni_lwir_syn`` - raw signed cosine (detached, logging).
            ``abs_cos_uni_vis_uni_lwir``, ``abs_cos_uni_vis_syn``,
            ``abs_cos_uni_lwir_syn`` - |cos| (detached, logging).
    """

    _DECORR_PAIRS: Tuple[Tuple[int, int, str], ...] = (
        (0, 1, 'uni_vis_uni_lwir'),
        (0, 2, 'uni_vis_syn'),
        (1, 2, 'uni_lwir_syn'),
    )
    _RED_MONITOR_PAIRS: Tuple[Tuple[int, int, str], ...] = (
        (0, 3, 'uni_vis_red'),
        (1, 3, 'uni_lwir_red'),
        (2, 3, 'syn_red'),
    )

    def __init__(self,
                 decorr_feat_levels: Tuple[int, ...] = (1, 2),
                 eps: float = 1e-8,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.decorr_feat_levels = tuple(decorr_feat_levels)
        self.eps = eps

    def _pooled_channel_vectors(self, expert_outputs, level: int):
        """GAP each of the 4 experts at the given FPN level -> [B, C] each."""
        vectors = []
        for feat_tuple in expert_outputs:
            pooled = self.gap(feat_tuple[level]).flatten(1)
            vectors.append(pooled)
        return vectors

    def forward(self, normal_outputs, perturbed_outputs,
                perturbed_modality: str):
        base = super().forward(
            normal_outputs, perturbed_outputs, perturbed_modality)

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
                cos = (vectors_norm[i] * vectors_norm[j]).sum(dim=1)  # [B]
                abs_cos = cos.abs()
                decorr_sum = decorr_sum + abs_cos.mean()
                pair_abs_means[name] = (
                    pair_abs_means[name] + abs_cos.mean().detach())
                pair_raw_means[name] = (
                    pair_raw_means[name] + cos.mean().detach())
            # Red-pair cosine values are monitoring metrics only.
            for i, j, name in self._RED_MONITOR_PAIRS:
                cos = (vectors_norm[i] * vectors_norm[j]).sum(dim=1)
                abs_cos = cos.abs()
                red_abs_means[name] = (
                    red_abs_means[name] + abs_cos.mean().detach())
                red_raw_means[name] = (
                    red_raw_means[name] + cos.mean().detach())

        n_pairs = len(self._DECORR_PAIRS)
        decorr = decorr_sum / (n_levels * n_pairs)

        total_main = (self.lambda_uni_active * base['uni_active']
                      + self.lambda_uni_stable * base['uni_stable']
                      + self.lambda_syn * base['syn']
                      + self.lambda_red * base['red'])

        out = {
            'uni_active': base['uni_active'],
            'uni_stable': base['uni_stable'],
            'syn': base['syn'],
            'red': base['red'],
            'total_main': total_main,
            'decorr': decorr,
        }
        for name in pair_abs_means:
            out[f'abs_cos_{name}'] = pair_abs_means[name] / n_levels
            out[f'cos_sim_{name}'] = pair_raw_means[name] / n_levels
        for name in red_abs_means:
            out[f'abs_cos_{name}'] = red_abs_means[name] / n_levels
            out[f'cos_sim_{name}'] = red_raw_means[name] / n_levels
        return out
