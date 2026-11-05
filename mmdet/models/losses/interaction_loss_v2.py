"""Interaction supervision for four experts sharing RGB and IR features.

Perturbing one modality changes its unique expert while the other modality's
unique expert remains stable. Synergistic and redundant experts receive
separate response constraints. The weighted total is optimized; component
values are available as detached diagnostics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.registry import MODELS


@MODELS.register_module()
class InteractionLossV2(nn.Module):
    """Symmetric interaction loss for four experts.

    Args:
        in_channels: channel count of expert features (shared FPN output).
        latent_dim: projector output dim.
        margin_triplet: triplet margin for ``uni_active``. Default 1.0.
        max_sim_syn: synergy cosine-similarity margin; the synergy term
            penalises ``cos - max_sim_syn`` clamped at 0.
        lambda_uni_active, lambda_uni_stable, lambda_syn, lambda_red:
            per-component weights combined inside ``total``. The default
            ``lambda_uni_stable=0.5`` is smaller than
            ``lambda_uni_active=1.0`` to weight stability separately.
        feat_levels: FPN levels at which to apply the loss. Default (1, 2).

    Returns (from :meth:`forward`):
        dict with keys ``uni_active``, ``uni_stable``, ``syn``, ``red``,
        ``total`` - the detector should back-prop ``total`` only (prefixing
        the log key as ``loss_interaction``) and log the other components
        under non-``loss`` keys so ``mmengine.model.BaseModel.parse_losses``
        does not double-count them.
    """

    def __init__(self,
                 in_channels: int = 256,
                 latent_dim: int = 128,
                 margin_triplet: float = 1.0,
                 max_sim_syn: float = 0.1,
                 lambda_uni_active: float = 1.0,
                 lambda_uni_stable: float = 0.5,
                 lambda_syn: float = 0.5,
                 lambda_red: float = 1.0,
                 feat_levels=(1, 2)) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.lambda_uni_active = lambda_uni_active
        self.lambda_uni_stable = lambda_uni_stable
        self.lambda_syn = lambda_syn
        self.lambda_red = lambda_red
        self.max_sim_syn = max_sim_syn
        self.feat_levels = tuple(feat_levels)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.projectors = nn.ModuleDict({
            'uni_vis': self._build_projector(in_channels, latent_dim),
            'uni_lwir': self._build_projector(in_channels, latent_dim),
            'syn': self._build_projector(in_channels, latent_dim),
            'red': self._build_projector(in_channels, latent_dim),
        })
        self.triplet_loss = nn.TripletMarginLoss(margin=margin_triplet, p=2)
        self.cos_sim = nn.CosineSimilarity(dim=1)

    @staticmethod
    def _build_projector(in_channels: int, latent_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_channels, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, latent_dim),
        )

    def _project(self, feat_tuple, level: int, expert_key: str):
        pooled = self.gap(feat_tuple[level]).flatten(1)
        return self.projectors[expert_key](pooled)

    def forward(self, normal_outputs, perturbed_outputs,
                perturbed_modality: str):
        """Compute symmetric expert-interaction loss.

        Args:
            normal_outputs: 4-tuple ``(uni_vis, uni_lwir, syn, red)`` of
                multi-scale feature tuples from the unperturbed forward.
            perturbed_outputs: same shape as ``normal_outputs`` after
                a modality perturbation.
            perturbed_modality: ``'vis'`` or ``'lwir'``.

        Returns:
            dict with ``uni_active``, ``uni_stable``, ``syn``, ``red``,
            ``total`` scalars.
        """
        if perturbed_modality not in ('vis', 'lwir'):
            raise ValueError(
                f"perturbed_modality must be 'vis' or 'lwir', got "
                f'{perturbed_modality!r}')

        x_uni_vis, x_uni_lwir, x_syn, x_red = normal_outputs
        x_uni_vis_p, x_uni_lwir_p, x_syn_p, x_red_p = perturbed_outputs

        total_active = 0.0
        total_stable = 0.0
        total_syn = 0.0
        total_red = 0.0
        n_levels = len(self.feat_levels)

        for level in self.feat_levels:
            if perturbed_modality == 'vis':
                active_anchor = self._project(x_uni_vis, level, 'uni_vis')
                active_neg = self._project(x_uni_vis_p, level, 'uni_vis')
                stable_anchor = self._project(x_uni_lwir, level, 'uni_lwir')
                stable_pair = self._project(x_uni_lwir_p, level, 'uni_lwir')
            else:
                active_anchor = self._project(x_uni_lwir, level, 'uni_lwir')
                active_neg = self._project(x_uni_lwir_p, level, 'uni_lwir')
                stable_anchor = self._project(x_uni_vis, level, 'uni_vis')
                stable_pair = self._project(x_uni_vis_p, level, 'uni_vis')

            pos = active_anchor.detach()
            loss_uni_active = self.triplet_loss(
                active_anchor, pos, active_neg)

            stable_cos = self.cos_sim(
                F.normalize(stable_anchor, p=2, dim=1),
                F.normalize(stable_pair, p=2, dim=1))
            loss_uni_stable = (1 - stable_cos).mean()

            syn_a = self._project(x_syn, level, 'syn')
            syn_b = self._project(x_syn_p, level, 'syn')
            syn_cos = self.cos_sim(
                F.normalize(syn_a, p=2, dim=1),
                F.normalize(syn_b, p=2, dim=1))
            loss_syn = torch.clamp(
                syn_cos - self.max_sim_syn, min=0).mean()

            red_a = self._project(x_red, level, 'red')
            red_b = self._project(x_red_p, level, 'red')
            red_cos = self.cos_sim(
                F.normalize(red_a, p=2, dim=1),
                F.normalize(red_b, p=2, dim=1))
            loss_red = (1 - red_cos).mean()

            total_active = total_active + loss_uni_active
            total_stable = total_stable + loss_uni_stable
            total_syn = total_syn + loss_syn
            total_red = total_red + loss_red

        uni_active = total_active / n_levels
        uni_stable = total_stable / n_levels
        syn = total_syn / n_levels
        red = total_red / n_levels

        total = (self.lambda_uni_active * uni_active
                 + self.lambda_uni_stable * uni_stable
                 + self.lambda_syn * syn
                 + self.lambda_red * red)

        return {
            'uni_active': uni_active,
            'uni_stable': uni_stable,
            'syn': syn,
            'red': red,
            'total': total,
        }
