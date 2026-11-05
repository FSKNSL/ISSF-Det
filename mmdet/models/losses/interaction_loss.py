import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS


@MODELS.register_module()
class InteractionLoss(nn.Module):
    """Latent-space interaction loss for the 2-modality ISSF-Det ISS-MoE variant.

    The original ISS-MoE interaction loss supervises expert behaviors under
    modality perturbation. For dense detection features, we first project
    FPN features into a compact latent space before applying the weak
    supervision. This keeps the interpretability signal while reducing the
    chance that auxiliary losses distort task-critical detection features.

    With only two modalities and a single perturbed forward pass per step,
    the uniqueness loss is implemented as a one-negative triplet:
    anchor=normal latent, positive=detached copy of the anchor, negative=
    perturbed latent of the currently perturbed modality expert. The
    redundancy loss follows ISS-MoE's invariant shared-information objective by
    matching normal and perturbed redundancy latents.
    """

    def __init__(self, in_channels=259, latent_dim=128, margin_triplet=1.0,
                 max_sim_syn=0.1,
                 lambda_uni=1.0, lambda_syn=0.5, lambda_red=1.0,
                 feat_levels=(1, 2)):
        super().__init__()
        self.latent_dim = latent_dim
        self.lambda_uni = lambda_uni
        self.lambda_syn = lambda_syn
        self.lambda_red = lambda_red
        self.max_sim_syn = max_sim_syn
        self.feat_levels = feat_levels
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
    def _build_projector(in_channels, latent_dim):
        return nn.Sequential(
            nn.Linear(in_channels, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, latent_dim),
        )

    def _project(self, feat_tuple, level, expert_key):
        pooled = self.gap(feat_tuple[level]).flatten(1)
        return self.projectors[expert_key](pooled)

    def forward(self, normal_outputs, perturbed_outputs, perturbed_modality):
        """Compute interaction losses.

        Args:
            normal_outputs: (x_uni_vis, x_uni_lwir, x_syn, x_red)
                from normal forward pass
            perturbed_outputs: expert outputs from a single perturbed forward
            perturbed_modality: 'vis' or 'lwir'

        Returns:
            Scalar interaction loss (always non-negative)
        """
        if perturbed_modality not in ('vis', 'lwir'):
            raise ValueError(
                f'perturbed_modality must be "vis" or "lwir", got '
                f'{perturbed_modality!r}')

        x_uni_vis, x_uni_lwir, x_syn, x_red = normal_outputs
        x_uni_vis_p, x_uni_lwir_p, x_syn_p, x_red_p = perturbed_outputs

        losses = []

        for level in self.feat_levels:
            if perturbed_modality == 'vis':
                anchor_uni = self._project(x_uni_vis, level, 'uni_vis')
                neg_uni = self._project(x_uni_vis_p, level, 'uni_vis')
            else:
                anchor_uni = self._project(x_uni_lwir, level, 'uni_lwir')
                neg_uni = self._project(x_uni_lwir_p, level, 'uni_lwir')
            pos_uni = anchor_uni.detach()
            loss_uni = self.triplet_loss(anchor_uni, pos_uni, neg_uni)

            anchor_syn = self._project(x_syn, level, 'syn')
            neg_syn = self._project(x_syn_p, level, 'syn')
            sim_syn = self.cos_sim(
                F.normalize(anchor_syn, p=2, dim=1),
                F.normalize(neg_syn, p=2, dim=1))
            loss_syn = torch.clamp(sim_syn - self.max_sim_syn, min=0).mean()

            anchor_red = self._project(x_red, level, 'red')
            pos_red = self._project(x_red_p, level, 'red')
            sim_red = self.cos_sim(
                F.normalize(anchor_red, p=2, dim=1),
                F.normalize(pos_red, p=2, dim=1))
            loss_red = (1 - sim_red).mean()

            level_loss = (self.lambda_uni * loss_uni +
                          self.lambda_syn * loss_syn +
                          self.lambda_red * loss_red)
            losses.append(level_loss)

        return sum(losses) / len(self.feat_levels)
