"""Train-only role response head for ISSFDet.2.

The head maps one expert feature level to a single response logit map. It is
used only inside ``detector.loss()`` to put role supervision directly on each
expert output; ``predict()`` never calls it, so inference keeps the shared
detection-head path unchanged.
"""

import torch.nn as nn

from mmdet.registry import MODELS


@MODELS.register_module()
class ExpertRoleResponseHead(nn.Module):
    """Lightweight response map head for one ISS-MoE expert."""

    def __init__(self,
                 in_channels: int = 256,
                 hidden_channels: int = 64,
                 num_groups: int = 16) -> None:
        super().__init__()
        if hidden_channels % num_groups != 0:
            raise ValueError(
                'hidden_channels must be divisible by num_groups, got '
                f'{hidden_channels} and {num_groups}')
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels, hidden_channels, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 1),
        )

    def forward(self, feats, levels=None):
        if levels is None:
            levels = range(len(feats))
        return tuple(self.stem(feats[level]) for level in levels)
