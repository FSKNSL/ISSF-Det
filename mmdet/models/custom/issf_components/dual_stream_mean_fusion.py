"""Parameter-free dual-stream mean fusion used by module ablations."""

from typing import Sequence

from mmengine.model import BaseModule
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import OptConfigType


@MODELS.register_module()
class DualStreamMeanFusion(BaseModule):
    """Average paired backbone features before a shared neck.

    The five-item return value follows the detector's fusion contract. This
    ablation has no expert outputs, expert weights, or auxiliary MI losses.
    """

    def __init__(self, neck: OptConfigType) -> None:
        super().__init__()
        self.neck = MODELS.build(neck)
        self.use_mi = False

    def forward(self, x_vis: Sequence[Tensor],
                x_lwir: Sequence[Tensor], **kwargs):
        if len(x_vis) != len(x_lwir):
            raise ValueError(
                'Visible and LWIR backbones must have the same levels, got '
                f'{len(x_vis)} and {len(x_lwir)}.')

        fused = []
        for level, (vis, lwir) in enumerate(zip(x_vis, x_lwir)):
            if vis.shape != lwir.shape:
                raise ValueError(
                    f'Feature shape mismatch at level {level}: '
                    f'{tuple(vis.shape)} vs {tuple(lwir.shape)}.')
            fused.append(0.5 * (vis + lwir))

        outputs = tuple(self.neck(tuple(fused)))
        zero = outputs[0].new_zeros(())
        return outputs, zero, zero, None, None
