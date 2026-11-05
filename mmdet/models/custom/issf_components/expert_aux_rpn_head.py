"""Train-only auxiliary RPN-lite head for the ISSFDet detector.

For each of the four experts (``uni_vis``, ``uni_lwir``, ``syn``, ``red``) we
instantiate one :class:`ExpertAuxRPNHead`. During training ``detector.loss()``
calls ``aux_head.loss(expert_k, rpn_data_samples)`` - the standard
``BaseDenseHead.loss`` entry, which internally runs ``forward`` then
``loss_by_feat`` and reuses the main RPN's anchor generator / bbox coder /
assigner / sampler verbatim (we just point at the same configs).

At inference the detector's ``predict()`` and ``_forward()`` never touch these
heads, so they add no forward-pass cost even though their parameters stay in
the ``state_dict``.

Implementation details:
    * ``_init_layers()`` is the only override. The shared stem is
      ``Conv3x3(in, feat) -> GroupNorm(num_groups, feat)`` - **without ReLU**.
      The parent :meth:`RPNHead.forward_single` applies ``F.relu`` right after
      ``self.rpn_conv(x)``, so adding a ReLU here would produce
      ``ReLU(ReLU(x))``.
    * The 4 experts do not share head parameters (that is the whole point of
      aux supervision). The 5 FPN levels share one head, identical to the
      main RPN behaviour.
"""

import torch.nn as nn

from mmdet.registry import MODELS
from mmdet.models.dense_heads import RPNHead
from mmdet.utils import MultiConfig


@MODELS.register_module()
class ExpertAuxRPNHead(RPNHead):
    """RPN-lite auxiliary head with Conv3x3 + GN stem (no ReLU here).

    Args:
        in_channels (int): channels of each expert feature (shared FPN output,
            defaults to 256).
        feat_channels (int): channels of the stem output. Kept at 256 so the
            ``cls`` / ``reg`` branches have the same shape as the main RPN.
        num_groups (int): GroupNorm groups for the stem. Defaults to 32
            (256 / 32 = 8 channels per group).
        init_cfg, **kwargs: forwarded to :class:`RPNHead` / :class:`AnchorHead`
            (anchor_generator, bbox_coder, loss_cls, loss_bbox, train_cfg,
            test_cfg, num_classes=1 by default).
    """

    def __init__(self,
                 in_channels: int = 256,
                 feat_channels: int = 256,
                 num_groups: int = 32,
                 init_cfg: MultiConfig = dict(
                     type='Normal', layer='Conv2d', std=0.01),
                 **kwargs) -> None:
        self._aux_num_groups = num_groups
        super().__init__(
            in_channels=in_channels,
            feat_channels=feat_channels,
            init_cfg=init_cfg,
            **kwargs)

    def _init_layers(self) -> None:
        """Conv3x3 + GN stem (ReLU left to parent ``forward_single``).

        Parent ``RPNHead._init_layers`` builds ``rpn_conv`` as either
        ``nn.Conv2d`` (num_convs=1) or ``nn.Sequential(*ConvModule)``
        (num_convs>1). We replace it with a Conv + GN sequential so the stem
        has a normalisation layer while still slotting into the parent's
        ``forward_single`` (which does ``rpn_conv -> F.relu -> cls/reg``).
        """
        self.rpn_conv = nn.Sequential(
            nn.Conv2d(
                self.in_channels, self.feat_channels, 3, padding=1, bias=True),
            nn.GroupNorm(self._aux_num_groups, self.feat_channels),
        )
        self.rpn_cls = nn.Conv2d(
            self.feat_channels,
            self.num_base_priors * self.cls_out_channels,
            1)
        reg_dim = self.bbox_coder.encode_size
        self.rpn_reg = nn.Conv2d(
            self.feat_channels, self.num_base_priors * reg_dim, 1)
