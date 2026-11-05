"""Single-backbone Faster R-CNN for unimodal ISSF-Det comparisons."""

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig

from .faster_rcnn import FasterRCNN


@MODELS.register_module()
class UnimodalFasterRCNN(FasterRCNN):
    """Faster R-CNN with the standard single backbone restored locally.

    The repository's shared ``TwoStageDetector`` intentionally does not build
    ``self.backbone`` because its multispectral subclasses own two backbones.
    This isolated detector keeps that global behavior unchanged while making
    RGB-only and IR-only baselines valid standard Faster R-CNN models.
    """

    def __init__(self,
                 backbone: ConfigType,
                 rpn_head: ConfigType,
                 roi_head: ConfigType,
                 train_cfg: ConfigType,
                 test_cfg: ConfigType,
                 neck: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
