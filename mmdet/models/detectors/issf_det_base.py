"""Base RGB-IR detector for interaction-structured expert fusion.

Pipeline:

    img_vis, img_lwir
      -> mrsc  (modality-wise residual spectral conditioning)
      -> backbone_vis, backbone_lwir -> x_vis, x_lwir
      -> fusion (ISSMoEFusionDualLatent): FPN + four spatial experts
      -> RPN / RoI head

The detector provides paired backbone feature extraction, expert fusion,
interaction supervision, and standard Faster R-CNN detection heads.
"""

import warnings
import copy
from typing import List, Tuple, Union

import torch
from torch import Tensor

from mmdet.registry import MODELS
from mmengine.config import ConfigDict
from mmdet.structures import SampleList
from mmdet.utils import OptConfigType, OptMultiConfig
from .two_stream_faster_rcnn import TwoStreamFasterRCNN


@MODELS.register_module()
class ISSFDetBase(TwoStreamFasterRCNN):
    """Same-input, same-structure ISS-MoE detector for RGB-IR detection.

    Args:
        backbone: shared backbone config used to build both
            ``backbone_vis`` and ``backbone_lwir``.
        mrsc: spectral front-end config (MRSC or the reference mask).
        fusion: fusion module config (typically
            :class:`ISSMoEFusionDualLatent`).
        rpn_head, roi_head, train_cfg, test_cfg: standard Faster R-CNN parts.
        interaction_loss: interaction supervision config; set to None
            to disable this optional training loss.
        interaction_loss_weight: weight applied to the ``total`` returned by
            the interaction loss before adding it to ``losses``.
        mi_loss_weight: weight for MI auxiliary losses. Defaults to 0.0;
            MI sub-modules are further gated by ``fusion.use_mi`` so that
            setting ``loss_MI=None`` in the fusion config truly prevents any
            MI parameter from being created.
        data_preprocessor: data preprocessor config
            (paired RGB-IR, see ``PairedDetDataPreprocessor``).
        init_cfg: initialisation config.
    """

    def __init__(self,
                 backbone: ConfigDict,
                 mrsc: OptConfigType,
                 fusion: ConfigDict,
                 rpn_head: ConfigDict,
                 roi_head: ConfigDict,
                 train_cfg: ConfigDict,
                 test_cfg: ConfigDict,
                 interaction_loss: OptConfigType = None,
                 interaction_loss_weight: float = 0.01,
                 mi_loss_weight: float = 0.0,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:

        super().__init__(
            backbone=backbone,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor)

        self.backbone_vis = MODELS.build(backbone)
        self.backbone_lwir = MODELS.build(backbone)

        self.Gmask = MODELS.build(mrsc) if mrsc is not None else None
        self.FeaFusion = MODELS.build(fusion)

        self._perturb_toggle = 0
        self.interaction_loss_weight = interaction_loss_weight
        self.mi_loss_weight = mi_loss_weight
        if interaction_loss is not None:
            self.interaction_loss_fn = MODELS.build(interaction_loss)
        else:
            self.interaction_loss_fn = None

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            rpn_head_num_classes = rpn_head_.get('num_classes', None)
            if rpn_head_num_classes is None:
                rpn_head_.update(num_classes=1)
            else:
                if rpn_head_num_classes != 1:
                    warnings.warn(
                        'The `num_classes` should be 1 in RPN, but get '
                        f'{rpn_head_num_classes}, please set '
                        'rpn_head.num_classes = 1 in your config file.')
                    rpn_head_.update(num_classes=1)
            self.rpn_head = MODELS.build(rpn_head_)

        if roi_head is not None:
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            self.roi_head = MODELS.build(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def _load_from_state_dict(self, state_dict: dict, prefix: str,
                              local_metadata: dict, strict: bool,
                              missing_keys: Union[List[str], str],
                              unexpected_keys: Union[List[str], str],
                              error_msgs: Union[List[str], str]) -> None:
        bbox_head_prefix = prefix + '.bbox_head' if prefix else 'bbox_head'
        bbox_head_keys = [
            k for k in state_dict.keys() if k.startswith(bbox_head_prefix)
        ]
        rpn_head_prefix = prefix + '.rpn_head' if prefix else 'rpn_head'
        rpn_head_keys = [
            k for k in state_dict.keys() if k.startswith(rpn_head_prefix)
        ]
        if len(bbox_head_keys) != 0 and len(rpn_head_keys) == 0:
            for bbox_head_key in bbox_head_keys:
                rpn_head_key = rpn_head_prefix + \
                               bbox_head_key[len(bbox_head_prefix):]
                state_dict[rpn_head_key] = state_dict.pop(bbox_head_key)
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)

    @property
    def with_rpn(self) -> bool:
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self) -> bool:
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        return self.backbone(batch_inputs)

    def extract_feat_vis(self, img: Tensor) -> Tuple[Tensor]:
        x = self.backbone_vis(img)
        self._check_feature_tuple(x, 'x_vis')
        return x

    def extract_feat_lwir(self, img: Tensor) -> Tuple[Tensor]:
        x = self.backbone_lwir(img)
        try:
            self._check_feature_tuple(x, 'x_lwir')
        except RuntimeError as exc:
            raise RuntimeError(
                f'{exc} | img_lwir_unique stats: '
                f'{self._tensor_stats(img)}') from exc
        return x

    @staticmethod
    def _ensure_finite(tensor: Tensor, name: str) -> None:
        if not torch.isfinite(tensor).all():
            raise RuntimeError(
                f'ISSFDetBase produced non-finite values in {name}')

    @staticmethod
    def _tensor_stats(tensor: Tensor) -> str:
        mean = tensor.mean().item()
        std = tensor.std(unbiased=False).item()
        min_value = tensor.min().item()
        max_value = tensor.max().item()
        return (
            f'mean={mean:.4f}, std={std:.4f}, '
            f'min={min_value:.4f}, max={max_value:.4f}')

    def _check_feature_tuple(self, feat_tuple, name: str) -> None:
        for level, feat in enumerate(feat_tuple):
            self._ensure_finite(feat, f'{name}[{level}]')

    def _match_sample_stats(self, unique_img: Tensor, reference_img: Tensor,
                            name: str) -> Tensor:
        """Align per-sample/channel mean-std of ``unique_img`` to
        ``reference_img`` and clamp to a 3-sigma box.

        Uses ``unbiased=False`` + ``clamp(min=1e-4)`` so divisions are always
        finite (consistent with the matched-noise logic in
        the fusion module's matched-noise perturbation helper).
        """
        self._ensure_finite(unique_img, f'{name}_pre_match')
        ref_mean = reference_img.mean(dim=(2, 3), keepdim=True)
        ref_std = reference_img.std(
            dim=(2, 3), keepdim=True, unbiased=False).clamp(min=1e-4)
        unique_mean = unique_img.mean(dim=(2, 3), keepdim=True)
        unique_std = unique_img.std(
            dim=(2, 3), keepdim=True, unbiased=False).clamp(min=1e-4)
        normalized = (unique_img - unique_mean) / unique_std
        matched = normalized * ref_std + ref_mean
        lower = ref_mean - 3 * ref_std
        upper = ref_mean + 3 * ref_std
        matched = matched.clamp(min=lower, max=upper)
        self._ensure_finite(matched, f'{name}_matched')
        return matched

    def _extract_unique_features(self, img_vis: Tensor,
                                 img_lwir: Tensor) -> Tuple[Tensor, Tensor]:
        if self.Gmask is None:
            return img_vis, img_lwir

        mask_vis, mask_lwir = self.Gmask(img_vis, img_lwir)

        vis_fre = torch.fft.fft2(img_vis)
        fre_m_vis = torch.abs(vis_fre)
        fre_m_vis = torch.fft.fftshift(fre_m_vis, dim=(-2, -1))
        fre_p_vis = torch.angle(vis_fre)
        masked_fre_m_vis = fre_m_vis * mask_vis
        masked_fre_m_vis = torch.fft.ifftshift(
            masked_fre_m_vis, dim=(-2, -1))
        fre_vis = masked_fre_m_vis * torch.e ** (1j * fre_p_vis)
        img_vis_unique = torch.real(torch.fft.ifft2(fre_vis))
        img_vis_unique = self._match_sample_stats(
            img_vis_unique, img_vis, 'img_vis_unique')

        lwir_fre = torch.fft.fft2(img_lwir)
        fre_m_lwir = torch.abs(lwir_fre)
        fre_m_lwir = torch.fft.fftshift(fre_m_lwir, dim=(-2, -1))
        fre_p_lwir = torch.angle(lwir_fre)
        masked_fre_m_lwir = fre_m_lwir * mask_lwir
        masked_fre_m_lwir = torch.fft.ifftshift(
            masked_fre_m_lwir, dim=(-2, -1))
        fre_lwir = masked_fre_m_lwir * torch.e ** (1j * fre_p_lwir)
        img_lwir_unique = torch.real(torch.fft.ifft2(fre_lwir))
        img_lwir_unique = self._match_sample_stats(
            img_lwir_unique, img_lwir, 'img_lwir_unique')

        self._ensure_finite(img_vis_unique, 'img_vis_unique')
        self._ensure_finite(img_lwir_unique, 'img_lwir_unique')
        return img_vis_unique, img_lwir_unique

    def _next_perturb_modality(self) -> str:
        modality = 'vis' if self._perturb_toggle % 2 == 0 else 'lwir'
        self._perturb_toggle += 1
        return modality

    def _forward(self, batch_inputs: Tensor,
                 batch_data_samples: SampleList) -> tuple:
        results = ()
        img_vis = batch_inputs['img_vis']
        img_lwir = batch_inputs['img_lwir']

        img_vis_unique, img_lwir_unique = self._extract_unique_features(
            img_vis, img_lwir)
        x_vis = self.extract_feat_vis(img_vis_unique)
        x_lwir = self.extract_feat_lwir(img_lwir_unique)
        fusion_kwargs = {}
        if getattr(self.FeaFusion, 'requires_raw_inputs', False):
            fusion_kwargs = dict(img_vis=img_vis, img_lwir=img_lwir)
        x, _, _, _, _ = self.FeaFusion(
            x_vis, x_lwir, **fusion_kwargs)

        if self.with_rpn:
            rpn_results_list = self.rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]
        roi_outs = self.roi_head.forward(x, rpn_results_list,
                                         batch_data_samples)
        results = results + (roi_outs, )
        return results

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> dict:
        losses = dict()
        img_vis = batch_inputs['img_vis']
        img_lwir = batch_inputs['img_lwir']

        img_vis_unique, img_lwir_unique = self._extract_unique_features(
            img_vis, img_lwir)
        x_vis = self.extract_feat_vis(img_vis_unique)
        x_lwir = self.extract_feat_lwir(img_lwir_unique)

        x, mi_loss_vis, mi_loss_lwir, weights, expert_outputs = \
            self.FeaFusion(x_vis, x_lwir)

        if self.mi_loss_weight > 0 and getattr(
                self.FeaFusion, 'use_mi', False):
            losses['loss_MI_vis'] = self.mi_loss_weight * mi_loss_vis
            losses['loss_MI_lwir'] = self.mi_loss_weight * mi_loss_lwir

        if self.interaction_loss_fn is not None:
            perturbed_modality = self._next_perturb_modality()
            perturbed_outputs = self.FeaFusion.forward_perturbed(
                x_vis, x_lwir, perturbed_modality)
            int_loss = self.interaction_loss_fn(
                expert_outputs, perturbed_outputs, perturbed_modality)
            if isinstance(int_loss, dict):
                total = int_loss['total']
                losses['loss_interaction'] = (
                    self.interaction_loss_weight * total)
                losses['interaction_total_raw'] = total.detach()
                for k in ('uni_active', 'uni_stable', 'syn', 'red'):
                    if k in int_loss:
                        losses[f'interaction_{k}'] = int_loss[k].detach()
            else:
                losses['loss_interaction'] = (
                    self.interaction_loss_weight * int_loss)

        if self.with_rpn:
            proposal_cfg = self.train_cfg.get(
                'rpn_proposal', self.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(batch_data_samples)
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = \
                    torch.zeros_like(data_sample.gt_instances.labels)
            rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
                x, rpn_data_samples, proposal_cfg=proposal_cfg)
            keys = rpn_losses.keys()
            for key in list(keys):
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rpn_losses)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        roi_losses = self.roi_head.loss(x, rpn_results_list,
                                        batch_data_samples)
        losses.update(roi_losses)
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        assert self.with_bbox, 'Bbox head must be implemented.'

        img_vis = batch_inputs['img_vis']
        img_lwir = batch_inputs['img_lwir']

        img_vis_unique, img_lwir_unique = self._extract_unique_features(
            img_vis, img_lwir)
        x_vis = self.extract_feat_vis(img_vis_unique)
        x_lwir = self.extract_feat_lwir(img_lwir_unique)

        x, _, _, weights, _ = self.FeaFusion(x_vis, x_lwir)

        if batch_data_samples[0].get('proposals', None) is None:
            rpn_results_list = self.rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        results_list = self.roi_head.predict(
            x, rpn_results_list, batch_data_samples, rescale=rescale)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        for i, data_sample in enumerate(batch_data_samples):
            data_sample.expert_weights = weights[i].detach().cpu()
        return batch_data_samples
