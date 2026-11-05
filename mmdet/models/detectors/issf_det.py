"""ISSF-Det with modality-wise spectral conditioning and spatial experts.

The detector uses MRSC before the two backbones and ISS-MoE for feature
fusion. Optional expert heads and interaction losses are used during training;
prediction exports expert weights for analysis.
"""

import copy
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmengine.logging import MessageHub
from mmdet.registry import MODELS
from mmengine.config import ConfigDict
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox2roi, bbox_overlaps, get_box_tensor
from mmdet.utils import OptConfigType, OptMultiConfig
from mmdet.models.utils import unpack_gt_instances
from .issf_det_base import ISSFDetBase


EXPERT_NAMES: Tuple[str, ...] = ('uni_vis', 'uni_lwir', 'syn', 'red')

KEY_INTERACTION_LOGS = frozenset({
    'interaction_uni_vis',
    'interaction_uni_lwir',
    'interaction_syn',
    'interaction_red',
    'interaction_region_uni_vis',
    'interaction_region_uni_lwir',
    'interaction_region_syn',
    'interaction_region_red',
    'interaction_region_total_raw',
    'interaction_region_valid',
    'interaction_response_syn_floor',
    'interaction_response_red_stable',
    'interaction_response_red_activity_floor',
    'interaction_response_uni_vis_d_vis',
    'interaction_response_uni_vis_d_lwir',
    'interaction_response_uni_vis_activity',
    'interaction_response_uni_lwir_d_vis',
    'interaction_response_uni_lwir_d_lwir',
    'interaction_response_uni_lwir_activity',
    'interaction_response_syn_d_vis',
    'interaction_response_syn_d_lwir',
    'interaction_response_syn_activity',
    'interaction_response_red_d_vis',
    'interaction_response_red_d_lwir',
    'interaction_response_red_activity',
    'interaction_region_response_uni_vis_d_vis',
    'interaction_region_response_uni_vis_d_lwir',
    'interaction_region_response_uni_vis_activity',
    'interaction_region_response_uni_lwir_d_vis',
    'interaction_region_response_uni_lwir_d_lwir',
    'interaction_region_response_uni_lwir_activity',
    'interaction_region_response_syn_d_vis',
    'interaction_region_response_syn_d_lwir',
    'interaction_region_response_syn_activity',
    'interaction_region_response_red_d_vis',
    'interaction_region_response_red_d_lwir',
    'interaction_region_response_red_activity',
    'interaction_region_mask_coverage_lvl0',
    'interaction_region_mask_coverage_lvl1',
    'interaction_region_valid_batch_frac_lvl0',
    'interaction_region_valid_batch_frac_lvl1',
    'interaction_syn_cos_raw',
    'interaction_syn_cos_raw_vismask',
    'interaction_syn_cos_raw_lwirmask',
    'interaction_syn_delta',
    'interaction_syn_delta_vismask',
    'interaction_syn_delta_lwirmask',
    'interaction_red_delta',
    'interaction_red_delta_vismask',
    'interaction_red_delta_lwirmask',
    'interaction_stable_delta',
    'interaction_stable_delta_vismask',
    'interaction_stable_delta_lwirmask',
    'interaction_syn_gap_norm_full',
    'interaction_syn_gap_norm_vismask',
    'interaction_syn_gap_norm_lwirmask',
    'interaction_syn_projected_norm_full',
    'interaction_syn_projected_norm_vismask',
    'interaction_syn_projected_norm_lwirmask',
    'interaction_abs_cos_uni_vis_uni_lwir',
    'interaction_abs_cos_uni_vis_syn',
    'interaction_abs_cos_uni_lwir_syn',
    'interaction_cos_sim_uni_vis_uni_lwir',
    'interaction_cos_sim_uni_vis_syn',
    'interaction_cos_sim_uni_lwir_syn',
    'interaction_uni_vis_region_delta_vismask',
    'interaction_uni_vis_region_delta_lwirmask',
    'interaction_uni_lwir_region_delta_vismask',
    'interaction_uni_lwir_region_delta_lwirmask',
    'interaction_syn_region_delta_vismask',
    'interaction_syn_region_delta_lwirmask',
    'interaction_red_region_delta_vismask',
    'interaction_red_region_delta_lwirmask',
})


@MODELS.register_module()
class ISSFDet(ISSFDetBase):
    """Same-input ISS-MoE detector with aux heads + decorrelation.

    Args:
        aux_rpn_head: config dict for :class:`ExpertAuxRPNHead`. The fields
            that are not specified (anchor_generator, bbox_coder, loss_cls,
            loss_bbox, num_classes, train_cfg, test_cfg) are filled in from
            the main ``rpn_head`` so aux and main RPN share assignment,
            bbox coding and loss behaviour.
        aux_head_weight: aux RPN loss weight applied after warmup.
        aux_head_weight_warmup_iters: iterations for the linear warmup on
            ``aux_head_weight``. With
            ``progress = min(cur_iter / warmup_iters, 1.0)`` the effective
            weight is ``aux_head_weight * progress``. 0 disables warmup.
            Defaults to 0.
        decorr_weight: independent outer weight for the decorrelation term
            returned by the interaction loss. Defaults to 0.1.
    """

    def __init__(self,
                 backbone: ConfigDict,
                 mrsc: OptConfigType,
                 fusion: ConfigDict,
                 rpn_head: ConfigDict,
                 roi_head: ConfigDict,
                 train_cfg: ConfigDict,
                 test_cfg: ConfigDict,
                 aux_rpn_head: OptConfigType = None,
                 aux_head_weight: float = 0.05,
                 aux_head_weight_warmup_iters: int = 0,
                 aux_roi_head: OptConfigType = None,
                 aux_roi_weight: float = 0.0,
                 aux_roi_warmup_iters: int = 0,
                 role_response_head: OptConfigType = None,
                 role_aux_weight: float = 0.0,
                 role_aux_warmup_iters: int = 0,
                 role_aux_feat_levels: Tuple[int, ...] = (1, 2),
                 role_aux_margin: float = 0.04,
                 role_aux_rank_margin: float = 0.01,
                 role_aux_tau: float = 0.02,
                 decorr_weight: float = 0.1,
                 paired_perturbation: bool = False,
                 roi_gate_diag: bool = False,
                 roi_gate_diag_hard_iou_thr: float = 0.75,
                 roi_gate_diag_near_neg_iou_thr: float = 0.30,
                 syn_aux_roi_probe: bool = False,
                 syn_aux_roi_probe_only: bool = False,
                 diag_gate_override: OptConfigType = None,
                 use_sparse_roi_syn_sem_loss: bool = False,
                 lambda_sparse_roi_syn_sem: float = 0.0025,
                 lambda_sparse_roi_syn_gap: float = 0.0025,
                 sparse_roi_syn_gap_margin: float = 0.05,
                 sparse_roi_syn_max_rois: int = 32,
                 sparse_roi_syn_min_iou: float = 0.50,
                 sparse_roi_syn_hard_iou_thr: float = 0.75,
                 use_sparse_roi_syn_fuse_sem_loss: bool = False,
                 lambda_sparse_roi_syn_fuse_sem: float = 0.0025,
                 lambda_sparse_roi_syn_fuse_gap: float = 0.0025,
                 sparse_roi_syn_fuse_gap_margin: float = 0.05,
                 sparse_roi_syn_fuse_max_rois: int = 32,
                 sparse_roi_syn_fuse_candidate_max_rois: int = 128,
                 sparse_roi_syn_fuse_min_iou: float = 0.50,
                 sparse_roi_syn_fuse_hard_iou_thr: float = 0.75,
                 sparse_roi_syn_fuse_single_fg_thr: float = 0.90,
                 use_syn_aux_head_loss: bool = False,
                 lambda_syn_aux_head_sem: float = 0.0025,
                 lambda_syn_aux_head_gap: float = 0.0025,
                 syn_aux_head_gap_margin: float = 0.05,
                 lambda_syn_aux_head_main_sem: float = 0.0,
                 lambda_syn_aux_head_main_gap: float = 0.0,
                 syn_aux_head_main_gap_margin: float = 0.05,
                 lambda_syn_aux_head_near_neg_bg: float = 0.0,
                 lambda_syn_aux_head_main_near_neg_bg: float = 0.0,
                 syn_aux_head_detach_near_neg_bg: bool = False,
                 lambda_syn_aux_head_comp_fg_floor: float = 0.0,
                 syn_aux_head_comp_fg_floor: float = 0.55,
                 syn_aux_head_max_rois: int = 32,
                 syn_aux_head_max_near_neg_rois: int = 64,
                 syn_aux_head_min_iou: float = 0.50,
                 syn_aux_head_hard_iou_thr: float = 0.75,
                 syn_aux_head_single_fg_thr: float = 0.90,
                 syn_aux_head_mode_split: bool = False,
                 syn_aux_head_single_min_fg: float = 0.55,
                 syn_aux_head_single_rel_margin: float = 0.10,
                 syn_aux_head_comp_max_fg: float = 0.55,
                 use_syn_boundary_gate_margin_loss: bool = False,
                 lambda_syn_boundary_gate_neg: float = 0.0,
                 lambda_syn_boundary_gate_near_neg: float = 0.0,
                 lambda_syn_boundary_gate_single: float = 0.0,
                 lambda_syn_boundary_gate_comp_floor: float = 0.0,
                 lambda_syn_boundary_gate_comp_rel: float = 0.0,
                 syn_boundary_gate_margin: float = 0.02,
                 syn_boundary_gate_comp_floor: float = 0.30,
                 syn_boundary_gate_comp_rel_ratio: float = 0.75,
                 syn_boundary_gate_comp_rel_margin: float = 0.0,
                 syn_boundary_gate_max_neg_rois: int = 64,
                 syn_boundary_gate_max_near_neg_rois: int = 64,
                 syn_boundary_gate_max_single_rois: int = 32,
                 syn_boundary_gate_max_comp_rois: int = 32,
                 use_syn_roi_gate_route_loss: bool = False,
                 lambda_syn_roi_gate_route_pos: float = 0.0,
                 lambda_syn_roi_gate_route_neg: float = 0.0,
                 lambda_syn_roi_gate_route_near_neg: float = 0.0,
                 syn_roi_gate_route_margin: float = 0.05,
                 syn_roi_gate_route_max_pos_rois: int = 32,
                 syn_roi_gate_route_max_neg_rois: int = 64,
                 syn_roi_gate_route_max_near_neg_rois: int = 64,
                 use_syn_causal_mode_loss: bool = False,
                 lambda_syn_causal_comp_hard: float = 0.0,
                 lambda_syn_causal_lwir_direct: float = 0.0,
                 lambda_syn_causal_vis_protect: float = 0.0,
                 lambda_syn_causal_near_neg: float = 0.0,
                 syn_causal_mode_fg_thr: float = 0.90,
                 syn_causal_mode_diff_gap: float = 0.30,
                 syn_causal_mode_single_vis_min_fg: float = 0.20,
                 syn_causal_mode_single_vis_rel_margin: float = 0.10,
                 syn_causal_mode_margin: float = 0.05,
                 syn_causal_mode_max_rois: int = 64,
                 syn_causal_mode_min_iou: float = 0.50,
                 use_cmm_roi_mode_route_loss: bool = False,
                 lambda_cmm_mode_single_lwir: float = 0.0,
                 lambda_cmm_mode_single_vis: float = 0.0,
                 lambda_cmm_mode_comp_hard: float = 0.0,
                 lambda_cmm_mode_near_neg: float = 0.0,
                 cmm_mode_single_vis_against_all_non_vis: bool = False,
                 cmm_mode_fg_thr: float = 0.75,
                 cmm_mode_diff_gap: float = 0.30,
                 cmm_mode_margin: float = 0.05,
                 cmm_mode_comp_syn_floor: float = 0.25,
                 cmm_mode_max_rois: int = 64,
                 cmm_mode_min_iou: float = 0.50,
                 use_qualified_syn_gate_nudge_loss: bool = False,
                 use_region_syn_rpn_sem_loss: bool = False,
                 lambda_region_syn_rpn_pos: float = 0.0025,
                 lambda_region_syn_rpn_gap: float = 0.0025,
                 region_syn_rpn_gap_margin: float = 0.05,
                 region_syn_rpn_feat_levels: Tuple[int, ...] = (1, 2),
                 region_syn_rpn_target_mask: str = 'target',
                 region_syn_rpn_core_ratio: float = 0.5,
                 enable_feature_interaction_loss: bool = True,
                 use_roi_logits_interaction_loss: bool = False,
                 roi_logits_interaction_weight: float = 0.01,
                 roi_logits_interaction_max_pos_per_img: int = 16,
                 roi_logits_interaction_margin: float = 1.0,
                 roi_logits_interaction_max_sim_syn: float = 0.1,
                 use_cada_roi: bool = False,
                 cada_freeze_detector: bool = False,
                 interaction_loss: OptConfigType = None,
                 interaction_loss_weight: float = 0.01,
                 mi_loss_weight: float = 0.0,
                 target_refiner: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        base_roi_head_cfg = copy.deepcopy(roi_head)
        super().__init__(
            backbone=backbone,
            mrsc=mrsc,
            fusion=fusion,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            interaction_loss=interaction_loss,
            interaction_loss_weight=interaction_loss_weight,
            mi_loss_weight=mi_loss_weight,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

        self.aux_head_weight = float(aux_head_weight)
        self.aux_head_weight_warmup_iters = int(aux_head_weight_warmup_iters)
        self.aux_roi_weight = float(aux_roi_weight)
        self.aux_roi_warmup_iters = int(aux_roi_warmup_iters)
        self.role_aux_weight = float(role_aux_weight)
        self.role_aux_warmup_iters = int(role_aux_warmup_iters)
        self.role_aux_feat_levels = tuple(role_aux_feat_levels)
        self.role_aux_margin = float(role_aux_margin)
        self.role_aux_rank_margin = float(role_aux_rank_margin)
        self.role_aux_tau = float(role_aux_tau)
        self.decorr_weight = float(decorr_weight)
        self.paired_perturbation = bool(paired_perturbation)
        self.roi_gate_diag = bool(roi_gate_diag)
        self.roi_gate_diag_hard_iou_thr = float(
            roi_gate_diag_hard_iou_thr)
        self.roi_gate_diag_near_neg_iou_thr = float(
            roi_gate_diag_near_neg_iou_thr)
        self.syn_aux_roi_probe = bool(syn_aux_roi_probe)
        self.syn_aux_roi_probe_only = bool(syn_aux_roi_probe_only)
        self.diag_gate_override = diag_gate_override
        self.use_sparse_roi_syn_sem_loss = bool(
            use_sparse_roi_syn_sem_loss)
        self.lambda_sparse_roi_syn_sem = float(lambda_sparse_roi_syn_sem)
        self.lambda_sparse_roi_syn_gap = float(lambda_sparse_roi_syn_gap)
        self.sparse_roi_syn_gap_margin = float(sparse_roi_syn_gap_margin)
        self.sparse_roi_syn_max_rois = int(sparse_roi_syn_max_rois)
        self.sparse_roi_syn_min_iou = float(sparse_roi_syn_min_iou)
        self.sparse_roi_syn_hard_iou_thr = float(
            sparse_roi_syn_hard_iou_thr)
        self.use_sparse_roi_syn_fuse_sem_loss = bool(
            use_sparse_roi_syn_fuse_sem_loss)
        self.lambda_sparse_roi_syn_fuse_sem = float(
            lambda_sparse_roi_syn_fuse_sem)
        self.lambda_sparse_roi_syn_fuse_gap = float(
            lambda_sparse_roi_syn_fuse_gap)
        self.sparse_roi_syn_fuse_gap_margin = float(
            sparse_roi_syn_fuse_gap_margin)
        self.sparse_roi_syn_fuse_max_rois = int(
            sparse_roi_syn_fuse_max_rois)
        self.sparse_roi_syn_fuse_candidate_max_rois = int(
            sparse_roi_syn_fuse_candidate_max_rois)
        self.sparse_roi_syn_fuse_min_iou = float(
            sparse_roi_syn_fuse_min_iou)
        self.sparse_roi_syn_fuse_hard_iou_thr = float(
            sparse_roi_syn_fuse_hard_iou_thr)
        self.sparse_roi_syn_fuse_single_fg_thr = float(
            sparse_roi_syn_fuse_single_fg_thr)
        self.use_syn_aux_head_loss = bool(use_syn_aux_head_loss)
        self.lambda_syn_aux_head_sem = float(lambda_syn_aux_head_sem)
        self.lambda_syn_aux_head_gap = float(lambda_syn_aux_head_gap)
        self.syn_aux_head_gap_margin = float(syn_aux_head_gap_margin)
        self.lambda_syn_aux_head_main_sem = float(
            lambda_syn_aux_head_main_sem)
        self.lambda_syn_aux_head_main_gap = float(
            lambda_syn_aux_head_main_gap)
        self.syn_aux_head_main_gap_margin = float(
            syn_aux_head_main_gap_margin)
        self.lambda_syn_aux_head_near_neg_bg = float(
            lambda_syn_aux_head_near_neg_bg)
        self.lambda_syn_aux_head_main_near_neg_bg = float(
            lambda_syn_aux_head_main_near_neg_bg)
        self.syn_aux_head_detach_near_neg_bg = bool(
            syn_aux_head_detach_near_neg_bg)
        self.lambda_syn_aux_head_comp_fg_floor = float(
            lambda_syn_aux_head_comp_fg_floor)
        self.syn_aux_head_comp_fg_floor = float(syn_aux_head_comp_fg_floor)
        self.syn_aux_head_max_rois = int(syn_aux_head_max_rois)
        self.syn_aux_head_max_near_neg_rois = int(
            syn_aux_head_max_near_neg_rois)
        self.syn_aux_head_min_iou = float(syn_aux_head_min_iou)
        self.syn_aux_head_hard_iou_thr = float(syn_aux_head_hard_iou_thr)
        self.syn_aux_head_single_fg_thr = float(syn_aux_head_single_fg_thr)
        self.syn_aux_head_mode_split = bool(syn_aux_head_mode_split)
        self.syn_aux_head_single_min_fg = float(
            syn_aux_head_single_min_fg)
        self.syn_aux_head_single_rel_margin = float(
            syn_aux_head_single_rel_margin)
        self.syn_aux_head_comp_max_fg = float(syn_aux_head_comp_max_fg)
        self.use_syn_boundary_gate_margin_loss = bool(
            use_syn_boundary_gate_margin_loss)
        self.lambda_syn_boundary_gate_neg = float(
            lambda_syn_boundary_gate_neg)
        self.lambda_syn_boundary_gate_near_neg = float(
            lambda_syn_boundary_gate_near_neg)
        self.lambda_syn_boundary_gate_single = float(
            lambda_syn_boundary_gate_single)
        self.lambda_syn_boundary_gate_comp_floor = float(
            lambda_syn_boundary_gate_comp_floor)
        self.lambda_syn_boundary_gate_comp_rel = float(
            lambda_syn_boundary_gate_comp_rel)
        self.syn_boundary_gate_margin = float(syn_boundary_gate_margin)
        self.syn_boundary_gate_comp_floor = float(
            syn_boundary_gate_comp_floor)
        self.syn_boundary_gate_comp_rel_ratio = float(
            syn_boundary_gate_comp_rel_ratio)
        self.syn_boundary_gate_comp_rel_margin = float(
            syn_boundary_gate_comp_rel_margin)
        self.syn_boundary_gate_max_neg_rois = int(
            syn_boundary_gate_max_neg_rois)
        self.syn_boundary_gate_max_near_neg_rois = int(
            syn_boundary_gate_max_near_neg_rois)
        self.syn_boundary_gate_max_single_rois = int(
            syn_boundary_gate_max_single_rois)
        self.syn_boundary_gate_max_comp_rois = int(
            syn_boundary_gate_max_comp_rois)
        self.use_syn_roi_gate_route_loss = bool(
            use_syn_roi_gate_route_loss)
        self.lambda_syn_roi_gate_route_pos = float(
            lambda_syn_roi_gate_route_pos)
        self.lambda_syn_roi_gate_route_neg = float(
            lambda_syn_roi_gate_route_neg)
        self.lambda_syn_roi_gate_route_near_neg = float(
            lambda_syn_roi_gate_route_near_neg)
        self.syn_roi_gate_route_margin = float(syn_roi_gate_route_margin)
        self.syn_roi_gate_route_max_pos_rois = int(
            syn_roi_gate_route_max_pos_rois)
        self.syn_roi_gate_route_max_neg_rois = int(
            syn_roi_gate_route_max_neg_rois)
        self.syn_roi_gate_route_max_near_neg_rois = int(
            syn_roi_gate_route_max_near_neg_rois)
        self.use_syn_causal_mode_loss = bool(use_syn_causal_mode_loss)
        self.lambda_syn_causal_comp_hard = float(
            lambda_syn_causal_comp_hard)
        self.lambda_syn_causal_lwir_direct = float(
            lambda_syn_causal_lwir_direct)
        self.lambda_syn_causal_vis_protect = float(
            lambda_syn_causal_vis_protect)
        self.lambda_syn_causal_near_neg = float(
            lambda_syn_causal_near_neg)
        self.syn_causal_mode_fg_thr = float(syn_causal_mode_fg_thr)
        self.syn_causal_mode_diff_gap = float(syn_causal_mode_diff_gap)
        self.syn_causal_mode_single_vis_min_fg = float(
            syn_causal_mode_single_vis_min_fg)
        self.syn_causal_mode_single_vis_rel_margin = float(
            syn_causal_mode_single_vis_rel_margin)
        self.syn_causal_mode_margin = float(syn_causal_mode_margin)
        self.syn_causal_mode_max_rois = int(syn_causal_mode_max_rois)
        self.syn_causal_mode_min_iou = float(syn_causal_mode_min_iou)
        self.use_cmm_roi_mode_route_loss = bool(use_cmm_roi_mode_route_loss)
        self.lambda_cmm_mode_single_lwir = float(
            lambda_cmm_mode_single_lwir)
        self.lambda_cmm_mode_single_vis = float(lambda_cmm_mode_single_vis)
        self.lambda_cmm_mode_comp_hard = float(lambda_cmm_mode_comp_hard)
        self.lambda_cmm_mode_near_neg = float(lambda_cmm_mode_near_neg)
        self.cmm_mode_single_vis_against_all_non_vis = bool(
            cmm_mode_single_vis_against_all_non_vis)
        self.cmm_mode_fg_thr = float(cmm_mode_fg_thr)
        self.cmm_mode_diff_gap = float(cmm_mode_diff_gap)
        self.cmm_mode_margin = float(cmm_mode_margin)
        self.cmm_mode_comp_syn_floor = float(cmm_mode_comp_syn_floor)
        self.cmm_mode_max_rois = int(cmm_mode_max_rois)
        self.cmm_mode_min_iou = float(cmm_mode_min_iou)
        if use_qualified_syn_gate_nudge_loss:
            raise ValueError(
                'qualified syn gate nudge experiments have been removed.')
        deprecated_syn_modes = []
        if self.use_sparse_roi_syn_sem_loss:
            deprecated_syn_modes.append('use_sparse_roi_syn_sem_loss')
        if self.use_sparse_roi_syn_fuse_sem_loss:
            deprecated_syn_modes.append('use_sparse_roi_syn_fuse_sem_loss')
        if self.use_syn_roi_gate_route_loss:
            deprecated_syn_modes.append('use_syn_roi_gate_route_loss')
        if self.use_syn_causal_mode_loss:
            deprecated_syn_modes.append('use_syn_causal_mode_loss')
        if self.use_cmm_roi_mode_route_loss:
            deprecated_syn_modes.append('use_cmm_roi_mode_route_loss')
        if deprecated_syn_modes:
            raise ValueError(
                'Deprecated syn diagnostic routes have been removed: '
                f'{", ".join(deprecated_syn_modes)}. Use the VIS anchor '
                'baseline plus train-only syn_aux_head comp-hard / near-neg '
                'constraints for the next syn-boundary diagnostic.')
        self.use_region_syn_rpn_sem_loss = bool(
            use_region_syn_rpn_sem_loss)
        self.lambda_region_syn_rpn_pos = float(lambda_region_syn_rpn_pos)
        self.lambda_region_syn_rpn_gap = float(lambda_region_syn_rpn_gap)
        self.region_syn_rpn_gap_margin = float(region_syn_rpn_gap_margin)
        self.region_syn_rpn_feat_levels = tuple(region_syn_rpn_feat_levels)
        self.region_syn_rpn_target_mask = str(region_syn_rpn_target_mask)
        self.region_syn_rpn_core_ratio = float(region_syn_rpn_core_ratio)
        self.enable_feature_interaction_loss = bool(
            enable_feature_interaction_loss)
        self.use_roi_logits_interaction_loss = bool(
            use_roi_logits_interaction_loss)
        self.roi_logits_interaction_weight = float(
            roi_logits_interaction_weight)
        self.roi_logits_interaction_max_pos_per_img = int(
            roi_logits_interaction_max_pos_per_img)
        self.roi_logits_interaction_margin = float(
            roi_logits_interaction_margin)
        self.roi_logits_interaction_max_sim_syn = float(
            roi_logits_interaction_max_sim_syn)
        self.use_cada_roi = bool(use_cada_roi)
        self.cada_freeze_detector = bool(cada_freeze_detector)
        if self.cada_freeze_detector and not self.use_cada_roi:
            raise ValueError(
                'cada_freeze_detector requires use_cada_roi=True.')
        self._local_iter = 0

        self.aux_heads: nn.ModuleList = self._build_aux_heads(
            aux_rpn_head=aux_rpn_head,
            rpn_head=rpn_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg)
        self.aux_roi_heads: nn.ModuleList = self._build_aux_roi_heads(
            aux_roi_head=aux_roi_head,
            roi_head=base_roi_head_cfg,
            train_cfg=train_cfg,
            test_cfg=test_cfg)
        self.role_heads: nn.ModuleList = self._build_role_heads(
            role_response_head)
        self.syn_aux_bbox_head = (
            MODELS.build(copy.deepcopy(base_roi_head_cfg['bbox_head']))
            if self.use_syn_aux_head_loss else None)
        self._syn_aux_head_copied = False
        self.target_refiner = (
            MODELS.build(copy.deepcopy(target_refiner))
            if target_refiner is not None else None)
        if self.use_cada_roi:
            if self.target_refiner is not None:
                raise ValueError('CADA-RoI requires target_refiner=None.')
            required = (
                'loss_with_cada_features', 'predict_with_cada_features',
                'cada_trainable_parameters', 'train_cada_components')
            if not all(hasattr(self.roi_head, name) for name in required):
                raise TypeError(
                    'use_cada_roi requires a compatible counterfactual '
                    'RoI head.')
        if self.cada_freeze_detector:
            for parameter in self.parameters():
                parameter.requires_grad_(False)
            for parameter in self.roi_head.cada_trainable_parameters():
                parameter.requires_grad_(True)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.cada_freeze_detector:
            for module in self.children():
                module.eval()
            self.roi_head.train_cada_components()
        return self

    def _build_aux_heads(self, aux_rpn_head, rpn_head, train_cfg,
                         test_cfg) -> nn.ModuleList:
        if aux_rpn_head is None:
            return nn.ModuleList()

        # Reuse the main RPN's anchor generator / bbox coder / losses /
        # train_cfg / test_cfg so aux and main RPN supervise with the same
        # assignment + coding rules.
        main_rpn = copy.deepcopy(rpn_head)
        rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
        main_rpn.update(
            train_cfg=rpn_train_cfg,
            test_cfg=test_cfg.rpn if test_cfg is not None else None)

        shared_keys = (
            'anchor_generator', 'bbox_coder', 'loss_cls', 'loss_bbox',
            'num_classes', 'train_cfg', 'test_cfg')

        heads = nn.ModuleList()
        for _ in range(len(EXPERT_NAMES)):
            cfg = copy.deepcopy(aux_rpn_head)
            for k in shared_keys:
                if k not in cfg and k in main_rpn:
                    cfg[k] = copy.deepcopy(main_rpn[k])
            # Force num_classes=1 (RPN contract).
            if cfg.get('num_classes', 1) != 1:
                cfg['num_classes'] = 1
            heads.append(MODELS.build(cfg))
        return heads

    def _build_role_heads(self, role_response_head) -> nn.ModuleList:
        if role_response_head is None:
            return nn.ModuleList()
        return nn.ModuleList([
            MODELS.build(copy.deepcopy(role_response_head))
            for _ in range(len(EXPERT_NAMES))
        ])

    def _build_aux_roi_heads(self, aux_roi_head, roi_head, train_cfg,
                             test_cfg) -> nn.ModuleList:
        if aux_roi_head is None:
            return nn.ModuleList()

        cfg = copy.deepcopy(roi_head)
        cfg.update(copy.deepcopy(aux_roi_head))
        rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
        cfg.update(
            train_cfg=rcnn_train_cfg,
            test_cfg=test_cfg.rcnn if test_cfg is not None else None)
        return nn.ModuleList([
            MODELS.build(copy.deepcopy(cfg)) for _ in range(len(EXPERT_NAMES))
        ])

    def _current_aux_weight(self) -> float:
        """Linearly warm up the auxiliary loss weight."""
        warmup = self.aux_head_weight_warmup_iters
        try:
            hub = MessageHub.get_current_instance()
            cur_iter = hub.get_info('iter')
            if cur_iter is None:
                cur_iter = self._local_iter
                self._local_iter += 1
        except Exception:
            cur_iter = self._local_iter
            self._local_iter += 1
        if warmup <= 0:
            return self.aux_head_weight
        progress = min(float(cur_iter) / float(warmup), 1.0)
        return self.aux_head_weight * progress

    def _current_role_weight(self) -> float:
        warmup = self.role_aux_warmup_iters
        try:
            hub = MessageHub.get_current_instance()
            cur_iter = hub.get_info('iter')
            if cur_iter is None:
                cur_iter = self._local_iter
        except Exception:
            cur_iter = self._local_iter
        if warmup <= 0:
            return self.role_aux_weight
        progress = min(float(cur_iter) / float(warmup), 1.0)
        return self.role_aux_weight * progress

    def _current_aux_roi_weight(self) -> float:
        warmup = self.aux_roi_warmup_iters
        try:
            hub = MessageHub.get_current_instance()
            cur_iter = hub.get_info('iter')
            if cur_iter is None:
                cur_iter = self._local_iter
                self._local_iter += 1
        except Exception:
            cur_iter = self._local_iter
            self._local_iter += 1
        if warmup <= 0:
            return self.aux_roi_weight
        progress = min(float(cur_iter) / float(warmup), 1.0)
        return self.aux_roi_weight * progress

    @staticmethod
    def _interaction_metric_key(key: str) -> str:
        return f"interaction_{key.replace('loss', 'metric')}"

    @staticmethod
    def _scale_aux_loss(val, scale: float):
        if isinstance(val, (list, tuple)):
            return [scale * v for v in val]
        return scale * val

    @staticmethod
    def _detach_metric(val):
        if torch.is_tensor(val):
            return val.detach()
        if isinstance(val, (list, tuple)):
            return [
                item.detach() if torch.is_tensor(item) else item
                for item in val
            ]
        return val

    def _sample_aux_roi_results(self, feats, rpn_results_list,
                                batch_data_samples):
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, _ = outputs

        sampling_results = []
        for img_idx, rpn_results in enumerate(rpn_results_list):
            if hasattr(rpn_results, 'bboxes'):
                rpn_results.priors = rpn_results.pop('bboxes')
            elif not hasattr(rpn_results, 'priors'):
                raise AttributeError(
                    'Aux ROI proposals must contain bboxes or priors.')

            assign_result = self.roi_head.bbox_assigner.assign(
                rpn_results, batch_gt_instances[img_idx],
                batch_gt_instances_ignore[img_idx])
            sampling_result = self.roi_head.bbox_sampler.sample(
                assign_result,
                rpn_results,
                batch_gt_instances[img_idx],
                feats=[lvl_feat[img_idx][None] for lvl_feat in feats])
            sampling_results.append(sampling_result)
        return sampling_results

    def _compute_aux_roi_losses(self, expert_outputs, rpn_results_list,
                                batch_data_samples, fused_feats):
        if len(self.aux_roi_heads) == 0:
            return {}

        cur_w = self._current_aux_roi_weight()
        per_expert_w = cur_w / float(len(self.aux_roi_heads))
        device_ref = expert_outputs[0][0]
        losses = {'aux_roi_weight': device_ref.new_tensor(cur_w).detach()}
        sampling_results = self._sample_aux_roi_results(
            fused_feats, rpn_results_list, batch_data_samples)

        for expert_idx, (head, expert_feats) in enumerate(
                zip(self.aux_roi_heads, expert_outputs)):
            name = EXPERT_NAMES[expert_idx]
            bbox_results = head.bbox_loss(expert_feats, sampling_results)
            raw_losses = bbox_results['loss_bbox']
            if 'loss_cls' in raw_losses:
                raw_cls = raw_losses['loss_cls']
                losses[f'loss_aux_roi_{name}_cls'] = self._scale_aux_loss(
                    raw_cls, per_expert_w)
                losses[f'aux_roi_raw_{name}_cls'] = self._detach_metric(
                    raw_cls)
            if 'loss_bbox' in raw_losses:
                raw_bbox = raw_losses['loss_bbox']
                losses[f'loss_aux_roi_{name}_bbox'] = self._scale_aux_loss(
                    raw_bbox, per_expert_w)
                losses[f'aux_roi_raw_{name}_bbox'] = self._detach_metric(
                    raw_bbox)
            if 'acc' in raw_losses:
                losses[f'aux_roi_acc_{name}'] = self._detach_metric(
                    raw_losses['acc'])
        return losses

    def _sample_roi_logits_positive_rois(self, fused_feats, rpn_results_list,
                                         batch_data_samples, ref: Tensor):
        sampling_results = self._sample_aux_roi_results(
            fused_feats, copy.deepcopy(rpn_results_list), batch_data_samples)
        max_per_img = int(self.roi_logits_interaction_max_pos_per_img)

        boxes_by_img = [
            ref.new_zeros((0, 4)) for _ in range(len(sampling_results))
        ]
        labels_by_img = [
            ref.new_zeros((0,), dtype=torch.long)
            for _ in range(len(sampling_results))
        ]
        ious_by_img = [
            ref.new_zeros((0,)) for _ in range(len(sampling_results))
        ]

        for batch_idx, result in enumerate(sampling_results):
            pos_boxes = self._as_bbox_tensor(result.pos_priors)
            pos_gt_boxes = self._as_bbox_tensor(result.pos_gt_bboxes)
            if (pos_boxes is None or pos_gt_boxes is None
                    or pos_boxes.numel() == 0
                    or pos_gt_boxes.numel() == 0):
                continue

            pos_boxes = pos_boxes.to(device=ref.device, dtype=ref.dtype)
            pos_gt_boxes = pos_gt_boxes.to(device=ref.device, dtype=ref.dtype)
            pos_ious = self._bbox_pair_iou(pos_boxes, pos_gt_boxes)
            if pos_ious.numel() == 0:
                continue

            pos_labels = getattr(result, 'pos_gt_labels', None)
            if pos_labels is None:
                pos_labels = pos_boxes.new_zeros(
                    (pos_boxes.shape[0],), dtype=torch.long)
            else:
                pos_labels = pos_labels.to(
                    device=ref.device, dtype=torch.long)

            keep = torch.argsort(pos_ious, descending=True)
            if max_per_img > 0:
                keep = keep[:max_per_img]
            boxes_by_img[batch_idx] = pos_boxes[keep]
            labels_by_img[batch_idx] = pos_labels[keep]
            ious_by_img[batch_idx] = pos_ious[keep]

        count = sum(int(boxes.shape[0]) for boxes in boxes_by_img)
        if count <= 0:
            return None, None, None
        return (
            bbox2roi(boxes_by_img),
            torch.cat(labels_by_img, dim=0),
            torch.cat(ious_by_img, dim=0))

    def _compute_roi_logits_interaction_loss(self, expert_outputs,
                                             perturbed_outputs, fused_feats,
                                             rpn_results_list,
                                             batch_data_samples):
        if not self.use_roi_logits_interaction_loss:
            return {}
        if not self.paired_perturbation:
            raise ValueError(
                'use_roi_logits_interaction_loss requires '
                'paired_perturbation=True.')
        if not isinstance(perturbed_outputs, dict) or not all(
                key in perturbed_outputs for key in ('vis', 'lwir')):
            raise ValueError(
                'use_roi_logits_interaction_loss requires paired perturbed '
                'outputs with "vis" and "lwir" keys.')
        if self.roi_logits_interaction_max_pos_per_img <= 0:
            raise ValueError(
                'roi_logits_interaction_max_pos_per_img must be positive, '
                f'got {self.roi_logits_interaction_max_pos_per_img}.')

        ref = expert_outputs[0][0]
        zero = ref.new_zeros(())
        rois, labels, ious = self._sample_roi_logits_positive_rois(
            fused_feats, rpn_results_list, batch_data_samples, ref)
        if rois is None:
            return {
                'loss_roi_logits_interaction': zero,
                'roi_logits_pos_count': zero.detach(),
                'roi_logits_pos_iou_mean': zero.detach(),
                'roi_logits_inter_uni_vis_raw': zero.detach(),
                'roi_logits_inter_uni_lwir_raw': zero.detach(),
                'roi_logits_inter_syn_raw': zero.detach(),
                'roi_logits_inter_red_raw': zero.detach(),
                'roi_logits_inter_total_raw': zero.detach(),
                'roi_logits_inter_weight': ref.new_tensor(
                    self.roi_logits_interaction_weight).detach(),
            }

        shared_head = getattr(self.roi_head, 'shared_head', None)
        shared_params = (
            list(shared_head.parameters()) if shared_head is not None else [])
        bbox_head_params = list(self.roi_head.bbox_head.parameters())
        frozen_params = shared_params + bbox_head_params
        old_requires_grad = [
            param.requires_grad for param in frozen_params
        ]
        for param in frozen_params:
            param.requires_grad_(False)

        try:
            full_scores = []
            vismask_scores = []
            lwirmask_scores = []
            for expert_idx in range(len(EXPERT_NAMES)):
                full_scores.append(self.roi_head._bbox_forward(
                    expert_outputs[expert_idx], rois)['cls_score'])
                # ``vis`` means VIS masked / only LWIR; ``lwir`` means LWIR
                # masked / only VIS in ISSMoEFusionDualLatent.
                vismask_scores.append(self.roi_head._bbox_forward(
                    perturbed_outputs['vis'][expert_idx], rois)['cls_score'])
                lwirmask_scores.append(self.roi_head._bbox_forward(
                    perturbed_outputs['lwir'][expert_idx], rois)['cls_score'])
        finally:
            for param, requires_grad in zip(frozen_params, old_requires_grad):
                param.requires_grad_(requires_grad)

        def embed(cls_score: Tensor) -> Tensor:
            return F.log_softmax(cls_score, dim=1)

        full = [embed(score) for score in full_scores]
        vismask = [embed(score) for score in vismask_scores]
        lwirmask = [embed(score) for score in lwirmask_scores]

        margin = float(self.roi_logits_interaction_margin)
        max_sim = ref.new_tensor(self.roi_logits_interaction_max_sim_syn)
        uni_vis_raw = F.triplet_margin_loss(
            full[0], lwirmask[0], vismask[0], margin=margin, p=2,
            reduction='mean')
        uni_lwir_raw = F.triplet_margin_loss(
            full[1], vismask[1], lwirmask[1], margin=margin, p=2,
            reduction='mean')
        syn_vis_raw = torch.relu(
            F.cosine_similarity(full[2], vismask[2], dim=1) - max_sim).mean()
        syn_lwir_raw = torch.relu(
            F.cosine_similarity(full[2], lwirmask[2], dim=1) - max_sim).mean()
        syn_raw = 0.5 * (syn_vis_raw + syn_lwir_raw)
        red_vis_raw = (
            1.0 - F.cosine_similarity(full[3], vismask[3], dim=1)).mean()
        red_lwir_raw = (
            1.0 - F.cosine_similarity(full[3], lwirmask[3], dim=1)).mean()
        red_raw = 0.5 * (red_vis_raw + red_lwir_raw)
        total_raw = torch.stack(
            (uni_vis_raw, uni_lwir_raw, syn_raw, red_raw)).mean()
        weight = float(self.roi_logits_interaction_weight)

        num_classes = int(self.roi_head.bbox_head.num_classes)
        logs = {
            'loss_roi_logits_interaction': weight * total_raw,
            'roi_logits_pos_count': ref.new_tensor(
                float(rois.shape[0])).detach(),
            'roi_logits_pos_iou_mean': ious.mean().detach(),
            'roi_logits_inter_uni_vis_raw': uni_vis_raw.detach(),
            'roi_logits_inter_uni_lwir_raw': uni_lwir_raw.detach(),
            'roi_logits_inter_syn_raw': syn_raw.detach(),
            'roi_logits_inter_red_raw': red_raw.detach(),
            'roi_logits_inter_total_raw': total_raw.detach(),
            'roi_logits_inter_weight': ref.new_tensor(weight).detach(),
        }
        for expert_idx, name in enumerate(EXPERT_NAMES):
            score = full_scores[expert_idx]
            logs[f'roi_logits_{name}_ce_full'] = F.cross_entropy(
                score, labels, reduction='mean').detach()
            logs[f'roi_logits_{name}_fg_full'] = F.softmax(
                score, dim=1)[:, :num_classes].sum(dim=1).mean().detach()
        return logs

    def _attach_expert_weights(self, batch_data_samples: SampleList,
                               weights) -> None:
        """Attach per-sample expert weights to each data sample.

        Handles both:
          * Image-level weights: ``weights`` is a ``Tensor`` with shape
            ``[B, S, K]``.
          * Spatial weights: ``weights`` is a ``tuple/list`` of 5
            tensors, each shaped ``[B, K, H_i, W_i]``.
        """
        if isinstance(weights, torch.Tensor) and weights.dim() == 3:
            for i, ds in enumerate(batch_data_samples):
                ds.expert_weights = weights[i].detach().cpu()
            return
        if (isinstance(weights, (tuple, list))
                and len(weights) > 0
                and isinstance(weights[0], torch.Tensor)
                and weights[0].dim() == 4):
            B = weights[0].shape[0]
            for i in range(B):
                batch_data_samples[i].expert_weights = tuple(
                    w[i].detach().cpu() for w in weights)
            return
        raise TypeError(
            f'Unsupported expert weights container: {type(weights)}')

    @staticmethod
    def _is_spatial_weights(weights) -> bool:
        return (
            isinstance(weights, (tuple, list))
            and len(weights) > 0
            and isinstance(weights[0], torch.Tensor)
            and weights[0].dim() == 4)

    def _active_expert_names(self) -> Tuple[str, ...]:
        return tuple(getattr(self.FeaFusion, 'expert_roles', EXPERT_NAMES))

    def _compute_spatial_gate_stats(self, weights, eps: float = 1e-8):
        """Compute detached gate entropy and expert load for monitoring."""
        if not self._is_spatial_weights(weights):
            return None

        stats = {}
        num_experts = weights[0].shape[1]
        expert_names = self._active_expert_names()
        if len(expert_names) != num_experts:
            raise ValueError(
                f'Fusion exposes {len(expert_names)} expert names but gate '
                f'has {num_experts} channels.')

        for level, w in enumerate(weights):
            entropy = -(w * (w + eps).log()).sum(dim=1).mean()
            stats[f'gate_entropy_lvl{level}'] = entropy.detach()

            u = w.mean(dim=(0, 2, 3))  # [K]
            for k, name in enumerate(expert_names):
                stats[f'gate_load_lvl{level}_{name}'] = u[k].detach()

        return stats

    @staticmethod
    def _get_bboxes_tensor(data_sample):
        gt_instances = getattr(data_sample, 'gt_instances', None)
        if gt_instances is None or not hasattr(gt_instances, 'bboxes'):
            return None
        bboxes = gt_instances.bboxes
        if hasattr(bboxes, 'tensor'):
            bboxes = bboxes.tensor
        if not isinstance(bboxes, torch.Tensor):
            return None
        if bboxes.numel() == 0:
            return None
        return bboxes[:, :4]

    @staticmethod
    def _smooth_hinge(violation: Tensor, tau: float) -> Tensor:
        if tau <= 0:
            return F.relu(violation)
        tau_tensor = violation.new_tensor(tau)
        return tau_tensor * F.softplus(violation / tau_tensor)

    def _build_target_region_masks(self, weights, batch_data_samples,
                                   input_hw: Tuple[int, int]):
        """Build per-level GT target masks aligned with spatial gates."""
        if not self._is_spatial_weights(weights):
            return None

        input_h, input_w = input_hw
        device = weights[0].device
        region_masks: Dict[str, list] = {'target': []}
        build_scale_aware = getattr(
            self.interaction_loss_fn, 'region_scale_aware', False)
        if build_scale_aware:
            scale_thresholds = self.interaction_loss_fn \
                .region_scale_thresholds
            scale_level_groups = self.interaction_loss_fn \
                .region_scale_level_groups
            max_scale_level = max(
                level for levels in scale_level_groups for level in levels)
            if max_scale_level >= len(weights):
                raise ValueError(
                    'Scale-aware target level exceeds available FPN levels: '
                    f'max level {max_scale_level}, got {len(weights)} levels')
            region_masks['target_scale_aware'] = []
        build_core = (
            self.use_region_syn_rpn_sem_loss
            and self.region_syn_rpn_target_mask == 'target_core')
        if build_core:
            if not (0.0 < self.region_syn_rpn_core_ratio <= 1.0):
                raise ValueError(
                    'region_syn_rpn_core_ratio must be in (0, 1], got '
                    f'{self.region_syn_rpn_core_ratio}')
            region_masks['target_core'] = []

        for level, gate in enumerate(weights):
            batch_size, _, gate_h, gate_w = gate.shape
            target = gate.new_zeros((batch_size, 1, gate_h, gate_w))
            target_scale_aware = (
                gate.new_zeros((batch_size, 1, gate_h, gate_w))
                if build_scale_aware else None)
            target_core = (
                gate.new_zeros((batch_size, 1, gate_h, gate_w))
                if build_core else None)
            scale_x = float(gate_w) / float(input_w)
            scale_y = float(gate_h) / float(input_h)

            for batch_idx, data_sample in enumerate(batch_data_samples):
                bboxes = self._get_bboxes_tensor(data_sample)
                if bboxes is None:
                    continue
                bboxes = bboxes.to(device=device, dtype=gate.dtype)

                for box_idx in range(bboxes.shape[0]):
                    box = bboxes[box_idx]
                    x1 = int(torch.floor(box[0] * scale_x).clamp(
                        0, gate_w - 1).item())
                    y1 = int(torch.floor(box[1] * scale_y).clamp(
                        0, gate_h - 1).item())
                    x2 = int(torch.ceil(box[2] * scale_x).clamp(
                        1, gate_w).item())
                    y2 = int(torch.ceil(box[3] * scale_y).clamp(
                        1, gate_h).item())
                    if x2 <= x1:
                        x2 = min(x1 + 1, gate_w)
                    if y2 <= y1:
                        y2 = min(y1 + 1, gate_h)

                    target[batch_idx, :, y1:y2, x1:x2] = 1.0
                    if target_scale_aware is not None:
                        box_w = (box[2] - box[0]).clamp(min=0)
                        box_h = (box[3] - box[1]).clamp(min=0)
                        scale_factor = data_sample.metainfo.get(
                            'scale_factor', None)
                        if scale_factor is not None:
                            scale_factor = torch.as_tensor(
                                scale_factor, device=device,
                                dtype=gate.dtype).flatten()
                            if scale_factor.numel() < 2:
                                raise ValueError(
                                    'scale_factor must provide width and '
                                    f'height scales, got {scale_factor}')
                            box_w = box_w / scale_factor[0].clamp(min=1e-6)
                            box_h = box_h / scale_factor[1].clamp(min=1e-6)
                        box_scale = torch.sqrt(box_w * box_h).item()
                        scale_group = sum(
                            box_scale >= threshold
                            for threshold in scale_thresholds)
                        if level in scale_level_groups[scale_group]:
                            target_scale_aware[
                                batch_idx, :, y1:y2, x1:x2] = 1.0
                    if target_core is not None:
                        ratio = self.region_syn_rpn_core_ratio
                        cx = 0.5 * (box[0] + box[2])
                        cy = 0.5 * (box[1] + box[3])
                        half_w = 0.5 * (box[2] - box[0]).clamp(min=0) * ratio
                        half_h = 0.5 * (box[3] - box[1]).clamp(min=0) * ratio
                        cx1 = int(torch.floor((cx - half_w) * scale_x).clamp(
                            0, gate_w - 1).item())
                        cy1 = int(torch.floor((cy - half_h) * scale_y).clamp(
                            0, gate_h - 1).item())
                        cx2 = int(torch.ceil((cx + half_w) * scale_x).clamp(
                            1, gate_w).item())
                        cy2 = int(torch.ceil((cy + half_h) * scale_y).clamp(
                            1, gate_h).item())
                        if cx2 <= cx1:
                            cx2 = min(cx1 + 1, gate_w)
                        if cy2 <= cy1:
                            cy2 = min(cy1 + 1, gate_h)
                        target_core[batch_idx, :, cy1:cy2, cx1:cx2] = 1.0

            region_masks['target'].append(target)
            if target_scale_aware is not None:
                region_masks['target_scale_aware'].append(
                    target_scale_aware)
            if target_core is not None:
                region_masks['target_core'].append(target_core)

        return {name: tuple(masks) for name, masks in region_masks.items()}

    def _rpn_objectness_logit(self, feat: Tensor) -> Tensor:
        x = self.rpn_head.rpn_conv(feat)
        x = F.relu(x)
        cls_score = self.rpn_head.rpn_cls(x)
        log_num_anchors = cls_score.new_tensor(
            float(cls_score.shape[1])).log()
        return torch.logsumexp(
            cls_score, dim=1, keepdim=True) - log_num_anchors

    @staticmethod
    def _target_masked_mean_per_image(value: Tensor,
                                      mask: Tensor,
                                      eps: float = 1e-6):
        if value.shape[-2:] != mask.shape[-2:]:
            raise ValueError(
                f'Mask/value spatial mismatch: value {value.shape[-2:]}, '
                f'mask {mask.shape[-2:]}')
        mask = mask.detach()
        denom = mask.sum(dim=(2, 3))
        valid = denom.squeeze(1) > eps
        pooled = (value * mask).sum(dim=(2, 3)) / denom.clamp(min=eps)
        return pooled.squeeze(1), valid

    def _compute_region_syn_rpn_sem_loss(self, expert_outputs,
                                         perturbed_outputs,
                                         region_masks):
        if not self.use_region_syn_rpn_sem_loss:
            return {}
        if not self.paired_perturbation:
            raise ValueError(
                'use_region_syn_rpn_sem_loss requires '
                'paired_perturbation=True.')
        if not isinstance(perturbed_outputs, dict) or not all(
                key in perturbed_outputs for key in ('vis', 'lwir')):
            raise ValueError(
                'use_region_syn_rpn_sem_loss requires paired perturbed '
                'outputs with "vis" and "lwir" keys.')
        target_mask_key = self.region_syn_rpn_target_mask
        if target_mask_key not in ('target', 'target_core'):
            raise ValueError(
                'region_syn_rpn_target_mask must be "target" or '
                f'"target_core", got {target_mask_key!r}.')
        if region_masks is None or target_mask_key not in region_masks:
            raise ValueError(
                'use_region_syn_rpn_sem_loss requires '
                f'{target_mask_key} region masks.')

        syn_full = expert_outputs[2]
        syn_vismask = perturbed_outputs['vis'][2]
        syn_lwirmask = perturbed_outputs['lwir'][2]
        target_masks = region_masks[target_mask_key]
        levels = self.region_syn_rpn_feat_levels
        if len(syn_full) <= max(levels) or len(target_masks) <= max(levels):
            raise ValueError(
                'region_syn_rpn_feat_levels exceed available feature or '
                f'mask levels: {levels}')

        rpn_params = list(self.rpn_head.rpn_conv.parameters())
        rpn_params += list(self.rpn_head.rpn_cls.parameters())
        old_requires_grad = [param.requires_grad for param in rpn_params]
        for param in rpn_params:
            param.requires_grad_(False)

        full_values = []
        vis_values = []
        lwir_values = []
        try:
            for level in levels:
                mask = target_masks[level]
                full_logit = self._rpn_objectness_logit(syn_full[level])
                full_mean, valid = self._target_masked_mean_per_image(
                    full_logit, mask)
                if not torch.any(valid):
                    continue
                with torch.no_grad():
                    vis_logit = self._rpn_objectness_logit(
                        syn_vismask[level].detach())
                    lwir_logit = self._rpn_objectness_logit(
                        syn_lwirmask[level].detach())
                    vis_mean, _ = self._target_masked_mean_per_image(
                        vis_logit, mask)
                    lwir_mean, _ = self._target_masked_mean_per_image(
                        lwir_logit, mask)
                full_values.append(full_mean[valid])
                vis_values.append(vis_mean[valid])
                lwir_values.append(lwir_mean[valid])
        finally:
            for param, requires_grad in zip(rpn_params, old_requires_grad):
                param.requires_grad_(requires_grad)

        ref = syn_full[0]
        if not full_values:
            zero = ref.sum() * 0.0
            return {
                'loss_region_syn_rpn_sem': zero,
                'region_syn_rpn_pos_raw': zero.detach(),
                'region_syn_rpn_gap_raw': zero.detach(),
                'region_syn_rpn_obj_full': zero.detach(),
                'region_syn_rpn_obj_vismask': zero.detach(),
                'region_syn_rpn_obj_lwirmask': zero.detach(),
                'region_syn_rpn_gap_vismask': zero.detach(),
                'region_syn_rpn_gap_lwirmask': zero.detach(),
                'region_syn_rpn_valid': zero.detach(),
            }

        full = torch.cat(full_values, dim=0)
        vis = torch.cat(vis_values, dim=0)
        lwir = torch.cat(lwir_values, dim=0)

        target = torch.ones_like(full)
        pos_raw = F.binary_cross_entropy_with_logits(
            full, target, reduction='mean')
        p_full = torch.sigmoid(full)
        p_vis = torch.sigmoid(vis)
        p_lwir = torch.sigmoid(lwir)
        gap_vis = p_full - p_vis
        gap_lwir = p_full - p_lwir
        margin = full.new_tensor(self.region_syn_rpn_gap_margin)
        gap_raw = 0.5 * (
            torch.relu(margin - gap_vis).mean()
            + torch.relu(margin - gap_lwir).mean())
        loss = (
            full.new_tensor(self.lambda_region_syn_rpn_pos) * pos_raw
            + full.new_tensor(self.lambda_region_syn_rpn_gap) * gap_raw)

        return {
            'loss_region_syn_rpn_sem': loss,
            'region_syn_rpn_pos_raw': pos_raw.detach(),
            'region_syn_rpn_gap_raw': gap_raw.detach(),
            'region_syn_rpn_obj_full': p_full.mean().detach(),
            'region_syn_rpn_obj_vismask': p_vis.mean().detach(),
            'region_syn_rpn_obj_lwirmask': p_lwir.mean().detach(),
            'region_syn_rpn_gap_vismask': gap_vis.mean().detach(),
            'region_syn_rpn_gap_lwirmask': gap_lwir.mean().detach(),
            'region_syn_rpn_valid': full.new_tensor(
                float(full.numel())).detach(),
        }

    def _ensure_syn_aux_head_ready(self, ref: Tensor) -> Tensor:
        if self.syn_aux_bbox_head is None:
            raise RuntimeError(
                'use_syn_aux_head_loss=True requires a registered '
                'syn_aux_bbox_head.')

        main_bbox_head = self.roi_head.bbox_head
        if self.syn_aux_bbox_head is main_bbox_head:
            raise RuntimeError(
                'syn_aux_bbox_head must be independent from main bbox_head.')
        if (hasattr(self.syn_aux_bbox_head, 'fc_cls')
                and hasattr(main_bbox_head, 'fc_cls')
                and id(self.syn_aux_bbox_head.fc_cls.weight)
                == id(main_bbox_head.fc_cls.weight)):
            raise RuntimeError(
                'syn_aux_bbox_head.fc_cls.weight must not share the main '
                'bbox_head.fc_cls.weight parameter object.')

        main_param_ids = {id(param) for param in main_bbox_head.parameters()}
        for param in self.syn_aux_bbox_head.parameters():
            if id(param) in main_param_ids:
                raise RuntimeError(
                    'syn_aux_bbox_head must not share any parameter object '
                    'with main bbox_head.')

        copied = ref.new_tensor(0.0)
        if not self._syn_aux_head_copied:
            state_dict = copy.deepcopy(main_bbox_head.state_dict())
            self.syn_aux_bbox_head.load_state_dict(state_dict, strict=True)
            self._syn_aux_head_copied = True
            copied = ref.new_tensor(1.0)
        return copied

    def _syn_aux_bbox_forward(self, feats, rois):
        bbox_roi_extractor = self.roi_head.bbox_roi_extractor
        bbox_feats = bbox_roi_extractor(
            feats[:bbox_roi_extractor.num_inputs], rois)
        if getattr(self.roi_head, 'with_shared_head', False):
            bbox_feats = self.roi_head.shared_head(bbox_feats)
        cls_score, bbox_pred = self.syn_aux_bbox_head(bbox_feats)
        return dict(
            cls_score=cls_score, bbox_pred=bbox_pred,
            bbox_feats=bbox_feats)

    def _split_syn_aux_roi_modes(self, p_uni_vis: Tensor,
                                 p_uni_lwir: Tensor) -> Dict[str, Tensor]:
        if not self.syn_aux_head_mode_split:
            main_fg = torch.maximum(p_uni_vis, p_uni_lwir)
            single_strong = main_fg >= main_fg.new_tensor(
                self.syn_aux_head_single_fg_thr)
            single_vis = single_strong & (p_uni_vis >= p_uni_lwir)
            single_lwir = single_strong & (p_uni_lwir > p_uni_vis)
            return {
                'comp_hard': ~single_strong,
                'single_strong': single_strong,
                'single_vis': single_vis,
                'single_lwir': single_lwir,
            }

        single_min = p_uni_vis.new_tensor(self.syn_aux_head_single_min_fg)
        rel_margin = p_uni_vis.new_tensor(
            self.syn_aux_head_single_rel_margin)
        comp_max = p_uni_vis.new_tensor(self.syn_aux_head_comp_max_fg)
        main_fg = torch.maximum(p_uni_vis, p_uni_lwir)
        single_vis = (
            (p_uni_vis >= single_min)
            & ((p_uni_vis - p_uni_lwir) >= rel_margin))
        single_lwir = (
            (p_uni_lwir >= single_min)
            & ((p_uni_lwir - p_uni_vis) >= rel_margin))
        single_strong = single_vis | single_lwir
        comp_hard = (~single_strong) & (main_fg < comp_max)
        return {
            'comp_hard': comp_hard,
            'single_strong': single_strong,
            'single_vis': single_vis,
            'single_lwir': single_lwir,
        }

    def _compute_syn_aux_head_loss(self, expert_outputs, perturbed_outputs,
                                   fused_feats, rpn_results_list,
                                   batch_data_samples):
        if not self.use_syn_aux_head_loss:
            return {}
        if not self.paired_perturbation:
            raise ValueError(
                'use_syn_aux_head_loss requires paired_perturbation=True.')
        if not isinstance(perturbed_outputs, dict) or not all(
                key in perturbed_outputs for key in ('vis', 'lwir')):
            raise ValueError(
                'use_syn_aux_head_loss requires paired perturbed outputs '
                'with "vis" and "lwir" keys.')
        if self.syn_aux_head_max_rois <= 0:
            raise ValueError(
                'syn_aux_head_max_rois must be positive, got '
                f'{self.syn_aux_head_max_rois}')

        ref = expert_outputs[2][0]
        copied = self._ensure_syn_aux_head_ready(ref)
        zero = ref.new_zeros(()).detach()
        num_classes = int(self.roi_head.bbox_head.num_classes)
        max_rois = int(self.syn_aux_head_max_rois)
        min_iou = ref.new_tensor(self.syn_aux_head_min_iou)
        hard_iou_thr = ref.new_tensor(self.syn_aux_head_hard_iou_thr)
        sampling_results = self._sample_aux_roi_results(
            fused_feats, copy.deepcopy(rpn_results_list), batch_data_samples)

        boxes_by_img = [
            ref.new_zeros((0, 4)) for _ in range(len(sampling_results))
        ]
        labels_by_img = [
            ref.new_zeros((0,), dtype=torch.long)
            for _ in range(len(sampling_results))
        ]
        ious_by_img = [
            ref.new_zeros((0,)) for _ in range(len(sampling_results))
        ]
        near_neg_boxes_by_img = [
            ref.new_zeros((0, 4)) for _ in range(len(sampling_results))
        ]
        for batch_idx, result in enumerate(sampling_results):
            pos_boxes = self._as_bbox_tensor(result.pos_priors)
            pos_gt_boxes = self._as_bbox_tensor(result.pos_gt_bboxes)
            if (pos_boxes is not None and pos_gt_boxes is not None
                    and pos_boxes.numel() > 0 and pos_gt_boxes.numel() > 0):
                pos_boxes = pos_boxes.to(device=ref.device, dtype=ref.dtype)
                pos_gt_boxes = pos_gt_boxes.to(
                    device=ref.device, dtype=ref.dtype)
                pos_ious = self._bbox_pair_iou(pos_boxes, pos_gt_boxes)
                hard_mask = (pos_ious >= min_iou) & (pos_ious < hard_iou_thr)
                if torch.any(hard_mask):
                    pos_labels = getattr(result, 'pos_gt_labels', None)
                    if pos_labels is None:
                        pos_labels = pos_boxes.new_zeros(
                            (pos_boxes.shape[0],), dtype=torch.long)
                    else:
                        pos_labels = pos_labels.to(
                            device=ref.device, dtype=torch.long)

                    boxes_by_img[batch_idx] = pos_boxes[hard_mask]
                    labels_by_img[batch_idx] = pos_labels[hard_mask]
                    ious_by_img[batch_idx] = pos_ious[hard_mask]

            if (self.lambda_syn_aux_head_near_neg_bg != 0.0
                    and self.syn_aux_head_max_near_neg_rois > 0):
                neg_boxes = self._as_bbox_tensor(result.neg_priors)
                if neg_boxes is not None and neg_boxes.numel() > 0:
                    neg_boxes = neg_boxes.to(device=ref.device,
                                             dtype=ref.dtype)
                    gt_boxes = self._get_bboxes_tensor(
                        batch_data_samples[batch_idx])
                    if gt_boxes is not None:
                        gt_boxes = gt_boxes.to(device=ref.device,
                                               dtype=ref.dtype)
                    neg_ious = self._bbox_max_iou(neg_boxes, gt_boxes)
                    near_mask = neg_ious >= neg_ious.new_tensor(
                        self.roi_gate_diag_near_neg_iou_thr)
                    if torch.any(near_mask):
                        near_neg_boxes_by_img[batch_idx] = (
                            neg_boxes[near_mask])

        def compute_near_neg_bg_logs():
            max_near_neg = int(self.syn_aux_head_max_near_neg_rois)
            use_aux_bg = self.lambda_syn_aux_head_near_neg_bg != 0.0
            use_main_bg = self.lambda_syn_aux_head_main_near_neg_bg != 0.0
            if (not use_aux_bg and not use_main_bg) or max_near_neg <= 0:
                return {}

            capped_boxes = []
            near_neg_count = 0
            for boxes in near_neg_boxes_by_img:
                remaining = max_near_neg - near_neg_count
                if remaining <= 0:
                    capped_boxes.append(boxes[:0])
                    continue
                cur_boxes = boxes[:remaining]
                capped_boxes.append(cur_boxes)
                near_neg_count += int(cur_boxes.shape[0])

            count_tensor = ref.new_tensor(float(near_neg_count)).detach()
            log_vars = {
                'syn_aux_head_near_neg_count': count_tensor,
            }
            if near_neg_count <= 0:
                if use_aux_bg:
                    log_vars.update({
                        'loss_syn_aux_head_near_neg_bg': ref.new_zeros(()),
                        'syn_aux_head_near_neg_bg_raw': zero,
                        'syn_aux_head_near_neg_fgprob': zero,
                    })
                if use_main_bg:
                    log_vars.update({
                        'loss_syn_aux_head_main_near_neg_bg':
                        ref.new_zeros(()),
                        'syn_aux_head_main_near_neg_bg_raw': zero,
                        'syn_aux_head_main_near_neg_fgprob': zero,
                    })
                return log_vars

            near_neg_rois = bbox2roi(capped_boxes)
            bg_labels = near_neg_rois.new_full(
                (near_neg_rois.shape[0],), num_classes, dtype=torch.long)

            if use_aux_bg:
                shared_head = getattr(self.roi_head, 'shared_head', None)
                shared_params = (
                    list(shared_head.parameters())
                    if shared_head is not None else [])
                old_requires_grad = [
                    param.requires_grad for param in shared_params
                ]
                for param in shared_params:
                    param.requires_grad_(False)

                try:
                    aux_bg_feats = expert_outputs[2]
                    if self.syn_aux_head_detach_near_neg_bg:
                        aux_bg_feats = tuple(
                            feat.detach() for feat in aux_bg_feats)
                    bg_results = self._syn_aux_bbox_forward(
                        aux_bg_feats, near_neg_rois)
                    bg_cls_score = bg_results['cls_score']
                    bg_raw = F.cross_entropy(
                        bg_cls_score, bg_labels, reduction='mean')
                    bg_fgprob = F.softmax(
                        bg_cls_score, dim=1
                    )[:, :num_classes].sum(dim=1).mean()
                finally:
                    for param, requires_grad in zip(
                            shared_params, old_requires_grad):
                        param.requires_grad_(requires_grad)

                bg_loss = (
                    ref.new_tensor(self.lambda_syn_aux_head_near_neg_bg)
                    * bg_raw)
                log_vars.update({
                    'loss_syn_aux_head_near_neg_bg': bg_loss,
                    'syn_aux_head_near_neg_bg_raw': bg_raw.detach(),
                    'syn_aux_head_near_neg_fgprob': bg_fgprob.detach(),
                })

            if use_main_bg:
                shared_head = getattr(self.roi_head, 'shared_head', None)
                shared_params = (
                    list(shared_head.parameters())
                    if shared_head is not None else [])
                main_bbox_params = list(self.roi_head.bbox_head.parameters())
                frozen_params = shared_params + main_bbox_params
                old_requires_grad = [
                    param.requires_grad for param in frozen_params
                ]
                for param in frozen_params:
                    param.requires_grad_(False)

                try:
                    main_bg_results = self.roi_head._bbox_forward(
                        expert_outputs[2], near_neg_rois)
                    main_bg_cls_score = main_bg_results['cls_score']
                    main_bg_raw = F.cross_entropy(
                        main_bg_cls_score, bg_labels, reduction='mean')
                    main_bg_fgprob = F.softmax(
                        main_bg_cls_score, dim=1
                    )[:, :num_classes].sum(dim=1).mean()
                finally:
                    for param, requires_grad in zip(
                            frozen_params, old_requires_grad):
                        param.requires_grad_(requires_grad)

                main_bg_loss = (
                    ref.new_tensor(
                        self.lambda_syn_aux_head_main_near_neg_bg)
                    * main_bg_raw)
                log_vars.update({
                    'loss_syn_aux_head_main_near_neg_bg': main_bg_loss,
                    'syn_aux_head_main_near_neg_bg_raw':
                    main_bg_raw.detach(),
                    'syn_aux_head_main_near_neg_fgprob':
                    main_bg_fgprob.detach(),
                })

            return log_vars

        candidate_count = sum(int(boxes.shape[0]) for boxes in boxes_by_img)
        candidate_count_tensor = ref.new_tensor(float(candidate_count)).detach()
        if candidate_count <= 0:
            log_vars = {
                'syn_aux_head_count': ref.new_zeros(()).detach(),
                'syn_aux_head_candidate_count': candidate_count_tensor,
                'syn_aux_head_copied_from_main': copied.detach(),
            }
            log_vars.update(compute_near_neg_bg_logs())
            return log_vars

        rois_all = bbox2roi(boxes_by_img)
        labels_all = torch.cat(labels_by_img, dim=0)
        ious_all = torch.cat(ious_by_img, dim=0)

        with torch.no_grad():
            uni_vis_results = self.roi_head._bbox_forward(
                expert_outputs[0], rois_all)
            uni_lwir_results = self.roi_head._bbox_forward(
                expert_outputs[1], rois_all)
            p_uni_vis = F.softmax(
                uni_vis_results['cls_score'], dim=1
            )[:, :num_classes].sum(dim=1)
            p_uni_lwir = F.softmax(
                uni_lwir_results['cls_score'], dim=1
            )[:, :num_classes].sum(dim=1)
            mode_masks = self._split_syn_aux_roi_modes(
                p_uni_vis, p_uni_lwir)
            keep = mode_masks['comp_hard']
            single_strong = mode_masks['single_strong']
            single_vis = mode_masks['single_vis']
            single_lwir = mode_masks['single_lwir']
            keep_idx = torch.nonzero(keep, as_tuple=False).flatten()

        comp_count_tensor = ref.new_tensor(
            float(int(keep.sum().item()))).detach()
        single_count_tensor = ref.new_tensor(
            float(int(single_strong.sum().item()))).detach()
        single_vis_count_tensor = ref.new_tensor(
            float(int(single_vis.sum().item()))).detach()
        single_lwir_count_tensor = ref.new_tensor(
            float(int(single_lwir.sum().item()))).detach()
        if torch.any(keep):
            comp_fg_uni_vis = p_uni_vis[keep].mean().detach()
            comp_fg_uni_lwir = p_uni_lwir[keep].mean().detach()
        else:
            comp_fg_uni_vis = zero
            comp_fg_uni_lwir = zero
        if torch.any(single_strong):
            single_fg_uni_vis = p_uni_vis[single_strong].mean().detach()
            single_fg_uni_lwir = p_uni_lwir[single_strong].mean().detach()
        else:
            single_fg_uni_vis = zero
            single_fg_uni_lwir = zero
        if torch.any(single_vis):
            single_vis_fg_uni_vis = p_uni_vis[single_vis].mean().detach()
        else:
            single_vis_fg_uni_vis = zero
        if torch.any(single_lwir):
            single_lwir_fg_uni_lwir = p_uni_lwir[single_lwir].mean().detach()
        else:
            single_lwir_fg_uni_lwir = zero
        split_logs = {
            'syn_aux_head_comp_hard_count': comp_count_tensor,
            'syn_aux_head_single_strong_count': single_count_tensor,
            'syn_aux_head_single_vis_count': single_vis_count_tensor,
            'syn_aux_head_single_lwir_count': single_lwir_count_tensor,
            'syn_aux_head_comp_hard_fg_uni_vis': comp_fg_uni_vis,
            'syn_aux_head_comp_hard_fg_uni_lwir': comp_fg_uni_lwir,
            'syn_aux_head_single_strong_fg_uni_vis': single_fg_uni_vis,
            'syn_aux_head_single_vis_fg_uni_vis': single_vis_fg_uni_vis,
            'syn_aux_head_single_lwir_fg_uni_lwir': single_lwir_fg_uni_lwir,
        }

        if keep_idx.numel() > max_rois:
            keep_idx = keep_idx[:max_rois]
        count = int(keep_idx.numel())
        count_tensor = ref.new_tensor(float(count)).detach()
        if count <= 0:
            log_vars = {
                'syn_aux_head_count': count_tensor,
                'syn_aux_head_candidate_count': candidate_count_tensor,
                'syn_aux_head_main_fg_uni_vis': p_uni_vis.mean().detach(),
                'syn_aux_head_main_fg_uni_lwir': p_uni_lwir.mean().detach(),
                'syn_aux_head_copied_from_main': copied.detach(),
            }
            log_vars.update(split_logs)
            log_vars.update(compute_near_neg_bg_logs())
            return log_vars

        rois = rois_all[keep_idx]
        labels = labels_all[keep_idx]
        ious = ious_all[keep_idx]
        p_uni_vis_sel = p_uni_vis[keep_idx]
        p_uni_lwir_sel = p_uni_lwir[keep_idx]
        margin = ref.new_tensor(self.syn_aux_head_gap_margin)
        main_margin = ref.new_tensor(self.syn_aux_head_main_gap_margin)
        use_main_probe = (
            self.lambda_syn_aux_head_main_sem != 0.0
            or self.lambda_syn_aux_head_main_gap != 0.0)

        shared_head = getattr(self.roi_head, 'shared_head', None)
        shared_params = (
            list(shared_head.parameters()) if shared_head is not None else [])
        main_bbox_params = list(self.roi_head.bbox_head.parameters())
        frozen_params = shared_params + main_bbox_params
        old_requires_grad = [
            param.requires_grad for param in frozen_params
        ]
        for param in frozen_params:
            param.requires_grad_(False)

        try:
            full_results = self._syn_aux_bbox_forward(
                expert_outputs[2], rois)
            cls_score_full = full_results['cls_score']
            sem_raw = F.cross_entropy(
                cls_score_full, labels, reduction='mean')
            p_full = F.softmax(
                cls_score_full, dim=1)[:, :num_classes].sum(dim=1)

            with torch.no_grad():
                syn_vismask = tuple(
                    feat.detach() for feat in perturbed_outputs['vis'][2])
                syn_lwirmask = tuple(
                    feat.detach() for feat in perturbed_outputs['lwir'][2])
                vis_results = self._syn_aux_bbox_forward(
                    syn_vismask, rois)
                lwir_results = self._syn_aux_bbox_forward(
                    syn_lwirmask, rois)
                p_vis = F.softmax(
                    vis_results['cls_score'], dim=1
                )[:, :num_classes].sum(dim=1)
                p_lwir = F.softmax(
                    lwir_results['cls_score'], dim=1
                )[:, :num_classes].sum(dim=1)

            if use_main_probe:
                main_full_results = self.roi_head._bbox_forward(
                    expert_outputs[2], rois)
                main_cls_score_full = main_full_results['cls_score']
                main_sem_raw = F.cross_entropy(
                    main_cls_score_full, labels, reduction='mean')
                main_p_full = F.softmax(
                    main_cls_score_full, dim=1
                )[:, :num_classes].sum(dim=1)

                with torch.no_grad():
                    main_vis_results = self.roi_head._bbox_forward(
                        syn_vismask, rois)
                    main_lwir_results = self.roi_head._bbox_forward(
                        syn_lwirmask, rois)
                    main_cls_score_lwir = main_lwir_results['cls_score']
                    main_lwir_ce = F.cross_entropy(
                        main_cls_score_lwir, labels, reduction='mean')
                    main_p_vis = F.softmax(
                        main_vis_results['cls_score'], dim=1
                    )[:, :num_classes].sum(dim=1)
                    main_p_lwir = F.softmax(
                        main_cls_score_lwir, dim=1
                    )[:, :num_classes].sum(dim=1)
        finally:
            for param, requires_grad in zip(
                    frozen_params, old_requires_grad):
                param.requires_grad_(requires_grad)

        gap_vis = p_full - p_vis
        gap_lwir = p_full - p_lwir
        gap_raw = 0.5 * (
            torch.relu(margin - gap_vis).mean()
            + torch.relu(margin - gap_lwir).mean())
        loss = (
            ref.new_tensor(self.lambda_syn_aux_head_sem) * sem_raw
            + ref.new_tensor(self.lambda_syn_aux_head_gap) * gap_raw)
        comp_fg_floor_raw = ref.new_zeros(())
        if self.lambda_syn_aux_head_comp_fg_floor != 0.0:
            comp_fg_floor = ref.new_tensor(self.syn_aux_head_comp_fg_floor)
            comp_fg_floor_raw = torch.relu(comp_fg_floor - p_full).mean()
            loss = (
                loss
                + ref.new_tensor(self.lambda_syn_aux_head_comp_fg_floor)
                * comp_fg_floor_raw)

        log_vars = {
            'loss_syn_aux_head': loss,
            'syn_aux_head_count': count_tensor,
            'syn_aux_head_candidate_count': candidate_count_tensor,
            'syn_aux_head_sem_raw': sem_raw.detach(),
            'syn_aux_head_gap_raw': gap_raw.detach(),
            'syn_aux_head_fg_full': p_full.mean().detach(),
            'syn_aux_head_fg_lwirmask': p_lwir.mean().detach(),
            'syn_aux_head_gap_lwirmask': gap_lwir.mean().detach(),
            'syn_aux_head_comp_fg_floor_raw': (
                comp_fg_floor_raw.detach()),
            'syn_aux_head_comp_fg_floor_value': p_full.mean().detach(),
            'syn_aux_head_main_fg_uni_vis': p_uni_vis_sel.mean().detach(),
            'syn_aux_head_main_fg_uni_lwir': p_uni_lwir_sel.mean().detach(),
            'syn_aux_head_iou': ious.mean().detach(),
        }
        log_vars.update(split_logs)
        log_vars.update(compute_near_neg_bg_logs())
        if use_main_probe:
            main_gap_vis = main_p_full - main_p_vis
            main_gap_lwir = main_p_full - main_p_lwir
            main_gap_raw = 0.5 * (
                torch.relu(main_margin - main_gap_vis).mean()
                + torch.relu(main_margin - main_gap_lwir).mean())
            log_vars['loss_syn_aux_head'] = (
                log_vars['loss_syn_aux_head']
                + ref.new_tensor(self.lambda_syn_aux_head_main_sem)
                * main_sem_raw
                + ref.new_tensor(self.lambda_syn_aux_head_main_gap)
                * main_gap_raw)
            log_vars.update({
                'syn_aux_head_main_sem_raw': main_sem_raw.detach(),
                'syn_aux_head_main_fg_full': main_p_full.mean().detach(),
                'syn_aux_head_main_fg_lwirmask': main_p_lwir.mean().detach(),
                'syn_aux_head_main_ce_vs_lwir': (
                    main_sem_raw - main_lwir_ce).detach(),
            })
        return log_vars

    @staticmethod
    def _masked_mean_scalar(value: Tensor,
                            mask: Tensor,
                            eps: float = 1e-6) -> Tuple[Tensor, bool]:
        denom = mask.sum()
        if denom.item() <= eps:
            return value.new_zeros(()), False
        return (value * mask).sum() / denom.clamp(min=eps), True

    def _response_delta_on_regions(self, normal_logits, perturbed_logits,
                                   masks) -> Tuple[Tensor, Tensor]:
        total = normal_logits[0].new_zeros(())
        valid_count = normal_logits[0].new_zeros(())
        for normal, perturbed, mask in zip(normal_logits, perturbed_logits,
                                           masks):
            normal_resp = torch.sigmoid(normal)
            perturbed_resp = torch.sigmoid(perturbed)
            delta, valid = self._masked_mean_scalar(
                (normal_resp - perturbed_resp).abs(), mask)
            if valid:
                total = total + delta
                valid_count = valid_count + 1.0
        if valid_count.item() <= 0:
            return total, valid_count
        return total / valid_count.clamp(min=1.0), valid_count

    @staticmethod
    def _stable_unique_idx(perturbed_modality: str) -> int:
        if perturbed_modality == 'vis':
            return 1
        if perturbed_modality == 'lwir':
            return 0
        raise ValueError(
            f"perturbed_modality must be 'vis' or 'lwir', got "
            f'{perturbed_modality!r}')

    @staticmethod
    def _active_unique_idx(perturbed_modality: str) -> int:
        if perturbed_modality == 'vis':
            return 0
        if perturbed_modality == 'lwir':
            return 1
        raise ValueError(
            f"perturbed_modality must be 'vis' or 'lwir', got "
            f'{perturbed_modality!r}')

    def _compute_role_response_loss_from_logits(self, normal_logits,
                                                perturbed_logits,
                                                perturbed_modality: str,
                                                target_masks, ref):
        active_idx = self._active_unique_idx(perturbed_modality)
        stable_idx = self._stable_unique_idx(perturbed_modality)
        active_delta, valid_count = self._response_delta_on_regions(
            normal_logits[active_idx], perturbed_logits[active_idx],
            target_masks)
        stable_delta, _ = self._response_delta_on_regions(
            normal_logits[stable_idx], perturbed_logits[stable_idx],
            target_masks)
        syn_delta, _ = self._response_delta_on_regions(
            normal_logits[2], perturbed_logits[2], target_masks)
        red_delta, _ = self._response_delta_on_regions(
            normal_logits[3], perturbed_logits[3], target_masks)

        zero = syn_delta.new_zeros(())
        if valid_count.item() <= 0:
            raw = zero
            syn_floor = zero
            syn_rank = zero
            active_floor = zero
            stable_penalty = zero
            red_penalty = zero
        else:
            margin = syn_delta.new_tensor(self.role_aux_margin)
            active_floor = self._smooth_hinge(
                margin - active_delta, self.role_aux_tau)
            syn_floor = self._smooth_hinge(
                margin - syn_delta, self.role_aux_tau)
            syn_rank = self._smooth_hinge(
                torch.maximum(red_delta, stable_delta)
                + syn_delta.new_tensor(self.role_aux_rank_margin)
                - syn_delta,
                self.role_aux_tau)
            stable_penalty = self._smooth_hinge(
                stable_delta - margin * 0.5, self.role_aux_tau)
            red_penalty = self._smooth_hinge(
                red_delta - margin * 0.5, self.role_aux_tau)
            raw = (
                active_floor + syn_floor + syn_rank
                + 0.5 * stable_penalty + 0.5 * red_penalty)

        cur_w = self._current_role_weight()
        return {
            'loss_role_aux': ref.new_tensor(cur_w) * raw,
            'role_aux_weight': ref.new_tensor(cur_w).detach(),
            'role_aux_valid': valid_count.detach(),
            'role_uni_active_delta': active_delta.detach(),
            'role_uni_stable_delta': stable_delta.detach(),
            'role_syn_delta': syn_delta.detach(),
            'role_red_stable_delta': red_delta.detach(),
            'role_active_floor': active_floor.detach(),
            'role_syn_floor': syn_floor.detach(),
            'role_syn_rank': syn_rank.detach(),
            'role_total_raw': raw.detach(),
        }

    def _compute_role_response_loss(self, expert_outputs, perturbed_outputs,
                                    perturbed_modality: str, region_masks):
        if len(self.role_heads) == 0:
            return {}
        if region_masks is None:
            raise ValueError(
                'role response supervision requires spatial GT region masks.')

        levels = self.role_aux_feat_levels
        target_masks = tuple(region_masks['target'][level] for level in levels)
        normal_logits = [
            self.role_heads[k](expert_outputs[k], levels=levels)
            for k in range(len(EXPERT_NAMES))
        ]
        ref = expert_outputs[0][0]

        if isinstance(perturbed_outputs, dict):
            per_mask = {}
            for modality in ('vis', 'lwir'):
                outputs = perturbed_outputs[modality]
                perturbed_logits = [
                    self.role_heads[k](outputs[k], levels=levels)
                    for k in range(len(EXPERT_NAMES))
                ]
                per_mask[modality] = (
                    self._compute_role_response_loss_from_logits(
                        normal_logits, perturbed_logits, modality,
                        target_masks, ref))

            out = {}
            keys = per_mask['vis'].keys()
            for key in keys:
                out[key] = 0.5 * (per_mask['vis'][key]
                                  + per_mask['lwir'][key])
            for modality in ('vis', 'lwir'):
                suffix = f'{modality}mask'
                for key, value in per_mask[modality].items():
                    if key.startswith('loss_') or key == 'role_aux_weight':
                        continue
                    out[f'{key}_{suffix}'] = value.detach()
            return out

        perturbed_logits = [
            self.role_heads[k](perturbed_outputs[k], levels=levels)
            for k in range(len(EXPERT_NAMES))
        ]
        return self._compute_role_response_loss_from_logits(
            normal_logits, perturbed_logits, perturbed_modality, target_masks,
            ref)

    def _compute_gt_gate_stats(self, weights, batch_data_samples,
                               input_hw: Tuple[int, int]):
        """Log GT-box-region gate usage for medium and large objects."""
        if not self._is_spatial_weights(weights):
            return {}

        input_h, input_w = input_hw
        num_experts = weights[0].shape[1]
        expert_names = self._active_expert_names()
        if len(expert_names) != num_experts:
            raise ValueError(
                f'Fusion exposes {len(expert_names)} expert names but gate '
                f'has {num_experts} channels.')
        device = weights[0].device
        sums = {
            'medium': weights[0].new_zeros(num_experts),
            'large': weights[0].new_zeros(num_experts),
        }
        counts = {
            'medium': weights[0].new_zeros(()),
            'large': weights[0].new_zeros(()),
        }

        for gate in weights:
            gate_detached = gate.detach()
            _, _, gate_h, gate_w = gate_detached.shape
            scale_x = float(gate_w) / float(input_w)
            scale_y = float(gate_h) / float(input_h)

            for batch_idx, data_sample in enumerate(batch_data_samples):
                bboxes = self._get_bboxes_tensor(data_sample)
                if bboxes is None:
                    continue
                bboxes = bboxes.to(device=device, dtype=gate_detached.dtype)
                box_w = (bboxes[:, 2] - bboxes[:, 0]).clamp(min=0)
                box_h = (bboxes[:, 3] - bboxes[:, 1]).clamp(min=0)
                areas = box_w * box_h
                bin_masks = {
                    'medium': (areas >= 32.0 ** 2) & (areas < 96.0 ** 2),
                    'large': areas >= 96.0 ** 2,
                }

                for bin_name, mask in bin_masks.items():
                    inds = torch.nonzero(mask, as_tuple=False).flatten()
                    for ind in inds.tolist():
                        box = bboxes[ind]
                        x1 = int(torch.floor(box[0] * scale_x).clamp(
                            0, gate_w - 1).item())
                        y1 = int(torch.floor(box[1] * scale_y).clamp(
                            0, gate_h - 1).item())
                        x2 = int(torch.ceil(box[2] * scale_x).clamp(
                            1, gate_w).item())
                        y2 = int(torch.ceil(box[3] * scale_y).clamp(
                            1, gate_h).item())
                        if x2 <= x1:
                            x2 = min(x1 + 1, gate_w)
                        if y2 <= y1:
                            y2 = min(y1 + 1, gate_h)
                        patch = gate_detached[
                            batch_idx, :, y1:y2, x1:x2]
                        if patch.numel() == 0:
                            continue
                        sums[bin_name] = sums[bin_name] + patch.mean(
                            dim=(1, 2))
                        counts[bin_name] = counts[bin_name] + 1.0

        stats = {}
        for bin_name in ('medium', 'large'):
            stats[f'gate_gt_{bin_name}_count'] = counts[bin_name].detach()
            if counts[bin_name].item() <= 0:
                continue
            values = sums[bin_name] / counts[bin_name].clamp(min=1.0)
            for k, name in enumerate(expert_names):
                stats[f'gate_gt_{bin_name}_{name}'] = values[k].detach()
        return stats

    @staticmethod
    def _as_bbox_tensor(boxes):
        if hasattr(boxes, 'tensor'):
            boxes = boxes.tensor
        if not isinstance(boxes, torch.Tensor):
            return None
        if boxes.numel() == 0:
            return boxes.new_zeros((0, 4))
        return boxes[:, :4]

    @staticmethod
    def _bbox_pair_iou(boxes_a: Tensor, boxes_b: Tensor,
                       eps: float = 1e-6) -> Tensor:
        if boxes_a.numel() == 0 or boxes_b.numel() == 0:
            return boxes_a.new_zeros((0,))
        lt = torch.maximum(boxes_a[:, :2], boxes_b[:, :2])
        rb = torch.minimum(boxes_a[:, 2:], boxes_b[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, 0] * wh[:, 1]
        area_a = (
            (boxes_a[:, 2] - boxes_a[:, 0]).clamp(min=0)
            * (boxes_a[:, 3] - boxes_a[:, 1]).clamp(min=0))
        area_b = (
            (boxes_b[:, 2] - boxes_b[:, 0]).clamp(min=0)
            * (boxes_b[:, 3] - boxes_b[:, 1]).clamp(min=0))
        return inter / (area_a + area_b - inter).clamp(min=eps)

    @staticmethod
    def _bbox_max_iou(boxes: Tensor, gt_boxes: Tensor,
                      eps: float = 1e-6) -> Tensor:
        if boxes.numel() == 0:
            return boxes.new_zeros((0,))
        if gt_boxes is None or gt_boxes.numel() == 0:
            return boxes.new_zeros((boxes.shape[0],))
        lt = torch.maximum(boxes[:, None, :2], gt_boxes[None, :, :2])
        rb = torch.minimum(boxes[:, None, 2:], gt_boxes[None, :, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[..., 0] * wh[..., 1]
        area_boxes = (
            (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
            * (boxes[:, 3] - boxes[:, 1]).clamp(min=0))
        area_gt = (
            (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp(min=0)
            * (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp(min=0))
        union = area_boxes[:, None] + area_gt[None, :] - inter
        return (inter / union.clamp(min=eps)).max(dim=1).values

    def _sample_roi_gate_values(self, weights, boxes: Tensor, batch_idx: int,
                                input_hw: Tuple[int, int],
                                detach: bool = True):
        if boxes.numel() == 0:
            return None

        boxes = boxes.detach()
        batch_col = boxes.new_full((boxes.shape[0], 1), float(batch_idx))
        rois = torch.cat([batch_col, boxes], dim=1)
        if hasattr(self.roi_head.bbox_roi_extractor, 'map_roi_levels'):
            levels = self.roi_head.bbox_roi_extractor.map_roi_levels(
                rois, len(weights))
        else:
            levels = boxes.new_zeros((boxes.shape[0],), dtype=torch.long)

        input_h, input_w = input_hw
        values = []
        for box_idx in range(boxes.shape[0]):
            level = int(levels[box_idx].item())
            gate = weights[level].detach() if detach else weights[level]
            _, _, gate_h, gate_w = gate.shape
            scale_x = float(gate_w) / float(input_w)
            scale_y = float(gate_h) / float(input_h)
            box = boxes[box_idx]
            x1 = int(torch.floor(box[0] * scale_x).clamp(
                0, gate_w - 1).item())
            y1 = int(torch.floor(box[1] * scale_y).clamp(
                0, gate_h - 1).item())
            x2 = int(torch.ceil(box[2] * scale_x).clamp(
                1, gate_w).item())
            y2 = int(torch.ceil(box[3] * scale_y).clamp(
                1, gate_h).item())
            if x2 <= x1:
                x2 = min(x1 + 1, gate_w)
            if y2 <= y1:
                y2 = min(y1 + 1, gate_h)
            patch = gate[batch_idx, :, y1:y2, x1:x2]
            if patch.numel() == 0:
                continue
            values.append(patch.mean(dim=(1, 2)))

        if not values:
            return None
        return torch.stack(values, dim=0)

    @staticmethod
    def _new_roi_gate_bucket(ref: Tensor):
        return {
            'count': ref.new_zeros(()),
            'iou_sum': ref.new_zeros(()),
            'expert_sum': ref.new_zeros(len(EXPERT_NAMES)),
            'syn_margin_sum': ref.new_zeros(()),
            'syn_top_sum': ref.new_zeros(()),
        }

    def _add_roi_gate_bucket(self, buckets, name: str, weights, boxes: Tensor,
                             ious: Tensor, batch_idx: int,
                             input_hw: Tuple[int, int]):
        if boxes is None or boxes.numel() == 0:
            return
        values = self._sample_roi_gate_values(
            weights, boxes, batch_idx, input_hw)
        if values is None:
            return
        count = values.new_tensor(float(values.shape[0]))
        buckets[name]['count'] = buckets[name]['count'] + count
        buckets[name]['expert_sum'] = (
            buckets[name]['expert_sum'] + values.sum(dim=0))
        buckets[name]['iou_sum'] = buckets[name]['iou_sum'] + ious.sum()
        best_other = torch.stack(
            (values[:, 0], values[:, 1], values[:, 3]), dim=1).max(
                dim=1).values
        buckets[name]['syn_margin_sum'] = (
            buckets[name]['syn_margin_sum'] + (values[:, 2] - best_other).sum())
        buckets[name]['syn_top_sum'] = (
            buckets[name]['syn_top_sum']
            + (values.argmax(dim=1) == 2).to(values.dtype).sum())

    def _compute_roi_gate_diag(self, weights, fused_feats, rpn_results_list,
                               batch_data_samples, expert_outputs,
                               input_hw: Tuple[int, int]):
        if not self.roi_gate_diag or not self._is_spatial_weights(weights):
            return {}

        ref = weights[0]
        buckets = {
            name: self._new_roi_gate_bucket(ref)
            for name in (
                'hard_pos', 'comp_hard', 'single_strong', 'single_vis',
                'single_lwir', 'neg', 'near_neg')
        }
        sampling_results = self._sample_aux_roi_results(
            fused_feats, copy.deepcopy(rpn_results_list), batch_data_samples)
        num_classes = int(self.roi_head.bbox_head.num_classes)

        for batch_idx, result in enumerate(sampling_results):
            pos_boxes = self._as_bbox_tensor(result.pos_priors)
            pos_gt_boxes = self._as_bbox_tensor(result.pos_gt_bboxes)
            if pos_boxes is not None and pos_boxes.numel() > 0:
                pos_ious = self._bbox_pair_iou(pos_boxes, pos_gt_boxes)
                hard_mask = pos_ious < pos_ious.new_tensor(
                    self.roi_gate_diag_hard_iou_thr)
                self._add_roi_gate_bucket(
                    buckets, 'hard_pos', weights, pos_boxes[hard_mask],
                    pos_ious[hard_mask], batch_idx, input_hw)
                hard_boxes = pos_boxes[hard_mask]
                hard_ious = pos_ious[hard_mask]
                if (expert_outputs is not None
                        and hard_boxes is not None
                        and hard_boxes.numel() > 0):
                    roi_inds = hard_boxes.new_full(
                        (hard_boxes.shape[0], 1), float(batch_idx))
                    rois = torch.cat((roi_inds, hard_boxes), dim=1)
                    with torch.no_grad():
                        uni_vis_results = self.roi_head._bbox_forward(
                            expert_outputs[0], rois)
                        uni_lwir_results = self.roi_head._bbox_forward(
                            expert_outputs[1], rois)
                        p_uni_vis = F.softmax(
                            uni_vis_results['cls_score'], dim=1
                        )[:, :num_classes].sum(dim=1)
                        p_uni_lwir = F.softmax(
                            uni_lwir_results['cls_score'], dim=1
                        )[:, :num_classes].sum(dim=1)
                        mode_masks = self._split_syn_aux_roi_modes(
                            p_uni_vis, p_uni_lwir)
                        comp_mask = mode_masks['comp_hard']
                        single_mask = mode_masks['single_strong']
                        single_vis_mask = mode_masks['single_vis']
                        single_lwir_mask = mode_masks['single_lwir']
                    self._add_roi_gate_bucket(
                        buckets, 'comp_hard', weights, hard_boxes[comp_mask],
                        hard_ious[comp_mask], batch_idx, input_hw)
                    self._add_roi_gate_bucket(
                        buckets, 'single_strong', weights,
                        hard_boxes[single_mask], hard_ious[single_mask],
                        batch_idx, input_hw)
                    self._add_roi_gate_bucket(
                        buckets, 'single_vis', weights,
                        hard_boxes[single_vis_mask],
                        hard_ious[single_vis_mask], batch_idx, input_hw)
                    self._add_roi_gate_bucket(
                        buckets, 'single_lwir', weights,
                        hard_boxes[single_lwir_mask],
                        hard_ious[single_lwir_mask], batch_idx, input_hw)

            neg_boxes = self._as_bbox_tensor(result.neg_priors)
            gt_boxes = self._get_bboxes_tensor(batch_data_samples[batch_idx])
            if gt_boxes is not None:
                gt_boxes = gt_boxes.to(
                    device=weights[0].device, dtype=weights[0].dtype)
            if neg_boxes is not None and neg_boxes.numel() > 0:
                neg_ious = self._bbox_max_iou(
                    neg_boxes.to(device=weights[0].device,
                                 dtype=weights[0].dtype),
                    gt_boxes)
                neg_boxes = neg_boxes.to(
                    device=weights[0].device, dtype=weights[0].dtype)
                max_neg = int(self.syn_aux_head_max_near_neg_rois)
                diag_neg_boxes = neg_boxes[:max_neg]
                diag_neg_ious = neg_ious[:max_neg]
                self._add_roi_gate_bucket(
                    buckets, 'neg', weights, diag_neg_boxes, diag_neg_ious,
                    batch_idx, input_hw)
                near_mask = neg_ious >= neg_ious.new_tensor(
                    self.roi_gate_diag_near_neg_iou_thr)
                self._add_roi_gate_bucket(
                    buckets, 'near_neg', weights, neg_boxes[near_mask],
                    neg_ious[near_mask], batch_idx, input_hw)

        logs = {}
        for bucket_name, bucket in buckets.items():
            count = bucket['count']
            logs[f'roi_diag_{bucket_name}_count'] = count.detach()
            if count.item() <= 0:
                continue
            denom = count.clamp(min=1.0)
            logs[f'roi_diag_{bucket_name}_iou'] = (
                bucket['iou_sum'] / denom).detach()
            experts = bucket['expert_sum'] / denom
            for expert_idx, expert_name in enumerate(EXPERT_NAMES):
                if expert_name == 'red':
                    continue
                logs[f'roi_diag_{bucket_name}_{expert_name}'] = (
                    experts[expert_idx].detach())
            logs[f'roi_diag_{bucket_name}_syn_margin'] = (
                bucket['syn_margin_sum'] / denom).detach()
            logs[f'roi_diag_{bucket_name}_syn_top_frac'] = (
                bucket['syn_top_sum'] / denom).detach()
        return logs

    def _compute_syn_boundary_gate_margin_loss(
            self, weights, fused_feats, rpn_results_list, batch_data_samples,
            expert_outputs, input_hw: Tuple[int, int]):
        if (not self.use_syn_boundary_gate_margin_loss
                or not self._is_spatial_weights(weights)):
            return {}

        use_neg = self.lambda_syn_boundary_gate_neg != 0.0
        use_near_neg = self.lambda_syn_boundary_gate_near_neg != 0.0
        use_single = self.lambda_syn_boundary_gate_single != 0.0
        use_comp_floor = self.lambda_syn_boundary_gate_comp_floor != 0.0
        use_comp_rel = self.lambda_syn_boundary_gate_comp_rel != 0.0
        use_comp = use_comp_floor or use_comp_rel
        if not use_neg and not use_near_neg and not use_single and not use_comp:
            return {}

        ref = weights[0]
        zero = ref.new_zeros(())
        num_classes = int(self.roi_head.bbox_head.num_classes)
        margin = ref.new_tensor(self.syn_boundary_gate_margin)
        comp_floor = ref.new_tensor(self.syn_boundary_gate_comp_floor)
        comp_rel_ratio = ref.new_tensor(self.syn_boundary_gate_comp_rel_ratio)
        comp_rel_margin = ref.new_tensor(
            self.syn_boundary_gate_comp_rel_margin)
        max_neg = int(self.syn_boundary_gate_max_neg_rois)
        max_near_neg = int(self.syn_boundary_gate_max_near_neg_rois)
        max_single = int(self.syn_boundary_gate_max_single_rois)
        max_comp = int(self.syn_boundary_gate_max_comp_rois)
        neg_count = 0
        near_count = 0
        single_count = 0
        comp_count = 0
        neg_terms = []
        neg_margins = []
        near_terms = []
        near_margins = []
        single_terms = []
        single_margins = []
        comp_terms = []
        comp_values = []
        comp_rel_terms = []
        comp_rel_margins = []
        comp_rel_targets = []

        sampling_results = self._sample_aux_roi_results(
            fused_feats, copy.deepcopy(rpn_results_list), batch_data_samples)

        for batch_idx, result in enumerate(sampling_results):
            use_pos_modes = (
                (use_single and single_count < max_single)
                or (use_comp and comp_count < max_comp))
            if use_pos_modes:
                pos_boxes = self._as_bbox_tensor(result.pos_priors)
                pos_gt_boxes = self._as_bbox_tensor(result.pos_gt_bboxes)
                if (pos_boxes is not None and pos_gt_boxes is not None
                        and pos_boxes.numel() > 0
                        and pos_gt_boxes.numel() > 0):
                    pos_boxes = pos_boxes.to(
                        device=ref.device, dtype=ref.dtype)
                    pos_gt_boxes = pos_gt_boxes.to(
                        device=ref.device, dtype=ref.dtype)
                    pos_ious = self._bbox_pair_iou(pos_boxes, pos_gt_boxes)
                    hard_mask = pos_ious < pos_ious.new_tensor(
                        self.roi_gate_diag_hard_iou_thr)
                    hard_boxes = pos_boxes[hard_mask]
                    if hard_boxes.numel() > 0:
                        roi_inds = hard_boxes.new_full(
                            (hard_boxes.shape[0], 1), float(batch_idx))
                        rois = torch.cat((roi_inds, hard_boxes), dim=1)
                        with torch.no_grad():
                            uni_vis_results = self.roi_head._bbox_forward(
                                expert_outputs[0], rois)
                            uni_lwir_results = self.roi_head._bbox_forward(
                                expert_outputs[1], rois)
                            p_uni_vis = F.softmax(
                                uni_vis_results['cls_score'], dim=1
                            )[:, :num_classes].sum(dim=1)
                            p_uni_lwir = F.softmax(
                                uni_lwir_results['cls_score'], dim=1
                            )[:, :num_classes].sum(dim=1)
                            mode_masks = self._split_syn_aux_roi_modes(
                                p_uni_vis, p_uni_lwir)

                        if use_comp and comp_count < max_comp:
                            cur_boxes = hard_boxes[mode_masks['comp_hard']]
                            if cur_boxes.numel() > 0:
                                remaining = max_comp - comp_count
                                cur_boxes = cur_boxes[:remaining]
                                values = self._sample_roi_gate_values(
                                    weights, cur_boxes, batch_idx, input_hw,
                                    detach=False)
                                if values is not None:
                                    syn_values = values[:, 2]
                                    comp_values.append(syn_values.detach())
                                    if use_comp_floor:
                                        comp_terms.append(
                                            torch.relu(comp_floor
                                                       - syn_values))
                                    if use_comp_rel:
                                        best_uni = torch.stack(
                                            (values[:, 0], values[:, 1]),
                                            dim=1).max(dim=1).values
                                        target = (comp_rel_ratio * best_uni
                                                  + comp_rel_margin)
                                        rel_margin = syn_values - target
                                        comp_rel_terms.append(
                                            torch.relu(-rel_margin))
                                        comp_rel_margins.append(
                                            rel_margin.detach())
                                        comp_rel_targets.append(
                                            target.detach())
                                    comp_count += int(values.shape[0])

                        single_masks = (
                            ('single_vis', mode_masks['single_vis'], 0),
                            ('single_lwir', mode_masks['single_lwir'], 1),
                        )
                        for _, mask, target_idx in single_masks:
                            if single_count >= max_single:
                                break
                            cur_boxes = hard_boxes[mask]
                            if cur_boxes.numel() == 0:
                                continue
                            remaining = max_single - single_count
                            cur_boxes = cur_boxes[:remaining]
                            values = self._sample_roi_gate_values(
                                weights, cur_boxes, batch_idx, input_hw,
                                detach=False)
                            if values is None:
                                continue
                            cur_margin = values[:, 2] - values[:, target_idx]
                            single_terms.append(
                                torch.relu(cur_margin + margin))
                            single_margins.append(cur_margin.detach())
                            single_count += int(values.shape[0])

            use_neg_modes = (
                (use_neg and neg_count < max_neg)
                or (use_near_neg and near_count < max_near_neg))
            if use_neg_modes:
                neg_boxes = self._as_bbox_tensor(result.neg_priors)
                gt_boxes = self._get_bboxes_tensor(
                    batch_data_samples[batch_idx])
                if gt_boxes is not None:
                    gt_boxes = gt_boxes.to(device=ref.device, dtype=ref.dtype)
                if neg_boxes is not None and neg_boxes.numel() > 0:
                    neg_boxes = neg_boxes.to(device=ref.device,
                                             dtype=ref.dtype)
                    neg_ious = self._bbox_max_iou(neg_boxes, gt_boxes)
                    near_iou_thr = neg_ious.new_tensor(
                        self.roi_gate_diag_near_neg_iou_thr)
                    if use_neg and neg_count < max_neg:
                        neg_mask = neg_ious < near_iou_thr
                        cur_boxes = neg_boxes[neg_mask]
                        if cur_boxes.numel() > 0:
                            remaining = max_neg - neg_count
                            cur_boxes = cur_boxes[:remaining]
                            values = self._sample_roi_gate_values(
                                weights, cur_boxes, batch_idx, input_hw,
                                detach=False)
                            if values is not None:
                                best_other = torch.stack(
                                    (values[:, 0], values[:, 1], values[:, 3]),
                                    dim=1).max(dim=1).values
                                cur_margin = values[:, 2] - best_other
                                neg_terms.append(
                                    torch.relu(cur_margin + margin))
                                neg_margins.append(cur_margin.detach())
                                neg_count += int(values.shape[0])

                    if use_near_neg and near_count < max_near_neg:
                        near_mask = neg_ious >= near_iou_thr
                        cur_boxes = neg_boxes[near_mask]
                        if cur_boxes.numel() == 0:
                            continue
                        remaining = max_near_neg - near_count
                        cur_boxes = cur_boxes[:remaining]
                        values = self._sample_roi_gate_values(
                            weights, cur_boxes, batch_idx, input_hw,
                            detach=False)
                        if values is not None:
                            best_other = torch.stack(
                                (values[:, 0], values[:, 1], values[:, 3]),
                                dim=1).max(dim=1).values
                            cur_margin = values[:, 2] - best_other
                            near_terms.append(
                                torch.relu(cur_margin + margin))
                            near_margins.append(cur_margin.detach())
                            near_count += int(values.shape[0])

        logs = {}
        if use_neg:
            if neg_terms:
                neg_raw = torch.cat(neg_terms).mean()
                neg_margin = torch.cat(neg_margins).mean()
            else:
                neg_raw = zero
                neg_margin = zero
            logs.update({
                'loss_syn_boundary_gate_neg':
                ref.new_tensor(self.lambda_syn_boundary_gate_neg)
                * neg_raw,
                'syn_boundary_gate_neg_raw': neg_raw.detach(),
                'syn_boundary_gate_neg_count':
                ref.new_tensor(float(neg_count)).detach(),
                'syn_boundary_gate_neg_margin': neg_margin.detach(),
            })
        if use_near_neg:
            if near_terms:
                near_raw = torch.cat(near_terms).mean()
                near_margin = torch.cat(near_margins).mean()
            else:
                near_raw = zero
                near_margin = zero
            logs.update({
                'loss_syn_boundary_gate_near_neg':
                ref.new_tensor(self.lambda_syn_boundary_gate_near_neg)
                * near_raw,
                'syn_boundary_gate_near_neg_raw': near_raw.detach(),
                'syn_boundary_gate_near_neg_count':
                ref.new_tensor(float(near_count)).detach(),
                'syn_boundary_gate_near_neg_margin':
                near_margin.detach(),
            })
        if use_single:
            if single_terms:
                single_raw = torch.cat(single_terms).mean()
                single_margin = torch.cat(single_margins).mean()
            else:
                single_raw = zero
                single_margin = zero
            logs.update({
                'loss_syn_boundary_gate_single':
                ref.new_tensor(self.lambda_syn_boundary_gate_single)
                * single_raw,
                'syn_boundary_gate_single_raw': single_raw.detach(),
                'syn_boundary_gate_single_count':
                ref.new_tensor(float(single_count)).detach(),
                'syn_boundary_gate_single_margin': single_margin.detach(),
            })
        if use_comp_floor:
            if comp_terms:
                comp_raw = torch.cat(comp_terms).mean()
                comp_value = torch.cat(comp_values).mean()
            else:
                comp_raw = zero
                comp_value = zero
            logs.update({
                'loss_syn_boundary_gate_comp_floor':
                ref.new_tensor(self.lambda_syn_boundary_gate_comp_floor)
                * comp_raw,
                'syn_boundary_gate_comp_floor_raw': comp_raw.detach(),
                'syn_boundary_gate_comp_floor_count':
                ref.new_tensor(float(comp_count)).detach(),
                'syn_boundary_gate_comp_floor_value':
                comp_value.detach(),
            })
        if use_comp_rel:
            if comp_rel_terms:
                comp_rel_raw = torch.cat(comp_rel_terms).mean()
                comp_rel_margin_value = torch.cat(comp_rel_margins).mean()
                comp_rel_target = torch.cat(comp_rel_targets).mean()
                comp_value = torch.cat(comp_values).mean()
            else:
                comp_rel_raw = zero
                comp_rel_margin_value = zero
                comp_rel_target = zero
                comp_value = zero
            logs.update({
                'loss_syn_boundary_gate_comp_rel':
                ref.new_tensor(self.lambda_syn_boundary_gate_comp_rel)
                * comp_rel_raw,
                'syn_boundary_gate_comp_rel_raw':
                comp_rel_raw.detach(),
                'syn_boundary_gate_comp_rel_count':
                ref.new_tensor(float(comp_count)).detach(),
                'syn_boundary_gate_comp_rel_margin':
                comp_rel_margin_value.detach(),
                'syn_boundary_gate_comp_rel_target':
                comp_rel_target.detach(),
                'syn_boundary_gate_comp_rel_value':
                comp_value.detach(),
            })
        return logs

    def _add_syn_probe_bucket(self, buckets, name: str, batch_idx: int,
                              boxes: Tensor, ious: Tensor, labels: Tensor):
        if boxes is None or boxes.numel() == 0:
            return
        buckets[name]['boxes'][batch_idx] = boxes
        buckets[name]['ious'][batch_idx] = ious
        buckets[name]['labels'][batch_idx] = labels

    def _compute_syn_aux_roi_probe(self, expert_outputs, fused_feats,
                                   rpn_results_list, batch_data_samples):
        if not self.syn_aux_roi_probe:
            return {}

        with torch.no_grad():
            ref = expert_outputs[0][0]
            num_imgs = len(batch_data_samples)
            bg_label = int(self.roi_head.bbox_head.num_classes)
            num_classes = int(self.roi_head.bbox_head.num_classes)
            buckets = {
                name: {
                    'boxes': [ref.new_zeros((0, 4)) for _ in range(num_imgs)],
                    'ious': [ref.new_zeros((0,)) for _ in range(num_imgs)],
                    'labels': [
                        ref.new_zeros((0,), dtype=torch.long)
                        for _ in range(num_imgs)
                    ],
                }
                for name in (
                    'comp_hard', 'single_strong', 'single_vis',
                    'single_lwir', 'near_neg')
            }
            sampling_results = self._sample_aux_roi_results(
                fused_feats, copy.deepcopy(rpn_results_list),
                batch_data_samples)

            for batch_idx, result in enumerate(sampling_results):
                pos_boxes = self._as_bbox_tensor(result.pos_priors)
                pos_gt_boxes = self._as_bbox_tensor(result.pos_gt_bboxes)
                if pos_boxes is not None and pos_boxes.numel() > 0:
                    pos_boxes = pos_boxes.to(device=ref.device,
                                             dtype=ref.dtype)
                    pos_gt_boxes = pos_gt_boxes.to(device=ref.device,
                                                   dtype=ref.dtype)
                    pos_ious = self._bbox_pair_iou(pos_boxes, pos_gt_boxes)
                    pos_labels = getattr(result, 'pos_gt_labels', None)
                    if pos_labels is None:
                        pos_labels = pos_boxes.new_zeros(
                            (pos_boxes.shape[0],), dtype=torch.long)
                    else:
                        pos_labels = pos_labels.to(
                            device=ref.device, dtype=torch.long)
                    hard_mask = pos_ious < pos_ious.new_tensor(
                        self.roi_gate_diag_hard_iou_thr)
                    hard_boxes = pos_boxes[hard_mask]
                    hard_ious = pos_ious[hard_mask]
                    hard_labels = pos_labels[hard_mask]
                    if hard_boxes.numel() > 0:
                        roi_inds = hard_boxes.new_full(
                            (hard_boxes.shape[0], 1), float(batch_idx))
                        rois = torch.cat((roi_inds, hard_boxes), dim=1)
                        uni_vis_results = self.roi_head._bbox_forward(
                            expert_outputs[0], rois)
                        uni_lwir_results = self.roi_head._bbox_forward(
                            expert_outputs[1], rois)
                        p_uni_vis = F.softmax(
                            uni_vis_results['cls_score'], dim=1
                        )[:, :num_classes].sum(dim=1)
                        p_uni_lwir = F.softmax(
                            uni_lwir_results['cls_score'], dim=1
                        )[:, :num_classes].sum(dim=1)
                        mode_masks = self._split_syn_aux_roi_modes(
                            p_uni_vis, p_uni_lwir)
                        comp_mask = mode_masks['comp_hard']
                        single_mask = mode_masks['single_strong']
                        single_vis_mask = mode_masks['single_vis']
                        single_lwir_mask = mode_masks['single_lwir']
                        self._add_syn_probe_bucket(
                            buckets, 'comp_hard', batch_idx,
                            hard_boxes[comp_mask], hard_ious[comp_mask],
                            hard_labels[comp_mask])
                        self._add_syn_probe_bucket(
                            buckets, 'single_strong', batch_idx,
                            hard_boxes[single_mask], hard_ious[single_mask],
                            hard_labels[single_mask])
                        self._add_syn_probe_bucket(
                            buckets, 'single_vis', batch_idx,
                            hard_boxes[single_vis_mask],
                            hard_ious[single_vis_mask],
                            hard_labels[single_vis_mask])
                        self._add_syn_probe_bucket(
                            buckets, 'single_lwir', batch_idx,
                            hard_boxes[single_lwir_mask],
                            hard_ious[single_lwir_mask],
                            hard_labels[single_lwir_mask])

                neg_boxes = self._as_bbox_tensor(result.neg_priors)
                gt_boxes = self._get_bboxes_tensor(batch_data_samples[batch_idx])
                if gt_boxes is not None:
                    gt_boxes = gt_boxes.to(device=ref.device, dtype=ref.dtype)
                if neg_boxes is not None and neg_boxes.numel() > 0:
                    neg_boxes = neg_boxes.to(device=ref.device,
                                             dtype=ref.dtype)
                    neg_ious = self._bbox_max_iou(neg_boxes, gt_boxes)
                    neg_labels = neg_boxes.new_full(
                        (neg_boxes.shape[0],), bg_label, dtype=torch.long)

                    near_mask = neg_ious >= neg_ious.new_tensor(
                        self.roi_gate_diag_near_neg_iou_thr)
                    self._add_syn_probe_bucket(
                        buckets, 'near_neg', batch_idx, neg_boxes[near_mask],
                        neg_ious[near_mask], neg_labels[near_mask])

            logs = {}
            for bucket_name, bucket in buckets.items():
                count = sum(int(boxes.shape[0]) for boxes in bucket['boxes'])
                logs[f'syn_probe_{bucket_name}_count'] = ref.new_tensor(
                    float(count)).detach()
                if count == 0:
                    continue

                rois = bbox2roi(bucket['boxes'])
                labels = torch.cat(bucket['labels'], dim=0)
                ious = torch.cat(bucket['ious'], dim=0)
                logs[f'syn_probe_{bucket_name}_iou'] = ious.mean().detach()

                ce_values = []
                fg_values = []
                for expert_idx, expert_feats in enumerate(expert_outputs):
                    bbox_results = self.roi_head._bbox_forward(
                        expert_feats, rois)
                    cls_score = bbox_results['cls_score']
                    cls_ce = F.cross_entropy(
                        cls_score, labels, reduction='mean')
                    prob = F.softmax(cls_score, dim=1)
                    fg_prob = prob[:, :num_classes].sum(dim=1).mean()
                    ce_values.append(cls_ce)
                    fg_values.append(fg_prob)

                    expert_name = EXPERT_NAMES[expert_idx]
                    if expert_name in ('uni_lwir', 'syn'):
                        logs[f'syn_probe_{bucket_name}_{expert_name}_ce'] = (
                            cls_ce.detach())
                        logs[
                            f'syn_probe_{bucket_name}_{expert_name}_fgprob'
                        ] = fg_prob.detach()

                other_ce = torch.stack(
                    (ce_values[0], ce_values[1], ce_values[3]))
                other_fg = torch.stack(
                    (fg_values[0], fg_values[1], fg_values[3]))
                logs[f'syn_probe_{bucket_name}_syn_ce_vs_lwir'] = (
                    ce_values[2] - ce_values[1]).detach()
                logs[f'syn_probe_{bucket_name}_syn_ce_vs_best_other'] = (
                    ce_values[2] - other_ce.min()).detach()
                logs[f'syn_probe_{bucket_name}_syn_fg_vs_lwir'] = (
                    fg_values[2] - fg_values[1]).detach()
                logs[f'syn_probe_{bucket_name}_syn_fg_vs_best_other'] = (
                    fg_values[2] - other_fg.max()).detach()
        return logs

    def _zero_probe_loss(self):
        for param in self.parameters():
            if param.requires_grad and param.numel() > 0:
                return param.reshape(-1)[0] * 0.0
        return torch.zeros(())

    def _extract_fusion_inputs(self, img_vis: Tensor,
                               img_lwir: Tensor) -> Tuple[tuple, tuple, dict]:
        img_vis_unique, img_lwir_unique = self._extract_unique_features(
            img_vis, img_lwir)
        x_vis = self.extract_feat_vis(img_vis_unique)
        x_lwir = self.extract_feat_lwir(img_lwir_unique)
        fusion_kwargs = {}
        if getattr(self.FeaFusion, 'requires_raw_inputs', False):
            fusion_kwargs = dict(img_vis=img_vis, img_lwir=img_lwir)
        return x_vis, x_lwir, fusion_kwargs

    def _fusion_aux_logs(self) -> dict:
        if not hasattr(self.FeaFusion, 'get_aux_refiner_logs'):
            return {}
        return self.FeaFusion.get_aux_refiner_logs()

    def _compute_extra_fusion_losses(self, batch_inputs: dict,
                                     batch_data_samples: SampleList,
                                     x_vis: tuple,
                                     x_lwir: tuple,
                                     fused_x: tuple,
                                     weights,
                                     expert_outputs,
                                     fusion_kwargs: dict) -> dict:
        return {}

    def _loss_syn_aux_roi_probe_only(self, batch_inputs: Tensor,
                                     batch_data_samples: SampleList) -> dict:
        logs: dict = {}
        with torch.no_grad():
            img_vis = batch_inputs['img_vis']
            img_lwir = batch_inputs['img_lwir']

            x_vis, x_lwir, fusion_kwargs = self._extract_fusion_inputs(
                img_vis, img_lwir)
            x, _, _, weights, expert_outputs = self.FeaFusion(
                x_vis, x_lwir, **fusion_kwargs)
            logs.update(self._fusion_aux_logs())
            if self.target_refiner is not None:
                x = self.target_refiner.refine(x)

            rpn_data_samples = copy.deepcopy(batch_data_samples)
            for ds in rpn_data_samples:
                ds.gt_instances.labels = torch.zeros_like(
                    ds.gt_instances.labels)

            if self.with_rpn:
                proposal_cfg = self.train_cfg.get(
                    'rpn_proposal', self.test_cfg.rpn)
                _, rpn_results_list = self.rpn_head.loss_and_predict(
                    x, rpn_data_samples, proposal_cfg=proposal_cfg)
            else:
                assert batch_data_samples[0].get('proposals', None) is not None
                rpn_results_list = [ds.proposals for ds in batch_data_samples]

            logs.update(self._compute_syn_aux_roi_probe(
                expert_outputs, x, rpn_results_list, batch_data_samples))
            logs.update(self._compute_roi_gate_diag(
                weights, x, rpn_results_list, batch_data_samples,
                expert_outputs,
                img_vis.shape[-2:]))

            gate_stats = self._compute_spatial_gate_stats(weights)
            if gate_stats is not None:
                logs.update(gate_stats)
                logs.update(self._compute_gt_gate_stats(
                    weights, batch_data_samples, img_vis.shape[-2:]))

        logs['loss_probe_zero'] = self._zero_probe_loss()
        return logs

    def _compute_cada_roi_losses(self, batch_inputs, x, expert_outputs,
                                 weights, rpn_results_list,
                                 batch_data_samples):
        """Dispatch the optional CADA RoI training loss.

        Subclasses may override this narrow hook to provide train-only
        teacher features without duplicating the detector's full loss path.
        """
        return self.roi_head.loss_with_cada_features(
            x, expert_outputs, weights, rpn_results_list,
            batch_data_samples)

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> dict:
        if self.syn_aux_roi_probe_only:
            return self._loss_syn_aux_roi_probe_only(
                batch_inputs, batch_data_samples)

        losses: dict = dict()
        img_vis = batch_inputs['img_vis']
        img_lwir = batch_inputs['img_lwir']

        x_vis, x_lwir, fusion_kwargs = self._extract_fusion_inputs(
            img_vis, img_lwir)

        feature_interaction_enabled = (
            self.enable_feature_interaction_loss
            and self.interaction_loss_fn is not None)
        needs_perturbed = (
            feature_interaction_enabled or len(self.role_heads) > 0
            or self.use_region_syn_rpn_sem_loss
            or self.use_syn_aux_head_loss
            or self.use_roi_logits_interaction_loss)
        perturbed_modality = None
        perturbed_outputs = None
        if (needs_perturbed and self.paired_perturbation
                and hasattr(self.FeaFusion, 'forward_with_perturbations')):
            fusion_outputs, perturbed_outputs = (
                self.FeaFusion.forward_with_perturbations(
                    x_vis, x_lwir, **fusion_kwargs))
            x, mi_loss_vis, mi_loss_lwir, weights, expert_outputs = (
                fusion_outputs)
            perturbed_modality = 'paired'
        else:
            x, mi_loss_vis, mi_loss_lwir, weights, expert_outputs = \
                self.FeaFusion(x_vis, x_lwir, **fusion_kwargs)

        losses.update(self._fusion_aux_logs())
        losses.update(self._compute_extra_fusion_losses(
            batch_inputs, batch_data_samples, x_vis, x_lwir, x, weights,
            expert_outputs, fusion_kwargs))

        if self.mi_loss_weight > 0 and getattr(
                self.FeaFusion, 'use_mi', False):
            losses['loss_MI_vis'] = self.mi_loss_weight * mi_loss_vis
            losses['loss_MI_lwir'] = self.mi_loss_weight * mi_loss_lwir

        needs_region_masks = (
            len(self.role_heads) > 0
            or (feature_interaction_enabled and getattr(
                self.interaction_loss_fn, 'accepts_region_masks', False))
            or self.use_region_syn_rpn_sem_loss)
        region_masks = (
            self._build_target_region_masks(
                weights, batch_data_samples, img_vis.shape[-2:])
            if needs_region_masks else None)

        if needs_perturbed and perturbed_outputs is None:
            if self.paired_perturbation:
                perturbed_modality = 'paired'
                perturbed_outputs = self.FeaFusion.forward_perturbed_pair(
                    x_vis, x_lwir)
            else:
                perturbed_modality = self._next_perturb_modality()
                perturbed_outputs = self.FeaFusion.forward_perturbed(
                    x_vis, x_lwir, perturbed_modality)

        if feature_interaction_enabled:
            int_kwargs = {}
            if getattr(
                    self.interaction_loss_fn, 'accepts_expert_weights',
                    False):
                int_kwargs['expert_weights'] = weights
            if getattr(
                    self.interaction_loss_fn, 'accepts_region_masks', False):
                if region_masks is None:
                    raise ValueError(
                        'interaction loss requires region masks but none were '
                        'built.')
                mask_key = (
                    'target_scale_aware'
                    if getattr(self.interaction_loss_fn,
                               'region_scale_aware', False)
                    else 'target')
                int_kwargs['target_region_masks'] = region_masks[mask_key]
            int_loss = self.interaction_loss_fn(
                expert_outputs, perturbed_outputs,
                perturbed_modality,
                **int_kwargs)
            cur_inter_w = float(self.interaction_loss_weight)

            if isinstance(int_loss, dict) and 'total_main' in int_loss:
                # InteractionLossV3 contract.
                total_main = int_loss['total_main']
                decorr = int_loss['decorr']
                cur_decorr_w = self.decorr_weight
                losses['loss_interaction'] = cur_inter_w * total_main
                losses['loss_decorr'] = cur_decorr_w * decorr
                losses['interaction_total_raw'] = total_main.detach()
                losses['interaction_decorr_raw'] = decorr.detach()
                losses['interaction_weight_cur'] = (
                    total_main.detach().new_tensor(cur_inter_w))
                losses['decorr_weight_cur'] = (
                    total_main.detach().new_tensor(cur_decorr_w))
                for k, v in int_loss.items():
                    if k in ('total_main', 'decorr'):
                        continue
                    if torch.is_tensor(v):
                        metric_key = self._interaction_metric_key(k)
                        if metric_key in KEY_INTERACTION_LOGS:
                            losses[metric_key] = v.detach()
            elif isinstance(int_loss, dict):
                # Interaction losses may return a single weighted total.
                total = int_loss['total']
                losses['loss_interaction'] = cur_inter_w * total
                losses['interaction_total_raw'] = total.detach()
                losses['interaction_weight_cur'] = (
                    total.detach().new_tensor(cur_inter_w))
                for k in ('uni_active', 'uni_stable', 'syn', 'red'):
                    if k in int_loss:
                        losses[f'interaction_{k}'] = int_loss[k].detach()
            else:
                losses['loss_interaction'] = cur_inter_w * int_loss
                losses['interaction_weight_cur'] = (
                    int_loss.detach().new_tensor(cur_inter_w))
        elif self.interaction_loss_fn is not None:
            device_ref = expert_outputs[0][0]
            losses['interaction_feature_disabled'] = (
                device_ref.new_tensor(1.0).detach())
            if self.decorr_weight == 0.0:
                losses['loss_decorr'] = device_ref.new_zeros(())

        if len(self.role_heads) > 0:
            losses.update(self._compute_role_response_loss(
                expert_outputs, perturbed_outputs, perturbed_modality,
                region_masks))

        losses.update(self._compute_region_syn_rpn_sem_loss(
            expert_outputs, perturbed_outputs, region_masks))

        if self.target_refiner is not None:
            x, target_refine_losses = self.target_refiner.loss_and_refine(
                x, batch_data_samples, img_vis.shape[-2:])
            losses.update(target_refine_losses)

        # Prepare binary (foreground-only) labels for RPN targets. Aux heads
        # reuse this exact data_samples so their supervision mirrors the
        # main RPN.
        rpn_data_samples = copy.deepcopy(batch_data_samples)
        for ds in rpn_data_samples:
            ds.gt_instances.labels = torch.zeros_like(
                ds.gt_instances.labels)

        # Auxiliary RPN-lite heads on each expert feature tuple.
        if len(self.aux_heads) > 0:
            cur_w = self._current_aux_weight()
            device_ref = expert_outputs[0][0]
            for k, expert_k in enumerate(expert_outputs):
                aux_losses_k = self.aux_heads[k].loss(
                    expert_k, rpn_data_samples)
                for key, val in aux_losses_k.items():
                    suffix = key[len('loss_'):] if key.startswith(
                        'loss_') else key
                    new_key = f'loss_aux_{EXPERT_NAMES[k]}_{suffix}'
                    losses[new_key] = self._scale_aux_loss(val, cur_w)
            losses['aux_head_weight'] = device_ref.new_tensor(cur_w).detach()

        # Main RPN and RoI heads.
        if self.with_rpn:
            if self.use_cada_roi and self.cada_freeze_detector:
                # A frozen post-hoc adapter must see the same proposals as
                # deployment. RPN supervision has no trainable recipient and
                # would only add detached losses and target-generation cost.
                with torch.no_grad():
                    rpn_results_list = self.rpn_head.predict(
                        x, batch_data_samples, rescale=False)
            else:
                proposal_cfg = self.train_cfg.get(
                    'rpn_proposal', self.test_cfg.rpn)
                rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
                    x, rpn_data_samples, proposal_cfg=proposal_cfg)
                for key in list(rpn_losses.keys()):
                    if 'loss' in key and 'rpn' not in key:
                        rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
                losses.update(rpn_losses)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            rpn_results_list = [
                ds.proposals for ds in batch_data_samples
            ]

        losses.update(self._compute_syn_aux_head_loss(
            expert_outputs, perturbed_outputs, x, rpn_results_list,
            batch_data_samples))
        losses.update(self._compute_syn_boundary_gate_margin_loss(
            weights, x, rpn_results_list, batch_data_samples, expert_outputs,
            img_vis.shape[-2:]))
        losses.update(self._compute_roi_logits_interaction_loss(
            expert_outputs, perturbed_outputs, x, rpn_results_list,
            batch_data_samples))
        aux_roi_rpn_results_list = (
            copy.deepcopy(rpn_results_list)
            if len(self.aux_roi_heads) > 0 else None)
        losses.update(self._compute_roi_gate_diag(
            weights, x, rpn_results_list, batch_data_samples, expert_outputs,
            img_vis.shape[-2:]))
        losses.update(self._compute_syn_aux_roi_probe(
            expert_outputs, x, rpn_results_list, batch_data_samples))
        if self.use_cada_roi:
            roi_losses = self._compute_cada_roi_losses(
                batch_inputs, x, expert_outputs, weights,
                rpn_results_list, batch_data_samples)
        else:
            roi_losses = self.roi_head.loss(
                x, rpn_results_list, batch_data_samples)
        losses.update(roi_losses)
        if aux_roi_rpn_results_list is not None:
            losses.update(self._compute_aux_roi_losses(
                expert_outputs, aux_roi_rpn_results_list, batch_data_samples,
                x))

        gate_stats = self._compute_spatial_gate_stats(weights)
        if gate_stats is not None:
            losses.update(gate_stats)
            losses.update(self._compute_gt_gate_stats(
                weights, batch_data_samples, img_vis.shape[-2:]))

        return losses

    def _override_diag_gate_weights(self, weights):
        cfg = self.diag_gate_override
        if cfg is None:
            return weights
        mode = cfg.get('mode', None)
        if mode in (None, 'normal'):
            return weights
        if not self._is_spatial_weights(weights):
            raise TypeError('diag_gate_override requires spatial gate weights.')

        level_cfg = cfg.get('level')
        if level_cfg == 'all':
            levels = range(len(weights))
        else:
            level = int(level_cfg)
            if level < 0 or level >= len(weights):
                raise ValueError(
                    f'diag_gate_override level {level} out of range for '
                    f'{len(weights)} levels.')
            levels = (level, )

        expert_idx = None
        if mode in ('onehot', 'one_hot'):
            expert = cfg.get('expert')
            if isinstance(expert, str):
                if expert not in EXPERT_NAMES:
                    raise ValueError(
                        f'Unknown diag expert {expert!r}; expected one of '
                        f'{EXPERT_NAMES}.')
                expert_idx = EXPERT_NAMES.index(expert)
            else:
                expert_idx = int(expert)

        new_weights = list(weights)
        for level in levels:
            base = weights[level]
            if mode == 'uniform':
                override = base.new_full(
                    base.shape, 1.0 / float(base.shape[1]))
            elif mode in ('onehot', 'one_hot'):
                if expert_idx < 0 or expert_idx >= base.shape[1]:
                    raise ValueError(
                        f'diag expert index {expert_idx} out of range for '
                        f'{base.shape[1]} experts.')
                override = torch.zeros_like(base)
                override[:, expert_idx:expert_idx + 1, :, :] = 1.0
            else:
                raise ValueError(
                    f'Unsupported diag_gate_override mode {mode!r}.')
            new_weights[level] = override
        return tuple(new_weights)

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        assert self.with_bbox, 'Bbox head must be implemented.'
        img_vis = batch_inputs['img_vis']
        img_lwir = batch_inputs['img_lwir']

        x_vis, x_lwir, fusion_kwargs = self._extract_fusion_inputs(
            img_vis, img_lwir)

        x, _, _, weights, expert_outputs = self.FeaFusion(
            x_vis, x_lwir, **fusion_kwargs)
        if self.diag_gate_override is not None:
            weights = self._override_diag_gate_weights(weights)
            x = self.FeaFusion._weighted_fuse(expert_outputs, weights)
        if self.target_refiner is not None:
            x = self.target_refiner.refine(x)

        if batch_data_samples[0].get('proposals', None) is None:
            rpn_results_list = self.rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            rpn_results_list = [
                ds.proposals for ds in batch_data_samples
            ]

        if self.use_cada_roi:
            results_list = self.roi_head.predict_with_cada_features(
                x, expert_outputs, weights, rpn_results_list,
                batch_data_samples, rescale=rescale)
        else:
            results_list = self.roi_head.predict(
                x, rpn_results_list, batch_data_samples, rescale=rescale)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)

        if weights is not None:
            self._attach_expert_weights(batch_data_samples, weights)
        return batch_data_samples
