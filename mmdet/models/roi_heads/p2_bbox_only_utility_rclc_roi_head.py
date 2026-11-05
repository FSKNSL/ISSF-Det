from typing import List, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from mmdet.registry import MODELS
from mmdet.structures import DetDataSample
from mmdet.structures.bbox import bbox2roi
from mmdet.utils import InstanceList
from .p2_counterfactual_credit_rclc_roi_head import (
    P2CounterfactualCreditRCLCRoIHead)


@MODELS.register_module()
class P2BBoxOnlyUtilityRCLCRoIHead(
        P2CounterfactualCreditRCLCRoIHead):
    """Route only P2 bbox deltas with counterfactual localization utility.

    The factual classification logits are always preserved.  Four signed
    utilities estimate the localization effect of half-strength
    leave-one-expert-out interventions; no-op has fixed utility zero.  An
    intervention is used only when its predicted utility reaches a positive
    margin.  Counterfactual targets are detached from the detector path.
    """

    def __init__(self,
                 utility_hidden_dim: int = 256,
                 utility_margin: float = 0.02,
                 threshold_gain_scale: float = 0.10,
                 unmatched_utility: float = -0.10,
                 positive_utility_weight: float = 2.0,
                 negative_utility_weight: float = 1.0,
                 utility_loss_beta: float = 0.05,
                 utility_loss_weight: float = 1.0,
                 utility_loss_warmup_iters: int = 500,
                 build_utility_router: bool = True,
                 *args,
                 **kwargs) -> None:
        super().__init__(
            credit_loss_weight=utility_loss_weight,
            credit_loss_warmup_iters=utility_loss_warmup_iters,
            build_credit_router=build_utility_router,
            *args,
            **kwargs)
        if utility_hidden_dim <= 0:
            raise ValueError('utility_hidden_dim must be positive.')
        if utility_margin < 0.0:
            raise ValueError('utility_margin must be non-negative.')
        if threshold_gain_scale <= 0.0:
            raise ValueError('threshold_gain_scale must be positive.')
        if unmatched_utility >= 0.0:
            raise ValueError('unmatched_utility must be negative.')
        if positive_utility_weight <= 0.0:
            raise ValueError('positive_utility_weight must be positive.')
        if negative_utility_weight <= 0.0:
            raise ValueError('negative_utility_weight must be positive.')
        if utility_loss_beta <= 0.0:
            raise ValueError('utility_loss_beta must be positive.')

        self.utility_margin = float(utility_margin)
        self.threshold_gain_scale = float(threshold_gain_scale)
        self.unmatched_utility = float(unmatched_utility)
        self.positive_utility_weight = float(positive_utility_weight)
        self.negative_utility_weight = float(negative_utility_weight)
        self.utility_loss_beta = float(utility_loss_beta)

        if build_utility_router:
            del self.credit_router
            num_cls_outputs = self.bbox_head.num_classes + 1
            num_bbox_outputs = (
                4 if self.bbox_head.reg_class_agnostic
                else 4 * self.bbox_head.num_classes)
            utility_input_dim = (
                self.bbox_head.in_channels
                + self.num_experts
                + num_cls_outputs
                + self.num_experts * num_bbox_outputs)
            self.utility_encoder = nn.Sequential(
                nn.Linear(utility_input_dim, utility_hidden_dim),
                nn.ReLU(inplace=True))
            self.utility_head = nn.Linear(
                utility_hidden_dim, self.num_experts)
            nn.init.zeros_(self.utility_head.weight)
            nn.init.zeros_(self.utility_head.bias)

    def cada_trainable_parameters(self):
        return tuple(self.utility_encoder.parameters()) + tuple(
            self.utility_head.parameters())

    def train_cada_components(self) -> None:
        self.eval()
        self.utility_encoder.train()
        self.utility_head.train()

    def _bbox_only_action_context(
            self, x: Tuple[Tensor], expert_outputs, weights, rois: Tensor,
            base_forward=None):
        if base_forward is None:
            base_forward = self._bbox_forward(x, rois)
        bbox_outputs = [base_forward['bbox_pred']]
        for expert_index in range(self.num_experts):
            candidate = self._counterfactual_features(
                x, expert_outputs, weights, expert_index)
            candidate_forward = self._bbox_forward(candidate, rois)
            bbox_outputs.append(candidate_forward['bbox_pred'])

        bbox_predictions = torch.stack(tuple(bbox_outputs), dim=1)
        base_probabilities = F.softmax(
            base_forward['cls_score'], dim=1)
        bbox_deltas = (
            bbox_predictions[:, 1:] - bbox_predictions[:, :1]).clamp(
                min=-self.response_delta_clip,
                max=self.response_delta_clip) / self.response_delta_clip
        pooled = F.adaptive_avg_pool2d(
            base_forward['bbox_feats'], 1).flatten(1)
        utility_features = torch.cat((
            pooled,
            self._roi_gate_values(weights, rois),
            base_probabilities,
            bbox_deltas.flatten(1),
        ), dim=1)
        return bbox_predictions, utility_features

    def _localization_utility_targets(self, iou: Tensor):
        thresholds = iou.new_tensor(self.coco_iou_thresholds)
        pass_counts = (
            iou[:, :, None] >= thresholds[None, None, :]).sum(dim=2)
        pass_gain = (
            pass_counts[:, 1:] - pass_counts[:, :1]).to(iou.dtype)
        iou_gain = iou[:, 1:] - iou[:, :1]
        utilities = self.threshold_gain_scale * pass_gain + iou_gain
        return utilities, pass_counts

    def _predict_action_indices(self, utility_features: Tensor):
        encoded = self.utility_encoder(utility_features)
        predicted_utilities = self.utility_head(encoded)
        best_utility, best_relative_action = predicted_utilities.max(dim=1)
        action_indices = torch.zeros_like(best_relative_action)
        active = best_utility >= self.utility_margin
        action_indices[active] = best_relative_action[active] + 1
        return action_indices, predicted_utilities

    def _utility_loss(self, predicted: Tensor, target: Tensor) -> Tensor:
        element_loss = F.smooth_l1_loss(
            predicted, target, beta=self.utility_loss_beta,
            reduction='none')
        weights = torch.ones_like(target)
        weights = torch.where(
            target >= self.utility_margin,
            weights.new_tensor(self.positive_utility_weight), weights)
        weights = torch.where(
            target < 0.0,
            weights.new_tensor(self.negative_utility_weight), weights)
        return (element_loss * weights).mean()

    def _empty_utility_logs(self) -> dict:
        zero = (
            self.utility_encoder[0].weight.sum()
            + self.utility_head.weight.sum()) * 0.0
        return {
            'loss_rclc_bbox_utility': zero,
            'rclc_utility_weight': zero.detach(),
            'rclc_candidate_count': zero.detach(),
            'rclc_matched_count': zero.detach(),
            'rclc_target_nonnoop_ratio': zero.detach(),
            'rclc_pred_nonnoop_ratio': zero.detach(),
            'rclc_action_accuracy': zero.detach(),
            'rclc_positive_action_accuracy': zero.detach(),
            'rclc_utility_mae': zero.detach(),
            'rclc_oracle_iou_gain': zero.detach(),
            'rclc_pred_iou_gain': zero.detach(),
            'rclc_pred_harm_ratio': zero.detach(),
        }

    def loss_with_cada_features(
            self, x: Tuple[Tensor], expert_outputs, weights,
            rpn_results_list: InstanceList,
            batch_data_samples: List[DetDataSample]) -> dict:
        proposals = [result.bboxes for result in rpn_results_list]
        rois = bbox2roi(proposals)
        if rois.shape[0] == 0:
            return self._empty_utility_logs()
        levels = self.bbox_roi_extractor.map_roi_levels(
            rois, self.bbox_roi_extractor.num_inputs)
        proposal_scores = self._proposal_scores(rpn_results_list)
        selected, matched, gt_boxes, gt_labels = self._select_training_rois(
            rois, levels, proposal_scores, batch_data_samples)
        if selected.numel() == 0:
            return self._empty_utility_logs()

        selected_rois = rois[selected]
        with torch.no_grad():
            bbox_predictions, utility_features = (
                self._bbox_only_action_context(
                    x, expert_outputs, weights, selected_rois))
            target_utilities = selected_rois.new_full(
                (selected.numel(), self.num_experts),
                self.unmatched_utility)
            candidate_iou = selected_rois.new_zeros(
                (selected.numel(), self.num_actions))
            if matched.any():
                matched_boxes = self._decode_matched_boxes(
                    selected_rois[matched], bbox_predictions[matched],
                    gt_labels[matched], batch_data_samples)
                matched_iou = self._aligned_iou(
                    matched_boxes, gt_boxes[matched])
                matched_utilities, _ = (
                    self._localization_utility_targets(matched_iou))
                target_utilities[matched] = matched_utilities
                candidate_iou[matched] = matched_iou

        encoded = self.utility_encoder(utility_features)
        predicted_utilities = self.utility_head(encoded)
        raw_loss = self._utility_loss(
            predicted_utilities, target_utilities)
        current_weight = self._current_credit_weight()
        loss = raw_loss * current_weight
        predicted_actions, _ = self._predict_action_indices(
            utility_features)
        target_best_utility, target_relative_action = (
            target_utilities.max(dim=1))
        target_actions = torch.zeros_like(target_relative_action)
        target_active = target_best_utility >= self.utility_margin
        target_actions[target_active] = (
            target_relative_action[target_active] + 1)

        if matched.any():
            rows = torch.arange(matched.sum(), device=rois.device)
            matched_iou = candidate_iou[matched]
            matched_targets = target_actions[matched]
            matched_predictions = predicted_actions[matched]
            oracle_iou_gain = (
                matched_iou[rows, matched_targets]
                - matched_iou[:, 0]).mean()
            predicted_iou_delta = (
                matched_iou[rows, matched_predictions]
                - matched_iou[:, 0])
            predicted_iou_gain = predicted_iou_delta.mean()
            harm_ratio = (
                predicted_iou_delta < -1e-8).float().mean()
        else:
            zero = loss.detach() * 0.0
            oracle_iou_gain = zero
            predicted_iou_gain = zero
            harm_ratio = zero
        positive = target_actions != 0
        positive_accuracy = (
            (predicted_actions[positive]
             == target_actions[positive]).float().mean()
            if positive.any() else loss.detach() * 0.0)

        return {
            'loss_rclc_bbox_utility': loss,
            'rclc_utility_raw': raw_loss.detach(),
            'rclc_utility_weight': loss.new_tensor(
                current_weight).detach(),
            'rclc_candidate_count': selected.new_tensor(
                float(selected.numel()), dtype=torch.float32).detach(),
            'rclc_matched_count': matched.sum().to(torch.float32).detach(),
            'rclc_target_nonnoop_ratio': positive.float().mean().detach(),
            'rclc_pred_nonnoop_ratio': (
                predicted_actions != 0).float().mean().detach(),
            'rclc_action_accuracy': (
                predicted_actions
                == target_actions).float().mean().detach(),
            'rclc_positive_action_accuracy': positive_accuracy.detach(),
            'rclc_utility_mae': (
                predicted_utilities
                - target_utilities).abs().mean().detach(),
            'rclc_oracle_iou_gain': oracle_iou_gain.detach(),
            'rclc_pred_iou_gain': predicted_iou_gain.detach(),
            'rclc_pred_harm_ratio': harm_ratio.detach(),
        }

    def predict_with_cada_features(
            self, x: Tuple[Tensor], expert_outputs, weights,
            rpn_results_list: InstanceList,
            batch_data_samples: List[DetDataSample],
            rescale: bool = True) -> InstanceList:
        proposals = [result.bboxes for result in rpn_results_list]
        proposal_counts = tuple(len(proposal) for proposal in proposals)
        rois = bbox2roi(proposals)
        if rois.shape[0] == 0:
            return self.predict(
                x, rpn_results_list, batch_data_samples, rescale=rescale)

        base_forward = self._bbox_forward(x, rois)
        cls_scores = base_forward['cls_score']
        bbox_predictions = base_forward['bbox_pred'].clone()
        levels = self.bbox_roi_extractor.map_roi_levels(
            rois, self.bbox_roi_extractor.num_inputs)
        route_indices = self._routed_indices(levels)
        if route_indices.numel() > 0 and not self.force_noop:
            p2_base = {
                key: value[route_indices]
                for key, value in base_forward.items()
            }
            action_bbox, utility_features = (
                self._bbox_only_action_context(
                    x, expert_outputs, weights, rois[route_indices],
                    p2_base))
            action_indices, predicted_utilities = (
                self._predict_action_indices(utility_features))
            rows = torch.arange(
                action_indices.numel(), device=rois.device)
            bbox_predictions[route_indices] = action_bbox[
                rows, action_indices]
        else:
            action_indices = rois.new_zeros(
                route_indices.numel(), dtype=torch.long)
            predicted_utilities = rois.new_zeros(
                (route_indices.numel(), self.num_experts))

        self.rclc_last_predict_nonnoop_ratio = (
            float((action_indices != 0).float().mean().item())
            if action_indices.numel() > 0 else 0.0)
        self.rclc_last_predict_action_histogram = tuple(
            torch.bincount(
                action_indices.detach().cpu(),
                minlength=self.num_actions).tolist())
        self.rclc_last_predict_utility_mean = (
            float(predicted_utilities.max(dim=1).values.mean().item())
            if predicted_utilities.numel() > 0 else 0.0)

        roi_parts = rois.split(proposal_counts, dim=0)
        cls_parts = cls_scores.split(proposal_counts, dim=0)
        bbox_parts = bbox_predictions.split(proposal_counts, dim=0)
        batch_img_metas = [
            sample.metainfo for sample in batch_data_samples]
        results = self.bbox_head.predict_by_feat(
            rois=roi_parts,
            cls_scores=cls_parts,
            bbox_preds=bbox_parts,
            batch_img_metas=batch_img_metas,
            rcnn_test_cfg=self.test_cfg,
            rescale=rescale)
        if self.with_mask:
            results = self.predict_mask(
                x, batch_img_metas, results, rescale=rescale)
        return results
