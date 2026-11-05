from typing import List, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from mmdet.registry import MODELS
from mmdet.structures import DetDataSample
from mmdet.structures.bbox import bbox2roi
from mmdet.utils import InstanceList
from .p2_bbox_only_utility_rclc_roi_head import (
    P2BBoxOnlyUtilityRCLCRoIHead)


@MODELS.register_module()
class P2BBoxOnlyBenefitRCLCRoIHead(P2BBoxOnlyUtilityRCLCRoIHead):
    """Route P2 bbox deltas with abstention and set-valued supervision.

    The policy first predicts whether any counterfactual action is beneficial,
    then ranks four expert-removal actions.  Near-best beneficial actions form
    a target set, avoiding an arbitrary single-action label.  Candidate RoI
    feature responses are exposed to the policy, while detector features and
    counterfactual targets remain detached.
    """

    def __init__(self,
                 activation_threshold: float = 0.55,
                 action_confidence_threshold: float = 0.55,
                 action_target_tolerance: float = 0.01,
                 activation_loss_weight: float = 1.0,
                 action_loss_weight: float = 1.0,
                 activation_positive_weight: float = 0.0,
                 noop_action_loss_weight: float = 0.0,
                 harmful_bbox_loss_weight: float = 0.0,
                 build_benefit_router: bool = True,
                 *args,
                 **kwargs) -> None:
        super().__init__(
            build_utility_router=build_benefit_router,
            *args,
            **kwargs)
        if not 0.0 <= activation_threshold <= 1.0:
            raise ValueError('activation_threshold must be in [0, 1].')
        if not 0.0 <= action_confidence_threshold <= 1.0:
            raise ValueError(
                'action_confidence_threshold must be in [0, 1].')
        if action_target_tolerance < 0.0:
            raise ValueError('action_target_tolerance must be non-negative.')
        if activation_loss_weight <= 0.0:
            raise ValueError('activation_loss_weight must be positive.')
        if action_loss_weight <= 0.0:
            raise ValueError('action_loss_weight must be positive.')
        if activation_positive_weight < 0.0:
            raise ValueError(
                'activation_positive_weight must be non-negative.')
        if noop_action_loss_weight < 0.0:
            raise ValueError(
                'noop_action_loss_weight must be non-negative.')
        if harmful_bbox_loss_weight < 0.0:
            raise ValueError(
                'harmful_bbox_loss_weight must be non-negative.')

        self.activation_threshold = float(activation_threshold)
        self.action_confidence_threshold = float(
            action_confidence_threshold)
        self.action_target_tolerance = float(action_target_tolerance)
        self.activation_loss_weight = float(activation_loss_weight)
        self.action_loss_weight = float(action_loss_weight)
        self.activation_positive_weight = float(
            activation_positive_weight)
        self.noop_action_loss_weight = float(
            noop_action_loss_weight)
        self.harmful_bbox_loss_weight = float(
            harmful_bbox_loss_weight)

        if build_benefit_router:
            legacy_input_dim = self.utility_encoder[0].in_features
            hidden_dim = self.utility_encoder[0].out_features
            del self.utility_encoder
            del self.utility_head
            expert_response_dim = (
                self.num_experts * self.bbox_head.in_channels)
            proposal_metadata_dim = 2
            selector_input_dim = (
                legacy_input_dim + expert_response_dim
                + proposal_metadata_dim)
            self.selector_encoder = nn.Sequential(
                nn.Linear(selector_input_dim, hidden_dim),
                nn.ReLU(inplace=True))
            self.activation_head = nn.Linear(hidden_dim, 1)
            self.action_head = nn.Linear(hidden_dim, self.num_experts)
            nn.init.zeros_(self.activation_head.weight)
            nn.init.zeros_(self.activation_head.bias)
            nn.init.zeros_(self.action_head.weight)
            nn.init.zeros_(self.action_head.bias)

    def cada_trainable_parameters(self):
        return tuple(self.selector_encoder.parameters()) + tuple(
            self.activation_head.parameters()) + tuple(
                self.action_head.parameters())

    def train_cada_components(self) -> None:
        self.eval()
        self.selector_encoder.train()
        self.activation_head.train()
        self.action_head.train()

    def _limit_positive_training_rois(
            self, positive_local: Tensor, best_iou: Tensor,
            candidate_scores: Tensor) -> Tensor:
        """Align capped training proposals with inference score ordering."""
        if positive_local.numel() <= self.max_matched_per_img:
            return positive_local
        order = candidate_scores[positive_local].argsort(
            descending=True)[:self.max_matched_per_img]
        return positive_local[order]

    @staticmethod
    def _proposal_route_metadata(rois: Tensor,
                                 proposal_scores: Tensor) -> Tensor:
        rank_scores = proposal_scores.new_zeros(proposal_scores.shape)
        if rois.numel() > 0:
            num_images = int(rois[:, 0].max().item()) + 1
            for image_index in range(num_images):
                indices = torch.nonzero(
                    rois[:, 0].long() == image_index,
                    as_tuple=False).flatten()
                if indices.numel() == 0:
                    continue
                order = proposal_scores[indices].argsort(descending=True)
                denominator = max(indices.numel() - 1, 1)
                ranks = torch.arange(
                    indices.numel(), device=rois.device,
                    dtype=proposal_scores.dtype)
                rank_scores[indices[order]] = 1.0 - ranks / denominator
        return torch.stack((proposal_scores, rank_scores), dim=1)

    def _bbox_only_action_context(
            self, x: Tuple[Tensor], expert_outputs, weights, rois: Tensor,
            base_forward=None, route_metadata=None):
        if base_forward is None:
            base_forward = self._bbox_forward(x, rois)
        base_pooled = F.adaptive_avg_pool2d(
            base_forward['bbox_feats'], 1).flatten(1)
        bbox_outputs = [base_forward['bbox_pred']]
        response_features = []
        for expert_index in range(self.num_experts):
            candidate = self._counterfactual_features(
                x, expert_outputs, weights, expert_index)
            candidate_forward = self._bbox_forward(candidate, rois)
            bbox_outputs.append(candidate_forward['bbox_pred'])
            candidate_pooled = F.adaptive_avg_pool2d(
                candidate_forward['bbox_feats'], 1).flatten(1)
            response_features.append(candidate_pooled - base_pooled)

        bbox_predictions = torch.stack(tuple(bbox_outputs), dim=1)
        base_probabilities = F.softmax(base_forward['cls_score'], dim=1)
        bbox_deltas = (
            bbox_predictions[:, 1:] - bbox_predictions[:, :1]).clamp(
                min=-self.response_delta_clip,
                max=self.response_delta_clip) / self.response_delta_clip
        if route_metadata is None:
            route_metadata = base_pooled.new_zeros(
                (base_pooled.shape[0], 2))
        selector_features = torch.cat((
            base_pooled,
            self._roi_gate_values(weights, rois),
            base_probabilities,
            bbox_deltas.flatten(1),
            torch.stack(tuple(response_features), dim=1).flatten(1),
            route_metadata,
        ), dim=1)
        return bbox_predictions, selector_features

    @staticmethod
    def _balanced_binary_loss(logits: Tensor, targets: Tensor) -> Tensor:
        positive = targets > 0.5
        parts = []
        if positive.any():
            parts.append(F.binary_cross_entropy_with_logits(
                logits[positive], targets[positive]))
        if (~positive).any():
            parts.append(F.binary_cross_entropy_with_logits(
                logits[~positive], targets[~positive]))
        if not parts:
            return logits.sum() * 0.0
        return torch.stack(parts).mean()

    def _activation_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        if self.activation_positive_weight == 0.0:
            return self._balanced_binary_loss(logits, targets)
        return F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=logits.new_tensor(
                self.activation_positive_weight))

    @staticmethod
    def _noop_action_loss(action_logits: Tensor,
                          active_targets: Tensor) -> Tensor:
        inactive = ~active_targets
        if not inactive.any():
            return action_logits.sum() * 0.0
        return F.binary_cross_entropy_with_logits(
            action_logits[inactive],
            torch.zeros_like(action_logits[inactive]))

    def _harmful_bbox_loss(
            self, bbox_predictions: Tensor,
            activation_probability: Tensor, action_logits: Tensor,
            rois: Tensor, gt_boxes: Tensor, gt_labels: Tensor,
            matched: Tensor) -> Tensor:
        if not matched.any():
            return activation_probability.sum() * 0.0
        action_weights = F.softmax(action_logits[matched], dim=1)
        expected_action = (
            action_weights[:, :, None]
            * bbox_predictions[matched, 1:]).sum(dim=1)
        base_prediction = bbox_predictions[matched, 0]
        executed_prediction = (
            base_prediction
            + activation_probability[matched, None]
            * (expected_action - base_prediction))
        executed_class_prediction = self._class_specific_bbox_pred(
            self.bbox_head, executed_prediction, gt_labels[matched])
        factual_class_prediction = self._class_specific_bbox_pred(
            self.bbox_head, base_prediction, gt_labels[matched])
        target_deltas = self.bbox_head.bbox_coder.encode(
            rois[matched, 1:5], gt_boxes[matched])
        executed_loss = F.smooth_l1_loss(
            executed_class_prediction,
            target_deltas,
            beta=1.0 / 9.0,
            reduction='none').mean(dim=1)
        factual_loss = F.smooth_l1_loss(
            factual_class_prediction,
            target_deltas,
            beta=1.0 / 9.0,
            reduction='none').mean(dim=1)
        return F.relu(executed_loss - factual_loss).mean()

    def _benefit_targets(self, utilities: Tensor, matched: Tensor):
        beneficial = utilities >= self.utility_margin
        active = beneficial.any(dim=1) & matched
        best_utility = utilities.max(dim=1, keepdim=True).values
        near_best = (
            utilities >= best_utility - self.action_target_tolerance)
        action_set = beneficial & near_best & matched[:, None]
        return active, action_set

    def _policy_outputs(self, selector_features: Tensor):
        encoded = self.selector_encoder(selector_features)
        activation_logits = self.activation_head(encoded).squeeze(1)
        action_logits = self.action_head(encoded)
        activation_probability = activation_logits.sigmoid()
        action_probabilities = action_logits.sigmoid()
        best_probability, best_relative_action = action_probabilities.max(
            dim=1)
        active = (
            (activation_probability > self.activation_threshold)
            & (best_probability > self.action_confidence_threshold))
        action_indices = torch.zeros_like(best_relative_action)
        action_indices[active] = best_relative_action[active] + 1
        return (
            action_indices, activation_logits, action_logits,
            activation_probability, action_probabilities)

    def _empty_benefit_logs(self) -> dict:
        zero = (
            self.activation_head.weight.sum()
            + self.action_head.weight.sum()) * 0.0
        return {
            'loss_rclc_activation': zero,
            'loss_rclc_action_rank': zero,
            'loss_rclc_noop_action': zero,
            'loss_rclc_bbox_harm': zero,
            'rclc_noop_action_raw': zero.detach(),
            'rclc_bbox_harm_raw': zero.detach(),
            'rclc_policy_weight': zero.detach(),
            'rclc_candidate_count': zero.detach(),
            'rclc_matched_count': zero.detach(),
            'rclc_target_nonnoop_ratio': zero.detach(),
            'rclc_pred_nonnoop_ratio': zero.detach(),
            'rclc_activation_precision': zero.detach(),
            'rclc_activation_recall': zero.detach(),
            'rclc_action_set_accuracy': zero.detach(),
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
            return self._empty_benefit_logs()
        levels = self.bbox_roi_extractor.map_roi_levels(
            rois, self.bbox_roi_extractor.num_inputs)
        proposal_scores = self._proposal_scores(rpn_results_list)
        selected, matched, gt_boxes, gt_labels = self._select_training_rois(
            rois, levels, proposal_scores, batch_data_samples)
        if selected.numel() == 0:
            return self._empty_benefit_logs()

        selected_rois = rois[selected]
        route_metadata = self._proposal_route_metadata(
            rois, proposal_scores)[selected]
        with torch.no_grad():
            bbox_predictions, selector_features = (
                self._bbox_only_action_context(
                    x, expert_outputs, weights, selected_rois,
                    route_metadata=route_metadata))
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
                matched_utilities, _ = self._localization_utility_targets(
                    matched_iou)
                target_utilities[matched] = matched_utilities
                candidate_iou[matched] = matched_iou
            active_targets, action_target_set = self._benefit_targets(
                target_utilities, matched)

        outputs = self._policy_outputs(selector_features)
        predicted_actions, activation_logits, action_logits = outputs[:3]
        activation_probability = outputs[3]
        activation_values = active_targets.to(activation_logits.dtype)
        raw_activation_loss = self._activation_loss(
            activation_logits, activation_values)
        if active_targets.any():
            action_values = action_target_set[active_targets].to(
                action_logits.dtype)
            raw_action_loss = self._balanced_binary_loss(
                action_logits[active_targets], action_values)
        else:
            raw_action_loss = action_logits.sum() * 0.0
        raw_noop_action_loss = self._noop_action_loss(
            action_logits, active_targets)
        if self.harmful_bbox_loss_weight > 0.0:
            raw_harmful_bbox_loss = self._harmful_bbox_loss(
                bbox_predictions,
                activation_probability,
                action_logits,
                selected_rois,
                gt_boxes,
                gt_labels,
                matched)
        else:
            raw_harmful_bbox_loss = action_logits.sum() * 0.0
        current_weight = self._current_credit_weight()
        activation_loss = (
            raw_activation_loss * self.activation_loss_weight
            * current_weight)
        action_loss = (
            raw_action_loss * self.action_loss_weight * current_weight)
        noop_action_loss = (
            raw_noop_action_loss * self.noop_action_loss_weight
            * current_weight)
        harmful_bbox_loss = (
            raw_harmful_bbox_loss * self.harmful_bbox_loss_weight
            * current_weight)

        predicted_active = predicted_actions != 0
        true_positive = predicted_active & active_targets
        precision = (
            true_positive.float().sum()
            / predicted_active.float().sum().clamp_min(1.0))
        recall = (
            true_positive.float().sum()
            / active_targets.float().sum().clamp_min(1.0))
        if active_targets.any():
            active_predictions = predicted_actions[active_targets] - 1
            valid_prediction = active_predictions >= 0
            set_correct = torch.zeros_like(
                active_predictions, dtype=torch.bool)
            if valid_prediction.any():
                rows = torch.arange(
                    active_predictions.numel(), device=rois.device)
                set_correct[valid_prediction] = action_target_set[
                    active_targets][rows[valid_prediction],
                                    active_predictions[valid_prediction]]
            action_set_accuracy = set_correct.float().mean()
        else:
            action_set_accuracy = activation_loss.detach() * 0.0

        if matched.any():
            rows = torch.arange(matched.sum(), device=rois.device)
            matched_iou = candidate_iou[matched]
            matched_utilities = target_utilities[matched]
            oracle_relative = matched_utilities.argmax(dim=1)
            oracle_actions = oracle_relative + 1
            oracle_active = matched_utilities.max(dim=1).values >= (
                self.utility_margin)
            oracle_actions = torch.where(
                oracle_active, oracle_actions,
                torch.zeros_like(oracle_actions))
            matched_predictions = predicted_actions[matched]
            oracle_iou_gain = (
                matched_iou[rows, oracle_actions]
                - matched_iou[:, 0]).mean()
            predicted_iou_delta = (
                matched_iou[rows, matched_predictions]
                - matched_iou[:, 0])
            predicted_iou_gain = predicted_iou_delta.mean()
            harm_ratio = (
                predicted_iou_delta < -1e-8).float().mean()
        else:
            zero = activation_loss.detach() * 0.0
            oracle_iou_gain = zero
            predicted_iou_gain = zero
            harm_ratio = zero

        return {
            'loss_rclc_activation': activation_loss,
            'loss_rclc_action_rank': action_loss,
            'loss_rclc_noop_action': noop_action_loss,
            'loss_rclc_bbox_harm': harmful_bbox_loss,
            'rclc_activation_raw': raw_activation_loss.detach(),
            'rclc_action_rank_raw': raw_action_loss.detach(),
            'rclc_noop_action_raw': raw_noop_action_loss.detach(),
            'rclc_bbox_harm_raw': raw_harmful_bbox_loss.detach(),
            'rclc_policy_weight': activation_loss.new_tensor(
                current_weight).detach(),
            'rclc_candidate_count': selected.new_tensor(
                float(selected.numel()), dtype=torch.float32).detach(),
            'rclc_matched_count': matched.sum().to(torch.float32).detach(),
            'rclc_target_nonnoop_ratio': active_targets.float().mean().detach(),
            'rclc_pred_nonnoop_ratio': predicted_active.float().mean().detach(),
            'rclc_activation_precision': precision.detach(),
            'rclc_activation_recall': recall.detach(),
            'rclc_action_set_accuracy': action_set_accuracy.detach(),
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
            proposal_scores = self._proposal_scores(rpn_results_list)
            route_metadata = self._proposal_route_metadata(
                rois, proposal_scores)[route_indices]
            p2_base = {
                key: value[route_indices]
                for key, value in base_forward.items()
            }
            action_bbox, selector_features = (
                self._bbox_only_action_context(
                    x, expert_outputs, weights, rois[route_indices], p2_base,
                    route_metadata))
            outputs = self._policy_outputs(selector_features)
            action_indices = outputs[0]
            activation_probability = outputs[3]
            action_probabilities = outputs[4]
            rows = torch.arange(
                action_indices.numel(), device=rois.device)
            bbox_predictions[route_indices] = action_bbox[
                rows, action_indices]
        else:
            action_indices = rois.new_zeros(
                route_indices.numel(), dtype=torch.long)
            activation_probability = rois.new_zeros(
                route_indices.numel())
            action_probabilities = rois.new_zeros(
                (route_indices.numel(), self.num_experts))

        self.rclc_last_predict_nonnoop_ratio = (
            float((action_indices != 0).float().mean().item())
            if action_indices.numel() > 0 else 0.0)
        self.rclc_last_predict_action_histogram = tuple(
            torch.bincount(
                action_indices.detach().cpu(),
                minlength=self.num_actions).tolist())
        self.rclc_last_predict_activation_mean = (
            float(activation_probability.mean().item())
            if activation_probability.numel() > 0 else 0.0)
        self.rclc_last_predict_action_confidence_mean = (
            float(action_probabilities.max(dim=1).values.mean().item())
            if action_probabilities.numel() > 0 else 0.0)

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
