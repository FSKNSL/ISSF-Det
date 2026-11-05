from typing import List, Tuple

import torch
from mmengine.logging import MessageHub
from torch import Tensor, nn
from torch.nn import functional as F

from mmdet.registry import MODELS
from mmdet.structures import DetDataSample
from mmdet.structures.bbox import (bbox2roi, bbox_overlaps, get_box_tensor)
from mmdet.utils import InstanceList
from .standard_roi_head import StandardRoIHead


@MODELS.register_module()
class P2CounterfactualCreditRCLCRoIHead(StandardRoIHead):
    """Route selected FPN RoIs with detector-utility counterfactual credit.

    The five actions are no-op and four half-strength leave-one-expert-out
    interventions.  A Pareto teacher accepts an intervention only when it
    does not reduce the GT-class probability, decoded-box IoU, or the number
    of passed COCO IoU thresholds.  The learned hard action selects both the
    classification logits and bbox deltas before decoding and NMS.

    Counterfactual targets are detached.  Consequently the ordinary RPN,
    RoI, MRSC and interaction losses keep training the detector, while this
    loss directly trains only the credit router in the same from-scratch run.
    """

    num_experts = 4
    num_actions = 5
    coco_iou_thresholds = tuple(
        0.50 + 0.05 * index for index in range(10))

    def __init__(self,
                 router_hidden_dim: int = 256,
                 removal_strength: float = 0.50,
                 route_levels: Tuple[int, ...] = (0,),
                 match_min_iou: float = 0.50,
                 max_matched_per_img: int = 256,
                 max_noop_negatives_per_img: int = 64,
                 response_delta_clip: float = 1.0,
                 credit_loss_weight: float = 1.0,
                 credit_loss_warmup_iters: int = 500,
                 force_noop: bool = False,
                 build_credit_router: bool = True,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if router_hidden_dim <= 0:
            raise ValueError('router_hidden_dim must be positive.')
        if not 0.0 < removal_strength <= 1.0:
            raise ValueError('removal_strength must be in (0, 1].')
        route_levels = tuple(int(level) for level in route_levels)
        if (not route_levels or len(set(route_levels)) != len(route_levels)
                or min(route_levels) < 0
                or max(route_levels) >= self.bbox_roi_extractor.num_inputs):
            raise ValueError(
                'route_levels must contain unique valid RoI FPN levels.')
        if not 0.0 <= match_min_iou <= 1.0:
            raise ValueError('match_min_iou must be in [0, 1].')
        if max_matched_per_img <= 0:
            raise ValueError('max_matched_per_img must be positive.')
        if max_noop_negatives_per_img < 0:
            raise ValueError(
                'max_noop_negatives_per_img must be non-negative.')
        if response_delta_clip <= 0.0:
            raise ValueError('response_delta_clip must be positive.')
        if credit_loss_weight < 0.0:
            raise ValueError('credit_loss_weight must be non-negative.')
        if credit_loss_warmup_iters < 0:
            raise ValueError(
                'credit_loss_warmup_iters must be non-negative.')

        self.removal_strength = float(removal_strength)
        self.route_levels = route_levels
        self.match_min_iou = float(match_min_iou)
        self.max_matched_per_img = int(max_matched_per_img)
        self.max_noop_negatives_per_img = int(
            max_noop_negatives_per_img)
        self.response_delta_clip = float(response_delta_clip)
        self.credit_loss_weight = float(credit_loss_weight)
        self.credit_loss_warmup_iters = int(credit_loss_warmup_iters)
        self.force_noop = bool(force_noop)

        self.credit_router = None
        if build_credit_router:
            num_cls_outputs = self.bbox_head.num_classes + 1
            num_bbox_outputs = (
                4 if self.bbox_head.reg_class_agnostic
                else 4 * self.bbox_head.num_classes)
            router_input_dim = (
                self.bbox_head.in_channels
                + self.num_experts
                + num_cls_outputs
                + self.num_experts * num_cls_outputs
                + self.num_experts * num_bbox_outputs)
            self.credit_router = nn.Sequential(
                nn.Linear(router_input_dim, router_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(router_hidden_dim, self.num_actions))
            nn.init.zeros_(self.credit_router[-1].weight)
            nn.init.zeros_(self.credit_router[-1].bias)

    def cada_trainable_parameters(self):
        return tuple(self.credit_router.parameters())

    def train_cada_components(self) -> None:
        self.eval()
        self.credit_router.train()

    def _current_credit_weight(self) -> float:
        if self.credit_loss_warmup_iters == 0:
            return self.credit_loss_weight
        try:
            current_iter = MessageHub.get_current_instance().get_info('iter')
        except Exception:
            current_iter = None
        if current_iter is None:
            return self.credit_loss_weight
        progress = min(
            float(current_iter + 1) / self.credit_loss_warmup_iters, 1.0)
        return self.credit_loss_weight * progress

    @staticmethod
    def _aligned_iou(boxes: Tensor, targets: Tensor) -> Tensor:
        left_top = torch.maximum(boxes[..., :2], targets[:, None, :2])
        right_bottom = torch.minimum(boxes[..., 2:], targets[:, None, 2:])
        intersection_wh = (right_bottom - left_top).clamp_min(0)
        intersection = intersection_wh[..., 0] * intersection_wh[..., 1]
        box_wh = (boxes[..., 2:] - boxes[..., :2]).clamp_min(0)
        target_wh = (targets[:, 2:] - targets[:, :2]).clamp_min(0)
        box_area = box_wh[..., 0] * box_wh[..., 1]
        target_area = target_wh[:, 0] * target_wh[:, 1]
        union = box_area + target_area[:, None] - intersection
        return intersection / union.clamp_min(1e-12)

    @classmethod
    def _pareto_action_targets(cls, iou: Tensor,
                               probabilities: Tensor) -> Tuple[Tensor, Tensor]:
        thresholds = iou.new_tensor(cls.coco_iou_thresholds)
        pass_counts = (
            iou[:, :, None] >= thresholds[None, None, :]).sum(dim=2)
        base_iou = iou[:, :1]
        base_probability = probabilities[:, :1]
        base_pass = pass_counts[:, :1]
        safe = (
            (pass_counts >= base_pass)
            & (iou >= base_iou - 1e-8)
            & (probabilities >= base_probability - 1e-8))
        strict_gain = (
            (pass_counts > base_pass)
            | (iou > base_iou + 1e-8)
            | (probabilities > base_probability + 1e-8))
        safe = safe & strict_gain
        safe[:, 0] = True
        utility = (
            pass_counts.to(iou.dtype) * 1_000_000.0
            + iou * 1_000.0 + probabilities)
        utility = utility.masked_fill(~safe, -torch.inf)
        return utility.argmax(dim=1), pass_counts

    def _counterfactual_level_feature(
            self, x: Tuple[Tensor], expert_outputs, weights,
            expert_index: int, level: int) -> Tensor:
        level_weights = weights[level]
        denominator = (
            1.0 - level_weights[:, expert_index:expert_index + 1]
        ).clamp_min(1e-6)
        removed = None
        for index, expert in enumerate(expert_outputs):
            if index == expert_index:
                continue
            contribution = (
                level_weights[:, index:index + 1] / denominator
                * expert[level])
            removed = (
                contribution if removed is None else removed + contribution)
        return x[level] + self.removal_strength * (removed - x[level])

    def _counterfactual_p2_feature(self, x: Tuple[Tensor], expert_outputs,
                                   weights, expert_index: int) -> Tensor:
        return self._counterfactual_level_feature(
            x, expert_outputs, weights, expert_index, level=0)

    def _counterfactual_features(self, x: Tuple[Tensor], expert_outputs,
                                 weights, expert_index: int) -> Tuple[Tensor]:
        candidate = list(x)
        for level in self.route_levels:
            candidate[level] = self._counterfactual_level_feature(
                x, expert_outputs, weights, expert_index, level)
        return tuple(candidate)

    def _routed_indices(self, levels: Tensor) -> Tensor:
        routed = torch.zeros_like(levels, dtype=torch.bool)
        for level in self.route_levels:
            routed |= levels == level
        return torch.nonzero(routed, as_tuple=False).flatten()

    def _roi_gate_values(self, weights, rois: Tensor) -> Tensor:
        levels = self.bbox_roi_extractor.map_roi_levels(
            rois, self.bbox_roi_extractor.num_inputs)
        values = weights[0].new_zeros((rois.shape[0], self.num_experts))
        for level in self.route_levels:
            selected = torch.nonzero(
                levels == level, as_tuple=False).flatten()
            if selected.numel() == 0:
                continue
            values[selected] = self.bbox_roi_extractor.roi_layers[level](
                weights[level], rois[selected]).mean(dim=(-1, -2))
        return values / values.sum(dim=1, keepdim=True).clamp_min(1e-8)

    def _action_context(self, x: Tuple[Tensor], expert_outputs, weights,
                        rois: Tensor, base_forward=None):
        if base_forward is None:
            base_forward = self._bbox_forward(x, rois)
        cls_outputs = [base_forward['cls_score']]
        bbox_outputs = [base_forward['bbox_pred']]
        for expert_index in range(self.num_experts):
            candidate = self._counterfactual_features(
                x, expert_outputs, weights, expert_index)
            candidate_forward = self._bbox_forward(candidate, rois)
            cls_outputs.append(candidate_forward['cls_score'])
            bbox_outputs.append(candidate_forward['bbox_pred'])

        cls_scores = torch.stack(tuple(cls_outputs), dim=1)
        bbox_predictions = torch.stack(tuple(bbox_outputs), dim=1)
        base_probabilities = F.softmax(cls_scores[:, 0], dim=1)
        probability_deltas = (
            F.softmax(cls_scores[:, 1:], dim=2)
            - base_probabilities[:, None, :])
        bbox_deltas = (
            bbox_predictions[:, 1:] - bbox_predictions[:, :1]).clamp(
                min=-self.response_delta_clip,
                max=self.response_delta_clip) / self.response_delta_clip
        pooled = F.adaptive_avg_pool2d(
            base_forward['bbox_feats'], 1).flatten(1)
        router_features = torch.cat((
            pooled,
            self._roi_gate_values(weights, rois),
            base_probabilities,
            probability_deltas.flatten(1),
            bbox_deltas.flatten(1),
        ), dim=1)
        return cls_scores, bbox_predictions, router_features

    @staticmethod
    def _class_specific_bbox_pred(bbox_head, bbox_pred: Tensor,
                                  labels: Tensor) -> Tensor:
        if bbox_head.reg_class_agnostic:
            return bbox_pred.view(bbox_pred.size(0), -1)
        reshaped = bbox_pred.view(
            bbox_pred.size(0), bbox_head.num_classes, -1)
        rows = torch.arange(labels.numel(), device=labels.device)
        return reshaped[rows, labels]

    def _decode_matched_boxes(
            self, rois: Tensor, bbox_predictions: Tensor, labels: Tensor,
            batch_data_samples: List[DetDataSample]) -> Tensor:
        decoded_actions = []
        for action_index in range(self.num_actions):
            class_prediction = self._class_specific_bbox_pred(
                self.bbox_head, bbox_predictions[:, action_index], labels)
            decoded = rois.new_zeros((rois.shape[0], 4))
            for image_index, data_sample in enumerate(batch_data_samples):
                selected = torch.nonzero(
                    rois[:, 0].long() == image_index,
                    as_tuple=False).flatten()
                if selected.numel() == 0:
                    continue
                boxes = self.bbox_head.bbox_coder.decode(
                    rois[selected, 1:5], class_prediction[selected],
                    max_shape=data_sample.metainfo['img_shape'])
                decoded[selected] = get_box_tensor(boxes)[:, :4]
            decoded_actions.append(decoded)
        return torch.stack(tuple(decoded_actions), dim=1)

    @staticmethod
    def _proposal_scores(rpn_results_list: InstanceList) -> Tensor:
        scores = []
        for result in rpn_results_list:
            if hasattr(result, 'scores'):
                scores.append(result.scores)
            else:
                scores.append(result.bboxes.new_zeros(len(result)))
        return torch.cat(tuple(scores))

    def _limit_positive_training_rois(
            self, positive_local: Tensor, best_iou: Tensor,
            candidate_scores: Tensor) -> Tensor:
        """Keep the highest-IoU matched proposals for the legacy router."""
        if positive_local.numel() <= self.max_matched_per_img:
            return positive_local
        order = best_iou[positive_local].argsort(
            descending=True)[:self.max_matched_per_img]
        return positive_local[order]

    def _select_training_rois(
            self, rois: Tensor, levels: Tensor, proposal_scores: Tensor,
            batch_data_samples: List[DetDataSample]):
        selected_parts = []
        matched_parts = []
        gt_box_parts = []
        gt_label_parts = []
        for image_index, data_sample in enumerate(batch_data_samples):
            image_indices = torch.nonzero(
                rois[:, 0].long() == image_index,
                as_tuple=False).flatten()
            routed_local = self._routed_indices(levels[image_indices])
            candidates = image_indices[routed_local]
            gt_instances = data_sample.gt_instances
            if candidates.numel() == 0:
                continue
            gt_boxes = get_box_tensor(gt_instances.bboxes)[:, :4].to(rois)
            gt_labels = gt_instances.labels.to(
                device=rois.device, dtype=torch.long)
            if gt_boxes.numel() == 0:
                positive = candidates.new_zeros((0,))
                best_gt = candidates.new_zeros((0,))
            else:
                overlaps = bbox_overlaps(rois[candidates, 1:5], gt_boxes)
                best_iou, best_gt_all = overlaps.max(dim=1)
                positive_local = torch.nonzero(
                    best_iou >= self.match_min_iou,
                    as_tuple=False).flatten()
                positive_local = self._limit_positive_training_rois(
                    positive_local, best_iou,
                    proposal_scores[candidates])
                positive = candidates[positive_local]
                best_gt = best_gt_all[positive_local]

            negative_mask = torch.ones(
                candidates.numel(), device=rois.device, dtype=torch.bool)
            if gt_boxes.numel() > 0:
                negative_mask = best_iou < self.match_min_iou
            negative_local = torch.nonzero(
                negative_mask, as_tuple=False).flatten()
            if self.max_noop_negatives_per_img == 0:
                negative_local = negative_local[:0]
            elif (negative_local.numel()
                  > self.max_noop_negatives_per_img):
                order = proposal_scores[candidates[negative_local]].argsort(
                    descending=True)[:self.max_noop_negatives_per_img]
                negative_local = negative_local[order]
            negative = candidates[negative_local]

            selected = torch.cat((positive, negative))
            if selected.numel() == 0:
                continue
            matched = torch.zeros(
                selected.numel(), device=rois.device, dtype=torch.bool)
            matched[:positive.numel()] = True
            selected_gt_boxes = rois.new_zeros((selected.numel(), 4))
            selected_gt_labels = torch.zeros(
                selected.numel(), device=rois.device, dtype=torch.long)
            if positive.numel() > 0:
                selected_gt_boxes[:positive.numel()] = gt_boxes[best_gt]
                selected_gt_labels[:positive.numel()] = gt_labels[best_gt]
            selected_parts.append(selected)
            matched_parts.append(matched)
            gt_box_parts.append(selected_gt_boxes)
            gt_label_parts.append(selected_gt_labels)

        if not selected_parts:
            empty = rois.new_zeros((0,), dtype=torch.long)
            return (
                empty,
                rois.new_zeros((0,), dtype=torch.bool),
                rois.new_zeros((0, 4)),
                empty)
        return (
            torch.cat(tuple(selected_parts)),
            torch.cat(tuple(matched_parts)),
            torch.cat(tuple(gt_box_parts)),
            torch.cat(tuple(gt_label_parts)))

    def _empty_credit_logs(self, reference: Tensor) -> dict:
        zero = reference.new_zeros(())
        return {
            'loss_rclc_credit': zero,
            'rclc_credit_weight': zero.detach(),
            'rclc_candidate_count': zero.detach(),
            'rclc_matched_count': zero.detach(),
            'rclc_target_nonnoop_ratio': zero.detach(),
            'rclc_pred_nonnoop_ratio': zero.detach(),
            'rclc_action_accuracy': zero.detach(),
            'rclc_oracle_iou_gain': zero.detach(),
            'rclc_oracle_prob_gain': zero.detach(),
            'rclc_pred_iou_gain': zero.detach(),
            'rclc_pred_prob_gain': zero.detach(),
            'rclc_pred_harm_ratio': zero.detach(),
        }

    def loss_with_cada_features(
            self, x: Tuple[Tensor], expert_outputs, weights,
            rpn_results_list: InstanceList,
            batch_data_samples: List[DetDataSample]) -> dict:
        proposals = [result.bboxes for result in rpn_results_list]
        rois = bbox2roi(proposals)
        if rois.shape[0] == 0:
            return self._empty_credit_logs(x[0])
        levels = self.bbox_roi_extractor.map_roi_levels(
            rois, self.bbox_roi_extractor.num_inputs)
        proposal_scores = self._proposal_scores(rpn_results_list)
        selected, matched, gt_boxes, gt_labels = self._select_training_rois(
            rois, levels, proposal_scores, batch_data_samples)
        if selected.numel() == 0:
            return self._empty_credit_logs(x[0])

        selected_rois = rois[selected]
        with torch.no_grad():
            cls_scores, bbox_predictions, router_features = (
                self._action_context(
                    x, expert_outputs, weights, selected_rois))
            targets = torch.zeros(
                selected.numel(), device=rois.device, dtype=torch.long)
            candidate_iou = rois.new_zeros(
                (selected.numel(), self.num_actions))
            candidate_probabilities = rois.new_zeros(
                (selected.numel(), self.num_actions))
            pass_counts = torch.zeros(
                (selected.numel(), self.num_actions),
                device=rois.device, dtype=torch.long)
            if matched.any():
                matched_boxes = self._decode_matched_boxes(
                    selected_rois[matched], bbox_predictions[matched],
                    gt_labels[matched], batch_data_samples)
                matched_iou = self._aligned_iou(
                    matched_boxes, gt_boxes[matched])
                probabilities = F.softmax(
                    cls_scores[matched], dim=2)
                rows = torch.arange(
                    matched.sum(), device=rois.device)
                matched_probabilities = probabilities[
                    rows[:, None],
                    torch.arange(self.num_actions, device=rois.device)[None],
                    gt_labels[matched, None]]
                matched_targets, matched_pass_counts = (
                    self._pareto_action_targets(
                        matched_iou, matched_probabilities))
                targets[matched] = matched_targets
                candidate_iou[matched] = matched_iou
                candidate_probabilities[matched] = matched_probabilities
                pass_counts[matched] = matched_pass_counts

        logits = self.credit_router(router_features)
        raw_loss = F.cross_entropy(logits, targets)
        current_weight = self._current_credit_weight()
        loss = raw_loss * current_weight
        predicted = logits.argmax(dim=1)

        if matched.any():
            rows = torch.arange(matched.sum(), device=rois.device)
            matched_targets = targets[matched]
            matched_predicted = predicted[matched]
            matched_iou = candidate_iou[matched]
            matched_probabilities = candidate_probabilities[matched]
            matched_pass = pass_counts[matched]
            oracle_iou_gain = (
                matched_iou[rows, matched_targets]
                - matched_iou[:, 0]).mean()
            oracle_prob_gain = (
                matched_probabilities[rows, matched_targets]
                - matched_probabilities[:, 0]).mean()
            pred_iou_gain = (
                matched_iou[rows, matched_predicted]
                - matched_iou[:, 0]).mean()
            pred_prob_gain = (
                matched_probabilities[rows, matched_predicted]
                - matched_probabilities[:, 0]).mean()
            harmful = (
                (matched_pass[rows, matched_predicted] < matched_pass[:, 0])
                | (matched_iou[rows, matched_predicted]
                   < matched_iou[:, 0] - 1e-8)
                | (matched_probabilities[rows, matched_predicted]
                   < matched_probabilities[:, 0] - 1e-8))
            harm_ratio = harmful.float().mean()
            target_nonnoop = (matched_targets != 0).float().mean()
        else:
            zero = loss.detach() * 0.0
            oracle_iou_gain = zero
            oracle_prob_gain = zero
            pred_iou_gain = zero
            pred_prob_gain = zero
            harm_ratio = zero
            target_nonnoop = zero

        return {
            'loss_rclc_credit': loss,
            'rclc_credit_raw': raw_loss.detach(),
            'rclc_credit_weight': loss.new_tensor(
                current_weight).detach(),
            'rclc_candidate_count': targets.new_tensor(
                float(targets.numel()), dtype=torch.float32).detach(),
            'rclc_matched_count': matched.sum().to(torch.float32).detach(),
            'rclc_target_nonnoop_ratio': target_nonnoop.detach(),
            'rclc_pred_nonnoop_ratio': (
                predicted != 0).float().mean().detach(),
            'rclc_action_accuracy': (
                predicted == targets).float().mean().detach(),
            'rclc_oracle_iou_gain': oracle_iou_gain.detach(),
            'rclc_oracle_prob_gain': oracle_prob_gain.detach(),
            'rclc_pred_iou_gain': pred_iou_gain.detach(),
            'rclc_pred_prob_gain': pred_prob_gain.detach(),
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
        cls_scores = base_forward['cls_score'].clone()
        bbox_predictions = base_forward['bbox_pred'].clone()
        levels = self.bbox_roi_extractor.map_roi_levels(
            rois, self.bbox_roi_extractor.num_inputs)
        route_indices = self._routed_indices(levels)
        if route_indices.numel() > 0 and not self.force_noop:
            p2_base = {
                key: value[route_indices]
                for key, value in base_forward.items()
            }
            action_cls, action_bbox, router_features = self._action_context(
                x, expert_outputs, weights, rois[route_indices], p2_base)
            action_indices = self.credit_router(router_features).argmax(dim=1)
            rows = torch.arange(action_indices.numel(), device=rois.device)
            cls_scores[route_indices] = action_cls[rows, action_indices]
            bbox_predictions[route_indices] = action_bbox[
                rows, action_indices]
        else:
            action_indices = rois.new_zeros(
                route_indices.numel(), dtype=torch.long)

        self.rclc_last_predict_nonnoop_ratio = (
            float((action_indices != 0).float().mean().item())
            if action_indices.numel() > 0 else 0.0)
        self.rclc_last_predict_action_histogram = tuple(
            torch.bincount(
                action_indices.detach().cpu(),
                minlength=self.num_actions).tolist())

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
