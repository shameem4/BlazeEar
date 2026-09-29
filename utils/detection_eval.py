"""
Dataset-level detection evaluation.

The previous metric path computed VOC07 11-point AP for each image separately and
averaged those per-image values. With one or two ground-truth boxes per image that
AP quantizes to a handful of discrete values, so the mean mostly reports how the
quantization fell rather than detector quality, and it is not comparable to any
published mAP.

This module accumulates `(score, is_true_positive)` for every detection across a
whole split, together with the total ground-truth count, and computes a single
pooled precision/recall curve at the end. That is the standard formulation used by
VOC (all-point interpolation) and COCO.

Boxes use the MediaPipe convention throughout: `[ymin, xmin, ymax, xmax]`.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch

# IoU thresholds for the COCO-style averaged metric: 0.50, 0.55, ... 0.95
COCO_IOU_THRESHOLDS: tuple[float, ...] = tuple(round(0.50 + 0.05 * i, 2) for i in range(10))


def pairwise_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """IoU between two sets of [ymin, xmin, ymax, xmax] boxes -> [N, M]."""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        device = boxes1.device if boxes1.numel() else boxes2.device
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), device=device)

    y_min = torch.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    x_min = torch.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    y_max = torch.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    x_max = torch.minimum(boxes1[:, None, 3], boxes2[None, :, 3])

    inter = torch.clamp(y_max - y_min, min=0) * torch.clamp(x_max - x_min, min=0)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2[None, :] - inter
    return inter / torch.clamp(union, min=1e-9)


def average_precision(
    scores: torch.Tensor,
    true_positives: torch.Tensor,
    num_ground_truth: int
) -> float:
    """
    All-point interpolated average precision from pooled detections.

    Args:
        scores: [N] confidence of every detection in the split
        true_positives: [N] 1.0 where that detection matched an unclaimed GT box
        num_ground_truth: total GT boxes in the split (including undetected ones)

    Returns:
        AP in [0, 1]. Returns 0.0 when there is no ground truth to recall.
    """
    if num_ground_truth == 0:
        return 0.0
    if scores.numel() == 0:
        return 0.0

    order = torch.argsort(scores, descending=True)
    tp = true_positives[order].to(torch.float64)
    fp = 1.0 - tp

    tp_cumsum = torch.cumsum(tp, dim=0)
    fp_cumsum = torch.cumsum(fp, dim=0)

    recall = tp_cumsum / float(num_ground_truth)
    precision = tp_cumsum / torch.clamp(tp_cumsum + fp_cumsum, min=1e-9)

    # All-point interpolation: make precision monotonically non-increasing when
    # scanned from the high-recall end, then integrate over recall steps.
    precision = torch.flip(torch.cummax(torch.flip(precision, [0]), dim=0).values, [0])

    zero = torch.zeros(1, dtype=recall.dtype, device=recall.device)
    recall = torch.cat([zero, recall])
    precision = torch.cat([precision[:1], precision])

    return float(torch.sum((recall[1:] - recall[:-1]) * precision[1:]).item())


class DetectionEvaluator:
    """
    Accumulates detections across a split and computes pooled detection metrics.

    Usage:
        evaluator = DetectionEvaluator()
        for image in split:
            evaluator.add_image(pred_boxes, pred_scores, gt_boxes)
        metrics = evaluator.compute()

    Greedy matching per image at each IoU threshold: detections are considered in
    descending score order and claim the highest-IoU unclaimed ground-truth box.
    Unmatched detections are false positives; unclaimed ground truth is a miss and
    is accounted for through the total GT count in the recall denominator.
    """

    def __init__(
        self,
        iou_thresholds: Sequence[float] = COCO_IOU_THRESHOLDS,
        primary_iou_threshold: float = 0.5,
        groups: bool = False,
        ignore_ioa_threshold: float = 0.5
    ):
        self.iou_thresholds = tuple(float(t) for t in iou_thresholds)
        if primary_iou_threshold not in self.iou_thresholds:
            self.iou_thresholds = tuple(sorted(self.iou_thresholds + (primary_iou_threshold,)))
        self.primary_iou_threshold = float(primary_iou_threshold)
        self.groups_enabled = groups
        self.ignore_ioa_threshold = float(ignore_ioa_threshold)
        self.reset()

    def reset(self) -> None:
        # Scores are kept per IoU threshold because the ignore mask depends on
        # whether a detection matched ground truth at that particular threshold.
        self._scores: Dict[float, List[torch.Tensor]] = {t: [] for t in self.iou_thresholds}
        self._tp: Dict[float, List[torch.Tensor]] = {t: [] for t in self.iou_thresholds}
        self._num_gt = 0
        self._num_images = 0
        self._num_ignored = 0
        self._matched_ious: List[torch.Tensor] = []
        self._group_scores: Dict[str, List[torch.Tensor]] = {}
        self._group_tp: Dict[str, List[torch.Tensor]] = {}
        self._group_num_gt: Dict[str, int] = {}

    @staticmethod
    def _greedy_match(
        iou: torch.Tensor,
        threshold: float
    ) -> torch.Tensor:
        """
        Greedy score-ordered matching. `iou` is [num_pred, num_gt] with predictions
        already sorted by descending score. Returns [num_pred] of 1.0 / 0.0.
        """
        num_pred, num_gt = iou.shape
        tp = torch.zeros(num_pred, dtype=torch.float32, device=iou.device)
        if num_gt == 0:
            return tp

        claimed = torch.zeros(num_gt, dtype=torch.bool, device=iou.device)
        for i in range(num_pred):
            candidates = iou[i].clone()
            candidates[claimed] = -1.0
            best_iou, best_gt = torch.max(candidates, dim=0)
            if best_iou >= threshold:
                tp[i] = 1.0
                claimed[best_gt] = True
        return tp

    def add_image(
        self,
        pred_boxes: torch.Tensor,
        pred_scores: torch.Tensor,
        gt_boxes: torch.Tensor,
        ignore_boxes: Optional[torch.Tensor] = None,
        group: Optional[str] = None
    ) -> None:
        """
        Add one image's post-NMS detections and ground truth.

        Ignore regions let a split be scored against a trusted subset of the
        labels without punishing the detector for the untrusted ones. A detection
        that matches no real ground truth but lands on an ignore region is dropped
        from the evaluation entirely rather than counted as a false positive.
        Ignore boxes never contribute to the recall denominator.

        The ignore decision is made per IoU threshold, using whether the detection
        matched real ground truth at that same threshold, which is why scores are
        accumulated per threshold rather than once.

        Args:
            pred_boxes: [P, 4] detections in [ymin, xmin, ymax, xmax]
            pred_scores: [P] detection confidences
            gt_boxes: [G, 4] ground-truth boxes, same convention
            ignore_boxes: [I, 4] regions that neither reward nor punish
            group: optional label (e.g. an `annotation_source`) to also report
                   this image's contribution under, when groups are enabled
        """
        self._num_images += 1
        num_gt = int(gt_boxes.shape[0]) if gt_boxes.numel() else 0
        self._num_gt += num_gt

        if self.groups_enabled and group is not None:
            self._group_num_gt[group] = self._group_num_gt.get(group, 0) + num_gt

        if pred_scores.numel() == 0:
            return

        order = torch.argsort(pred_scores, descending=True)
        scores_sorted = pred_scores[order].detach().float().cpu()
        boxes_sorted = pred_boxes[order].detach().float().cpu()
        gt = gt_boxes.detach().float().cpu() if num_gt else torch.zeros((0, 4))

        iou = pairwise_iou(boxes_sorted, gt)

        # Overlap with ignore regions, measured as intersection over the
        # *detection* area: a small detection sitting inside a large ignore
        # region should be ignored even though their IoU is low.
        on_ignore = torch.zeros(boxes_sorted.shape[0], dtype=torch.bool)
        if ignore_boxes is not None and ignore_boxes.numel():
            ign = ignore_boxes.detach().float().cpu()
            y_min = torch.maximum(boxes_sorted[:, None, 0], ign[None, :, 0])
            x_min = torch.maximum(boxes_sorted[:, None, 1], ign[None, :, 1])
            y_max = torch.minimum(boxes_sorted[:, None, 2], ign[None, :, 2])
            x_max = torch.minimum(boxes_sorted[:, None, 3], ign[None, :, 3])
            inter = torch.clamp(y_max - y_min, min=0) * torch.clamp(x_max - x_min, min=0)
            det_area = torch.clamp(
                (boxes_sorted[:, 2] - boxes_sorted[:, 0]) * (boxes_sorted[:, 3] - boxes_sorted[:, 1]),
                min=1e-9
            )
            ioa = inter / det_area[:, None]
            on_ignore = (ioa >= self.ignore_ioa_threshold).any(dim=1)

        primary_tp: Optional[torch.Tensor] = None
        primary_keep: Optional[torch.Tensor] = None

        for threshold in self.iou_thresholds:
            tp = self._greedy_match(iou, threshold)
            # Keep true positives always; drop unmatched detections that sit on
            # an ignore region.
            keep = (tp > 0.5) | (~on_ignore)
            self._scores[threshold].append(scores_sorted[keep])
            self._tp[threshold].append(tp[keep])
            if threshold == self.primary_iou_threshold:
                primary_tp, primary_keep = tp, keep

        if primary_keep is not None:
            self._num_ignored += int((~primary_keep).sum().item())

        # IoU of the detections that matched at the primary threshold, for a
        # localization-quality readout that (unlike the positive-anchor IoU) is
        # computed on real post-NMS detections.
        if num_gt and primary_tp is not None and primary_tp.any():
            self._matched_ious.append(iou.max(dim=1).values[primary_tp > 0.5])

        if self.groups_enabled and group is not None and primary_keep is not None:
            self._group_scores.setdefault(group, []).append(scores_sorted[primary_keep])
            self._group_tp.setdefault(group, []).append(primary_tp[primary_keep])

    def compute(self) -> Dict[str, float]:
        """
        Compute pooled metrics over everything added since the last reset.

        Returns a dict with `map_50`, `map_50_95`, `detection_iou` (mean IoU of
        matched detections), `num_detections`, `num_ground_truth`, `num_images`,
        and `map_50/<group>` entries when groups are enabled.
        """
        primary_scores = self._scores[self.primary_iou_threshold]
        if not primary_scores:
            return {
                'map_50': 0.0,
                'map_50_95': 0.0,
                'detection_iou': 0.0,
                'num_detections': 0.0,
                'num_ignored': float(self._num_ignored),
                'num_ground_truth': float(self._num_gt),
                'num_images': float(self._num_images),
            }

        per_threshold = {
            threshold: average_precision(
                torch.cat(self._scores[threshold]),
                torch.cat(self._tp[threshold]),
                self._num_gt
            )
            for threshold in self.iou_thresholds
        }
        scores = torch.cat(primary_scores)

        detection_iou = (
            float(torch.cat(self._matched_ious).mean().item())
            if self._matched_ious else 0.0
        )

        metrics: Dict[str, float] = {
            'map_50': per_threshold[self.primary_iou_threshold],
            'map_50_95': sum(per_threshold.values()) / len(per_threshold),
            'detection_iou': detection_iou,
            'num_detections': float(scores.numel()),
            'num_ignored': float(self._num_ignored),
            'num_ground_truth': float(self._num_gt),
            'num_images': float(self._num_images),
        }

        # Iterate over every group that contributed ground truth, not only those
        # that produced detections: a group the detector missed entirely must
        # report 0.0 rather than disappear from the breakdown.
        for group, group_num_gt in self._group_num_gt.items():
            group_scores = self._group_scores.get(group)
            if group_scores:
                metrics[f'map_50/{group}'] = average_precision(
                    torch.cat(group_scores),
                    torch.cat(self._group_tp[group]),
                    group_num_gt
                )
            else:
                metrics[f'map_50/{group}'] = 0.0

        return metrics
