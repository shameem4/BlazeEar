"""
Metrics and evaluation utilities (IoU, matching, etc.).

Uses consolidated IoU functions from utils.iou module.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch

from utils.iou import (
    compute_iou_legacy_xywh,
    compute_iou_torch,
    compute_iou_batch_np,
    compute_mean_iou_torch as _compute_mean_iou_torch,
)


def compute_iou(
    box1: tuple[int, int, int, int],
    box2: np.ndarray
) -> float:
    """
    Compute IoU between a ground truth box (xywh) and a detection box (yxyx).
    
    Wrapper around utils.iou.compute_iou_legacy_xywh for backward compatibility.
    """
    return compute_iou_legacy_xywh(box1, box2)


def match_detections_to_ground_truth(
    gt_boxes: List[tuple[int, int, int, int]],
    detections: np.ndarray,
    iou_threshold: float = 0.3
) -> Tuple[List[int], List[float]]:
    """Match detections to ground truth boxes using IoU (greedy)."""
    if len(detections) == 0 or len(gt_boxes) == 0:
        return [], []

    num_gt = len(gt_boxes)
    num_det = len(detections)

    # Convert gt_boxes from xywh to yxyx for vectorized IoU
    gt_array = np.array(gt_boxes, dtype=np.float32)  # [N, 4] in xywh
    gt_yxyx = np.column_stack([
        gt_array[:, 1],  # ymin = y
        gt_array[:, 0],  # xmin = x
        gt_array[:, 1] + gt_array[:, 3],  # ymax = y + h
        gt_array[:, 0] + gt_array[:, 2]   # xmax = x + w
    ])
    
    # detections are in yxyx format, extract box coords (first 4 columns)
    det_boxes = detections[:, :4]
    
    # Compute all IoUs at once (vectorized)
    iou_matrix = compute_iou_batch_np(gt_yxyx, det_boxes, format="yxyx")

    matched_det_indices: List[int] = []
    matched_ious: List[float] = []
    used_detections = np.zeros(num_det, dtype=bool)

    gt_order = np.argsort(-iou_matrix.max(axis=1))

    for gt_idx in gt_order:
        # Vectorized: find best unmatched detection
        ious = iou_matrix[gt_idx]
        ious_masked = np.where(used_detections, -1.0, ious)
        best_det_idx = np.argmax(ious_masked)
        best_iou = ious_masked[best_det_idx]

        if best_iou >= iou_threshold:
            matched_det_indices.append(int(best_det_idx))
            matched_ious.append(float(best_iou))
            used_detections[best_det_idx] = True
        else:
            matched_det_indices.append(-1)
            matched_ious.append(0.0)

    return matched_det_indices, matched_ious


def compute_mean_iou_torch(
    pred_boxes: torch.Tensor,
    true_boxes: torch.Tensor,
    scale: float = 128.0
) -> torch.Tensor:
    """
    Mean IoU between predicted and true boxes (MediaPipe convention).
    
    Wrapper around utils.iou.compute_mean_iou_torch.
    """
    return _compute_mean_iou_torch(pred_boxes, true_boxes, scale=scale, format="yxyx")


def compute_map_torch(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    true_boxes: torch.Tensor,
    iou_threshold: float = 0.5
) -> torch.Tensor:
    """Compute mAP for single-class detection (vectorized IoU computation)."""
    if pred_boxes.numel() == 0 or true_boxes.numel() == 0:
        return torch.tensor(0.0, device=pred_boxes.device)

    sorted_indices = torch.argsort(pred_scores, descending=True)
    pred_boxes_sorted = pred_boxes[sorted_indices]

    num_pred = pred_boxes_sorted.shape[0]
    num_gt = true_boxes.shape[0]
    
    # Compute all IoUs at once [num_pred, num_gt]
    all_ious = compute_iou_torch(pred_boxes_sorted, true_boxes)
    max_ious, max_indices = all_ious.max(dim=1)  # [num_pred]
    
    # Pre-allocate TP/FP arrays
    tp = torch.zeros(num_pred, dtype=torch.float32, device=pred_boxes.device)
    fp = torch.ones(num_pred, dtype=torch.float32, device=pred_boxes.device)
    gt_matched = torch.zeros(num_gt, dtype=torch.bool, device=pred_boxes.device)

    # Sequential matching (inherently sequential due to greedy assignment)
    for i in range(num_pred):
        max_iou = max_ious[i]
        max_idx = max_indices[i]
        
        if max_iou >= iou_threshold and not gt_matched[max_idx]:
            tp[i] = 1.0
            fp[i] = 0.0
            gt_matched[max_idx] = True

    tp_cumsum = torch.cumsum(tp, dim=0)
    fp_cumsum = torch.cumsum(fp, dim=0)

    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    recall = tp_cumsum / (num_gt + 1e-6)

    # Vectorized AP computation using 11-point interpolation
    thresholds = torch.linspace(0, 1, 11, device=pred_boxes.device)
    ap = torch.zeros(1, device=pred_boxes.device)
    for t in thresholds:
        mask = recall >= t
        if mask.any():
            ap += precision[mask].max()
    ap /= 11
    return ap
