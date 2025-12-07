"""
Unified IoU (Intersection over Union) computation utilities.

Consolidates IoU implementations from across the codebase:
- NumPy version for data loading and evaluation
- PyTorch version for training and loss computation
- Batched versions for NMS and matching

All box formats supported:
- xyxy: [x1, y1, x2, y2] (corners)
- xywh: [x, y, width, height] (top-left + size)
- yxyx: [ymin, xmin, ymax, xmax] (MediaPipe convention)
"""
from __future__ import annotations

from typing import Tuple, Union

import numpy as np
import torch


# =============================================================================
# NumPy Implementations (for data loading, evaluation, CPU operations)
# =============================================================================

def compute_iou_np(
    box1: np.ndarray,
    box2: np.ndarray,
    box1_format: str = "yxyx",
    box2_format: str = "yxyx"
) -> float:
    """
    Compute IoU between two boxes using NumPy.
    
    Args:
        box1: First box
        box2: Second box
        box1_format: Format of box1 - "xyxy", "xywh", or "yxyx"
        box2_format: Format of box2 - "xyxy", "xywh", or "yxyx"
        
    Returns:
        IoU value (0.0 to 1.0)
    """
    # Convert to xyxy format for computation
    b1 = _convert_to_xyxy_np(box1, box1_format)
    b2 = _convert_to_xyxy_np(box2, box2_format)
    
    # Compute intersection
    x_min = max(b1[0], b2[0])
    y_min = max(b1[1], b2[1])
    x_max = min(b1[2], b2[2])
    y_max = min(b1[3], b2[3])
    
    inter_w = max(0.0, x_max - x_min)
    inter_h = max(0.0, y_max - y_min)
    inter_area = inter_w * inter_h
    
    # Compute areas
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union_area = area1 + area2 - inter_area
    
    if union_area <= 0:
        return 0.0
    
    return float(inter_area / union_area)


def compute_iou_batch_np(
    boxes1: np.ndarray,
    boxes2: np.ndarray,
    format: str = "yxyx"
) -> np.ndarray:
    """
    Compute pairwise IoU between two sets of boxes.
    
    Args:
        boxes1: [N, 4] array of boxes
        boxes2: [M, 4] array of boxes
        format: Box format - "xyxy", "xywh", or "yxyx"
        
    Returns:
        [N, M] array of IoU values
    """
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)
    
    # Convert to xyxy (vectorized)
    b1 = _convert_to_xyxy_np_batch(np.asarray(boxes1, dtype=np.float32), format)
    b2 = _convert_to_xyxy_np_batch(np.asarray(boxes2, dtype=np.float32), format)
    
    # Compute intersections [N, M]
    x_min = np.maximum(b1[:, None, 0], b2[None, :, 0])
    y_min = np.maximum(b1[:, None, 1], b2[None, :, 1])
    x_max = np.minimum(b1[:, None, 2], b2[None, :, 2])
    y_max = np.minimum(b1[:, None, 3], b2[None, :, 3])
    
    inter_w = np.maximum(0, x_max - x_min)
    inter_h = np.maximum(0, y_max - y_min)
    inter_area = inter_w * inter_h
    
    # Compute areas
    area1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
    area2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
    union_area = area1[:, None] + area2[None, :] - inter_area
    
    return inter_area / (union_area + 1e-6)


def _convert_to_xyxy_np(box: np.ndarray, format: str) -> np.ndarray:
    """Convert single box to xyxy format."""
    box = np.asarray(box, dtype=np.float32)
    if format == "xyxy":
        return box
    elif format == "xywh":
        return np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]])
    elif format == "yxyx":
        return np.array([box[1], box[0], box[3], box[2]])
    else:
        raise ValueError(f"Unknown format: {format}")


def _convert_to_xyxy_np_batch(boxes: np.ndarray, format: str) -> np.ndarray:
    """Convert batch of boxes to xyxy format (vectorized)."""
    if format == "xyxy":
        return boxes
    elif format == "xywh":
        return np.column_stack([
            boxes[:, 0], boxes[:, 1],
            boxes[:, 0] + boxes[:, 2], boxes[:, 1] + boxes[:, 3]
        ])
    elif format == "yxyx":
        return boxes[:, [1, 0, 3, 2]]
    else:
        raise ValueError(f"Unknown format: {format}")


# =============================================================================
# PyTorch Implementations (for training, loss, GPU operations)
# =============================================================================

def compute_iou_torch(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    format: str = "yxyx"
) -> torch.Tensor:
    """
    Compute pairwise IoU between two sets of boxes using PyTorch.
    
    Args:
        boxes1: [N, 4] tensor of boxes
        boxes2: [M, 4] tensor of boxes
        format: Box format - "xyxy", "xywh", or "yxyx"
        
    Returns:
        [N, M] tensor of IoU values
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros(
            (boxes1.shape[0], boxes2.shape[0]),
            device=boxes1.device if boxes1.numel() else boxes2.device,
            dtype=torch.float32
        )
    
    # Convert to xyxy format
    b1 = _convert_to_xyxy_torch(boxes1, format)
    b2 = _convert_to_xyxy_torch(boxes2, format)
    
    # Compute intersections [N, M]
    x_min = torch.maximum(b1[:, None, 0], b2[None, :, 0])
    y_min = torch.maximum(b1[:, None, 1], b2[None, :, 1])
    x_max = torch.minimum(b1[:, None, 2], b2[None, :, 2])
    y_max = torch.minimum(b1[:, None, 3], b2[None, :, 3])
    
    inter_w = torch.clamp(x_max - x_min, min=0)
    inter_h = torch.clamp(y_max - y_min, min=0)
    inter_area = inter_w * inter_h
    
    # Compute areas
    area1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
    area2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
    union_area = area1[:, None] + area2[None, :] - inter_area
    
    return inter_area / (union_area + 1e-6)


def compute_iou_elementwise_torch(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    format: str = "yxyx"
) -> torch.Tensor:
    """
    Compute element-wise IoU between aligned box tensors.
    
    Args:
        boxes1: [N, 4] tensor of boxes
        boxes2: [N, 4] tensor of boxes (same N as boxes1)
        format: Box format - "xyxy", "xywh", or "yxyx"
        
    Returns:
        [N] tensor of IoU values
    """
    if boxes1.numel() == 0:
        return torch.zeros(0, device=boxes1.device, dtype=torch.float32)
    
    # Convert to xyxy format
    b1 = _convert_to_xyxy_torch(boxes1, format)
    b2 = _convert_to_xyxy_torch(boxes2, format)
    
    # Compute intersections
    x_min = torch.maximum(b1[:, 0], b2[:, 0])
    y_min = torch.maximum(b1[:, 1], b2[:, 1])
    x_max = torch.minimum(b1[:, 2], b2[:, 2])
    y_max = torch.minimum(b1[:, 3], b2[:, 3])
    
    inter_w = torch.clamp(x_max - x_min, min=0)
    inter_h = torch.clamp(y_max - y_min, min=0)
    inter_area = inter_w * inter_h
    
    # Compute areas
    area1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
    area2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
    union_area = area1 + area2 - inter_area
    
    return inter_area / (union_area + 1e-6)


def _convert_to_xyxy_torch(boxes: torch.Tensor, format: str) -> torch.Tensor:
    """Convert boxes to xyxy format."""
    if format == "xyxy":
        return boxes
    elif format == "xywh":
        return torch.stack([
            boxes[:, 0],
            boxes[:, 1],
            boxes[:, 0] + boxes[:, 2],
            boxes[:, 1] + boxes[:, 3]
        ], dim=1)
    elif format == "yxyx":
        return torch.stack([
            boxes[:, 1],  # xmin
            boxes[:, 0],  # ymin
            boxes[:, 3],  # xmax
            boxes[:, 2]   # ymax
        ], dim=1)
    else:
        raise ValueError(f"Unknown format: {format}")


# =============================================================================
# NMS Helper Functions (used by BlazeDetector)
# =============================================================================

def intersect_torch(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
    """
    Compute intersection area between two sets of boxes.
    
    Used for NMS calculations. Boxes should be in [ymin, xmin, ymax, xmax] format.
    
    Args:
        box_a: [A, 4] tensor of boxes
        box_b: [B, 4] tensor of boxes
        
    Returns:
        [A, B] tensor of intersection areas
    """
    A = box_a.size(0)
    B = box_b.size(0)
    
    # box format: [ymin, xmin, ymax, xmax]
    max_xy = torch.min(
        box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
        box_b[:, 2:].unsqueeze(0).expand(A, B, 2)
    )
    min_xy = torch.max(
        box_a[:, :2].unsqueeze(1).expand(A, B, 2),
        box_b[:, :2].unsqueeze(0).expand(A, B, 2)
    )
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard_torch(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
    """
    Compute Jaccard overlap (IoU) between two sets of boxes.
    
    Boxes should be in [ymin, xmin, ymax, xmax] format.
    
    Args:
        box_a: [A, 4] tensor of boxes
        box_b: [B, 4] tensor of boxes
        
    Returns:
        [A, B] tensor of IoU values
    """
    inter = intersect_torch(box_a, box_b)
    
    # box format: [ymin, xmin, ymax, xmax]
    area_a = ((box_a[:, 2] - box_a[:, 0]) * 
              (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)
    area_b = ((box_b[:, 2] - box_b[:, 0]) * 
              (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)
    
    union = area_a + area_b - inter
    return inter / union


def overlap_similarity_torch(box: torch.Tensor, other_boxes: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between a single box and a set of other boxes.
    
    Used for NMS. Boxes should be in [ymin, xmin, ymax, xmax] format.
    
    Args:
        box: [4] tensor - single box
        other_boxes: [N, 4] tensor of boxes
        
    Returns:
        [N] tensor of IoU values
    """
    return jaccard_torch(box.unsqueeze(0), other_boxes).squeeze(0)


# =============================================================================
# Legacy Compatibility Functions
# =============================================================================

def compute_iou_legacy_xywh(
    gt_box: Tuple[int, int, int, int],
    det_box: np.ndarray
) -> float:
    """
    Compute IoU between ground truth (xywh) and detection (yxyx).
    
    Legacy function for metrics.py compatibility.
    
    Args:
        gt_box: (x1, y1, w, h) ground truth box
        det_box: [ymin, xmin, ymax, xmax] detection array
        
    Returns:
        IoU value
    """
    gt_x1, gt_y1, gt_w, gt_h = gt_box
    gt_x2 = gt_x1 + gt_w
    gt_y2 = gt_y1 + gt_h
    
    det_y1, det_x1, det_y2, det_x2 = det_box[0], det_box[1], det_box[2], det_box[3]
    
    inter_x1 = max(gt_x1, det_x1)
    inter_y1 = max(gt_y1, det_y1)
    inter_x2 = min(gt_x2, det_x2)
    inter_y2 = min(gt_y2, det_y2)
    
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    
    gt_area = gt_w * gt_h
    det_area = (det_x2 - det_x1) * (det_y2 - det_y1)
    union_area = gt_area + det_area - inter_area
    
    if union_area == 0:
        return 0.0
    return inter_area / union_area


def compute_mean_iou_torch(
    pred_boxes: torch.Tensor,
    true_boxes: torch.Tensor,
    scale: float = 128.0,
    format: str = "yxyx"
) -> torch.Tensor:
    """
    Compute mean IoU between predicted and true boxes.
    
    Args:
        pred_boxes: [N, 4] predicted boxes
        true_boxes: [N, 4] ground truth boxes
        scale: Scale factor (for normalized coords)
        format: Box format
        
    Returns:
        Mean IoU tensor (scalar)
    """
    if pred_boxes.numel() == 0:
        return torch.tensor(0.0, device=pred_boxes.device)
    
    pred_scaled = pred_boxes * scale
    true_scaled = true_boxes * scale
    
    ious = compute_iou_elementwise_torch(pred_scaled, true_scaled, format)
    return ious.mean()
