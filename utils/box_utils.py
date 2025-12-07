"""
Unified box decoding and encoding utilities.

Consolidates box transformation functions from blazedetector.py and loss_functions.py:
- Anchor prediction decoding (raw outputs -> absolute coordinates)
- Box format conversions

MediaPipe convention: boxes are [ymin, xmin, ymax, xmax] normalized to [0, 1]
"""
from __future__ import annotations

import torch


def decode_boxes(
    anchor_predictions: torch.Tensor,
    reference_anchors: torch.Tensor,
    scale: float = 128.0
) -> torch.Tensor:
    """
    Decode anchor predictions to absolute box coordinates.

    Following vincent1bt/blazeface-tensorflow decoding (no anchor w/h scaling):
    - x_center = anchor_x + (pred_x / scale)
    - y_center = anchor_y + (pred_y / scale)
    - w = pred_w / scale
    - h = pred_h / scale

    Args:
        anchor_predictions: [B, 896, 4+] predicted offsets [dx, dy, w, h, ...]
        reference_anchors: [896, 2] or [896, 4] anchor centers [x, y, ...]
        scale: Image scale for decoding (128 for front, 256 for back)

    Returns:
        [B, 896, 4] decoded boxes [ymin, xmin, ymax, xmax] in normalized coords
    """
    # Handle different anchor formats
    if reference_anchors.shape[1] >= 4:
        anchor_x = reference_anchors[:, 0:1]
        anchor_y = reference_anchors[:, 1:2]
        anchor_w = reference_anchors[:, 2:3]
        anchor_h = reference_anchors[:, 3:4]
    else:
        anchor_x = reference_anchors[:, 0:1]
        anchor_y = reference_anchors[:, 1:2]
        anchor_w = anchor_h = torch.ones_like(anchor_x)

    # Decode center and size (raw layout = [dx, dy, w, h])
    x_center = anchor_x + (anchor_predictions[..., 0:1] / scale) * anchor_w
    y_center = anchor_y + (anchor_predictions[..., 1:2] / scale) * anchor_h

    w = (anchor_predictions[..., 2:3] / scale) * anchor_w
    h = (anchor_predictions[..., 3:4] / scale) * anchor_h

    # Convert to corners - [ymin, xmin, ymax, xmax] to match MediaPipe format
    y_min = y_center - h / 2
    x_min = x_center - w / 2
    y_max = y_center + h / 2
    x_max = x_center + w / 2

    return torch.cat([y_min, x_min, y_max, x_max], dim=-1)


def decode_boxes_with_keypoints(
    raw_boxes: torch.Tensor,
    anchors: torch.Tensor,
    x_scale: float = 128.0,
    y_scale: float = 128.0,
    w_scale: float = 128.0,
    h_scale: float = 128.0,
    num_keypoints: int = 6
) -> torch.Tensor:
    """
    Decode anchor predictions including keypoint coordinates.
    
    Used by BlazeDetector for full detection output.
    
    Args:
        raw_boxes: [B, 896, num_coords] raw predictions
        anchors: [896, 4] anchor boxes [x, y, w, h]
        x_scale, y_scale, w_scale, h_scale: Scale factors
        num_keypoints: Number of keypoints to decode
        
    Returns:
        [B, 896, num_coords] decoded boxes with keypoints
    """
    boxes = torch.zeros_like(raw_boxes)

    # Decode center and size (raw layout = [dx, dy, w, h])
    x_center = raw_boxes[..., 0] / x_scale * anchors[:, 2] + anchors[:, 0]
    y_center = raw_boxes[..., 1] / y_scale * anchors[:, 3] + anchors[:, 1]

    w = raw_boxes[..., 2] / w_scale * anchors[:, 2]
    h = raw_boxes[..., 3] / h_scale * anchors[:, 3]

    boxes[..., 0] = y_center - h / 2.  # ymin
    boxes[..., 1] = x_center - w / 2.  # xmin
    boxes[..., 2] = y_center + h / 2.  # ymax
    boxes[..., 3] = x_center + w / 2.  # xmax

    # Decode keypoint coordinates (vectorized)
    # MediaPipe stores x,y pairs after the box coords
    if num_keypoints > 0 and raw_boxes.shape[-1] > 4:
        kp_end = min(4 + num_keypoints * 2, raw_boxes.shape[-1])
        num_coords = kp_end - 4
        if num_coords >= 2:
            # Decode all keypoints at once using slicing
            # x coordinates are at indices 4, 6, 8, ... (even offsets from 4)
            # y coordinates are at indices 5, 7, 9, ... (odd offsets from 4)
            boxes[..., 4:kp_end:2] = raw_boxes[..., 4:kp_end:2] / x_scale * anchors[:, 2:3] + anchors[:, 0:1]
            boxes[..., 5:kp_end:2] = raw_boxes[..., 5:kp_end:2] / y_scale * anchors[:, 3:4] + anchors[:, 1:2]

    return boxes


def xyxy_to_yxyx(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from [x1, y1, x2, y2] to [y1, x1, y2, x2] format.
    
    Args:
        boxes: [..., 4] tensor in xyxy format
        
    Returns:
        [..., 4] tensor in yxyx format
    """
    return torch.stack([
        boxes[..., 1],  # y1
        boxes[..., 0],  # x1
        boxes[..., 3],  # y2
        boxes[..., 2]   # x2
    ], dim=-1)


def yxyx_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from [y1, x1, y2, x2] to [x1, y1, x2, y2] format.
    
    Args:
        boxes: [..., 4] tensor in yxyx format
        
    Returns:
        [..., 4] tensor in xyxy format
    """
    return torch.stack([
        boxes[..., 1],  # x1
        boxes[..., 0],  # y1
        boxes[..., 3],  # x2
        boxes[..., 2]   # y2
    ], dim=-1)


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from [x, y, w, h] to [x1, y1, x2, y2] format.
    
    Args:
        boxes: [..., 4] tensor in xywh format
        
    Returns:
        [..., 4] tensor in xyxy format
    """
    return torch.stack([
        boxes[..., 0],
        boxes[..., 1],
        boxes[..., 0] + boxes[..., 2],
        boxes[..., 1] + boxes[..., 3]
    ], dim=-1)


def xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from [x1, y1, x2, y2] to [x, y, w, h] format.
    
    Args:
        boxes: [..., 4] tensor in xyxy format
        
    Returns:
        [..., 4] tensor in xywh format
    """
    return torch.stack([
        boxes[..., 0],
        boxes[..., 1],
        boxes[..., 2] - boxes[..., 0],
        boxes[..., 3] - boxes[..., 1]
    ], dim=-1)


def clip_boxes(boxes: torch.Tensor, min_val: float = 0.0, max_val: float = 1.0) -> torch.Tensor:
    """
    Clip box coordinates to valid range.
    
    Args:
        boxes: [..., 4] tensor of boxes
        min_val: Minimum value (default 0.0)
        max_val: Maximum value (default 1.0)
        
    Returns:
        Clipped boxes tensor
    """
    return torch.clamp(boxes, min=min_val, max=max_val)
