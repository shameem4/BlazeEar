"""
Unified anchor generation and encoding utilities.

Consolidates anchor-related functions from blazebase.py and dataloader.py:
- Reference anchor generation for BlazeFace (896 anchors)
- Box-to-anchor encoding for training
- Anchor target flattening

MediaPipe convention: boxes are [ymin, xmin, ymax, xmax] normalized to [0, 1]
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


# =============================================================================
# Reference Anchor Generation
# =============================================================================

def generate_reference_anchors(
    input_size: int = 128,
    fixed_anchor_size: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate reference anchor centers for BlazeFace detector.
    
    Creates a grid of anchor centers for two scales:
    - 16x16 grid with 2 anchors per cell = 512 small anchors
    - 8x8 grid with 6 anchors per cell = 384 big anchors
    Total: 896 anchors
    
    Args:
        input_size: Input image size (default 128)
        fixed_anchor_size: If True, all anchors have w=h=1.0 (default).
                          If False, could support variable anchor sizes in future.
        
    Returns:
        reference_anchors: [896, 4] tensor of (x_center, y_center, width, height)
        small_anchors: [512, 4] tensor for 16x16 grid
        big_anchors: [384, 4] tensor for 8x8 grid
    """
    # Small anchors: 16x16 grid, size 0.0625 (1/16)
    # Centers at 0.03125, 0.09375, ..., 0.96875
    small_boxes = torch.linspace(0.03125, 0.96875, 16)
    
    # Big anchors: 8x8 grid, size 0.125 (1/8)  
    # Centers at 0.0625, 0.1875, ..., 0.9375
    big_boxes = torch.linspace(0.0625, 0.9375, 8)
    
    # Create grid for small anchors (16x16 with 2 anchors per cell = 512)
    # x coordinates: repeat each x 2 times, then tile 16 times
    small_x = small_boxes.repeat_interleave(2).repeat(16)  # 512
    # y coordinates: repeat each y 32 times (2 anchors * 16 x positions)
    small_y = small_boxes.repeat_interleave(32)  # 512
    # Width and height: 1.0 for fixed_anchor_size=True
    small_w = torch.ones_like(small_x)
    small_h = torch.ones_like(small_x)
    small_anchors = torch.stack([small_x, small_y, small_w, small_h], dim=1)  # [512, 4]
    
    # Create grid for big anchors (8x8 with 6 anchors per cell = 384)
    # x coordinates: repeat each x 6 times, then tile 8 times
    big_x = big_boxes.repeat_interleave(6).repeat(8)  # 384
    # y coordinates: repeat each y 48 times (6 anchors * 8 x positions)
    big_y = big_boxes.repeat_interleave(48)  # 384
    # Width and height: 1.0 for fixed_anchor_size=True
    big_w = torch.ones_like(big_x)
    big_h = torch.ones_like(big_x)
    big_anchors = torch.stack([big_x, big_y, big_w, big_h], dim=1)  # [384, 4]
    
    # Combine: small first, then big (matching model output order)
    reference_anchors = torch.cat([small_anchors, big_anchors], dim=0)  # [896, 4]
    
    return reference_anchors, small_anchors, big_anchors


# =============================================================================
# Box-to-Anchor Encoding (for training data preparation)
# =============================================================================

def _compute_iou_vectorized(
    box: np.ndarray,
    anchor_boxes: np.ndarray
) -> np.ndarray:
    """
    Compute IoU between one box and multiple anchor boxes (vectorized).
    
    Args:
        box: [4] single box [ymin, xmin, ymax, xmax]
        anchor_boxes: [N, 4] anchor boxes [ymin, xmin, ymax, xmax]
        
    Returns:
        [N] IoU values
    """
    # Intersection
    ymin = np.maximum(box[0], anchor_boxes[:, 0])
    xmin = np.maximum(box[1], anchor_boxes[:, 1])
    ymax = np.minimum(box[2], anchor_boxes[:, 2])
    xmax = np.minimum(box[3], anchor_boxes[:, 3])
    
    inter_h = np.maximum(0, ymax - ymin)
    inter_w = np.maximum(0, xmax - xmin)
    intersection = inter_h * inter_w
    
    # Areas
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    anchor_area = (anchor_boxes[:, 2] - anchor_boxes[:, 0]) * (anchor_boxes[:, 3] - anchor_boxes[:, 1])
    union = box_area + anchor_area - intersection
    
    return np.where(union > 0, intersection / union, 0.0)


def _assign_box_to_grid(
    box_coords: np.ndarray,
    encoded_box: np.ndarray,
    coords: np.ndarray,
    anchor_size: float,
    anchor_tensor: np.ndarray,
    occupied_ious: np.ndarray,
    input_size: int
) -> None:
    """
    Assign a single box to the best available anchor cell (vectorized).
    
    Args:
        box_coords: [ymin, xmin, ymax, xmax] normalized box
        encoded_box: [class, ymin, xmin, ymax, xmax] encoded target
        coords: Grid coordinates (16 or 8 values)
        anchor_size: Size of anchor (0.0625 or 0.125)
        anchor_tensor: [grid, grid, 5] output tensor to fill
        occupied_ious: [grid, grid] IoU values for occupied cells
        input_size: Image size for IoU computation
    """
    grid_size = coords.shape[0]
    
    # Build all anchor boxes at once using broadcasting
    # coords is [grid_size], we need [grid_size, grid_size] for y and x
    y_coords = coords[:, np.newaxis]  # [grid_size, 1]
    x_coords = coords[np.newaxis, :]  # [1, grid_size]
    
    # Broadcast to [grid_size, grid_size, 4]
    anchor_ymin = (y_coords - anchor_size) * input_size
    anchor_xmin = (x_coords - anchor_size) * input_size
    anchor_ymax = (y_coords + anchor_size) * input_size
    anchor_xmax = (x_coords + anchor_size) * input_size
    
    anchor_boxes = np.stack([
        np.broadcast_to(anchor_ymin, (grid_size, grid_size)),
        np.broadcast_to(anchor_xmin, (grid_size, grid_size)),
        np.broadcast_to(anchor_ymax, (grid_size, grid_size)),
        np.broadcast_to(anchor_xmax, (grid_size, grid_size))
    ], axis=-1).reshape(-1, 4)  # [grid_size*grid_size, 4]
    
    # Compute all IoUs at once
    box_scaled = box_coords * input_size
    iou_flat = _compute_iou_vectorized(box_scaled, anchor_boxes)
    iou_grid = iou_flat.reshape(grid_size, grid_size)

    flat_indices = np.argsort(iou_grid.ravel())[::-1]

    for flat_idx in flat_indices:
        iou = iou_grid.ravel()[flat_idx]
        if iou <= 0:
            break
        y_idx, x_idx = divmod(flat_idx, grid_size)
        if occupied_ious[y_idx, x_idx] == 0:
            anchor_tensor[y_idx, x_idx] = encoded_box
            occupied_ious[y_idx, x_idx] = iou
            return

    # No free slot with IoU>0, optionally replace if better
    best_flat = flat_indices[0]
    best_iou = iou_grid.ravel()[best_flat]
    if best_iou > 0:
        y_idx, x_idx = divmod(best_flat, grid_size)
        if best_iou > occupied_ious[y_idx, x_idx]:
            anchor_tensor[y_idx, x_idx] = encoded_box
            occupied_ious[y_idx, x_idx] = best_iou


def encode_boxes_to_anchors(
    boxes: np.ndarray,
    input_size: int = 128
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encode normalized boxes into MediaPipe anchor grids.
    
    Boxes should be in [ymin, xmin, ymax, xmax] format, normalized to [0, 1].
    
    Args:
        boxes: [N, 4] array of boxes in [ymin, xmin, ymax, xmax] format
        input_size: Input image size (default 128)
        
    Returns:
        small_anchors: [16, 16, 5] targets for 16x16 grid
        big_anchors: [8, 8, 5] targets for 8x8 grid
    """
    small_size = 0.0625
    big_size = 0.125
    small_coords = np.linspace(0.03125, 0.96875, 16, dtype=np.float32)
    big_coords = np.linspace(0.0625, 0.9375, 8, dtype=np.float32)

    small_anchor = np.zeros((16, 16, 5), dtype=np.float32)
    big_anchor = np.zeros((8, 8, 5), dtype=np.float32)
    small_ious = np.zeros((16, 16), dtype=np.float32)
    big_ious = np.zeros((8, 8), dtype=np.float32)

    for box in boxes:
        # box format: [ymin, xmin, ymax, xmax]
        encoded = np.array([1.0, box[0], box[1], box[2], box[3]], dtype=np.float32)
        _assign_box_to_grid(box, encoded, small_coords, small_size, small_anchor, small_ious, input_size)
        _assign_box_to_grid(box, encoded, big_coords, big_size, big_anchor, big_ious, input_size)

    return small_anchor, big_anchor


def flatten_anchor_targets(
    small_anchors: np.ndarray,
    big_anchors: np.ndarray
) -> np.ndarray:
    """
    Flatten anchor targets to (896, 5) layout.
    
    Args:
        small_anchors: [16, 16, 5] from 16x16 grid
        big_anchors: [8, 8, 5] from 8x8 grid
        
    Returns:
        [896, 5] array with repeated anchors per cell
    """
    # Small: 16x16 grid with 2 anchors per cell -> 512
    small_flat = np.repeat(small_anchors.reshape(-1, 5), 2, axis=0)
    # Big: 8x8 grid with 6 anchors per cell -> 384
    big_flat = np.repeat(big_anchors.reshape(-1, 5), 6, axis=0)
    return np.concatenate([small_flat, big_flat], axis=0)


def flatten_anchor_targets_torch(
    small_targets: np.ndarray,
    big_targets: np.ndarray
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Flatten anchor targets to match model output shape and return as tensors.
    
    Args:
        small_targets: [16, 16, 5] from 16x16 grid
        big_targets: [8, 8, 5] from 8x8 grid
        
    Returns:
        classes: [896] tensor of class labels (0 or 1)
        coords: [896, 4] tensor of box coordinates [ymin, xmin, ymax, xmax]
    """
    all_targets = flatten_anchor_targets(small_targets, big_targets)
    
    classes = torch.from_numpy(all_targets[:, 0])  # [896]
    coords = torch.from_numpy(all_targets[:, 1:])  # [896, 4]
    
    return classes, coords


# =============================================================================
# Anchor Options (MediaPipe configuration)
# =============================================================================

anchor_options = {
    "num_layers": 4,
    "min_scale": 0.1484375,
    "max_scale": 0.75,
    "input_size_height": 128,
    "input_size_width": 128,
    "anchor_offset_x": 0.5,
    "anchor_offset_y": 0.5,
    "strides": [8, 16, 16, 16],
    "aspect_ratios": [1.0],
    "reduce_boxes_in_lowest_layer": False,
    "interpolated_scale_aspect_ratio": 1.0,
    "fixed_anchor_size": True,
}
