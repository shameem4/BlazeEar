"""
Unified anchor generation and encoding utilities.

Consolidates anchor-related functions from blazebase.py and dataloader.py:
- Reference anchor generation for BlazeFace (896 anchors)
- Anchor generation from explicit (width, height) priors
- Per-anchor target assignment with an ignore band

The previous cell-based encoder (encode_boxes_to_anchors / flatten_anchor_targets)
has been removed. It emitted one target per grid *cell* and repeated it 2x and 6x,
so every anchor in a cell necessarily shared a target and per-anchor priors could
not be expressed; it also hardcoded square anchor sizes instead of reading the
anchor tensor, so the two could silently disagree. Use assign_anchor_targets.

MediaPipe convention: boxes are [ymin, xmin, ymax, xmax] normalized to [0, 1]
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


# =============================================================================
# Reference Anchor Generation
# =============================================================================

def _calculate_scale(
    min_scale: float,
    max_scale: float,
    stride_index: int,
    num_strides: int
) -> float:
    if num_strides <= 1:
        return (min_scale + max_scale) / 2.0
    return min_scale + (max_scale - min_scale) * stride_index / (num_strides - 1.0)


def _generate_variable_anchors_from_options(options: dict) -> torch.Tensor:
    """Generate variable-size anchors following MediaPipe BlazeFace options."""
    strides = options["strides"]
    num_layers = options["num_layers"]
    assert num_layers == len(strides)

    anchors: list[list[float]] = []
    layer_id = 0
    strides_size = len(strides)

    while layer_id < strides_size:
        anchor_height: list[float] = []
        anchor_width: list[float] = []
        aspect_ratios: list[float] = []
        scales: list[float] = []

        last_same_stride_layer = layer_id
        while (
            last_same_stride_layer < strides_size
            and strides[last_same_stride_layer] == strides[layer_id]
        ):
            scale = _calculate_scale(
                options["min_scale"],
                options["max_scale"],
                last_same_stride_layer,
                strides_size,
            )

            if last_same_stride_layer == 0 and options.get("reduce_boxes_in_lowest_layer", False):
                aspect_ratios.extend([1.0, 2.0, 0.5])
                scales.extend([0.1, scale, scale])
            else:
                for aspect_ratio in options["aspect_ratios"]:
                    aspect_ratios.append(float(aspect_ratio))
                    scales.append(float(scale))

                interpolated = float(options.get("interpolated_scale_aspect_ratio", 0.0))
                if interpolated > 0.0:
                    if last_same_stride_layer == strides_size - 1:
                        scale_next = 1.0
                    else:
                        scale_next = _calculate_scale(
                            options["min_scale"],
                            options["max_scale"],
                            last_same_stride_layer + 1,
                            strides_size,
                        )
                    scales.append(float(np.sqrt(scale * scale_next)))
                    aspect_ratios.append(interpolated)

            last_same_stride_layer += 1

        for i in range(len(aspect_ratios)):
            ratio_sqrt = float(np.sqrt(aspect_ratios[i]))
            anchor_height.append(scales[i] / ratio_sqrt)
            anchor_width.append(scales[i] * ratio_sqrt)

        stride = strides[layer_id]
        feature_map_height = int(np.ceil(options["input_size_height"] / stride))
        feature_map_width = int(np.ceil(options["input_size_width"] / stride))

        for y in range(feature_map_height):
            for x in range(feature_map_width):
                x_center = (x + options["anchor_offset_x"]) / feature_map_width
                y_center = (y + options["anchor_offset_y"]) / feature_map_height
                for aw, ah in zip(anchor_width, anchor_height):
                    anchors.append([x_center, y_center, aw, ah])

        layer_id = last_same_stride_layer

    return torch.tensor(anchors, dtype=torch.float32)


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
                          If False, anchor w/h are derived from MediaPipe
                          `anchor_options` (variable-size anchors).
        
    Returns:
        reference_anchors: [896, 4] tensor of (x_center, y_center, width, height)
        small_anchors: [512, 4] tensor for 16x16 grid
        big_anchors: [384, 4] tensor for 8x8 grid
    """
    if fixed_anchor_size:
        # Small anchors: 16x16 grid, size 0.0625 (1/16)
        # Centers at 0.03125, 0.09375, ..., 0.96875
        small_boxes = torch.linspace(0.03125, 0.96875, 16)

        # Big anchors: 8x8 grid, size 0.125 (1/8)
        # Centers at 0.0625, 0.1875, ..., 0.9375
        big_boxes = torch.linspace(0.0625, 0.9375, 8)

        small_x = small_boxes.repeat_interleave(2).repeat(16)  # 512
        small_y = small_boxes.repeat_interleave(32)  # 512
        small_w = torch.ones_like(small_x)
        small_h = torch.ones_like(small_x)
        small_anchors = torch.stack([small_x, small_y, small_w, small_h], dim=1)

        big_x = big_boxes.repeat_interleave(6).repeat(8)  # 384
        big_y = big_boxes.repeat_interleave(48)  # 384
        big_w = torch.ones_like(big_x)
        big_h = torch.ones_like(big_x)
        big_anchors = torch.stack([big_x, big_y, big_w, big_h], dim=1)

        reference_anchors = torch.cat([small_anchors, big_anchors], dim=0)
        return reference_anchors, small_anchors, big_anchors

    options = dict(anchor_options)
    options["input_size_height"] = input_size
    options["input_size_width"] = input_size
    options["fixed_anchor_size"] = False

    reference_anchors = _generate_variable_anchors_from_options(options)
    small_count = 16 * 16 * 2
    small_anchors = reference_anchors[:small_count]
    big_anchors = reference_anchors[small_count:]
    return reference_anchors, small_anchors, big_anchors


# =============================================================================
# Box-to-Anchor Encoding (for training data preparation)
# =============================================================================

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


# =============================================================================
# Anchor generation from explicit priors
# =============================================================================

def generate_anchors_from_priors(
    small_priors=None,
    big_priors=None,
    small_grid: int = 16,
    big_grid: int = 8
) -> torch.Tensor:
    """
    Build the 896-anchor tensor from explicit (width, height) priors.

    Keeps the MediaPipe layout -- `small_grid**2 * len(small_priors)` anchors
    followed by `big_grid**2 * len(big_priors)` -- so the exported graph shape
    and the decode convention are unchanged; only the w/h priors differ.

    Returns:
        [A, 4] tensor of (x_center, y_center, width, height), normalized.
    """
    from utils.config import EAR_ANCHOR_PRIORS_SMALL, EAR_ANCHOR_PRIORS_BIG

    small_priors = EAR_ANCHOR_PRIORS_SMALL if small_priors is None else small_priors
    big_priors = EAR_ANCHOR_PRIORS_BIG if big_priors is None else big_priors

    step_small = 1.0 / (2 * small_grid)
    step_big = 1.0 / (2 * big_grid)
    small_centres = np.linspace(step_small, 1.0 - step_small, small_grid)
    big_centres = np.linspace(step_big, 1.0 - step_big, big_grid)

    anchors = []
    for centres, priors in ((small_centres, small_priors), (big_centres, big_priors)):
        for y in centres:
            for x in centres:
                for w, h in priors:
                    anchors.append([float(x), float(y), float(w), float(h)])

    return torch.tensor(anchors, dtype=torch.float32)


def anchors_to_corners(anchors: np.ndarray) -> np.ndarray:
    """(x, y, w, h) -> [ymin, xmin, ymax, xmax], matching the box convention."""
    x, y, w, h = anchors[:, 0], anchors[:, 1], anchors[:, 2], anchors[:, 3]
    return np.stack([y - h / 2, x - w / 2, y + h / 2, x + w / 2], axis=1)


def _iou_matrix(boxes: np.ndarray, anchor_corners: np.ndarray) -> np.ndarray:
    """IoU between [G, 4] boxes and [A, 4] anchors, both [ymin, xmin, ymax, xmax]."""
    if len(boxes) == 0:
        return np.zeros((0, len(anchor_corners)), dtype=np.float32)

    ymin = np.maximum(boxes[:, None, 0], anchor_corners[None, :, 0])
    xmin = np.maximum(boxes[:, None, 1], anchor_corners[None, :, 1])
    ymax = np.minimum(boxes[:, None, 2], anchor_corners[None, :, 2])
    xmax = np.minimum(boxes[:, None, 3], anchor_corners[None, :, 3])

    inter = np.clip(ymax - ymin, 0, None) * np.clip(xmax - xmin, 0, None)
    area_b = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))[:, None]
    area_a = ((anchor_corners[:, 2] - anchor_corners[:, 0])
              * (anchor_corners[:, 3] - anchor_corners[:, 1]))[None, :]
    union = np.clip(area_b + area_a - inter, 1e-12, None)
    return (inter / union).astype(np.float32)


def assign_anchor_targets(
    boxes: np.ndarray,
    anchors: np.ndarray,
    top_k: int = 3,
    ignore_iou: float = 0.35
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assign ground-truth boxes to anchors, best-match first, with an ignore band.

    Assignment is per anchor and driven by the actual anchor tensor, rather than
    per grid *cell* with hardcoded square sizes. The previous encoder produced a
    [16,16,5] / [8,8,5] target that was then repeated 2x and 6x, so every anchor
    in a cell necessarily shared one target and per-anchor priors could not be
    expressed at all.

    Each box claims its `top_k` highest-IoU anchors, so every box gets
    supervision regardless of how poorly the priors fit -- necessary here, since
    only 23.8% of ears reach IoU 0.5 with even fitted priors on this grid.
    Conflicts go to the box with the higher IoU.

    Anchors that are not positive but overlap some box by at least `ignore_iou`
    are marked ignore: excluded from the positive set *and* from hard negative
    mining. Without this, an anchor overlapping an ear at IoU 0.9 that merely
    lost the top-k race is trained as background, and hard negative mining
    specifically seeks out such high-scoring "negatives".

    Args:
        boxes: [G, 4] ground truth, [ymin, xmin, ymax, xmax] normalized
        anchors: [A, 4] anchors as (x_center, y_center, w, h) normalized
        top_k: positives per ground-truth box
        ignore_iou: overlap above which a non-positive anchor is ignored

    Returns:
        targets: [A, 5] of (class, ymin, xmin, ymax, xmax)
        ignore:  [A] bool, True where the anchor is neither positive nor negative
    """
    num_anchors = len(anchors)
    targets = np.zeros((num_anchors, 5), dtype=np.float32)
    ignore = np.zeros(num_anchors, dtype=bool)

    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if len(boxes) == 0:
        return targets, ignore

    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes = boxes[valid]
    if len(boxes) == 0:
        return targets, ignore

    iou = _iou_matrix(boxes, anchors_to_corners(np.asarray(anchors, dtype=np.float32)))

    k = int(min(max(1, top_k), num_anchors))
    candidate_anchors = np.argpartition(-iou, k - 1, axis=1)[:, :k]      # [G, k]
    candidate_gt = np.repeat(np.arange(len(boxes)), k)
    candidate_anchors = candidate_anchors.reshape(-1)
    candidate_iou = iou[candidate_gt, candidate_anchors]

    # Strongest claim wins each anchor.
    assigned_iou = np.full(num_anchors, -1.0, dtype=np.float32)
    for order in np.argsort(-candidate_iou):
        a = candidate_anchors[order]
        score = candidate_iou[order]
        if score <= 0.0 or score <= assigned_iou[a]:
            continue
        assigned_iou[a] = score
        targets[a, 0] = 1.0
        targets[a, 1:] = boxes[candidate_gt[order]]

    positive = targets[:, 0] > 0.5
    ignore = (iou.max(axis=0) >= ignore_iou) & ~positive
    return targets, ignore


# =============================================================================
# Canonical anchors
# =============================================================================

_ANCHOR_CACHE: dict = {}


def get_anchors(device=None) -> torch.Tensor:
    """
    The one anchor tensor. Every path must use this.

    Anchors were previously generated in four places -- the dataloader for
    assignment, the trainer for decoding inside the loss, `BlazeEar.generate_anchors`
    for `process()`, and `BlazeEarInference._generate_anchors` for the exported
    graph. Three of them independently hardcoded w=h=1.0. Changing the priors in
    one place silently left the others behind, which is survivable only while
    every copy happens to agree.

    Decode reads w/h as the scale the network's raw prediction is multiplied by,
    so these priors are not only a matching device: the network regresses
    relative to them. Training and inference must therefore use identical values
    or the boxes come out at the wrong scale.

    Returns:
        [896, 4] of (x_center, y_center, width, height), normalized.
    """
    key = str(device)
    if key not in _ANCHOR_CACHE:
        _ANCHOR_CACHE[key] = generate_anchors_from_priors().to(device) if device is not None \
            else generate_anchors_from_priors()
    return _ANCHOR_CACHE[key]
