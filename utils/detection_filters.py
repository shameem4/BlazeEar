"""Duplicate suppression helpers shared across BlazeEar inference utilities."""
from __future__ import annotations

from typing import Tuple

import numpy as np

NEAR_CENTER_DISTANCE_FRAC = 0.55
NEAR_MIN_AREA_RATIO = 0.35
NEAR_MIN_COVERAGE = 0.8


def boxes_are_near_xyxy_np(
    candidate: np.ndarray,
    boxes: np.ndarray,
    max_center_distance_frac: float = NEAR_CENTER_DISTANCE_FRAC,
    min_area_ratio: float = NEAR_MIN_AREA_RATIO
) -> np.ndarray:
    """Return mask for boxes whose centers and areas nearly match candidate."""
    if boxes.size == 0:
        return np.zeros((0,), dtype=bool)

    candidate = candidate.astype(np.float32, copy=False)
    boxes = boxes.astype(np.float32, copy=False)

    cand_h = np.clip(candidate[2] - candidate[0], 1e-6, None)
    cand_w = np.clip(candidate[3] - candidate[1], 1e-6, None)
    cand_area = cand_h * cand_w
    cand_center_y = (candidate[0] + candidate[2]) / 2.0
    cand_center_x = (candidate[1] + candidate[3]) / 2.0

    box_h = np.clip(boxes[:, 2] - boxes[:, 0], 1e-6, None)
    box_w = np.clip(boxes[:, 3] - boxes[:, 1], 1e-6, None)
    box_area = box_h * box_w
    box_center_y = (boxes[:, 0] + boxes[:, 2]) / 2.0
    box_center_x = (boxes[:, 1] + boxes[:, 3]) / 2.0

    center_distance = np.sqrt(
        (cand_center_y - box_center_y) ** 2 + (cand_center_x - box_center_x) ** 2
    )
    other_max_dim = np.maximum(box_h, box_w)
    max_dim = np.maximum(np.maximum(cand_h, cand_w), other_max_dim)
    center_close = center_distance <= (max_center_distance_frac * max_dim)

    area_ratio = np.minimum(cand_area, box_area) / np.maximum(cand_area, box_area)
    return center_close & (area_ratio >= min_area_ratio)


def box_covers_target_xyxy_np(
    candidate: np.ndarray,
    boxes: np.ndarray,
    min_coverage: float = NEAR_MIN_COVERAGE
) -> np.ndarray:
    """Return mask for boxes that are mostly covered by candidate."""
    if boxes.size == 0:
        return np.zeros((0,), dtype=bool)

    candidate = candidate.astype(np.float32, copy=False)
    boxes = boxes.astype(np.float32, copy=False)

    y_min = np.maximum(candidate[0], boxes[:, 0])
    x_min = np.maximum(candidate[1], boxes[:, 1])
    y_max = np.minimum(candidate[2], boxes[:, 2])
    x_max = np.minimum(candidate[3], boxes[:, 3])

    inter_h = np.clip(y_max - y_min, 0.0, None)
    inter_w = np.clip(x_max - x_min, 0.0, None)
    inter_area = inter_h * inter_w
    target_area = np.clip((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), 1e-6, None)
    coverage = inter_area / target_area
    return coverage >= min_coverage


def _compute_keep_indices(
    boxes: np.ndarray,
    scores: np.ndarray,
    center_distance_frac: float,
    min_area_ratio: float,
    min_coverage: float
) -> np.ndarray:
    """Return indices of boxes that survive duplicate suppression."""
    if boxes.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    if boxes.shape[0] == 1:
        return np.array([0], dtype=np.int64)

    order = np.argsort(scores)[::-1]
    keep: list[int] = []

    for idx in order:
        candidate = boxes[idx]
        duplicate = False
        for kept_idx in keep:
            kept_box = boxes[kept_idx]
            if box_covers_target_xyxy_np(kept_box, candidate[np.newaxis, :], min_coverage)[0]:
                duplicate = True
                break
            if boxes_are_near_xyxy_np(kept_box, candidate[np.newaxis, :], center_distance_frac, min_area_ratio)[0]:
                duplicate = True
                break
        if not duplicate:
            keep.append(idx)

    if not keep:
        return np.zeros((0,), dtype=np.int64)

    return np.array(keep, dtype=np.int64)


def filter_duplicate_boxes(
    boxes: np.ndarray,
    scores: np.ndarray,
    center_distance_frac: float = NEAR_CENTER_DISTANCE_FRAC,
    min_area_ratio: float = NEAR_MIN_AREA_RATIO,
    min_coverage: float = NEAR_MIN_COVERAGE
) -> Tuple[np.ndarray, np.ndarray]:
    """Suppress duplicate predictions that differ mainly by scale."""
    if boxes.shape[0] <= 1:
        return boxes, scores

    keep = _compute_keep_indices(boxes, scores, center_distance_frac, min_area_ratio, min_coverage)
    if keep.size == 0:
        return boxes[:0], scores[:0]
    return boxes[keep], scores[keep]


def filter_duplicate_detections(
    detections: np.ndarray,
    center_distance_frac: float = NEAR_CENTER_DISTANCE_FRAC,
    min_area_ratio: float = NEAR_MIN_AREA_RATIO,
    min_coverage: float = NEAR_MIN_COVERAGE
) -> np.ndarray:
    """Filter duplicate detections while preserving score columns when present."""
    if detections is None:
        return detections

    dets = np.asarray(detections)
    if dets.ndim == 1:
        dets = dets.reshape(1, -1)

    if dets.shape[0] <= 1:
        return dets

    boxes = dets[:, :4].astype(np.float32, copy=False)
    if dets.shape[1] > 4:
        scores = dets[:, -1].astype(np.float32, copy=False)
    else:
        scores = np.ones(dets.shape[0], dtype=np.float32)

    keep = _compute_keep_indices(boxes, scores, center_distance_frac, min_area_ratio, min_coverage)
    if keep.size == 0:
        return dets[:0]
    return dets[keep]
