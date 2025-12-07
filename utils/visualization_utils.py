"""Shared visualization helpers for BlazeFace debugging and documentation."""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


def compute_resize_metadata(
    orig_h: int,
    orig_w: int,
    target_size: Tuple[int, int]
) -> Tuple[float, int, int]:
    """Recreate resize/pad parameters used during preprocessing."""
    target_h, target_w = target_size

    if orig_h >= orig_w:
        scale = target_h / orig_h
        new_h = target_h
        new_w = int(round(orig_w * scale))
    else:
        scale = target_w / orig_w
        new_w = target_w
        new_h = int(round(orig_h * scale))

    pad_h = target_h - new_h
    pad_w = target_w - new_w
    pad_top = pad_h // 2
    pad_left = pad_w // 2

    return scale, pad_top, pad_left


def map_preprocessed_boxes_to_original(
    boxes_xyxy: np.ndarray,
    orig_shape: Tuple[int, int],
    target_size: Tuple[int, int],
    scale: float,
    pad_top: int,
    pad_left: int
) -> np.ndarray:
    """Map boxes from preprocessed (padded) space back to original resolution."""
    if boxes_xyxy.size == 0:
        return np.empty((0, 4), dtype=np.float32)

    target_h, target_w = target_size
    orig_h, orig_w = orig_shape
    boxes = np.array(boxes_xyxy, dtype=np.float32, copy=True)

    boxes[:, [0, 2]] *= target_w  # x coords
    boxes[:, [1, 3]] *= target_h  # y coords
    boxes[:, [0, 2]] -= pad_left
    boxes[:, [1, 3]] -= pad_top
    boxes /= scale

    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h - 1)

    return boxes


def convert_ymin_xmin_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert [ymin, xmin, ymax, xmax] boxes to [xmin, ymin, xmax, ymax]."""
    if boxes.size == 0:
        return np.empty((0, 4), dtype=np.float32)
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.ndim == 1:
        boxes = boxes[None, :]
    return boxes[:, [1, 0, 3, 2]]


def draw_box(
    image: np.ndarray,
    box: np.ndarray,
    color: Tuple[int, int, int],
    label: Optional[str] = None,
    thickness: int = 2
) -> None:
    """Draw a single rectangle (with optional label) on the target image."""
    x1, y1, x2, y2 = box.astype(int).tolist()
    x1 = int(np.clip(x1, 0, image.shape[1] - 1))
    x2 = int(np.clip(x2, 0, image.shape[1] - 1))
    y1 = int(np.clip(y1, 0, image.shape[0] - 1))
    y2 = int(np.clip(y2, 0, image.shape[0] - 1))

    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        (text_w, text_h), baseline = cv2.getTextSize(label, font, scale, 1)
        text_y = max(y1 - baseline, text_h + 2)
        cv2.rectangle(
            image,
            (x1, text_y - text_h - baseline),
            (x1 + text_w, text_y + baseline // 2),
            color,
            thickness=cv2.FILLED
        )
        cv2.putText(
            image,
            label,
            (x1, text_y - 2),
            font,
            scale,
            (0, 0, 0),
            thickness=1,
            lineType=cv2.LINE_AA
        )
