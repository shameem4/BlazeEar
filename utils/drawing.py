"""
Drawing utilities for visualizing detections and ground truth.
"""
import cv2
import torch
import numpy as np
from pathlib import Path


def draw_detections(
    img: np.ndarray,
    detections: torch.Tensor | np.ndarray,
    indices: list[int] | None = None,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    label: str = "Det"
) -> None:
    """Draw bounding boxes and confidence scores from detections.

    Args:
        img: Image to draw on (modified in place)
        detections: Detection tensor [N, 5] with format [ymin, xmin, ymax, xmax, score]
        indices: Optional list of detection indices to draw (if None, draws all)
        color: BGR color tuple (default: green for detections)
        thickness: Line thickness
        label: Label prefix for detections
    """
    if isinstance(detections, torch.Tensor):
        detections = detections.cpu().numpy()

    if len(detections) == 0:
        return

    if detections.ndim == 1:
        detections = np.expand_dims(detections, axis=0)

    # Determine which detections to draw
    if indices is None:
        indices_to_draw = list(range(detections.shape[0]))
    else:
        # Filter out -1 (unmatched GT boxes)
        indices_to_draw = [idx for idx in indices if idx != -1]

    for display_idx, det_idx in enumerate(indices_to_draw):
        ymin = int(detections[det_idx, 0])
        xmin = int(detections[det_idx, 1])
        ymax = int(detections[det_idx, 2])
        xmax = int(detections[det_idx, 3])

        # Get confidence score (last column when >4 dims)
        if detections.shape[1] > 4:
            score = detections[det_idx, detections.shape[1] - 1]
        else:
            score = 0.0

        # Draw axis-aligned bounding box
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, thickness)

        # Draw confidence score and label above the box
        label_text = f"{label} {display_idx+1}: {score:.2f}"
        (label_w, label_h), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        # Position label above box, or inside if at top edge
        label_y = ymin - 5 if ymin > label_h + 5 else ymin + label_h + 5
        cv2.putText(
            img, label_text, (xmin, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
        )


def draw_ground_truth_boxes(
    img: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    ious: list[float] | None = None,
    color: tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
    label: str = "GT"
) -> None:
    """Draw ground truth bounding boxes from CSV annotations.

    Args:
        img: Image to draw on (modified in place)
        boxes: List of boxes in format (x1, y1, w, h)
        ious: Optional list of IoU values for each box
        color: BGR color tuple (default: blue for ground truth)
        thickness: Line thickness
        label: Label text to show above boxes
    """
    for i, (x1, y1, w, h) in enumerate(boxes):
        # Convert (x1, y1, w, h) to (x1, y1, x2, y2)
        x2 = x1 + w
        y2 = y1 + h

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # Draw label above box with IoU if available
        if ious is not None and i < len(ious):
            label_text = f"{label} {i+1} (IoU: {ious[i]:.2f})"
        else:
            label_text = f"{label} {i+1}"

        (label_w, label_h), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        label_y = y1 - 5 if y1 > label_h + 5 else y1 + label_h + 5
        cv2.putText(
            img, label_text, (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
        )


def draw_fps(img: np.ndarray, fps: float) -> None:
    """Draw FPS counter on image.

    Args:
        img: Image to draw on (modified in place)
        fps: FPS value to display
    """
    text = f"FPS: {fps:.1f}"
    cv2.putText(
        img, text, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA
    )


def draw_info_text(
    img: np.ndarray,
    image_idx: int,
    total_images: int,
    num_gt_boxes: int,
    num_detections: int,
    num_matched: int,
    avg_iou: float,
    image_path: str,
    detection_only: bool = False
) -> None:
    """Draw information text at the top of the image.

    Args:
        img: Image to draw on (modified in place)
        image_idx: Current image index
        total_images: Total number of images
        num_gt_boxes: Number of ground truth boxes
        num_detections: Number of detections
        num_matched: Number of matched detections
        avg_iou: Average IoU of matches
        image_path: Path to current image
        detection_only: If True, only show detection count (no GT/IoU info)
    """
    # Background for text
    overlay = img.copy()
    height = 80 if detection_only else 100
    cv2.rectangle(overlay, (0, 0), (img.shape[1], height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)

    # Text lines
    if detection_only:
        text_lines = [
            f"Image {image_idx + 1}/{total_images}: {Path(image_path).name}",
            f"Detections: {num_detections} boxes",
            "Controls: A/D or Arrow keys to navigate | Q/ESC to quit"
        ]
    else:
        text_lines = [
            f"Image {image_idx + 1}/{total_images}: {Path(image_path).name}",
            f"Ground Truth: {num_gt_boxes} boxes | Total Detections: {num_detections} | Matched: {num_matched}",
            f"Average IoU: {avg_iou:.3f}" if num_matched > 0 else "Average IoU: N/A",
            "Controls: A/D or Arrow keys to navigate | Q/ESC to quit"
        ]

    y_offset = 20
    for line in text_lines:
        cv2.putText(
            img, line, (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )
        y_offset += 20
