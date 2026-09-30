"""The one non-maximum suppression used by every detection path.

There were five: a Python loop in `blazedetector`, torchvision in
`blazeear_inference`, a second Python loop in the trainer's evaluation,
torchvision again in the ONNX export, and a third loop in the two-stage
evaluator. They disagreed on the boundary case and ran at three different IoU
thresholds, so the mAP the trainer reported described post-processing that no
deployed path performed.

`torchvision.ops.nms` is the implementation: it is exact, it suppresses at
IoU strictly above the threshold (which is what three of the five did), and
it is a single fused op rather than a Python loop that synchronises the GPU
on every iteration.
"""
from typing import Optional, Tuple

import torch

from utils.box_utils import yxyx_to_xyxy
from utils.config import MAX_DETECTIONS, NMS_IOU_THRESHOLD


def nms_indices(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = NMS_IOU_THRESHOLD,
    max_detections: Optional[int] = MAX_DETECTIONS,
) -> torch.Tensor:
    """Indices to keep, highest score first.

    Args:
        boxes: [N, 4] as [ymin, xmin, ymax, xmax].
        scores: [N].
        iou_threshold: suppress a box overlapping a kept one above this.
        max_detections: truncate to this many, or None for all.

    Returns:
        A LongTensor of indices into `boxes`.
    """
    if scores.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=scores.device)

    import torchvision.ops

    keep = torchvision.ops.nms(
        yxyx_to_xyxy(boxes[:, :4]).float(), scores.float(), iou_threshold)
    if max_detections is not None and keep.numel() > max_detections:
        keep = keep[:max_detections]
    return keep


def suppress_overlapping(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = NMS_IOU_THRESHOLD,
    max_detections: Optional[int] = MAX_DETECTIONS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """`nms_indices`, already applied. Returns (boxes, scores)."""
    keep = nms_indices(boxes, scores, iou_threshold, max_detections)
    return boxes[keep], scores[keep]
