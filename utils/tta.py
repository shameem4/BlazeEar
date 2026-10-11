"""Test-time augmentation for the two-stage pipeline.

One ear is detected several times through nuisance-perturbed face crops, and
the agreeing boxes are fused into one. The ear never moved, so the spread
between those detections is the model's own noise; averaging cancels part of
it. Measured on the common held-out set, this is the single largest gain
available without retraining or relabelling.

A fused box takes the HIGHEST score among its members. Dividing the summed
score by the view count instead ("vote") was tried first, on the theory that
quartering a box only one view in four saw would suppress false positives. It
does the opposite: at the full score range it demotes real ears that a single
view found, wrecking the ranking mAP depends on. Measured on the common
held-out set at four views, mAP@0.5 is 0.6295 under `max` against 0.5887 under
`vote`, and at two views `vote` (0.5530) is worse than no TTA at all (0.5809).
An earlier comparison that found the two rules equivalent had thresholded every
view at 0.70 before fusing, which hides exactly the part of the curve the rules
disagree about.

`VIEWS` must stay in step with the view list in docs/blazeear_inference.js, the
same way crop_window() and _cropWindow() must; test_tta.py pins the fusion
against hand-computed cases that the JS is checked against too.
"""
from __future__ import annotations

from typing import Sequence

import torch

from utils.detection_eval import pairwise_iou
from utils.nms import suppress_overlapping

# (expand, flip). The default view first, so a single-view run is exactly the
# pipeline without TTA. Measured: the flip carries most of the gain, and the
# two scale views add the rest; beyond four the curve is flat.
VIEWS: tuple[tuple[float, bool], ...] = (
    (1.50, False),
    (1.50, True),
    (1.35, False),
    (1.65, False),
)

FUSION_IOU_THRESHOLD = 0.5


def merge_views(
    boxes: Sequence[torch.Tensor],
    scores: Sequence[torch.Tensor],
    num_views: int,
    iou_threshold: float = FUSION_IOU_THRESHOLD,
    rule: str = 'max',
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse per-view detections into one set of boxes.

    Boxes are clustered greedily from the highest-scoring one down; each
    cluster becomes the score-weighted mean of its members.

    Args:
        boxes: one (N, 4) tensor per view, in a common coordinate frame
        scores: one (N,) tensor per view
        num_views: how many views were run, which divides each fused score
        iou_threshold: overlap at which two detections are the same ear
        rule: how a cluster's score is formed -- 'max' (the default, and the
            only one measured to help), 'mean', or 'vote' (sum / num_views)

    Fusion clusters at `iou_threshold`, which is deliberately looser than the
    pipeline's NMS. Detections of one ear that disagree by more than that stay
    separate, so the fused set is then put through the ordinary NMS as well --
    without it, running more views simply emits more boxes. Measured in the
    browser on a two-face image, the counts went 7 / 12 / 17 for 1 / 2 / 4
    views before this suppression was added.

    Returns:
        (M, 4) fused boxes and (M,) fused scores, highest score first.
    """
    if num_views < 1:
        raise ValueError('num_views must be at least 1')
    boxes = [b for b in boxes if len(b)]
    scores = [s for s in scores if len(s)]
    if not boxes:
        return torch.zeros((0, 4)), torch.zeros((0,))

    all_boxes = torch.cat(boxes).float()
    all_scores = torch.cat(scores).float()
    ious = pairwise_iou(all_boxes, all_boxes)

    order = torch.argsort(all_scores, descending=True)
    used = torch.zeros(len(all_boxes), dtype=torch.bool, device=all_boxes.device)
    fused_boxes, fused_scores = [], []
    for idx in order.tolist():
        if used[idx]:
            continue
        members = ((ious[idx] >= iou_threshold) & ~used).nonzero().flatten()
        used[members] = True
        weights = all_scores[members].unsqueeze(1)
        fused_boxes.append((all_boxes[members] * weights).sum(0) / weights.sum())
        member_scores = all_scores[members]
        if rule == 'vote':
            fused_scores.append(member_scores.sum() / num_views)
        elif rule == 'max':
            fused_scores.append(member_scores.max())
        elif rule == 'mean':
            fused_scores.append(member_scores.mean())
        else:
            raise ValueError(f'unknown fusion rule {rule!r}')
    return suppress_overlapping(torch.stack(fused_boxes),
                                torch.stack(fused_scores))


def flip_boxes_x(boxes: torch.Tensor, width: float) -> torch.Tensor:
    """Mirror yxyx boxes back out of a horizontally flipped crop."""
    if not len(boxes):
        return boxes
    out = boxes.clone()
    out[:, 1] = width - boxes[:, 3]
    out[:, 3] = width - boxes[:, 1]
    return out
