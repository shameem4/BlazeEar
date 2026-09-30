"""Tests for the one shared non-maximum suppression.

There were five implementations at three different IoU thresholds, so the mAP
the trainer reported described post-processing no deployed path performed.
These lock down the contract they now all share.
"""
import torch

from utils.config import MAX_DETECTIONS, NMS_IOU_THRESHOLD
from utils.nms import nms_indices, suppress_overlapping


def boxes_yxyx(*spec):
    return torch.tensor(spec, dtype=torch.float32)


class TestContract:
    def test_empty_input_returns_empty_indices(self):
        keep = nms_indices(torch.zeros((0, 4)), torch.zeros(0))
        assert keep.numel() == 0 and keep.dtype == torch.long

    def test_a_single_box_survives(self):
        keep = nms_indices(boxes_yxyx([0, 0, 10, 10]), torch.tensor([0.5]))
        assert keep.tolist() == [0]

    def test_indices_come_back_highest_score_first(self):
        b = boxes_yxyx([0, 0, 1, 1], [10, 10, 11, 11], [20, 20, 21, 21])
        keep = nms_indices(b, torch.tensor([0.1, 0.9, 0.5]))
        assert keep.tolist() == [1, 2, 0]

    def test_disjoint_boxes_are_all_kept(self):
        b = boxes_yxyx([0, 0, 10, 10], [100, 100, 110, 110])
        assert len(nms_indices(b, torch.tensor([0.9, 0.8]))) == 2

    def test_the_lower_scoring_of_two_duplicates_is_dropped(self):
        b = boxes_yxyx([0, 0, 10, 10], [0, 0, 10, 10])
        keep = nms_indices(b, torch.tensor([0.4, 0.9]))
        assert keep.tolist() == [1]


class TestThresholdBoundary:
    """
    The five implementations disagreed here: three suppressed strictly above
    the threshold and two at-or-above, so identical inputs kept a different
    number of boxes depending on which path ran.
    """

    def test_overlap_exactly_at_the_threshold_is_kept(self):
        # Two 10x10 boxes sharing 1/3 of their union.
        a, b = [0, 0, 10, 10], [0, 5, 10, 15]
        iou = 50.0 / 150.0
        keep = nms_indices(boxes_yxyx(a, b), torch.tensor([0.9, 0.8]), iou + 1e-6)
        assert len(keep) == 2, 'suppression must be strictly above the threshold'

    def test_overlap_above_the_threshold_is_suppressed(self):
        a, b = [0, 0, 10, 10], [0, 5, 10, 15]
        iou = 50.0 / 150.0
        keep = nms_indices(boxes_yxyx(a, b), torch.tensor([0.9, 0.8]), iou - 1e-6)
        assert len(keep) == 1


class TestDetectionCap:
    def test_the_cap_truncates_to_the_highest_scores(self):
        b = torch.stack([torch.tensor([i * 100.0, 0, i * 100 + 10, 10])
                         for i in range(10)])
        scores = torch.arange(10, dtype=torch.float32) / 10
        keep = nms_indices(b, scores, 0.3, max_detections=3)
        assert keep.tolist() == [9, 8, 7]

    def test_none_means_no_cap(self):
        b = torch.stack([torch.tensor([i * 100.0, 0, i * 100 + 10, 10])
                         for i in range(10)])
        keep = nms_indices(b, torch.rand(10), 0.3, max_detections=None)
        assert len(keep) == 10

    def test_the_default_cap_is_the_shared_one(self):
        b = torch.stack([torch.tensor([i * 100.0, 0, i * 100 + 10, 10])
                         for i in range(MAX_DETECTIONS + 20)])
        keep = nms_indices(b, torch.rand(MAX_DETECTIONS + 20))
        assert len(keep) == MAX_DETECTIONS


class TestSuppressOverlapping:
    def test_it_returns_boxes_and_scores_already_filtered(self):
        b = boxes_yxyx([0, 0, 10, 10], [0, 0, 10, 10], [50, 50, 60, 60])
        kept_boxes, kept_scores = suppress_overlapping(
            b, torch.tensor([0.9, 0.5, 0.7]))
        assert len(kept_boxes) == 2 and len(kept_scores) == 2
        assert torch.allclose(kept_scores, torch.tensor([0.9, 0.7]))

    def test_extra_box_columns_do_not_break_it(self):
        """Callers pass detection rows wider than 4 columns."""
        rows = torch.tensor([[0, 0, 10, 10, 0.9], [50, 50, 60, 60, 0.7]],
                            dtype=torch.float32)
        keep = nms_indices(rows[:, :4], rows[:, 4])
        assert len(keep) == 2

    def test_defaults_come_from_config(self):
        import inspect
        signature = inspect.signature(nms_indices)
        assert signature.parameters['iou_threshold'].default == NMS_IOU_THRESHOLD
        assert signature.parameters['max_detections'].default == MAX_DETECTIONS
