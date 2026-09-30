"""Tests for per-anchor target assignment with an ignore band."""
import numpy as np
import pytest

from utils.anchor_utils import (
    anchors_to_corners,
    assign_anchor_targets,
    generate_anchors_from_priors,
)
from utils.config import EAR_ANCHOR_PRIORS_BIG, EAR_ANCHOR_PRIORS_SMALL


@pytest.fixture(scope='module')
def anchors():
    return generate_anchors_from_priors().numpy()


def box(ymin, xmin, ymax, xmax):
    return np.array([[ymin, xmin, ymax, xmax]], dtype=np.float32)


class TestAnchorGeneration:
    def test_keeps_the_896_layout(self, anchors):
        expected = 16 * 16 * len(EAR_ANCHOR_PRIORS_SMALL) + 8 * 8 * len(EAR_ANCHOR_PRIORS_BIG)
        assert anchors.shape == (expected, 4) == (896, 4)

    def test_priors_are_distinct(self, anchors):
        # The whole point: the old fixed anchors collapsed to one (w, h).
        assert len(np.unique(anchors[:, 2:], axis=0)) == 8

    def test_centres_cover_the_unit_square(self, anchors):
        assert anchors[:, 0].min() > 0 and anchors[:, 0].max() < 1
        assert anchors[:, 1].min() > 0 and anchors[:, 1].max() < 1

    def test_small_grid_comes_first(self, anchors):
        small = anchors[:512]
        assert len(np.unique(small[:, 2:], axis=0)) == len(EAR_ANCHOR_PRIORS_SMALL)

    def test_corners_round_trip(self):
        a = np.array([[0.5, 0.5, 0.2, 0.4]], dtype=np.float32)
        c = anchors_to_corners(a)
        assert np.allclose(c, [[0.3, 0.4, 0.7, 0.6]])


class TestAssignment:
    def test_every_box_gets_top_k_positives(self, anchors):
        targets, _ = assign_anchor_targets(box(0.40, 0.45, 0.50, 0.49), anchors, top_k=3)
        assert int((targets[:, 0] > 0.5).sum()) == 3

    def test_assignment_succeeds_for_boxes_no_anchor_overlaps_well(self, anchors):
        # A tiny ear that reaches nowhere near IoU 0.5 must still be supervised;
        # an IoU-thresholded assignment would silently drop it.
        targets, _ = assign_anchor_targets(box(0.50, 0.50, 0.512, 0.506), anchors, top_k=3)
        assert int((targets[:, 0] > 0.5).sum()) == 3

    def test_positive_targets_carry_the_box(self, anchors):
        b = box(0.40, 0.45, 0.50, 0.49)
        targets, _ = assign_anchor_targets(b, anchors, top_k=3)
        pos = targets[targets[:, 0] > 0.5]
        for row in pos:
            assert np.allclose(row[1:], b[0], atol=1e-6)

    def test_two_boxes_get_separate_anchors(self, anchors):
        boxes = np.array([
            [0.10, 0.10, 0.20, 0.15],
            [0.70, 0.70, 0.80, 0.75],
        ], dtype=np.float32)
        targets, _ = assign_anchor_targets(boxes, anchors, top_k=3)
        pos = targets[targets[:, 0] > 0.5]
        assert len(pos) == 6
        assert len(np.unique(pos[:, 1:], axis=0)) == 2

    def test_conflicting_claims_go_to_the_higher_iou(self):
        # Two anchors; both boxes want anchor 0, but box 1 matches it exactly.
        anchors = np.array([[0.5, 0.5, 0.1, 0.1], [0.9, 0.9, 0.1, 0.1]], dtype=np.float32)
        boxes = np.array([
            [0.30, 0.30, 0.70, 0.70],   # loose overlap with anchor 0
            [0.45, 0.45, 0.55, 0.55],   # exactly anchor 0
        ], dtype=np.float32)
        targets, _ = assign_anchor_targets(boxes, anchors, top_k=1)
        assert np.allclose(targets[0, 1:], boxes[1], atol=1e-6)

    def test_empty_input_produces_all_background(self, anchors):
        targets, ignore = assign_anchor_targets(np.zeros((0, 4), np.float32), anchors)
        assert targets.shape == (896, 5)
        assert targets[:, 0].sum() == 0
        assert not ignore.any()

    def test_degenerate_boxes_are_dropped(self, anchors):
        degenerate = np.array([[0.5, 0.5, 0.5, 0.5], [0.6, 0.6, 0.5, 0.5]], dtype=np.float32)
        targets, _ = assign_anchor_targets(degenerate, anchors)
        assert targets[:, 0].sum() == 0

    def test_top_k_is_clamped_to_at_least_one(self, anchors):
        targets, _ = assign_anchor_targets(box(0.4, 0.45, 0.5, 0.49), anchors, top_k=0)
        assert int((targets[:, 0] > 0.5).sum()) == 1


class TestIgnoreBand:
    def test_near_miss_anchors_are_ignored_not_background(self):
        # Five near-identical anchors on one box, but only one may be positive.
        anchors = np.array(
            [[0.5, 0.5 + 0.002 * i, 0.20, 0.20] for i in range(5)], dtype=np.float32
        )
        targets, ignore = assign_anchor_targets(
            box(0.40, 0.40, 0.60, 0.60), anchors, top_k=1, ignore_iou=0.35
        )
        positive = targets[:, 0] > 0.5
        assert positive.sum() == 1
        # The four losers overlap heavily and must not be trained as background.
        assert ignore.sum() == 4
        assert not (positive & ignore).any()

    def test_distant_anchors_stay_negative(self):
        anchors = np.array([[0.5, 0.5, 0.2, 0.2], [0.05, 0.05, 0.02, 0.02]], dtype=np.float32)
        targets, ignore = assign_anchor_targets(
            box(0.40, 0.40, 0.60, 0.60), anchors, top_k=1, ignore_iou=0.35
        )
        assert not ignore[1], 'a far-away anchor is a legitimate negative'

    def test_positive_anchors_are_never_ignored(self, anchors):
        targets, ignore = assign_anchor_targets(
            box(0.30, 0.30, 0.70, 0.70), anchors, top_k=5, ignore_iou=0.1
        )
        assert not ((targets[:, 0] > 0.5) & ignore).any()

    def test_higher_threshold_ignores_fewer(self, anchors):
        b = box(0.30, 0.30, 0.70, 0.70)
        _, loose = assign_anchor_targets(b, anchors, top_k=1, ignore_iou=0.10)
        _, tight = assign_anchor_targets(b, anchors, top_k=1, ignore_iou=0.60)
        assert loose.sum() >= tight.sum()

    def test_ignore_band_is_empty_without_boxes(self, anchors):
        _, ignore = assign_anchor_targets(np.zeros((0, 4), np.float32), anchors, ignore_iou=0.0)
        assert not ignore.any()


class TestExplicitIgnoreRegions:
    """
    Regions holding a real but unlearnable ear (blurred, tiny, occluded). They
    must be neither targets nor background: labelling them teaches an
    unlearnable example, dropping them makes a real ear a hard negative.
    """

    def test_ignore_region_marks_anchors(self, anchors):
        gt = box(0.10, 0.10, 0.20, 0.15)
        region = box(0.60, 0.60, 0.80, 0.70)
        _, without = assign_anchor_targets(gt, anchors)
        _, with_region = assign_anchor_targets(gt, anchors, ignore_boxes=region)
        assert with_region.sum() > without.sum()

    def test_ignore_region_adds_no_positives(self, anchors):
        gt = box(0.10, 0.10, 0.20, 0.15)
        region = box(0.60, 0.60, 0.80, 0.70)
        base, _ = assign_anchor_targets(gt, anchors)
        with_region, _ = assign_anchor_targets(gt, anchors, ignore_boxes=region)
        assert (base[:, 0] == with_region[:, 0]).all()

    def test_positives_win_over_ignore_regions(self, anchors):
        """An anchor assigned to a real box stays positive even under a region."""
        gt = box(0.30, 0.30, 0.70, 0.70)
        targets, ignore = assign_anchor_targets(gt, anchors, ignore_boxes=gt.copy())
        assert (targets[:, 0] > 0.5).sum() == 3
        assert not ((targets[:, 0] > 0.5) & ignore).any()

    def test_degenerate_ignore_regions_are_dropped(self, anchors):
        gt = box(0.10, 0.10, 0.20, 0.15)
        bad = np.array([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32)
        _, a = assign_anchor_targets(gt, anchors)
        _, b = assign_anchor_targets(gt, anchors, ignore_boxes=bad)
        assert a.sum() == b.sum()

    def test_ignore_only_image_has_no_positives(self, anchors):
        targets, ignore = assign_anchor_targets(
            np.zeros((0, 4), np.float32), anchors,
            ignore_boxes=box(0.3, 0.3, 0.7, 0.7))
        assert targets[:, 0].sum() == 0
        assert ignore.sum() > 0
