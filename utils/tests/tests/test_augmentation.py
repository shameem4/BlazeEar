"""Tests for augmentation label validity and geometry."""
import numpy as np
import pytest

from utils import augmentation
from utils.augmentation import _EarCoverage


def make_image(h=400, w=600, value=200):
    return np.full((h, w, 3), value, dtype=np.uint8)


def ear_box(ymin=0.4, xmin=0.4, ymax=0.6, xmax=0.5):
    return np.array([[ymin, xmin, ymax, xmax]], dtype=np.float32)


def visible_fraction(image, original, bboxes):
    """Fraction of each labelled box whose pixels are unchanged."""
    h, w = image.shape[:2]
    out = []
    for ymin, xmin, ymax, xmax in bboxes:
        y1, y2 = int(ymin * h), max(int(ymin * h) + 1, int(ymax * h))
        x1, x2 = int(xmin * w), max(int(xmin * w) + 1, int(xmax * w))
        patch = image[y1:y2, x1:x2]
        ref = original[y1:y2, x1:x2]
        out.append(float(np.all(patch == ref, axis=-1).mean()))
    return out


class TestEarCoverage:
    def test_no_boxes_never_exceeds(self):
        cov = _EarCoverage(None, 100, 100, 0.5)
        assert not cov.would_exceed(0, 0, 100, 100)

    def test_full_overlap_exceeds(self):
        cov = _EarCoverage(ear_box(0.0, 0.0, 1.0, 1.0), 100, 100, 0.5)
        assert cov.would_exceed(0, 0, 100, 100)

    def test_disjoint_rect_does_not_exceed(self):
        cov = _EarCoverage(ear_box(0.0, 0.0, 0.2, 0.2), 100, 100, 0.5)
        assert not cov.would_exceed(50, 50, 100, 100)

    def test_coverage_accumulates_across_rects(self):
        # Box spans the top-left quarter; two rects each covering 40% of it are
        # individually fine but together exceed the 50% budget.
        cov = _EarCoverage(ear_box(0.0, 0.0, 1.0, 1.0), 100, 100, 0.5)
        assert not cov.would_exceed(0, 0, 40, 100)
        cov.commit(0, 0, 40, 100)
        assert cov.would_exceed(40, 0, 80, 100)


@pytest.mark.parametrize('seed', range(25))
class TestOcclusionsPreserveLabels:
    """No occlusion may hide a labelled ear while its positive label survives."""

    def test_face_cutout_leaves_ear_visible(self, seed):
        np.random.seed(seed)
        boxes = ear_box()
        original = make_image()
        out = augmentation.augment_face_cutout(original.copy(), boxes)
        assert visible_fraction(out, original, boxes)[0] >= 0.5 - 1e-6

    def test_targeted_occlusion_leaves_ear_visible(self, seed):
        np.random.seed(seed)
        boxes = ear_box()
        original = make_image()
        out = augmentation.augment_targeted_ear_occlusion(original.copy(), boxes)
        assert visible_fraction(out, original, boxes)[0] >= 0.5 - 1e-6

    def test_synthetic_occlusion_leaves_ear_visible(self, seed):
        np.random.seed(seed)
        boxes = ear_box()
        original = make_image()
        out = augmentation.augment_synthetic_occlusion(original.copy(), boxes, num_occlusions=3)
        assert visible_fraction(out, original, boxes)[0] >= 0.5 - 1e-6

    def test_cutout_leaves_ear_visible(self, seed):
        np.random.seed(seed)
        boxes = ear_box()
        original = make_image()
        out = augmentation.augment_cutout(original.copy(), boxes, num_holes=3)
        assert visible_fraction(out, original, boxes)[0] >= 0.5 - 1e-6


class TestOcclusionsStillDoSomething:
    def test_face_cutout_modifies_the_image(self):
        # The coverage guard must not turn the augmentation into a no-op.
        changed = 0
        for seed in range(40):
            np.random.seed(seed)
            original = make_image()
            out = augmentation.augment_face_cutout(original.copy(), ear_box())
            if not np.array_equal(out, original):
                changed += 1
        assert changed > 10, f'face cutout fired only {changed}/40 times'

    def test_cutout_scales_with_image_size(self):
        # Sizes are fractions, so a big image gets a proportionally big hole.
        np.random.seed(0)
        small = augmentation.augment_cutout(make_image(100, 100), None, num_holes=1)
        np.random.seed(0)
        large = augmentation.augment_cutout(make_image(1000, 1000), None, num_holes=1)
        small_frac = (small != 200).any(axis=-1).mean()
        large_frac = (large != 200).any(axis=-1).mean()
        assert small_frac > 0
        assert abs(small_frac - large_frac) < 0.02


class TestHorizontalFlip:
    def test_returns_contiguous_image(self):
        image = make_image(64, 64)
        out, _ = augmentation.augment_horizontal_flip(image, ear_box())
        assert out.flags['C_CONTIGUOUS']

    def test_does_not_mutate_caller_boxes(self):
        boxes = ear_box(0.1, 0.2, 0.3, 0.4)
        before = boxes.copy()
        augmentation.augment_horizontal_flip(make_image(64, 64), boxes)
        assert np.array_equal(boxes, before)

    def test_box_is_mirrored(self):
        boxes = ear_box(0.1, 0.2, 0.3, 0.4)
        _, flipped = augmentation.augment_horizontal_flip(make_image(64, 64), boxes)
        assert np.allclose(flipped[0], [0.1, 0.6, 0.3, 0.8])

    def test_flip_is_an_involution(self):
        boxes = ear_box(0.1, 0.2, 0.3, 0.4)
        image = make_image(64, 64)
        once_img, once_box = augmentation.augment_horizontal_flip(image.copy(), boxes)
        twice_img, twice_box = augmentation.augment_horizontal_flip(once_img, once_box)
        assert np.allclose(twice_box, boxes)
        assert np.array_equal(twice_img, image)
