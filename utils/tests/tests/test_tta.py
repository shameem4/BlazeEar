"""Tests for test-time augmentation fusion.

The Python and JavaScript view lists are pinned against each other here, the
same way crop_window() and _cropWindow() are: the browser must run the same
views in the same order as everything that was measured, or it reports
different boxes from the numbers in the README.
"""
import re
import unittest
from pathlib import Path

import torch

from utils.config import (DEFAULT_INPUT_SIZE, DETECTION_SCORE_THRESHOLD,
                          FACE_CROP_EXPAND, FACE_INPUT_SIZE,
                          TTA_SCORE_THRESHOLD)
from utils.tta import VIEWS, flip_boxes_x, merge_views

JS_PATH = Path(__file__).resolve().parents[3] / 'docs' / 'blazeear_inference.js'


def box(y0, x0, y1, x1):
    return torch.tensor([[float(y0), float(x0), float(y1), float(x1)]])


class TestMergeViews(unittest.TestCase):
    def test_identical_detections_fuse_to_one(self):
        b = box(10, 10, 20, 20)
        boxes, scores = merge_views([b, b.clone()],
                                    [torch.tensor([0.8]), torch.tensor([0.6])], 2)
        self.assertEqual(len(boxes), 1)
        torch.testing.assert_close(boxes[0], b[0])

    def test_max_rule_takes_the_highest_member_score(self):
        b = box(10, 10, 20, 20)
        _, scores = merge_views([b, b.clone()],
                                [torch.tensor([0.8]), torch.tensor([0.6])], 2)
        self.assertAlmostEqual(float(scores[0]), 0.8, places=5)

    def test_vote_rule_divides_by_the_view_count_not_the_member_count(self):
        """The distinction is the whole point: a box two of four views found
        keeps half its summed score, not all of it."""
        b = box(10, 10, 20, 20)
        _, scores = merge_views([b, b.clone()],
                                [torch.tensor([0.8]), torch.tensor([0.6])],
                                num_views=4, rule='vote')
        self.assertAlmostEqual(float(scores[0]), (0.8 + 0.6) / 4, places=5)

    def test_coordinates_are_score_weighted(self):
        # IoU 81/119 = 0.68, comfortably over the 0.5 fusion threshold.
        a, b = box(0, 0, 10, 10), box(1, 1, 11, 11)
        boxes, _ = merge_views([a, b], [torch.tensor([3.0]), torch.tensor([1.0])], 2)
        self.assertEqual(len(boxes), 1)
        # weighted mean of 0 and 1 at weights 3:1
        self.assertAlmostEqual(float(boxes[0][0]), 0.25, places=4)
        self.assertAlmostEqual(float(boxes[0][2]), 10.25, places=4)

    def test_disjoint_detections_stay_separate(self):
        a, b = box(0, 0, 10, 10), box(500, 500, 510, 510)
        boxes, scores = merge_views([a, b],
                                    [torch.tensor([0.9]), torch.tensor([0.4])], 2)
        self.assertEqual(len(boxes), 2)
        self.assertGreater(float(scores[0]), float(scores[1]))   # score-ordered

    def test_empty_input(self):
        boxes, scores = merge_views([], [], 4)
        self.assertEqual(len(boxes), 0)
        self.assertEqual(len(scores), 0)

    def test_unknown_rule_is_rejected(self):
        b = box(0, 0, 1, 1)
        with self.assertRaises(ValueError):
            merge_views([b], [torch.tensor([0.5])], 1, rule='average')

    def test_zero_views_is_rejected(self):
        b = box(0, 0, 1, 1)
        with self.assertRaises(ValueError):
            merge_views([b], [torch.tensor([0.5])], 0)


class TestFlipBoxes(unittest.TestCase):
    def test_flip_is_its_own_inverse(self):
        b = box(5, 10, 25, 40)
        torch.testing.assert_close(flip_boxes_x(flip_boxes_x(b, 128.0), 128.0), b)

    def test_flip_mirrors_about_the_centre(self):
        b = box(0, 10, 10, 30)
        f = flip_boxes_x(b, 100.0)
        self.assertAlmostEqual(float(f[0][1]), 70.0, places=5)   # 100 - 30
        self.assertAlmostEqual(float(f[0][3]), 90.0, places=5)   # 100 - 10
        self.assertAlmostEqual(float(f[0][0]), 0.0, places=5)    # y untouched

    def test_empty(self):
        self.assertEqual(len(flip_boxes_x(torch.zeros((0, 4)), 128.0)), 0)


class TestViewsMatchJs(unittest.TestCase):
    def _js_views(self):
        text = JS_PATH.read_text(encoding='utf-8')
        block = re.search(r'const TTA_VIEWS = \[(.*?)\];', text, re.S)
        self.assertIsNotNone(block, 'TTA_VIEWS not found in the browser pipeline')
        return [(float(e), f == 'true') for e, f in
                re.findall(r'\[\s*([\d.]+)\s*,\s*(true|false)\s*\]', block.group(1))]

    def test_js_runs_the_same_views_in_the_same_order(self):
        self.assertEqual(self._js_views(), [(e, f) for e, f in VIEWS])

    def test_first_view_is_the_plain_pipeline(self):
        """views=1 must be exactly the shipped single-crop behaviour."""
        self.assertEqual(VIEWS[0], (FACE_CROP_EXPAND, False))

    def _js_const(self, name):
        text = JS_PATH.read_text(encoding='utf-8')
        match = re.search(rf'const {name} = ([\d.]+);', text)
        self.assertIsNotNone(match, f'{name} not found in the browser pipeline')
        return float(match.group(1))

    def test_js_uses_the_same_operating_point(self):
        """The browser page is where the miscalibration actually shipped.

        The fused path needs a higher threshold than the plain one, because
        `max` fusion gives a cluster its best member's score. Running TTA at
        the plain threshold measured WORSE on F1 than no TTA at all, so a drift
        between these two files is not a cosmetic inconsistency.
        """
        self.assertEqual(self._js_const('SCORE_THRESHOLD'),
                         DETECTION_SCORE_THRESHOLD)
        self.assertEqual(self._js_const('TTA_SCORE_THRESHOLD'),
                         TTA_SCORE_THRESHOLD)

    def test_fused_threshold_is_above_the_plain_one(self):
        self.assertGreater(TTA_SCORE_THRESHOLD, DETECTION_SCORE_THRESHOLD)

    def test_js_uses_the_same_input_sizes(self):
        """The two stages run at different input edges now.

        The browser preprocesses to whatever these say, so a drift here feeds
        a graph the wrong input edge -- which ONNX reports as a shape error for
        the ear model, but which would silently letterbox differently if the
        two ever happened to be swapped.
        """
        self.assertEqual(self._js_const('EAR_INPUT_SIZE'), DEFAULT_INPUT_SIZE)
        self.assertEqual(self._js_const('FACE_INPUT_SIZE'), FACE_INPUT_SIZE)

    def test_intermediate_letterbox_matches_python(self):
        """resize_pad goes source -> 256 -> input_size in both languages."""
        import inspect
        from utils import preprocess
        sig = inspect.signature(preprocess.resize_pad)
        self.assertEqual(self._js_const('INTERMEDIATE_SIZE'),
                         sig.parameters['intermediate_size'].default)


if __name__ == '__main__':
    unittest.main()
