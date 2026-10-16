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

    def test_demo_tracker_constants_are_the_measured_ones(self):
        """The demo's tracker constants, including the slider's HTML default.

        That default is the defect this test exists for. `loadModel()` passes
        the slider's `value` attribute straight into `applyThresholds`, so the
        attribute -- not `TRACK_CONFIRM` -- is what governs at page load. It
        sat at 0.70 while the constants said 0.85, which ran the page at the
        worst measured configuration for midline false positives (40 against
        9 on the clips corpus) and was invisible to a test that checked only
        the constants.

        0.80 / 0.65 / minHits 3 is measured on 7,500 frames of 25
        conversational clips: +25 lateral confirmations against the previous
        pair, 95% CI [+6, +50], with midline indistinguishable. The confirm
        score is deliberately below TTA_SCORE_THRESHOLD, because a track
        confirming over minHits frames has evidence a single frame does not.
        """
        page = (JS_PATH.parent / 'index.html').read_text(encoding='utf-8')

        def const(name):
            m = re.search(rf'const {name} = ([\d.]+);', page)
            self.assertIsNotNone(m, f'{name} not found in the demo page')
            return float(m.group(1))

        slider = re.search(r'id="confSlider"[^>]*value="([\d.]+)"', page)
        self.assertIsNotNone(slider, 'confSlider default not found')
        label = re.search(r'id="confValue">([\d.]+)<', page)
        self.assertIsNotNone(label, 'confValue label not found')

        confirm = const('TRACK_CONFIRM')
        self.assertEqual(confirm, 0.80)
        self.assertEqual(const('MIN_HITS'), 3)
        self.assertEqual(round(confirm - const('CONTINUE_MARGIN'), 10), 0.65)
        self.assertEqual(float(slider.group(1)), confirm,
                         "the slider's value attribute is what runs at load; "
                         'it must equal TRACK_CONFIRM')
        self.assertEqual(float(label.group(1)), confirm,
                         'the displayed label must match what the page does')
        self.assertLess(confirm, TTA_SCORE_THRESHOLD,
                        'a track confirming over several frames should need '
                        'less than a single frame does')
        self.assertLess(confirm - const('CONTINUE_MARGIN'),
                        DETECTION_SCORE_THRESHOLD,
                        'a running track should continue below the '
                        'single-frame gate; that is the hysteresis')

    def test_demo_tracker_matches_against_the_raw_box(self):
        """Matching must see the last matched detection, not the EMA blend.

        These were one field until `headsize v2` read the demo against
        `measure_video_dynamics.py` and found the comment describing an intent
        the code did not implement. The Python that measured 0.80 / 0.65 /
        minHits 3 matches against the raw box; the JS was matching against the
        smoothed one. Replaying the 25 clips both ways gives 189 lateral / 9
        midline confirmations as measured against 212 / 14 as shipped, a
        divergence the same size as the +25 lateral the constants were chosen
        for. So this is not a style question: if the two languages disagree
        here, the shipped constants are not the measured ones.
        """
        page = (JS_PATH.parent / 'index.html').read_text(encoding='utf-8')
        body = re.search(r'class EarTracker \{(.*?)\n        \}', page, re.S)
        self.assertIsNotNone(body, 'EarTracker not found in the demo page')
        body = body.group(1)
        self.assertIn('EarTracker.iou(track.rawBox', body,
                      'IoU matching must use the raw box, which is what '
                      'measure_video_dynamics.py matches against')
        self.assertNotIn('EarTracker.iou(track.box', body,
                         'matching against the smoothed box re-denominates '
                         'the measured constants')
        self.assertEqual(body.count('rawBox = {') + body.count('rawBox: {'), 2,
                         'rawBox must be set both when a track is created '
                         'and when it matches, or matching sees a stale box')

    def test_fused_threshold_is_not_below_the_plain_one(self):
        """Relaxed from `>` to `>=` when both paths were set to 0.70.

        The strict form encoded a measured finding: fusion shifts the score
        distribution up, so the fused path wanted a higher gate, and running
        TTA at the plain threshold was worse on F1 than no TTA at all. That
        finding still stands and is recorded in `utils/config.py`; what
        changed is that a single deployed default was wanted across all paths.
        Equal is therefore allowed, but below the plain gate is still a bug --
        it could only ever admit more of a distribution that already sits
        higher.
        """
        self.assertGreaterEqual(TTA_SCORE_THRESHOLD, DETECTION_SCORE_THRESHOLD)

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
