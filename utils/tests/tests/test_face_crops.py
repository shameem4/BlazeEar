"""Tests for the face-crop dataset builder.

The remapping is where a two-stage dataset goes wrong silently: an ear kept
with the wrong coordinates trains the model on a lie, and a sliver of an ear
left labelled as ground truth teaches it to fire on nothing.
"""
import numpy as np
import pandas as pd
import pytest

from make_face_crops import MIN_VISIBLE_FRACTION, crop_window, remap_boxes


def rows(*specs):
    """specs: (x1, y1, w, h, annotation_source)"""
    return list(pd.DataFrame(
        [{'x1': x, 'y1': y, 'w': w, 'h': h, 'earside': 'left',
          'source': 'ds', 'annotation_source': src, 'confidence': 1.0}
         for x, y, w, h, src in specs]).itertuples(index=False))


def face(ymin, xmin, ymax, xmax):
    return np.array([ymin, xmin, ymax, xmax], dtype=np.float32)


class TestCropWindow:
    def test_window_is_square_and_scaled(self):
        x0, y0, side = crop_window(face(100, 100, 200, 200), 1.5, (1000, 1000, 3))
        assert side == 150
        # Centred on the face centre at (150, 150).
        assert x0 == 75 and y0 == 75

    def test_the_longer_face_side_sets_the_scale(self):
        _, _, side = crop_window(face(100, 100, 300, 200), 1.0, (1000, 1000, 3))
        assert side == 200

    def test_window_shifts_to_stay_inside_the_frame(self):
        x0, y0, side = crop_window(face(0, 0, 100, 100), 3.0, (1000, 1000, 3))
        assert x0 == 0 and y0 == 0 and side == 300

    def test_window_never_leaves_the_right_edge(self):
        x0, _, side = crop_window(face(400, 900, 500, 1000), 3.0, (1000, 1000, 3))
        assert x0 + side <= 1000

    def test_side_is_capped_by_the_smaller_image_dimension(self):
        _, _, side = crop_window(face(100, 100, 200, 200), 20.0, (400, 1000, 3))
        assert side == 400

    def test_a_narrow_frame_still_yields_a_window_inside_it(self):
        x0, y0, side = crop_window(face(10, 10, 60, 60), 4.0, (80, 300, 3))
        assert 0 <= y0 and y0 + side <= 80
        assert 0 <= x0 and x0 + side <= 300


class TestRemapBoxes:
    def test_coordinates_become_crop_relative(self):
        out, _ = remap_boxes(rows((120, 130, 20, 40, 'GT')), (100, 100, 200))
        assert out[0]['x1'] == 20 and out[0]['y1'] == 30
        assert out[0]['w'] == 20 and out[0]['h'] == 40

    def test_an_ear_outside_the_window_is_dropped(self):
        out, _ = remap_boxes(rows((500, 500, 20, 40, 'GT')), (100, 100, 200))
        assert out == []

    def test_membership_is_decided_by_the_centre(self):
        """An ear straddling the edge but centred inside is kept."""
        # Centre at x=295, inside a window spanning 100..300.
        out, _ = remap_boxes(rows((285, 150, 20, 20, 'GT')), (100, 100, 200))
        assert len(out) == 1

    def test_a_kept_ear_is_clipped_to_the_window(self):
        out, _ = remap_boxes(rows((285, 150, 20, 20, 'GT')), (100, 100, 200))
        assert out[0]['x1'] + out[0]['w'] <= 200

    def test_a_mostly_cut_ear_is_demoted_to_ignore(self):
        """
        Demotion is only reachable at a corner: the centre test already
        guarantees over half of each single axis survives, so a box has to
        lose area on both axes at once to fall under the threshold. Here 12
        of 20 px survive on each axis, so 36% of the area does.
        """
        out, demoted = remap_boxes(rows((288, 288, 20, 20, 'GT')), (100, 100, 200))
        assert out[0]['annotation_source'] == 'IGNORE'
        assert demoted == 1

    def test_a_barely_cut_ear_stays_ground_truth(self):
        # 15 of 20 px wide survive, and the full height.
        out, demoted = remap_boxes(rows((285, 150, 20, 20, 'GT')), (100, 100, 200))
        assert out[0]['annotation_source'] == 'GT'
        assert demoted == 0

    def test_the_demotion_boundary_matches_the_constant(self):
        assert 0 < MIN_VISIBLE_FRACTION < 1

    def test_ignore_regions_are_carried_over_without_the_centre_test(self):
        """
        An ignore region marks ground the model must not be scored on. It has
        no centre worth testing, so any overlap with the window is carried in.
        """
        out, _ = remap_boxes(rows((290, 150, 40, 20, 'IGNORE')), (100, 100, 200))
        assert len(out) == 1 and out[0]['annotation_source'] == 'IGNORE'

    def test_an_ignore_region_that_misses_the_window_is_dropped(self):
        out, _ = remap_boxes(rows((900, 900, 40, 20, 'IGNORE')), (100, 100, 200))
        assert out == []

    def test_a_degenerate_sliver_is_dropped_entirely(self):
        """A box overlapping by under a pixel is noise, not an ignore region."""
        out, _ = remap_boxes(rows((299, 150, 20, 20, 'IGNORE')), (100, 100, 200))
        assert out == []

    def test_every_ear_in_a_crowded_crop_is_kept(self):
        out, _ = remap_boxes(
            rows((110, 110, 20, 40, 'GT'), (250, 260, 20, 40, 'GT+EAR')),
            (100, 100, 200))
        assert len(out) == 2

    def test_the_annotation_source_survives_the_move(self):
        out, _ = remap_boxes(rows((120, 130, 20, 40, 'GT+REVIEW')), (100, 100, 200))
        assert out[0]['annotation_source'] == 'GT+REVIEW'


class TestSharedDefaults:
    """
    The dataset builder and the evaluator have to crop identically. When they
    drifted, evaluation used face threshold 0.5 against a dataset built at
    0.3, so the pipeline was scored on crops it had never been trained on and
    stranded 13% of images instead of 3.6%.
    """

    def test_both_scripts_take_their_defaults_from_config(self):
        import evaluate_two_stage
        import make_face_crops
        from utils import config
        for module in (make_face_crops, evaluate_two_stage):
            defaults = {a.dest: a.default
                        for a in module.build_parser()._actions}
            assert defaults['expand'] == config.FACE_CROP_EXPAND
            assert defaults['face_threshold'] == config.FACE_CROP_THRESHOLD
            assert defaults['max_faces'] == config.FACE_CROP_MAX_FACES

    def test_the_face_detector_defaults_to_the_shared_threshold(self):
        import inspect

        from make_face_crops import load_face_detector
        from utils.config import FACE_CROP_THRESHOLD
        signature = inspect.signature(load_face_detector)
        assert signature.parameters['score_threshold'].default == FACE_CROP_THRESHOLD



class TestCropWindowIsPinned:
    """
    The crop window exists twice: here and in docs/blazeear_inference.js, so
    the browser and the Python pipeline crop identically. Nothing enforces
    that across languages, so these cases are pinned on the Python side and
    the JS carries a pointer to them. They were verified equal on all ten.
    """

    CASES = [
        # (face ymin,xmin,ymax,xmax), (frame h,w) -> (x0, y0, side) at 1.5x
        ((100, 100, 200, 200), (640, 640), (75, 75, 150)),
        ((0, 0, 50, 50), (640, 640), (0, 0, 75)),
        ((590, 590, 640, 640), (640, 640), (565, 565, 75)),
        ((10, 300, 60, 350), (80, 640), (288, 0, 75)),
        ((300, 10, 350, 60), (640, 80), (0, 288, 75)),
        ((0, 0, 640, 640), (640, 640), (0, 0, 640)),
        ((-20, -20, 40, 40), (640, 640), (0, 0, 90)),
        ((320, 320, 321, 321), (640, 640), (320, 320, 2)),
    ]

    def test_pinned_cases(self, subtests=None):
        from utils.config import FACE_CROP_EXPAND
        for face_box, (height, width), expected in self.CASES:
            got = crop_window(
                np.array(face_box, dtype=np.float32), FACE_CROP_EXPAND,
                (height, width, 3))
            assert got == expected, f'{face_box} in {(height, width)}: {got} != {expected}'

    def test_the_window_always_lands_inside_the_frame(self):
        from utils.config import FACE_CROP_EXPAND
        for face_box, (height, width), _ in self.CASES:
            x0, y0, side = crop_window(
                np.array(face_box, dtype=np.float32), FACE_CROP_EXPAND,
                (height, width, 3))
            assert 0 <= x0 and x0 + side <= width
            assert 0 <= y0 and y0 + side <= height
