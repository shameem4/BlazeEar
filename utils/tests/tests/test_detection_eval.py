"""Tests for pooled dataset-level detection metrics."""
import torch

from utils.detection_eval import (
    DetectionEvaluator,
    average_precision,
    pairwise_iou,
)


def box(ymin, xmin, ymax, xmax):
    return [float(ymin), float(xmin), float(ymax), float(xmax)]


class TestAveragePrecision:
    def test_no_ground_truth_is_zero(self):
        scores = torch.tensor([0.9, 0.8])
        tp = torch.tensor([0.0, 0.0])
        assert average_precision(scores, tp, num_ground_truth=0) == 0.0

    def test_no_detections_is_zero(self):
        empty = torch.zeros(0)
        assert average_precision(empty, empty, num_ground_truth=5) == 0.0

    def test_perfect_detector(self):
        scores = torch.tensor([0.9, 0.8])
        tp = torch.tensor([1.0, 1.0])
        assert average_precision(scores, tp, num_ground_truth=2) == 1.0

    def test_trailing_false_positive_does_not_reduce_ap(self):
        # One GT, recalled by the top detection; a lower-scoring FP follows.
        # Interpolated precision at full recall stays 1.0.
        scores = torch.tensor([0.9, 0.8])
        tp = torch.tensor([1.0, 0.0])
        assert average_precision(scores, tp, num_ground_truth=1) == 1.0

    def test_leading_false_positive(self):
        # 2 GT; a FP outranks the single TP. recall 0.5 at precision 0.5.
        scores = torch.tensor([0.9, 0.8])
        tp = torch.tensor([0.0, 1.0])
        assert abs(average_precision(scores, tp, num_ground_truth=2) - 0.25) < 1e-9

    def test_all_point_interpolation(self):
        # 2 GT, detections TP / FP / TP -> VOC all-point AP = 5/6.
        scores = torch.tensor([0.9, 0.8, 0.7])
        tp = torch.tensor([1.0, 0.0, 1.0])
        assert abs(average_precision(scores, tp, num_ground_truth=2) - 5.0 / 6.0) < 1e-6

    def test_undetected_ground_truth_caps_recall(self):
        # One perfect detection but four GT boxes: recall cannot exceed 0.25.
        scores = torch.tensor([0.9])
        tp = torch.tensor([1.0])
        assert abs(average_precision(scores, tp, num_ground_truth=4) - 0.25) < 1e-9


class TestPairwiseIoU:
    def test_identical_boxes(self):
        b = torch.tensor([box(0, 0, 10, 10)])
        assert abs(pairwise_iou(b, b)[0, 0].item() - 1.0) < 1e-6

    def test_disjoint_boxes(self):
        a = torch.tensor([box(0, 0, 10, 10)])
        b = torch.tensor([box(50, 50, 60, 60)])
        assert pairwise_iou(a, b)[0, 0].item() == 0.0

    def test_half_overlap(self):
        a = torch.tensor([box(0, 0, 10, 10)])
        b = torch.tensor([box(0, 5, 10, 15)])
        # intersection 50, union 150
        assert abs(pairwise_iou(a, b)[0, 0].item() - 1.0 / 3.0) < 1e-6

    def test_empty_inputs(self):
        a = torch.zeros((0, 4))
        b = torch.tensor([box(0, 0, 10, 10)])
        assert pairwise_iou(a, b).shape == (0, 1)
        assert pairwise_iou(b, a).shape == (1, 0)


class TestDetectionEvaluator:
    def test_perfect_split(self):
        ev = DetectionEvaluator()
        for _ in range(3):
            gt = torch.tensor([box(10, 10, 20, 20)])
            ev.add_image(gt.clone(), torch.tensor([0.99]), gt)
        m = ev.compute()
        assert m['map_50'] == 1.0
        assert m['map_50_95'] == 1.0
        assert abs(m['detection_iou'] - 1.0) < 1e-6
        assert m['num_images'] == 3.0
        assert m['num_ground_truth'] == 3.0

    def test_missed_ground_truth_is_counted(self):
        # Two images, one GT each; only the first is detected.
        ev = DetectionEvaluator()
        gt = torch.tensor([box(10, 10, 20, 20)])
        ev.add_image(gt.clone(), torch.tensor([0.9]), gt)
        ev.add_image(torch.zeros((0, 4)), torch.zeros(0), gt)
        m = ev.compute()
        # Recall caps at 0.5 with precision 1.0 -> AP 0.5.
        assert abs(m['map_50'] - 0.5) < 1e-6
        assert m['num_ground_truth'] == 2.0

    def test_duplicate_detection_is_a_false_positive(self):
        # Two detections on one GT: the second cannot claim the same box.
        ev = DetectionEvaluator()
        gt = torch.tensor([box(10, 10, 20, 20)])
        preds = torch.tensor([box(10, 10, 20, 20), box(10, 10, 20, 20)])
        ev.add_image(preds, torch.tensor([0.9, 0.8]), gt)
        m = ev.compute()
        assert m['num_detections'] == 2.0
        # Trailing FP leaves interpolated precision at 1.0.
        assert abs(m['map_50'] - 1.0) < 1e-6

    def test_detections_on_image_without_ground_truth_hurt(self):
        ev = DetectionEvaluator()
        gt = torch.tensor([box(10, 10, 20, 20)])
        ev.add_image(gt.clone(), torch.tensor([0.8]), gt)
        # A higher-scoring false positive on an empty image.
        ev.add_image(torch.tensor([box(0, 0, 5, 5)]), torch.tensor([0.95]), torch.zeros((0, 4)))
        m = ev.compute()
        assert abs(m['map_50'] - 0.5) < 1e-6

    def test_pooling_differs_from_per_image_average(self):
        # The regression this module exists to prevent: per-image AP averaging
        # rates these two images at (1.0 + 0.0) / 2 = 0.5, but pooled ranking
        # puts the confident false positive above the correct detection.
        ev = DetectionEvaluator()
        gt = torch.tensor([box(10, 10, 20, 20)])
        ev.add_image(gt.clone(), torch.tensor([0.4]), gt)
        ev.add_image(torch.tensor([box(80, 80, 90, 90)]), torch.tensor([0.99]), gt)
        m = ev.compute()
        assert m['map_50'] < 0.5

    def test_iou_threshold_sweep(self):
        # A loose box: IoU 0.25 with GT, so it is a TP at 0.5 for neither
        # threshold and map_50 is 0.
        ev = DetectionEvaluator()
        gt = torch.tensor([box(0, 0, 10, 10)])
        loose = torch.tensor([box(0, 0, 20, 20)])  # IoU = 100/400 = 0.25
        ev.add_image(loose, torch.tensor([0.9]), gt)
        m = ev.compute()
        assert m['map_50'] == 0.0
        assert m['map_50_95'] == 0.0

    def test_group_breakdown(self):
        ev = DetectionEvaluator(groups=True)
        gt = torch.tensor([box(10, 10, 20, 20)])
        ev.add_image(gt.clone(), torch.tensor([0.9]), gt, group='GT')
        ev.add_image(torch.zeros((0, 4)), torch.zeros(0), gt, group='POSE')
        m = ev.compute()
        assert abs(m['map_50/GT'] - 1.0) < 1e-6
        assert m['map_50/POSE'] == 0.0
        # Pooled number sits between the two groups.
        assert abs(m['map_50'] - 0.5) < 1e-6

class TestIgnoreRegions:
    def test_detection_on_ignore_region_is_not_a_false_positive(self):
        # One real GT (detected) plus a detection on an untrusted region.
        # Without ignore this would be a FP and halve precision.
        gt = torch.tensor([box(10, 10, 20, 20)])
        pseudo = torch.tensor([box(80, 80, 90, 90)])
        preds = torch.tensor([box(80, 80, 90, 90), box(10, 10, 20, 20)])
        scores = torch.tensor([0.95, 0.9])

        without = DetectionEvaluator()
        without.add_image(preds, scores, gt)
        with_ignore = DetectionEvaluator()
        with_ignore.add_image(preds, scores, gt, ignore_boxes=pseudo)

        assert without.compute()['map_50'] < 1.0
        assert with_ignore.compute()['map_50'] == 1.0
        assert with_ignore.compute()['num_ignored'] == 1.0

    def test_ignore_does_not_inflate_recall(self):
        # The ignore region must not count as recallable ground truth.
        gt = torch.tensor([box(10, 10, 20, 20)])
        pseudo = torch.tensor([box(80, 80, 90, 90)])
        ev = DetectionEvaluator()
        # Detect only the ignore region, miss the real GT.
        ev.add_image(torch.tensor([box(80, 80, 90, 90)]), torch.tensor([0.9]), gt,
                     ignore_boxes=pseudo)
        m = ev.compute()
        assert m['num_ground_truth'] == 1.0
        assert m['map_50'] == 0.0

    def test_true_positive_overlapping_ignore_is_still_counted(self):
        # A detection that matches real GT is kept even if an ignore region
        # happens to cover the same area.
        gt = torch.tensor([box(10, 10, 20, 20)])
        ev = DetectionEvaluator()
        ev.add_image(gt.clone(), torch.tensor([0.9]), gt, ignore_boxes=gt.clone())
        m = ev.compute()
        assert m['map_50'] == 1.0
        assert m['num_ignored'] == 0.0

    def test_small_detection_inside_large_ignore_region(self):
        # IoU would be tiny here; intersection-over-detection-area is what matters.
        gt = torch.zeros((0, 4))
        big_ignore = torch.tensor([box(0, 0, 100, 100)])
        tiny = torch.tensor([box(40, 40, 45, 45)])
        ev = DetectionEvaluator()
        ev.add_image(tiny, torch.tensor([0.9]), gt, ignore_boxes=big_ignore)
        assert ev.compute()['num_ignored'] == 1.0

    def test_detection_outside_ignore_still_penalised(self):
        gt = torch.tensor([box(10, 10, 20, 20)])
        pseudo = torch.tensor([box(80, 80, 90, 90)])
        ev = DetectionEvaluator()
        ev.add_image(
            torch.tensor([box(40, 40, 50, 50), box(10, 10, 20, 20)]),
            torch.tensor([0.95, 0.9]), gt, ignore_boxes=pseudo,
        )
        m = ev.compute()
        assert m['num_ignored'] == 0.0
        assert m['map_50'] < 1.0


class TestReset:
    def test_reset_clears_state(self):
        ev = DetectionEvaluator()
        gt = torch.tensor([box(10, 10, 20, 20)])
        ev.add_image(gt.clone(), torch.tensor([0.9]), gt)
        ev.reset()
        m = ev.compute()
        assert m['num_images'] == 0.0
        assert m['map_50'] == 0.0
