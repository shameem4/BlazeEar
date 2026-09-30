import json
import unittest
from pathlib import Path

import numpy as np
import torch

from dataloader import (
    CSVDetectorDataset,
    assign_anchor_targets,
    generate_anchors_from_priors,
)
from loss_functions import BlazeEarDetectionLoss
from blazebase import generate_reference_anchors


ASSETS_ROOT = Path("utils/tests/assets")
CSV_PATH = ASSETS_ROOT / "test_data.csv"
EXPECTED_PATH = ASSETS_ROOT / "expected_outputs.json"


with EXPECTED_PATH.open("r", encoding="utf-8") as f:
    EXPECTED = json.load(f)


def _load_dataset() -> CSVDetectorDataset:
    return CSVDetectorDataset(
        csv_path=str(CSV_PATH),
        root_dir=str(ASSETS_ROOT),
        target_size=(128, 128),
        augment=False,
    )


def _compute_normalized_boxes(image: np.ndarray, boxes_px: np.ndarray) -> np.ndarray:
    if len(boxes_px) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    orig_h, orig_w = image.shape[:2]
    y1 = boxes_px[:, 1]
    x1 = boxes_px[:, 0]
    w = boxes_px[:, 2]
    h = boxes_px[:, 3]
    ymin = np.clip(y1 / orig_h, 0, 1)
    xmin = np.clip(x1 / orig_w, 0, 1)
    ymax = np.clip((y1 + h) / orig_h, 0, 1)
    xmax = np.clip((x1 + w) / orig_w, 0, 1)
    return np.stack([ymin, xmin, ymax, xmax], axis=1).astype(np.float32)


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = _load_dataset()
        cls.reference_anchors, _, _ = generate_reference_anchors()
        cls.loss = BlazeEarDetectionLoss()

    def test_normalization_matches_expected(self):
        for sample in self.dataset.samples:
            image = self.dataset._load_image(sample["image_path"])
            norm_boxes = _compute_normalized_boxes(image, sample["boxes"])
            name = Path(sample["image_path"]).stem
            expected_norm = np.asarray(EXPECTED[name]["normalized_boxes"], dtype=np.float32)
            np.testing.assert_allclose(
                norm_boxes, expected_norm, atol=1e-6, err_msg=f"Mismatch for {name}"
            )

    def test_resize_and_pad_matches_expected(self):
        for sample in self.dataset.samples:
            image = self.dataset._load_image(sample["image_path"])
            norm_boxes = _compute_normalized_boxes(image, sample["boxes"])
            _, resized_boxes = self.dataset._resize_and_pad(image, norm_boxes.copy())
            name = Path(sample["image_path"]).stem
            expected = np.asarray(EXPECTED[name]["resized_boxes"], dtype=np.float32)
            np.testing.assert_allclose(
                resized_boxes, expected, atol=1e-6, err_msg=f"Resize mismatch for {name}"
            )

    def test_anchor_assignment_invariants(self):
        """Properties that must hold regardless of the anchor priors."""
        anchors = generate_anchors_from_priors().numpy()
        for sample in self.dataset.samples:
            image = self.dataset._load_image(sample["image_path"])
            norm_boxes = _compute_normalized_boxes(image, sample["boxes"])
            targets, ignore = assign_anchor_targets(norm_boxes, anchors, top_k=3)
            positives = targets[:, 0] > 0.5

            # Every box is supervised, and nothing is both positive and ignored.
            self.assertEqual(int(positives.sum()), 3 * len(norm_boxes))
            self.assertFalse(bool((positives & ignore).any()))
            # Each positive target is one of the input boxes.
            for row in targets[positives]:
                self.assertTrue(
                    any(np.allclose(row[1:], b, atol=1e-6) for b in norm_boxes)
                )

    def test_anchor_assignment_is_unchanged(self):
        """
        Change detector, not a correctness oracle: the expected indices were
        generated from this same implementation, so this catches unintended
        drift in assignment, not a wrong assignment. Correctness lives in
        test_anchor_assignment.py and in test_anchor_assignment_invariants.
        """
        for sample in self.dataset.samples:
            image = self.dataset._load_image(sample["image_path"])
            norm_boxes = _compute_normalized_boxes(image, sample["boxes"])
            anchor_targets, _ = assign_anchor_targets(norm_boxes, generate_anchors_from_priors().numpy())
            positives = np.where(anchor_targets[:, 0] == 1)[0].tolist()
            name = Path(sample["image_path"]).stem
            self.assertEqual(
                positives,
                EXPECTED[name]["positive_indices"],
                f"Anchor indices mismatch for {name}",
            )

    def test_decode_roundtrip_matches_targets(self):
        for sample in self.dataset.samples:
            image = self.dataset._load_image(sample["image_path"])
            norm_boxes = _compute_normalized_boxes(image, sample["boxes"])
            anchor_targets, _ = assign_anchor_targets(norm_boxes, generate_anchors_from_priors().numpy())
            predictions = torch.zeros((self.reference_anchors.shape[0], 4), dtype=torch.float32)
            for anchor_idx, target_row in enumerate(anchor_targets):
                if target_row[0] != 1:
                    continue
                y_min, x_min, y_max, x_max = target_row[1:]
                x_center = float((x_min + x_max) / 2.0)
                y_center = float((y_min + y_max) / 2.0)
                width = float(x_max - x_min)
                height = float(y_max - y_min)
                anchor = self.reference_anchors[anchor_idx]
                anchor_w = anchor[2].item()
                anchor_h = anchor[3].item()
                predictions[anchor_idx, 0] = ((x_center - anchor[0].item()) / anchor_w) * self.loss.scale
                predictions[anchor_idx, 1] = ((y_center - anchor[1].item()) / anchor_h) * self.loss.scale
                predictions[anchor_idx, 2] = (width / anchor_w) * self.loss.scale
                predictions[anchor_idx, 3] = (height / anchor_h) * self.loss.scale

            decoded = self.loss.decode_boxes(predictions.unsqueeze(0), self.reference_anchors).squeeze(0).numpy()

            for anchor_idx, target_row in enumerate(anchor_targets):
                if target_row[0] != 1:
                    continue
                np.testing.assert_allclose(
                    decoded[anchor_idx],
                    target_row[1:],
                    atol=1e-6,
                    err_msg=f"Decode mismatch for {Path(sample['image_path']).stem} anchor {anchor_idx}",
                )


if __name__ == "__main__":
    unittest.main()


class TestIgnoreRegionsSurviveTransforms(unittest.TestCase):
    """
    The ignore flag travels as a fifth column through resize and augmentation.
    `_resize_and_pad` multiplied the whole array by the resize scale, which
    turned a flag of 1.0 into 0.2 and silently reclassified the region as
    ground truth: gt count rose and the region trained as a positive.
    """

    def test_resize_preserves_the_flag_column(self):
        from dataloader import CSVDetectorDataset
        dataset = CSVDetectorDataset(
            str(ASSETS_ROOT / "test_data.csv"), str(ASSETS_ROOT), augment=False)
        boxes = np.array([[0.2, 0.2, 0.4, 0.4, 0.0],
                          [0.6, 0.6, 0.8, 0.8, 1.0]], dtype=np.float32)
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        _, out = dataset._resize_and_pad(image, boxes.copy())
        self.assertEqual(out.shape[1], 5)
        np.testing.assert_allclose(out[:, 4], [0.0, 1.0], atol=1e-6)

    def test_rotation_preserves_the_flag_column(self):
        from utils import augmentation
        boxes = np.array([[0.3, 0.3, 0.5, 0.5, 1.0]], dtype=np.float32)
        np.random.seed(0)
        _, out = augmentation.augment_rotation(
            np.zeros((200, 200, 3), dtype=np.uint8), boxes.copy(), angle_range=(5, 5))
        self.assertEqual(out.shape[1], 5)
        self.assertEqual(float(out[0, 4]), 1.0)
