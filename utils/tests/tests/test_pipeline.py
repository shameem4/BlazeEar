import json
import unittest
from pathlib import Path

import numpy as np
import torch

from dataloader import (
    CSVDetectorDataset,
    encode_boxes_to_anchors,
    flatten_anchor_targets,
)
from loss_functions import BlazeFaceDetectionLoss
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
        cls.loss = BlazeFaceDetectionLoss()

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

    def test_anchor_encoding_positive_indices_match_expected(self):
        for sample in self.dataset.samples:
            image = self.dataset._load_image(sample["image_path"])
            norm_boxes = _compute_normalized_boxes(image, sample["boxes"])
            small, big = encode_boxes_to_anchors(norm_boxes, input_size=self.dataset.target_size[0])
            anchor_targets = flatten_anchor_targets(small, big)
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
            small, big = encode_boxes_to_anchors(norm_boxes, input_size=self.dataset.target_size[0])
            anchor_targets = flatten_anchor_targets(small, big)
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
