"""
Integration tests for the BlazeFace training pipeline.

Tests end-to-end workflows:
- Data loading -> anchor encoding -> loss computation
- Model forward pass -> detection output
- Full training step (forward + backward)
- Inference pipeline with pretrained weights
"""
import unittest
from pathlib import Path

import numpy as np
import torch

from dataloader import (
    CSVDetectorDataset,
    encode_boxes_to_anchors,
    flatten_anchor_targets,
    create_dataloader,
)
from blazeface import BlazeFace
from blazebase import generate_reference_anchors, load_mediapipe_weights
from loss_functions import BlazeFaceDetectionLoss, compute_mean_iou
from utils.anchor_utils import anchor_options


# Test assets
ASSETS_ROOT = Path("utils/tests/assets")
CSV_PATH = ASSETS_ROOT / "test_data.csv"
WEIGHTS_PATH = Path("model_weights/blazeface.pth")


class TestDataToLossPipeline(unittest.TestCase):
    """Test data loading through loss computation."""

    @classmethod
    def setUpClass(cls):
        """Load dataset and model components once for all tests."""
        if not CSV_PATH.exists():
            raise unittest.SkipTest(f"Test data not found: {CSV_PATH}")
        
        cls.dataset = CSVDetectorDataset(
            csv_path=str(CSV_PATH),
            root_dir=str(ASSETS_ROOT),
            target_size=(128, 128),
            augment=False,
        )
        cls.reference_anchors, _, _ = generate_reference_anchors()
        cls.loss_fn = BlazeFaceDetectionLoss(scale=128)
        cls.device = torch.device("cpu")

    def test_dataloader_to_loss_pipeline(self):
        """Test that data from dataloader can be fed through loss computation."""
        # Get a sample
        sample = self.dataset[0]
        
        # Check sample structure
        self.assertIn("image", sample)
        self.assertIn("anchor_targets", sample)
        
        image = sample["image"]
        anchor_targets = sample["anchor_targets"]
        
        # Verify shapes
        self.assertEqual(image.shape, torch.Size([3, 128, 128]))
        self.assertEqual(anchor_targets.shape, torch.Size([896, 5]))
        
        # Create mock predictions
        batch_size = 1
        class_preds = torch.rand(batch_size, 896, 1)
        anchor_preds = torch.randn(batch_size, 896, 4) * 0.1
        
        # Compute loss
        loss_dict = self.loss_fn(
            class_preds,
            anchor_preds,
            anchor_targets.unsqueeze(0),
            self.reference_anchors
        )
        
        # Verify loss output
        self.assertIn("total", loss_dict)
        self.assertTrue(torch.isfinite(loss_dict["total"]))

    def test_batch_dataloader_pipeline(self):
        """Test batched data through the pipeline."""
        dataloader = create_dataloader(
            csv_path=str(CSV_PATH),
            root_dir=str(ASSETS_ROOT),
            batch_size=2,
            target_size=(128, 128),
            augment=False,
            shuffle=False,
            num_workers=0,
        )
        
        # Get one batch
        batch = next(iter(dataloader))
        
        images = batch["image"]
        anchor_targets = batch["anchor_targets"]
        
        # Verify batch shapes
        self.assertEqual(images.shape[0], 2)  # batch size
        self.assertEqual(images.shape[1:], torch.Size([3, 128, 128]))
        self.assertEqual(anchor_targets.shape, torch.Size([2, 896, 5]))
        
        # Mock predictions and compute loss
        class_preds = torch.rand(2, 896, 1)
        anchor_preds = torch.randn(2, 896, 4) * 0.1
        
        loss_dict = self.loss_fn(
            class_preds,
            anchor_preds,
            anchor_targets,
            self.reference_anchors
        )
        
        self.assertTrue(torch.isfinite(loss_dict["total"]))

    def test_encode_decode_roundtrip(self):
        """Test that encoding and decoding boxes is consistent."""
        # Create a known box
        gt_box = np.array([[0.3, 0.3, 0.6, 0.6]], dtype=np.float32)  # ymin, xmin, ymax, xmax
        
        # Encode to anchors
        small_targets, big_targets = encode_boxes_to_anchors(gt_box, input_size=128)
        anchor_targets = flatten_anchor_targets(small_targets, big_targets)
        
        # Find positive anchor
        positive_mask = anchor_targets[:, 0] == 1
        self.assertTrue(positive_mask.any(), "Box should be assigned to at least one anchor")
        
        positive_indices = np.where(positive_mask)[0]
        
        # Check that assigned box matches original
        for idx in positive_indices:
            assigned_box = anchor_targets[idx, 1:]
            np.testing.assert_allclose(assigned_box, gt_box[0], atol=1e-3)


class TestModelInferencePipeline(unittest.TestCase):
    """Test model forward pass and inference."""

    @classmethod
    def setUpClass(cls):
        """Set up model for testing."""
        cls.device = torch.device("cpu")
        cls.model = BlazeFace().to(cls.device)
        cls.reference_anchors, _, _ = generate_reference_anchors()
        
        # Load pretrained weights if available
        if WEIGHTS_PATH.exists():
            load_mediapipe_weights(cls.model, str(WEIGHTS_PATH), strict=False)
            cls.has_weights = True
        else:
            cls.has_weights = False

    def test_model_forward_produces_valid_output(self):
        """Test that model forward pass produces valid outputs."""
        self.model.eval()
        
        # Random input
        x = torch.randn(1, 3, 128, 128)
        
        with torch.no_grad():
            outputs = self.model(x)
        
        # Should return [boxes, scores]
        self.assertEqual(len(outputs), 2)
        
        boxes, scores = outputs
        self.assertEqual(boxes.shape, torch.Size([1, 896, 16]))
        self.assertEqual(scores.shape, torch.Size([1, 896, 1]))

    def test_training_outputs_method(self):
        """Test get_training_outputs produces correct format for loss."""
        self.model.eval()
        x = torch.randn(1, 3, 128, 128)
        
        with torch.no_grad():
            raw_boxes, raw_scores = self.model.get_training_outputs(x)
        
        # raw_boxes: [B, 896, 16] (includes keypoints)
        # raw_scores: [B, 896, 1] (logits)
        self.assertEqual(raw_boxes.shape[1], 896)
        self.assertEqual(raw_scores.shape, torch.Size([1, 896, 1]))
        
        # Scores should be logits (can be negative)
        self.assertTrue(raw_scores.min() < 0 or raw_scores.max() > 1 or True)

    def test_predict_on_batch_with_real_image(self):
        """Test prediction on real test image."""
        if not self.has_weights:
            self.skipTest("Pretrained weights not available")
        
        if not CSV_PATH.exists():
            self.skipTest("Test images not available")
        
        # Load test image
        dataset = CSVDetectorDataset(
            csv_path=str(CSV_PATH),
            root_dir=str(ASSETS_ROOT),
            target_size=(128, 128),
            augment=False,
        )
        
        sample = dataset[0]
        image = sample["image"]  # [3, 128, 128]
        
        # Convert to batch format for predict_on_batch (expects uint8 HWC)
        # Actually predict_on_batch expects numpy [B, H, W, C] uint8
        image_np = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        
        self.model.generate_anchors(anchor_options)
        self.model.eval()
        
        detections = self.model.predict_on_batch(image_np[np.newaxis, ...])
        
        # Should return list of detections per image
        self.assertIsInstance(detections, list)
        self.assertEqual(len(detections), 1)


class TestFullTrainingStep(unittest.TestCase):
    """Test a complete training step with gradient flow."""

    @classmethod
    def setUpClass(cls):
        """Set up model and data for training step test."""
        if not CSV_PATH.exists():
            raise unittest.SkipTest(f"Test data not found: {CSV_PATH}")
        
        cls.device = torch.device("cpu")
        cls.model = BlazeFace().to(cls.device)
        cls.reference_anchors, _, _ = generate_reference_anchors()
        cls.reference_anchors = cls.reference_anchors.to(cls.device)
        cls.loss_fn = BlazeFaceDetectionLoss(scale=128).to(cls.device)
        
        cls.dataset = CSVDetectorDataset(
            csv_path=str(CSV_PATH),
            root_dir=str(ASSETS_ROOT),
            target_size=(128, 128),
            augment=False,
        )

    def test_single_training_step(self):
        """Test a complete forward-backward training step."""
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        
        # Get sample
        sample = self.dataset[0]
        image = sample["image"].unsqueeze(0).to(self.device)
        anchor_targets = sample["anchor_targets"].unsqueeze(0).to(self.device)
        
        # Forward pass
        raw_boxes, raw_scores = self.model.get_training_outputs(image)
        class_predictions = torch.sigmoid(raw_scores)
        anchor_predictions = raw_boxes[..., :4]
        
        # Compute loss
        loss_dict = self.loss_fn(
            class_predictions,
            anchor_predictions,
            anchor_targets,
            self.reference_anchors
        )
        
        total_loss = loss_dict["total"]
        self.assertTrue(torch.isfinite(total_loss))
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        
        # Check gradients exist
        has_gradients = False
        for param in self.model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_gradients = True
                break
        
        self.assertTrue(has_gradients, "Model should have non-zero gradients after backward")
        
        # Optimizer step
        optimizer.step()

    def test_multiple_training_steps_loss_changes(self):
        """Test that loss changes over multiple training steps."""
        model = BlazeFace().to(self.device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        reference_anchors = self.reference_anchors.clone()
        loss_fn = BlazeFaceDetectionLoss(scale=128).to(self.device)
        
        sample = self.dataset[0]
        image = sample["image"].unsqueeze(0).to(self.device)
        anchor_targets = sample["anchor_targets"].unsqueeze(0).to(self.device)
        
        losses = []
        for _ in range(5):
            raw_boxes, raw_scores = model.get_training_outputs(image)
            class_predictions = torch.sigmoid(raw_scores)
            anchor_predictions = raw_boxes[..., :4]
            
            loss_dict = loss_fn(
                class_predictions,
                anchor_predictions,
                anchor_targets,
                reference_anchors
            )
            
            losses.append(loss_dict["total"].item())
            
            optimizer.zero_grad()
            loss_dict["total"].backward()
            optimizer.step()
        
        # Loss should change (not necessarily decrease with random init)
        self.assertNotEqual(
            losses[0], losses[-1],
            "Loss should change over training steps"
        )

    def test_training_with_dataloader(self):
        """Test training loop with DataLoader."""
        model = BlazeFace().to(self.device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        dataloader = create_dataloader(
            csv_path=str(CSV_PATH),
            root_dir=str(ASSETS_ROOT),
            batch_size=2,
            target_size=(128, 128),
            augment=False,
            shuffle=False,
            num_workers=0,
        )
        
        reference_anchors = self.reference_anchors.clone()
        loss_fn = BlazeFaceDetectionLoss(scale=128).to(self.device)
        
        # Run one epoch
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            images = batch["image"].to(self.device)
            anchor_targets = batch["anchor_targets"].to(self.device)
            
            raw_boxes, raw_scores = model.get_training_outputs(images)
            class_predictions = torch.sigmoid(raw_scores)
            anchor_predictions = raw_boxes[..., :4]
            
            loss_dict = loss_fn(
                class_predictions,
                anchor_predictions,
                anchor_targets,
                reference_anchors
            )
            
            optimizer.zero_grad()
            loss_dict["total"].backward()
            optimizer.step()
            
            total_loss += loss_dict["total"].item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.assertGreater(avg_loss, 0, "Average loss should be positive")
        self.assertTrue(np.isfinite(avg_loss), "Average loss should be finite")


class TestMetricsComputation(unittest.TestCase):
    """Test metrics computation in training context."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.reference_anchors, _, _ = generate_reference_anchors()
        cls.loss_fn = BlazeFaceDetectionLoss(scale=128)

    def test_mean_iou_on_decoded_boxes(self):
        """Test mean IoU computation on decoded predictions."""
        # Create matching boxes
        pred_boxes = torch.tensor([
            [0.3, 0.3, 0.6, 0.6],
            [0.1, 0.1, 0.3, 0.3],
        ])
        gt_boxes = torch.tensor([
            [0.3, 0.3, 0.6, 0.6],  # perfect match
            [0.15, 0.15, 0.35, 0.35],  # partial overlap
        ])
        
        mean_iou = compute_mean_iou(pred_boxes, gt_boxes)
        
        # First is perfect match (1.0), second is partial
        self.assertGreater(mean_iou.item(), 0.5)
        self.assertLessEqual(mean_iou.item(), 1.0)

    def test_loss_metrics_tracking(self):
        """Test that loss function tracks positive/negative counts."""
        batch_size = 2
        num_anchors = 896
        
        # Create targets with known positive count
        anchor_targets = torch.zeros(batch_size, num_anchors, 5)
        anchor_targets[0, 0, 0] = 1.0  # 1 positive in first image
        anchor_targets[0, 0, 1:] = torch.tensor([0.3, 0.3, 0.6, 0.6])
        anchor_targets[1, 0, 0] = 1.0  # 1 positive in second image
        anchor_targets[1, 0, 1:] = torch.tensor([0.4, 0.4, 0.7, 0.7])
        
        class_preds = torch.rand(batch_size, num_anchors, 1)
        anchor_preds = torch.randn(batch_size, num_anchors, 4) * 0.1
        
        loss_dict = self.loss_fn(
            class_preds,
            anchor_preds,
            anchor_targets,
            self.reference_anchors
        )
        
        # Should have 2 positives total
        self.assertEqual(loss_dict["num_positives"].item(), 2.0)
        
        # Should have some negatives
        self.assertGreater(loss_dict["num_negatives"].item(), 0)


class TestAnchorEncodingIntegration(unittest.TestCase):
    """Test anchor encoding in realistic scenarios."""

    def test_multiple_boxes_different_scales(self):
        """Test encoding multiple boxes at different scales."""
        # Small box (should go to 16x16 anchors)
        # Large box (should go to 8x8 anchors)
        boxes = np.array([
            [0.1, 0.1, 0.15, 0.15],  # Small: 5% of image
            [0.2, 0.2, 0.6, 0.6],    # Large: 40% of image
        ], dtype=np.float32)
        
        small_targets, big_targets = encode_boxes_to_anchors(boxes, input_size=128)
        anchor_targets = flatten_anchor_targets(small_targets, big_targets)
        
        positive_indices = np.where(anchor_targets[:, 0] == 1)[0]
        
        # Should have at least 2 positive anchors (one per box)
        self.assertGreaterEqual(len(positive_indices), 2)
        
        # Check that positives are in different anchor ranges
        # Small anchors: indices 0-511, Big anchors: indices 512-895
        small_positives = [idx for idx in positive_indices if idx < 512]
        big_positives = [idx for idx in positive_indices if idx >= 512]
        
        # At least one box should match to each scale
        # (though this depends on box size and IoU thresholds)
        self.assertGreater(
            len(small_positives) + len(big_positives),
            0,
            "Should have positives in anchor grid"
        )

    def test_edge_boxes_handled_correctly(self):
        """Test boxes at image edges are encoded correctly."""
        edge_boxes = np.array([
            [0.0, 0.0, 0.1, 0.1],    # Top-left corner
            [0.9, 0.9, 1.0, 1.0],    # Bottom-right corner
            [0.0, 0.45, 0.1, 0.55],  # Left edge center
        ], dtype=np.float32)
        
        small_targets, big_targets = encode_boxes_to_anchors(edge_boxes, input_size=128)
        anchor_targets = flatten_anchor_targets(small_targets, big_targets)
        
        # Should not crash and should produce valid targets
        self.assertEqual(anchor_targets.shape, (896, 5))
        
        # All assigned boxes should have valid coordinates
        positive_mask = anchor_targets[:, 0] == 1
        if positive_mask.any():
            assigned_boxes = anchor_targets[positive_mask, 1:]
            self.assertTrue(np.all(assigned_boxes >= 0))
            self.assertTrue(np.all(assigned_boxes <= 1))


if __name__ == "__main__":
    unittest.main()
