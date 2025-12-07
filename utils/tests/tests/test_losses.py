"""
Unit tests for loss functions.
"""
import unittest

import torch

from loss_functions import BlazeFaceDetectionLoss, get_loss
from utils.anchor_utils import generate_reference_anchors


class TestLosses(unittest.TestCase):
    """Tests for BlazeFaceDetectionLoss."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.batch_size = 2
        self.num_anchors = 896
        self.loss_fn = BlazeFaceDetectionLoss(
            hard_negative_ratio=1.0,
            detection_weight=150.0,
            classification_weight=35.0,
            scale=128,
            min_negatives_per_image=10
        )
        # Generate reference anchors (returns tuple: full, small, big)
        reference_anchors, _, _ = generate_reference_anchors()
        self.reference_anchors = reference_anchors.float()

    def _create_dummy_inputs(
        self,
        num_positives_per_batch: int = 5
    ):
        """
        Create dummy inputs for loss computation.
        
        Args:
            num_positives_per_batch: Number of positive anchors per image
            
        Returns:
            Tuple of (class_predictions, anchor_predictions, anchor_targets)
        """
        B = self.batch_size
        N = self.num_anchors
        
        # Class predictions: sigmoid outputs in [0, 1]
        class_predictions = torch.rand(B, N, 1) * 0.3  # mostly background
        
        # Anchor predictions: [dx, dy, w, h] offsets
        anchor_predictions = torch.randn(B, N, 4) * 0.1
        
        # Anchor targets: [class, ymin, xmin, ymax, xmax]
        anchor_targets = torch.zeros(B, N, 5)
        
        # Set some positives
        for b in range(B):
            for i in range(num_positives_per_batch):
                idx = i * 10  # spread out
                anchor_targets[b, idx, 0] = 1.0  # class = 1
                # Random box
                ymin, xmin = torch.rand(2) * 0.5
                ymax = ymin + torch.rand(1) * 0.3 + 0.1
                xmax = xmin + torch.rand(1) * 0.3 + 0.1
                anchor_targets[b, idx, 1:5] = torch.tensor([ymin, xmin, ymax.item(), xmax.item()])
                # Higher prediction for positives
                class_predictions[b, idx, 0] = 0.7 + torch.rand(1) * 0.2
        
        return class_predictions, anchor_predictions, anchor_targets

    def test_loss_computation(self):
        """Test that loss computation produces valid outputs."""
        class_preds, anchor_preds, anchor_targets = self._create_dummy_inputs(
            num_positives_per_batch=5
        )
        
        # Compute loss
        loss_dict = self.loss_fn(
            class_preds,
            anchor_preds,
            anchor_targets,
            self.reference_anchors
        )
        
        # Check all expected keys are present
        expected_keys = ['total', 'detection', 'background', 'positive', 'num_positives', 'num_negatives']
        for key in expected_keys:
            self.assertIn(key, loss_dict, f"Missing key: {key}")
        
        # Check losses are finite and non-negative
        for key in ['total', 'detection', 'background', 'positive']:
            self.assertTrue(
                torch.isfinite(loss_dict[key]),
                f"{key} loss is not finite: {loss_dict[key]}"
            )
            self.assertGreaterEqual(
                loss_dict[key].item(), 0,
                f"{key} loss should be non-negative"
            )
        
        # Check total loss is weighted sum
        expected_total = (
            loss_dict['detection'] * 150.0 +
            loss_dict['background'] * 35.0 +
            loss_dict['positive'] * 35.0
        )
        self.assertAlmostEqual(
            loss_dict['total'].item(),
            expected_total.item(),
            places=5,
            msg="Total loss doesn't match weighted sum"
        )
        
        # Check num_positives matches what we set
        self.assertEqual(
            loss_dict['num_positives'].item(),
            self.batch_size * 5,
            "num_positives should be batch_size * 5"
        )

    def test_loss_computation_no_positives(self):
        """Test loss computation with no positive samples."""
        B, N = self.batch_size, self.num_anchors
        
        # All background
        class_preds = torch.rand(B, N, 1) * 0.3
        anchor_preds = torch.randn(B, N, 4) * 0.1
        anchor_targets = torch.zeros(B, N, 5)  # all zeros = all background
        
        loss_dict = self.loss_fn(
            class_preds,
            anchor_preds,
            anchor_targets,
            self.reference_anchors
        )
        
        # Detection loss should be 0 (no positives to regress)
        self.assertEqual(loss_dict['detection'].item(), 0.0)
        
        # Positive loss should be 0
        self.assertEqual(loss_dict['positive'].item(), 0.0)
        
        # Background loss should be non-zero
        self.assertGreater(loss_dict['background'].item(), 0.0)
        
        # num_positives should be 0
        self.assertEqual(loss_dict['num_positives'].item(), 0.0)

    def test_hard_negative_mining(self):
        """Test that hard negative mining selects highest-scoring backgrounds."""
        B, N = 1, self.num_anchors
        
        # Create specific class predictions
        class_preds = torch.zeros(B, N, 1)
        
        # Set one positive at index 0
        anchor_targets = torch.zeros(B, N, 5)
        anchor_targets[0, 0, 0] = 1.0  # positive
        anchor_targets[0, 0, 1:5] = torch.tensor([0.3, 0.3, 0.6, 0.6])
        class_preds[0, 0, 0] = 0.9  # high positive prediction
        
        # Set specific background scores - create clear hard negatives
        # Indices 1-10: low scores (0.1)
        class_preds[0, 1:11, 0] = 0.1
        # Indices 11-20: high scores (0.8) - these should be selected as hard negatives
        class_preds[0, 11:21, 0] = 0.8
        # Rest: medium scores (0.5)
        class_preds[0, 21:, 0] = 0.5
        
        anchor_preds = torch.randn(B, N, 4) * 0.1
        
        # Create loss with specific settings
        loss_fn = BlazeFaceDetectionLoss(
            hard_negative_ratio=10.0,  # 10 negatives per positive
            min_negatives_per_image=5
        )
        
        loss_dict = loss_fn(
            class_preds,
            anchor_preds,
            anchor_targets,
            self.reference_anchors
        )
        
        # Should have 1 positive
        self.assertEqual(loss_dict['num_positives'].item(), 1.0)
        
        # Should have max(1*10, 5) = 10 negatives (but capped by hard_negative_ratio)
        # Actually: background_num = max(int(1 * 10) // 1, 5) = max(10, 5) = 10
        self.assertEqual(loss_dict['num_negatives'].item(), 10)
        
        # Background loss should be higher because hard negatives have high scores
        # BCE(-log(1-0.8)) is high for background with 0.8 prediction
        self.assertGreater(loss_dict['background'].item(), 0.5)

    def test_hard_negative_mining_min_negatives(self):
        """Test min_negatives_per_image is respected."""
        B, N = 1, self.num_anchors
        
        # Single positive
        class_preds = torch.rand(B, N, 1) * 0.3
        anchor_targets = torch.zeros(B, N, 5)
        anchor_targets[0, 0, 0] = 1.0
        anchor_targets[0, 0, 1:5] = torch.tensor([0.3, 0.3, 0.6, 0.6])
        class_preds[0, 0, 0] = 0.9
        
        anchor_preds = torch.randn(B, N, 4) * 0.1
        
        # Low ratio but high min
        loss_fn = BlazeFaceDetectionLoss(
            hard_negative_ratio=0.1,  # Would give 0 negatives for 1 positive
            min_negatives_per_image=20
        )
        
        loss_dict = loss_fn(
            class_preds,
            anchor_preds,
            anchor_targets,
            self.reference_anchors
        )
        
        # Should use min_negatives_per_image
        self.assertEqual(loss_dict['num_negatives'].item(), 20)

    def test_focal_loss(self):
        """Test focal loss option."""
        loss_fn_focal = BlazeFaceDetectionLoss(
            use_focal_loss=True,
            focal_alpha=0.25,
            focal_gamma=2.0
        )
        
        class_preds, anchor_preds, anchor_targets = self._create_dummy_inputs()
        
        loss_dict = loss_fn_focal(
            class_preds,
            anchor_preds,
            anchor_targets,
            self.reference_anchors
        )
        
        # Should produce valid losses
        self.assertTrue(torch.isfinite(loss_dict['total']))
        self.assertGreater(loss_dict['total'].item(), 0)

    def test_focal_loss_vs_bce(self):
        """Test that focal loss down-weights easy examples compared to BCE."""
        # Easy example: high confidence correct prediction
        pred_easy = torch.tensor([0.99])  # confident correct
        target = torch.tensor([1.0])
        
        loss_fn_bce = BlazeFaceDetectionLoss(use_focal_loss=False)
        loss_fn_focal = BlazeFaceDetectionLoss(use_focal_loss=True, focal_gamma=2.0)
        
        bce_loss = loss_fn_bce.bce_loss(pred_easy, target)
        focal_loss = loss_fn_focal.focal_loss(pred_easy, target)
        
        # Focal loss should be lower for easy examples
        self.assertLess(
            focal_loss.item(),
            bce_loss.item(),
            "Focal loss should be lower than BCE for easy examples"
        )
        
        # Hard example: low confidence for positive
        pred_hard = torch.tensor([0.3])  # uncertain
        bce_hard = loss_fn_bce.bce_loss(pred_hard, target)
        focal_hard = loss_fn_focal.focal_loss(pred_hard, target)
        
        # Both should be relatively high, but ratio should be different
        # Focal loss focuses on hard examples
        easy_ratio = focal_loss / bce_loss
        hard_ratio = focal_hard / bce_hard
        
        # Hard examples should have a higher ratio (less down-weighted)
        self.assertGreater(
            hard_ratio.item(),
            easy_ratio.item(),
            "Focal loss should down-weight easy examples more than hard ones"
        )

    def test_get_loss_factory(self):
        """Test get_loss factory function."""
        loss_fn = get_loss(
            hard_negative_ratio=2.0,
            detection_weight=100.0,
            use_focal_loss=True
        )
        
        self.assertIsInstance(loss_fn, BlazeFaceDetectionLoss)
        self.assertEqual(loss_fn.hard_negative_ratio, 2.0)
        self.assertEqual(loss_fn.detection_weight, 100.0)
        self.assertTrue(loss_fn.use_focal_loss)

    def test_loss_gradients(self):
        """Test that gradients flow correctly through loss."""
        class_preds, anchor_preds, anchor_targets = self._create_dummy_inputs()
        
        # Enable gradients
        class_preds.requires_grad_(True)
        anchor_preds.requires_grad_(True)
        
        loss_dict = self.loss_fn(
            class_preds,
            anchor_preds,
            anchor_targets,
            self.reference_anchors
        )
        
        # Backward pass
        loss_dict['total'].backward()
        
        # Check gradients exist
        self.assertIsNotNone(class_preds.grad)
        self.assertIsNotNone(anchor_preds.grad)
        
        # Check gradients are finite
        if class_preds.grad is not None:
            self.assertTrue(torch.isfinite(class_preds.grad).all())
        if anchor_preds.grad is not None:
            self.assertTrue(torch.isfinite(anchor_preds.grad).all())


if __name__ == "__main__":
    unittest.main()
