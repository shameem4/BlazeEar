"""
Unit tests for BlazeFace model.
"""
import unittest
from pathlib import Path

import torch

from blazeface import BlazeFace
from blazebase import load_mediapipe_weights
from utils.anchor_utils import anchor_options


class TestBlazeFace(unittest.TestCase):
    """Tests for BlazeFace model."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.model = BlazeFace().to(self.device)
        self.input_size = 128
        self.num_anchors = 896
        self.num_coords = 16  # 4 box + 12 keypoint coords

    def test_forward_shape(self):
        """Test that forward pass produces correct output shapes."""
        batch_sizes = [1, 2, 4]
        
        for batch_size in batch_sizes:
            with self.subTest(batch_size=batch_size):
                # Create random input tensor (B, C, H, W)
                x = torch.randn(batch_size, 3, self.input_size, self.input_size)
                
                # Forward pass
                self.model.eval()
                with torch.no_grad():
                    outputs = self.model(x)
                
                # Check output is a list of 2 tensors [boxes, scores]
                self.assertIsInstance(outputs, list)
                self.assertEqual(len(outputs), 2)
                
                boxes, scores = outputs
                
                # Check boxes shape: (B, 896, 16)
                self.assertEqual(
                    boxes.shape, 
                    torch.Size([batch_size, self.num_anchors, self.num_coords]),
                    f"Expected boxes shape ({batch_size}, {self.num_anchors}, {self.num_coords}), "
                    f"got {boxes.shape}"
                )
                
                # Check scores shape: (B, 896, 1)
                self.assertEqual(
                    scores.shape,
                    torch.Size([batch_size, self.num_anchors, 1]),
                    f"Expected scores shape ({batch_size}, {self.num_anchors}, 1), "
                    f"got {scores.shape}"
                )

    def test_forward_dtype(self):
        """Test that forward pass maintains float32 dtype."""
        x = torch.randn(1, 3, self.input_size, self.input_size)
        
        self.model.eval()
        with torch.no_grad():
            boxes, scores = self.model(x)
        
        self.assertEqual(boxes.dtype, torch.float32)
        self.assertEqual(scores.dtype, torch.float32)

    def test_load_weights(self):
        """Test loading MediaPipe pretrained weights."""
        weights_path = Path("model_weights/blazeface.pth")
        
        if not weights_path.exists():
            self.skipTest(f"Weights file not found: {weights_path}")
        
        # Create fresh model
        model = BlazeFace().to(self.device)
        
        # Load weights
        missing_keys, unexpected_keys = load_mediapipe_weights(
            model, 
            str(weights_path), 
            strict=False
        )
        
        # Check that critical layers were loaded (no missing backbone keys)
        backbone_missing = [k for k in missing_keys if "backbone" in k]
        self.assertEqual(
            len(backbone_missing), 0,
            f"Missing backbone keys: {backbone_missing}"
        )
        
        # Verify model can still forward after loading
        model.eval()
        x = torch.randn(1, 3, self.input_size, self.input_size)
        with torch.no_grad():
            outputs = model(x)
        
        self.assertEqual(len(outputs), 2)
        self.assertEqual(outputs[0].shape[1], self.num_anchors)

    def test_generate_anchors(self):
        """Test anchor generation."""
        model = BlazeFace().to(self.device)
        
        # Generate anchors
        model.generate_anchors(anchor_options)
        
        # Check anchors were created
        self.assertTrue(hasattr(model, 'anchors'))
        self.assertEqual(model.anchors.shape, torch.Size([self.num_anchors, 4]))
        
        # Check anchor values are in valid range [0, 1]
        self.assertTrue(torch.all(model.anchors[:, :2] >= 0))  # x, y centers
        self.assertTrue(torch.all(model.anchors[:, :2] <= 1))

    def test_predict_on_batch(self):
        """Test predict_on_batch returns filtered detections."""
        weights_path = Path("model_weights/blazeface.pth")
        
        if not weights_path.exists():
            self.skipTest(f"Weights file not found: {weights_path}")
        
        model = BlazeFace().to(self.device)
        load_mediapipe_weights(model, str(weights_path), strict=False)
        model.generate_anchors(anchor_options)
        model.eval()
        
        # Create batch of images
        batch_size = 2
        x = torch.randint(0, 256, (batch_size, self.input_size, self.input_size, 3), dtype=torch.uint8)
        
        # Predict
        detections = model.predict_on_batch(x.numpy())
        
        # Check output structure
        self.assertIsInstance(detections, list)
        self.assertEqual(len(detections), batch_size)
        
        # Each detection should be a tensor with shape (N, num_coords+1)
        for det in detections:
            self.assertIsInstance(det, torch.Tensor)
            if det.numel() > 0:
                self.assertEqual(det.shape[1], self.num_coords + 1)  # coords + score

    def test_model_parameters(self):
        """Test model has expected number of parameters."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # BlazeFace should have around 100k parameters
        self.assertGreater(total_params, 50000)
        self.assertLess(total_params, 500000)
        
        # All params should be trainable by default
        self.assertEqual(total_params, trainable_params)

    def test_freeze_keypoint_regressors(self):
        """Test freezing keypoint regressor layers."""
        model = BlazeFace()
        
        # Initially all trainable
        kp_params_before = sum(
            p.numel() for p in model.regressor_8_kp.parameters() if p.requires_grad
        ) + sum(
            p.numel() for p in model.regressor_16_kp.parameters() if p.requires_grad
        )
        self.assertGreater(kp_params_before, 0)
        
        # Freeze keypoint regressors
        model.freeze_keypoint_regressors()
        
        # Check they're frozen
        kp_params_after = sum(
            p.numel() for p in model.regressor_8_kp.parameters() if p.requires_grad
        ) + sum(
            p.numel() for p in model.regressor_16_kp.parameters() if p.requires_grad
        )
        self.assertEqual(kp_params_after, 0)


if __name__ == "__main__":
    unittest.main()
