"""
Unit tests for geometry utilities (IoU computations).
"""
import unittest

import numpy as np
import torch

from utils.iou import (
    compute_iou_np,
    compute_iou_torch,
    compute_iou_elementwise_torch,
    compute_iou_batch_np,
)


class TestIoU(unittest.TestCase):
    """Tests for IoU computation functions."""

    def test_iou_perfect_overlap(self):
        """Test IoU = 1.0 when boxes are identical."""
        # yxyx format: [ymin, xmin, ymax, xmax]
        box = np.array([0.0, 0.0, 1.0, 1.0])
        
        iou = compute_iou_np(box, box, box1_format="yxyx", box2_format="yxyx")
        self.assertAlmostEqual(iou, 1.0, places=5)
        
        # Test with different box positions
        box2 = np.array([0.25, 0.25, 0.75, 0.75])
        iou2 = compute_iou_np(box2, box2, box1_format="yxyx", box2_format="yxyx")
        self.assertAlmostEqual(iou2, 1.0, places=5)
        
        # Test with xyxy format
        box_xyxy = np.array([0.0, 0.0, 1.0, 1.0])
        iou_xyxy = compute_iou_np(box_xyxy, box_xyxy, box1_format="xyxy", box2_format="xyxy")
        self.assertAlmostEqual(iou_xyxy, 1.0, places=5)

    def test_iou_no_overlap(self):
        """Test IoU = 0.0 when boxes don't overlap."""
        # yxyx format: [ymin, xmin, ymax, xmax]
        box1 = np.array([0.0, 0.0, 0.5, 0.5])  # Top-left quadrant
        box2 = np.array([0.6, 0.6, 1.0, 1.0])  # Bottom-right quadrant
        
        iou = compute_iou_np(box1, box2, box1_format="yxyx", box2_format="yxyx")
        self.assertAlmostEqual(iou, 0.0, places=5)
        
        # Test with horizontally separated boxes
        box3 = np.array([0.0, 0.0, 0.5, 0.3])
        box4 = np.array([0.0, 0.5, 0.5, 1.0])
        iou2 = compute_iou_np(box3, box4, box1_format="yxyx", box2_format="yxyx")
        self.assertAlmostEqual(iou2, 0.0, places=5)
        
        # Test with vertically separated boxes
        box5 = np.array([0.0, 0.0, 0.3, 0.5])
        box6 = np.array([0.5, 0.0, 1.0, 0.5])
        iou3 = compute_iou_np(box5, box6, box1_format="yxyx", box2_format="yxyx")
        self.assertAlmostEqual(iou3, 0.0, places=5)

    def test_iou_torch_matches_numpy(self):
        """Test that PyTorch IoU matches NumPy IoU."""
        # Generate random test boxes
        np.random.seed(42)
        
        for _ in range(10):
            # Random boxes in yxyx format
            y1, x1 = np.random.rand(2) * 0.5
            y2, x2 = y1 + np.random.rand() * 0.5, x1 + np.random.rand() * 0.5
            box1_np = np.array([y1, x1, y2, x2])
            
            y1, x1 = np.random.rand(2) * 0.5
            y2, x2 = y1 + np.random.rand() * 0.5, x1 + np.random.rand() * 0.5
            box2_np = np.array([y1, x1, y2, x2])
            
            # Compute NumPy IoU
            iou_np = compute_iou_np(box1_np, box2_np, box1_format="yxyx", box2_format="yxyx")
            
            # Compute PyTorch IoU
            box1_torch = torch.tensor([box1_np])
            box2_torch = torch.tensor([box2_np])
            iou_torch = compute_iou_torch(box1_torch, box2_torch, format="yxyx")
            
            self.assertAlmostEqual(
                iou_np, 
                iou_torch[0, 0].item(), 
                places=5,
                msg=f"IoU mismatch for boxes {box1_np} and {box2_np}"
            )

    def test_iou_partial_overlap(self):
        """Test IoU for partial overlap cases."""
        # 50% overlap case
        # Box1: [0, 0, 1, 1] area=1
        # Box2: [0, 0.5, 1, 1.5] area=1
        # Intersection: [0, 0.5, 1, 1] area=0.5
        # Union: 1 + 1 - 0.5 = 1.5
        # IoU: 0.5 / 1.5 = 0.333...
        box1 = np.array([0.0, 0.0, 1.0, 1.0])
        box2 = np.array([0.0, 0.5, 1.0, 1.5])
        
        iou = compute_iou_np(box1, box2, box1_format="yxyx", box2_format="yxyx")
        expected_iou = 0.5 / 1.5  # = 1/3
        self.assertAlmostEqual(iou, expected_iou, places=5)

    def test_iou_batch_np(self):
        """Test batch IoU computation with NumPy."""
        boxes1 = np.array([
            [0.0, 0.0, 0.5, 0.5],
            [0.5, 0.5, 1.0, 1.0],
        ])
        boxes2 = np.array([
            [0.0, 0.0, 0.5, 0.5],
            [0.25, 0.25, 0.75, 0.75],
            [0.5, 0.5, 1.0, 1.0],
        ])
        
        iou_matrix = compute_iou_batch_np(boxes1, boxes2, format="yxyx")
        
        self.assertEqual(iou_matrix.shape, (2, 3))
        # First box with itself should be 1.0
        self.assertAlmostEqual(iou_matrix[0, 0], 1.0, places=5)
        # Second box with third (same box) should be 1.0
        self.assertAlmostEqual(iou_matrix[1, 2], 1.0, places=5)

    def test_iou_elementwise_torch(self):
        """Test element-wise IoU computation with PyTorch."""
        boxes1 = torch.tensor([
            [0.0, 0.0, 0.5, 0.5],
            [0.5, 0.5, 1.0, 1.0],
        ])
        boxes2 = torch.tensor([
            [0.0, 0.0, 0.5, 0.5],  # Same as boxes1[0]
            [0.5, 0.5, 1.0, 1.0],  # Same as boxes1[1]
        ])
        
        ious = compute_iou_elementwise_torch(boxes1, boxes2, format="yxyx")
        
        self.assertEqual(ious.shape, (2,))
        self.assertAlmostEqual(ious[0].item(), 1.0, places=5)
        self.assertAlmostEqual(ious[1].item(), 1.0, places=5)

    def test_iou_empty_boxes(self):
        """Test IoU with empty box arrays."""
        empty_np = np.zeros((0, 4))
        boxes_np = np.array([[0.0, 0.0, 0.5, 0.5]])
        
        iou_matrix = compute_iou_batch_np(empty_np, boxes_np, format="yxyx")
        self.assertEqual(iou_matrix.shape, (0, 1))
        
        iou_matrix2 = compute_iou_batch_np(boxes_np, empty_np, format="yxyx")
        self.assertEqual(iou_matrix2.shape, (1, 0))
        
        # PyTorch version
        empty_torch = torch.zeros((0, 4))
        boxes_torch = torch.tensor([[0.0, 0.0, 0.5, 0.5]])
        
        iou_torch = compute_iou_torch(empty_torch, boxes_torch, format="yxyx")
        self.assertEqual(iou_torch.shape, (0, 1))


if __name__ == "__main__":
    unittest.main()
