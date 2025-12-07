"""
Loss functions for ear detection models.

Provides:
- BlazeFaceDetectionLoss: Primary loss with BCE or focal loss option

Based on vincent1bt/blazeface-tensorflow loss implementation:
- Hard negative mining for background samples
- Huber/Smooth L1 for box regression
- BCE or Focal loss for classification
- Loss formula: detection_loss * 150 + background_loss * 35 + positive_loss * 35
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from utils.metrics import compute_mean_iou_torch, compute_map_torch
from utils.box_utils import decode_boxes as _decode_boxes_util


class BlazeFaceDetectionLoss(nn.Module):
    """
    Loss function for BlazeFace detector following vincent1bt approach.
    
    Combines:
    - Classification loss (BCE or Focal loss with hard negative mining)
    - Regression loss (Huber/Smooth L1 for box coordinates)
    
    Key features:
    - Hard negative mining: Selects highest-scoring background anchors
    - Decodes anchor predictions to absolute coordinates for loss
    - Optional focal loss for better handling of class imbalance
    - Weight formula: detection * 150 + background * 35 + positive * 35
    """
    
    def __init__(
        self,
        hard_negative_ratio: float = 1.0,
        detection_weight: float = 150.0,
        classification_weight: float = 35.0,
        scale: int = 128,
        min_negatives_per_image: int = 10,
        use_focal_loss: bool = False,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        positive_classification_weight: Optional[float] = None
    ):
        """
        Args:
            hard_negative_ratio: Ratio of negatives to positives for hard mining
            detection_weight: Weight for box regression loss
            classification_weight: Weight for both positive and background loss
            scale: Image scale for decoding (128 for front model)
            min_negatives_per_image: Minimum number of negatives per image
            use_focal_loss: Whether to use focal loss instead of BCE
            focal_alpha: Focal loss alpha parameter (weight for positive class)
            focal_gamma: Focal loss gamma parameter (focusing parameter)
            positive_classification_weight: Optional override for positive classification loss weight.
                Defaults to classification_weight when not provided.
        """
        super().__init__()
        self.hard_negative_ratio = hard_negative_ratio
        self.detection_weight = detection_weight
        self.classification_weight = classification_weight
        self.positive_classification_weight = \
            positive_classification_weight if positive_classification_weight is not None else classification_weight
        self.scale = scale
        self.min_negatives_per_image = min_negatives_per_image
        self.use_focal_loss = use_focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        self.huber_loss = nn.SmoothL1Loss(reduction='mean')
    
    def focal_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute focal loss for classification.
        
        Focal loss down-weights easy examples and focuses on hard negatives.
        FL(pt) = -alpha * (1 - pt)^gamma * log(pt)
        
        Args:
            pred: Predicted probabilities (after sigmoid) [N]
            target: Target labels (0 or 1) [N]
            
        Returns:
            Focal loss value (scalar)
        """
        # Clamp for numerical stability
        pred = torch.clamp(pred, min=1e-7, max=1 - 1e-7)
        
        # Binary cross entropy per element
        bce = -target * torch.log(pred) - (1 - target) * torch.log(1 - pred)
        
        # Focal weight: (1 - pt)^gamma
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_weight = (1 - pt) ** self.focal_gamma
        
        # Alpha weighting
        alpha_weight = torch.where(target == 1, self.focal_alpha, 1 - self.focal_alpha)
        
        loss = alpha_weight * focal_weight * bce
        return loss.mean()
    
    def bce_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute binary cross entropy loss.
        
        Args:
            pred: Predicted probabilities (after sigmoid) [N]
            target: Target labels (0 or 1) [N]
            
        Returns:
            BCE loss value (scalar)
        """
        # Clamp for numerical stability
        pred = torch.clamp(pred, min=1e-7, max=1 - 1e-7)
        bce = -target * torch.log(pred) - (1 - target) * torch.log(1 - pred)
        return bce.mean()
    
    def classification_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute classification loss (BCE or Focal).
        
        Args:
            pred: Predicted probabilities [N]
            target: Target labels [N]
            
        Returns:
            Classification loss value
        """
        if self.use_focal_loss:
            return self.focal_loss(pred, target)
        else:
            return self.bce_loss(pred, target)
    
    def decode_boxes(
        self,
        anchor_predictions: torch.Tensor,
        reference_anchors: torch.Tensor
    ) -> torch.Tensor:
        """
        Decode anchor predictions to absolute box coordinates.

        Wrapper around utils.box_utils.decode_boxes.

        Args:
            anchor_predictions: [B, 896, 4] predicted offsets [dx, dy, w, h]
            reference_anchors: [896, 2] or [896, 4] anchor centers [x, y, ...]

        Returns:
            [B, 896, 4] decoded boxes [ymin, xmin, ymax, xmax] in normalized coords
        """
        return _decode_boxes_util(anchor_predictions, reference_anchors, scale=self.scale)
    
    def forward(
        self,
        class_predictions: torch.Tensor,
        anchor_predictions: torch.Tensor,
        anchor_targets: torch.Tensor,
        reference_anchors: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute BlazeFace detection loss.
        
        Following vincent1bt/blazeface-tensorflow loss_functions.py:
        - Hard negative mining for background
        - Huber loss for box regression (only positive anchors)
        - BCE for classification
        
        Args:
            class_predictions: [B, 896, 1] predicted class scores (sigmoid applied)
            anchor_predictions: [B, 896, 4] predicted box offsets [dx, dy, w, h]
            anchor_targets: [B, 896, 5] targets [class, ymin, xmin, ymax, xmax]
            reference_anchors: [896, 2] anchor centers [x, y]
            
        Returns:
            Dict with 'total', 'detection', 'background', 'positive' losses
        """
        B = class_predictions.shape[0]
        
        # Extract targets
        true_classes = anchor_targets[:, :, 0]  # [B, 896]
        true_coords = anchor_targets[:, :, 1:]  # [B, 896, 4]
        
        # Boolean mask for positive anchors
        faces_mask_bool = true_classes > 0.5  # [B, 896]
        
        # Count positives
        faces_num = faces_mask_bool.float().sum()
        
        # Hard negative mining: select ratio * num_positives negatives per batch
        # Select the highest-scoring background predictions
        background_num = max(
            int(faces_num * self.hard_negative_ratio) // B,
            self.min_negatives_per_image
        )
        
        # Squeeze class predictions
        class_pred_squeezed = class_predictions.squeeze(-1)  # [B, 896]
        
        # For hard negative mining: set positive locations to very low score
        # so they won't be selected as negatives
        predicted_classes_scores = torch.where(
            faces_mask_bool,
            torch.full_like(class_pred_squeezed, -99.0),
            class_pred_squeezed
        )  # [B, 896]
        
        # Sort and select top-k background predictions (hard negatives)
        sorted_scores, _ = torch.sort(predicted_classes_scores, dim=-1, descending=True)
        background_class_predictions = sorted_scores[:, :background_num]  # [B, background_num]
        
        # Get positive predictions
        positive_class_predictions = class_pred_squeezed[faces_mask_bool]  # [num_positives]
        
        # === Classification Loss ===
        # Background loss: predict 0 for background
        if background_class_predictions.numel() > 0:
            background_loss = self.classification_loss(
                background_class_predictions,
                torch.zeros_like(background_class_predictions)
            )
        else:
            background_loss = torch.tensor(0.0, device=class_predictions.device)
        
        # Positive loss: predict 1 for faces/ears
        if positive_class_predictions.numel() > 0:
            positive_loss = self.classification_loss(
                positive_class_predictions,
                torch.ones_like(positive_class_predictions)
            )
        else:
            positive_loss = torch.tensor(0.0, device=class_predictions.device)
        
        # === Regression Loss ===
        # Decode predictions to absolute coordinates
        offset_boxes = self.decode_boxes(anchor_predictions, reference_anchors)
        
        # Get predicted and true coords for positive anchors only
        filtered_pred_coords = offset_boxes[faces_mask_bool]  # [num_positives, 4]
        filtered_true_coords = true_coords[faces_mask_bool]   # [num_positives, 4]
        
        if filtered_pred_coords.numel() > 0:
            detection_loss = self.huber_loss(filtered_pred_coords, filtered_true_coords)
        else:
            detection_loss = torch.tensor(0.0, device=class_predictions.device)
        
        # === Combined Loss ===
        # Following vincent1bt formula: detection * 150 + background * 35 + positive * 35
        total_loss = (
            detection_loss * self.detection_weight +
            background_loss * self.classification_weight +
            positive_loss * self.positive_classification_weight
        )
        
        return {
            'total': total_loss,
            'detection': detection_loss,
            'background': background_loss,
            'positive': positive_loss,
            'num_positives': faces_num,
            'num_negatives': torch.tensor(background_num * B, device=class_predictions.device)
        }


# Backwards compatibility re-exports (used in training scripts)
compute_mean_iou = compute_mean_iou_torch
compute_map = compute_map_torch


def get_loss(**kwargs) -> nn.Module:
    """
    Factory function to get BlazeFace detection loss.

    Args:
        **kwargs: Arguments for BlazeFaceDetectionLoss
            - hard_negative_ratio: Ratio of negatives to positives (default: 1.0)
            - detection_weight: Weight for box regression (default: 150.0)
            - classification_weight: Weight for classification (default: 35.0)
            - scale: Image scale for decoding (default: 128)
            - use_focal_loss: Use focal loss instead of BCE (default: False)
            - focal_alpha: Focal loss alpha (default: 0.25)
            - focal_gamma: Focal loss gamma (default: 2.0)
        
    Returns:
        BlazeFaceDetectionLoss module
    """
    return BlazeFaceDetectionLoss(**kwargs)
