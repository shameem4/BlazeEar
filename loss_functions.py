"""
Loss functions for ear detection models.

Provides:
- BlazeEarDetectionLoss: Primary loss with BCE or focal loss option

Based on vincent1bt/blazeface-tensorflow loss implementation:
- Hard negative mining for background samples
- Huber/Smooth L1 for box regression
- BCE or Focal loss for classification
- Loss formula: detection_loss * 150 + background_loss * 35 + positive_loss * 35
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config import (DEFAULT_INPUT_SIZE, HARD_NEGATIVE_PRIOR_CAP,
                          HARD_NEGATIVE_RATIO)

from utils.metrics import compute_mean_iou_torch, compute_map_torch
from utils.box_utils import decode_boxes as _decode_boxes_util


class BlazeEarDetectionLoss(nn.Module):
    """
    Loss function for BlazeEar detector following vincent1bt approach.

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
        hard_negative_ratio: float = HARD_NEGATIVE_RATIO,
        hard_negative_prior_cap: float = HARD_NEGATIVE_PRIOR_CAP,
        detection_weight: float = 150.0,
        classification_weight: float = 35.0,
        scale: int = DEFAULT_INPUT_SIZE,
        min_negatives_per_image: int = 10,
        use_focal_loss: bool = False,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        positive_classification_weight: Optional[float] = None,
        regression_beta: float = 0.1
    ):
        """
        Args:
            hard_negative_ratio: Ratio of negatives to positives for hard mining.
                The budget is derived from the batch's MEAN positive count, so
                changing the share of all-background images in the data also
                changes mining strength on the images that do have ears.
            detection_weight: Weight for box regression loss
            classification_weight: Weight for both positive and background loss
            scale: Image scale for decoding (128 for front model)
            min_negatives_per_image: Minimum number of negatives per image
            use_focal_loss: Whether to use focal loss instead of BCE
            focal_alpha: Focal loss alpha parameter (weight for positive class)
            focal_gamma: Focal loss gamma parameter (focusing parameter)
            positive_classification_weight: Optional override for positive classification loss weight.
                Defaults to classification_weight when not provided.
            regression_beta: SmoothL1 transition point, in the same units as
                the decoded boxes (normalized [0, 1]). 0.1 is about 13 px at the
                128 px input: errors below it are treated as fine localization
                and get the quadratic branch, larger ones are outliers and get
                the linear branch. The previous default of 1.0 is the width of
                the whole image, so a 6 px error received a gradient of 0.05
                while a gross error received one 20x larger -- fine localization
                was effectively untrained. Measured on the epoch-94 checkpoint,
                48.3% of positive-anchor errors fell below 1.0.
        """
        super().__init__()
        self.hard_negative_ratio = hard_negative_ratio
        self.hard_negative_prior_cap = float(hard_negative_prior_cap)
        # Anchor index -> prior group, derived lazily from the anchor tensor
        # the caller passes and cached, since it never changes within a run.
        self._prior_groups: Optional[list[torch.Tensor]] = None
        self._prior_groups_key: Optional[tuple] = None
        self.detection_weight = detection_weight
        self.classification_weight = classification_weight
        self.positive_classification_weight = \
            positive_classification_weight if positive_classification_weight is not None else classification_weight
        self.scale = scale
        self.min_negatives_per_image = min_negatives_per_image
        self.use_focal_loss = use_focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        self.regression_beta = float(regression_beta)
        self.huber_loss = nn.SmoothL1Loss(reduction='mean', beta=self.regression_beta)
    
    def focal_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute focal loss for classification, from logits.

        FL(pt) = -alpha * (1 - pt)^gamma * log(pt)

        Args:
            logits: Raw classification logits (no sigmoid applied) [N]
            target: Target labels (0 or 1) [N]

        Returns:
            Focal loss value (scalar)
        """
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')

        # pt = p for positives, 1 - p for negatives. exp(-bce) recovers it
        # without a second sigmoid and without the saturation that made the
        # probability-space version lose gradient on confident mistakes.
        pt = torch.exp(-bce)
        focal_weight = (1 - pt) ** self.focal_gamma
        alpha_weight = torch.where(target == 1, self.focal_alpha, 1 - self.focal_alpha)

        return (alpha_weight * focal_weight * bce).mean()
    
    def bce_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute binary cross entropy from logits.

        Args:
            logits: Raw classification logits (no sigmoid applied) [N]
            target: Target labels (0 or 1) [N]

        Returns:
            BCE loss value (scalar)
        """
        return F.binary_cross_entropy_with_logits(logits, target)
    
    def classification_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute classification loss (BCE or Focal) from logits.

        Args:
            logits: Raw classification logits [N]
            target: Target labels [N]

        Returns:
            Classification loss value
        """
        if self.use_focal_loss:
            return self.focal_loss(logits, target)
        else:
            return self.bce_loss(logits, target)
    
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
    
    def _groups_for(self, reference_anchors: torch.Tensor) -> list[torch.Tensor]:
        """Anchor indices grouped by (w, h) prior, cached per anchor tensor.

        `reference_anchors` is [N, 4] as [x, y, w, h]; a [N, 2] tensor carries
        no prior and yields a single group, which disables the cap.
        """
        key = (int(reference_anchors.shape[0]), int(reference_anchors.shape[1]),
               str(reference_anchors.device))
        if self._prior_groups is not None and self._prior_groups_key == key:
            return self._prior_groups

        if reference_anchors.shape[1] < 4:
            groups = [torch.arange(reference_anchors.shape[0],
                                   device=reference_anchors.device)]
        else:
            wh = reference_anchors[:, 2:4]
            unique = torch.unique(wh, dim=0)
            groups = [torch.nonzero((wh == row).all(dim=1), as_tuple=True)[0]
                      for row in unique]
        self._prior_groups = groups
        self._prior_groups_key = key
        return groups

    def _mine_negatives(
        self,
        scores: torch.Tensor,
        budget: int,
        reference_anchors: torch.Tensor,
    ) -> torch.Tensor:
        """The `budget` hardest background scores per image, [B, budget].

        With `hard_negative_prior_cap >= 1.0` this is a plain global top-k,
        which is what it was. Below 1.0, each prior first offers only its own
        hardest `ceil(cap * budget)` candidates and the global top-k is taken
        from that pool, so no prior can exceed the ceiling.

        The ceiling is one-sided on purpose: a prior whose logits are not
        competitive still contributes nothing, so this removes a concentration
        without manufacturing background pressure anywhere. The total stays
        `budget`, so the loss scale does not move.
        """
        if budget <= 0:
            return scores[:, :0]

        groups = self._groups_for(reference_anchors)
        if self.hard_negative_prior_cap >= 1.0 or len(groups) < 2:
            return torch.topk(scores, k=budget, dim=-1).values

        # Enough per group that the pool can always fill the budget, even if
        # the cap alone would leave it short.
        per_group = max(
            int(-(-budget * self.hard_negative_prior_cap // 1)),
            int(-(-budget // len(groups))),
            1,
        )

        floor_value = torch.finfo(scores.dtype).min
        offers = []
        for index in groups:
            take = min(per_group, int(index.numel()))
            group_top = torch.topk(scores.index_select(1, index), k=take,
                                   dim=-1).values
            if take < per_group:
                group_top = F.pad(group_top, (0, per_group - take),
                                  value=floor_value)
            offers.append(group_top)

        pool = torch.cat(offers, dim=-1)
        return torch.topk(pool, k=min(budget, pool.shape[-1]), dim=-1).values

    def forward(
        self,
        class_logits: torch.Tensor,
        anchor_predictions: torch.Tensor,
        anchor_targets: torch.Tensor,
        reference_anchors: torch.Tensor,
        anchor_ignore: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute BlazeEar detection loss.

        Following vincent1bt/blazeface-tensorflow loss_functions.py:
        - Hard negative mining for background
        - Huber loss for box regression (only positive anchors)
        - BCE for classification

        Args:
            class_logits: [B, 896, 1] raw classification logits (no sigmoid)
            anchor_predictions: [B, 896, 4] predicted box offsets [dx, dy, w, h]
            anchor_targets: [B, 896, 5] targets [class, ymin, xmin, ymax, xmax]
            reference_anchors: [896, 2] anchor centers [x, y]
            anchor_ignore: [B, 896] bool. Anchors that overlap a box but did not
                win assignment. They are neither positives nor negatives, and are
                excluded from hard negative mining in particular: mining selects
                the *highest-scoring* backgrounds, which is exactly an anchor
                sitting on a real ear that lost the top-k race.

        Returns:
            Dict with 'total', 'detection', 'background', 'positive' losses
        """
        B = class_logits.shape[0]
        
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
        
        # Squeeze class logits
        class_pred_squeezed = class_logits.squeeze(-1)  # [B, 896]

        # For hard negative mining: push positive locations below every real
        # logit so they cannot be selected as negatives. A fixed sentinel such
        # as -99 is not safe in logit space, where a confident negative can go
        # lower than that.
        exclude_from_negatives = faces_mask_bool
        if anchor_ignore is not None:
            exclude_from_negatives = exclude_from_negatives | anchor_ignore.bool()

        predicted_classes_scores = class_pred_squeezed.masked_fill(
            exclude_from_negatives, torch.finfo(class_pred_squeezed.dtype).min
        )  # [B, 896]

        # Do not mine more negatives than actually exist, or the pool refills
        # with the masked-out sentinel values.
        available_negatives = int((~exclude_from_negatives).sum(dim=1).min().item())
        background_num = min(background_num, max(available_negatives, 0))
        
        # Select the top-k background predictions (hard negatives), subject to
        # a per-prior ceiling so no single prior can take the whole budget.
        background_class_predictions = self._mine_negatives(
            predicted_classes_scores, background_num, reference_anchors)
        
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
            background_loss = torch.tensor(0.0, device=class_logits.device)
        
        # Positive loss: predict 1 for faces/ears
        if positive_class_predictions.numel() > 0:
            positive_loss = self.classification_loss(
                positive_class_predictions,
                torch.ones_like(positive_class_predictions)
            )
        else:
            positive_loss = torch.tensor(0.0, device=class_logits.device)
        
        # === Regression Loss ===
        # Decode predictions to absolute coordinates
        offset_boxes = self.decode_boxes(anchor_predictions, reference_anchors)
        
        # Get predicted and true coords for positive anchors only
        filtered_pred_coords = offset_boxes[faces_mask_bool]  # [num_positives, 4]
        filtered_true_coords = true_coords[faces_mask_bool]   # [num_positives, 4]
        
        if filtered_pred_coords.numel() > 0:
            detection_loss = self.huber_loss(filtered_pred_coords, filtered_true_coords)
        else:
            detection_loss = torch.tensor(0.0, device=class_logits.device)
        
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
            'num_negatives': torch.tensor(background_num * B, device=class_logits.device),
            'num_ignored': (
                anchor_ignore.float().sum() if anchor_ignore is not None
                else torch.tensor(0.0, device=class_logits.device)
            )
        }


# Backwards compatibility re-exports (used in training scripts)
compute_mean_iou = compute_mean_iou_torch
compute_map = compute_map_torch


def get_loss(**kwargs) -> nn.Module:
    """
    Factory function to get BlazeEar detection loss.

    Args:
        **kwargs: Arguments for BlazeEarDetectionLoss
            - hard_negative_ratio: Ratio of negatives to positives (default: 1.0)
            - detection_weight: Weight for box regression (default: 150.0)
            - classification_weight: Weight for classification (default: 35.0)
            - scale: Image scale for decoding (default: 128)
            - use_focal_loss: Use focal loss instead of BCE (default: False)
            - focal_alpha: Focal loss alpha (default: 0.25)
            - focal_gamma: Focal loss gamma (default: 2.0)

    Returns:
        BlazeEarDetectionLoss module
    """
    return BlazeEarDetectionLoss(**kwargs)
