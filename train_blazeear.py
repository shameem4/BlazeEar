"""
Training script for BlazeEar Ear detector.

Complete training pipeline following vincent1bt/blazeface-tensorflow methodology:
- Anchor-based target encoding (from dataloader)
- Hard negative mining loss
- BCE or Focal loss for classification
- Huber loss for box regression

Usage (NPY format):
    python train_blazeear.py --train-data data/preprocessed/train_detector.npy
    python train_blazeear.py --train-data data/preprocessed/train_detector.npy --val-data data/preprocessed/val_detector.npy
    python train_blazeear.py --train-data data/preprocessed/train_detector.npy --epochs 500 --lr 1e-4

Usage (CSV format):
    # Default: MediaPipe weight initialization with auto-resume
    python train_blazeear.py --csv-format --train-data data/splits/train.csv --val-data data/splits/val.csv --data-root data/raw/blazeear

    # Train from scratch (random initialization)
    python train_blazeear.py --csv-format --train-data data/splits/train.csv --val-data data/splits/val.csv --data-root data/raw/blazeear --init-weights scratch

    # Start fresh (disable auto-resume, but use MediaPipe weights)
    python train_blazeear.py --csv-format --train-data data/splits/train.csv --val-data data/splits/val.csv --data-root data/raw/blazeear --no-auto-resume
"""

import argparse
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import time
from pathlib import Path
from collections.abc import Sized
from typing import Callable, Dict, List, Optional, TypedDict, cast

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LRScheduler
from torch.cuda.amp import autocast, GradScaler

from blazeear import BlazeEar
from blazebase import checkpoint_is_folded, load_checkpoint_state, load_mediapipe_weights
from utils.anchor_utils import get_anchors
from dataloader import create_dataloader
from loss_functions import BlazeEarDetectionLoss, compute_mean_iou
from utils.detection_eval import DetectionEvaluator
from utils.nms import nms_indices
from utils.config import (
    DEFAULT_BEST_CHECKPOINT,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_DATA_ROOT,
    DEFAULT_EPOCHS,
    DEFAULT_INPUT_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LOG_DIR,
    MAX_DETECTIONS,
    NMS_IOU_THRESHOLD,
    DEFAULT_NUM_WORKERS,
    DEFAULT_SAVE_EVERY,
    DEFAULT_TRAIN_CSV,
    DEFAULT_VAL_CSV,
    DEFAULT_WEIGHTS_PATH,
    DEFAULT_WEIGHT_DECAY,
)


def _dataset_length(loader: DataLoader) -> int:
    """Return the size of a DataLoader's underlying dataset."""
    dataset = loader.dataset
    if not isinstance(dataset, Sized):
        raise TypeError("Dataset must implement __len__() for progress reporting")
    return len(cast(Sized, dataset))


class BlazeEarTrainer:
    """
    Trainer for BlazeEar Ear detector.

    Handles training loop, validation, checkpointing, and logging.
    Following vincent1bt methodology for loss computation and metrics.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        loss_fn: Optional[BlazeEarDetectionLoss] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[LRScheduler] = None,
        device: str = 'cuda',
        checkpoint_dir: str = 'checkpoints',
        log_dir: str = 'logs',
        model_name: str = 'BlazeEar',
        scale: int = 128,
        compute_train_map: bool = False,
        eval_score_threshold: float = 0.1,
        nms_iou_threshold: float = NMS_IOU_THRESHOLD,
        max_eval_detections: int = 75,
        metric_threshold: float = 0.45,
        use_amp: bool = True
    ):
        """
        Args:
            model: BlazeEar model
            train_loader: Training data loader
            val_loader: Optional validation data loader
            loss_fn: Loss function (BlazeEarDetectionLoss if None)
            optimizer: Optimizer (AdamW if None)
            scheduler: Learning rate scheduler
            device: Device to train on
            checkpoint_dir: Directory for saving checkpoints
            log_dir: Directory for TensorBoard logs
            model_name: Name for saving checkpoints
            scale: Image scale for decoding (128 for front, 256 for back)
        """
        self.model: nn.Module = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.model_name = model_name
        self.scale = scale
        self.compute_train_map = compute_train_map

        # Mixed precision (only meaningful on CUDA)
        self.use_amp = bool(use_amp and str(device).startswith("cuda") and torch.cuda.is_available())
        self.scaler = GradScaler(enabled=self.use_amp)
        
        # The canonical anchors, shared with the dataloader's assignment and
        # with every inference path.
        self.reference_anchors = get_anchors().float().to(device)
        
        # Setup loss function
        self.loss_fn = loss_fn if loss_fn else BlazeEarDetectionLoss(scale=scale)
        self.loss_fn = self.loss_fn.to(device)
        
        # Setup optimizer
        self.optimizer = optimizer if optimizer else optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=1e-4
        )
        
        self.scheduler: LRScheduler | None = scheduler
        
        # Setup directories
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup TensorBoard
        self.writer = SummaryWriter(self.log_dir / model_name)
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_val_map = -1.0
        
        # Metrics tracking
        self.metrics = {
            'positive_correct': 0,
            'positive_total': 0,
            'background_correct': 0,
            'background_total': 0
        }
        self.eval_score_threshold = eval_score_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_eval_detections = max_eval_detections
        self.max_map_candidates = 200
        self.metric_threshold = metric_threshold
    
    def _get_training_outputs(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get raw training outputs from BlazeEar model.

        BlazeEar model returns (raw_boxes, raw_scores) from get_training_outputs()
        for training, which bypasses post-processing.

        Args:
            images: [B, 3, H, W] input images

        Returns:
            class_logits: [B, 896, 1] raw classification logits (no sigmoid).
                The loss applies binary_cross_entropy_with_logits itself, which
                is numerically stable under autocast; pre-applying sigmoid in
                fp16 and taking log of it afterwards was not.
            anchor_predictions: [B, 896, 4] box predictions
        """
        # Use training output method that bypasses NMS
        get_outputs = getattr(self.model, "get_training_outputs", None)
        if callable(get_outputs):
            raw_boxes, raw_scores = cast(
                Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
                get_outputs,
            )(images)
            # raw_boxes: [B, 896, 16] or [B, 896, 4]
            # raw_scores: [B, 896, 1] - raw logits

            # Extract first 4 coords if model outputs 16 (for keypoints)
            if raw_boxes.shape[-1] > 4:
                raw_boxes = raw_boxes[..., :4]

            return raw_scores, raw_boxes
        else:
            # Fallback: call model directly
            output = self.model(images)
            if isinstance(output, tuple):
                return output[1], output[0]  # logits, boxes
            raise ValueError("Model must have get_training_outputs() method")

    @staticmethod
    def _pairwise_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """
        Compute IoU between two sets of boxes ([ymin, xmin, ymax, xmax]).
        """
        if boxes1.numel() == 0 or boxes2.numel() == 0:
            return torch.zeros(
                (boxes1.shape[0], boxes2.shape[0]),
                device=boxes1.device if boxes1.numel() else boxes2.device
            )
        y_min = torch.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
        x_min = torch.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
        y_max = torch.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
        x_max = torch.minimum(boxes1[:, None, 3], boxes2[None, :, 3])

        inter_h = torch.clamp(y_max - y_min, min=0)
        inter_w = torch.clamp(x_max - x_min, min=0)
        intersection = inter_h * inter_w

        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union = area1[:, None] + area2[None, :] - intersection
        return intersection / (union + 1e-6)

    def _nms(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        iou_threshold: float
    ) -> torch.Tensor:
        """Indices to keep, via the one shared suppression.

        This was a Python while-loop over CUDA tensors, run once per
        validation image, which synchronised the GPU on every iteration and
        dominated epoch time. It also disagreed with the deployed paths on the
        IoU threshold, so the reported mAP described post-processing that
        nothing shipped.
        """
        return nms_indices(boxes, scores, iou_threshold, self.max_eval_detections)

    def _detections_for_image(
        self,
        scores: torch.Tensor,
        decoded_boxes: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run the evaluation detection path for a single image.

        Mirrors inference: take the top candidates, drop anything under the score
        threshold, then suppress duplicates. Returns (boxes, scores), possibly empty.

        Args:
            scores: [896] per-anchor confidences (sigmoid already applied)
            decoded_boxes: [896, 4] decoded boxes [ymin, xmin, ymax, xmax]
        """
        if scores.numel() and (scores.min() < 0.0 or scores.max() > 1.0):
            raise ValueError(
                "scores must be probabilities in [0, 1], got range "
                f"[{scores.min():.3f}, {scores.max():.3f}]. _get_training_outputs "
                "returns logits; apply sigmoid before calling this."
            )

        candidate_k = min(self.max_map_candidates, scores.numel())
        candidate_scores, candidate_indices = torch.topk(scores, k=candidate_k)
        candidate_boxes = decoded_boxes[candidate_indices]

        score_mask = candidate_scores > self.eval_score_threshold
        filtered_scores = candidate_scores[score_mask]
        filtered_boxes = candidate_boxes[score_mask]

        if filtered_scores.numel() == 0:
            return filtered_boxes, filtered_scores

        keep = self._nms(filtered_boxes, filtered_scores, self.nms_iou_threshold)
        if keep.numel() == 0:
            return filtered_boxes[:0], filtered_scores[:0]

        return filtered_boxes[keep], filtered_scores[keep]

    @staticmethod
    def _build_gt_from_targets(anchor_targets: torch.Tensor) -> List[torch.Tensor]:
        """
        Fallback GT extraction from anchor targets when dataset GT boxes are unavailable.
        """
        gt_boxes = []
        for b in range(anchor_targets.shape[0]):
            mask = anchor_targets[b, :, 0] > 0.5
            gt_boxes.append(anchor_targets[b, mask, 1:])
        return gt_boxes
    
    def _compute_metrics(
        self,
        class_logits: torch.Tensor,
        anchor_targets: torch.Tensor,
        anchor_predictions: torch.Tensor,
        gt_boxes_tensor: Optional[torch.Tensor] = None,
        gt_box_counts: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
        evaluator: Optional[DetectionEvaluator] = None,
        compute_iou_flag: bool = True
    ) -> Dict[str, float]:
        """
        Compute per-batch anchor-level diagnostics.

        Detection quality (mAP, post-NMS IoU) is not returned here: it is pooled
        across the whole split by `evaluator`, because averaging per-image AP over
        images with one or two boxes each measures quantization more than it
        measures the detector. Call `evaluator.compute()` once the split is done.

        Metrics returned:
        - positive_acc: % of positive anchors correctly classified
        - background_acc: % of background anchors correctly classified
        - positive_anchor_iou: regression quality on target-positive anchors only

        Args:
            class_logits: [B, 896, 1] raw classification logits
            anchor_targets: [B, 896, 5] targets [class, ymin, xmin, ymax, xmax]
            anchor_predictions: [B, 896, 4] predicted boxes
            gt_boxes_tensor: [B, max_gt, 4] padded ground-truth boxes
            gt_box_counts: [B] real box count per image
            threshold: Classification threshold
            evaluator: Optional split-level detection evaluator to feed

        Returns:
            Dictionary of per-batch diagnostics
        """
        true_classes = anchor_targets[:, :, 0]  # [B, 896]
        true_coords = anchor_targets[:, :, 1:]  # [B, 896, 4]
        # Metrics are thresholded and ranked in probability space.
        pred_scores = torch.sigmoid(class_logits).squeeze(-1)  # [B, 896]

        # Positive accuracy
        positive_mask = true_classes > 0.5
        if positive_mask.sum() > 0:
            positive_preds = pred_scores[positive_mask] > threshold
            positive_acc = positive_preds.float().mean().item()
        else:
            positive_acc = 0.0

        # Background accuracy
        background_mask = true_classes < 0.5
        if background_mask.sum() > 0:
            background_preds = pred_scores[background_mask] < threshold
            background_acc = background_preds.float().mean().item()
        else:
            background_acc = 1.0

        # Regression quality on anchors the target marks positive. This is a
        # training diagnostic only: false positives and missed detections cannot
        # affect it, so it must not be read as detection quality. The comparable
        # number is `detection_iou`, which the evaluator computes on post-NMS
        # detections.
        positive_anchor_iou = 0.0

        decoded_boxes = None
        if positive_mask.sum() > 0 and compute_iou_flag:
            decoded_boxes = self.loss_fn.decode_boxes(
                anchor_predictions, self.reference_anchors
            )
            pred_coords = decoded_boxes[positive_mask]
            gt_coords = true_coords[positive_mask]
            positive_anchor_iou = compute_mean_iou(pred_coords, gt_coords, scale=self.scale).item()

        # Feed the split-level evaluator, which pools detections across every
        # image before computing one PR curve. Unlike the previous per-image AP
        # average, images with no ground truth are still accumulated so that
        # false positives on them are counted.
        if evaluator is not None:
            if decoded_boxes is None:
                decoded_boxes = self.loss_fn.decode_boxes(
                    anchor_predictions, self.reference_anchors
                )

            batch_size = class_logits.shape[0]
            fallback_gt = None
            if gt_boxes_tensor is None or gt_box_counts is None:
                fallback_gt = self._build_gt_from_targets(anchor_targets)

            for b in range(batch_size):
                if gt_boxes_tensor is not None and gt_box_counts is not None:
                    count = int(gt_box_counts[b].item())
                    gt_boxes_batch = gt_boxes_tensor[b, :count]
                elif fallback_gt is not None and b < len(fallback_gt):
                    gt_boxes_batch = fallback_gt[b]
                else:
                    gt_boxes_batch = anchor_predictions.new_zeros((0, 4))

                det_boxes, det_scores = self._detections_for_image(
                    pred_scores[b], decoded_boxes[b]
                )
                evaluator.add_image(det_boxes, det_scores, gt_boxes_batch)

        return {
            'positive_acc': positive_acc,
            'background_acc': background_acc,
            'positive_anchor_iou': positive_anchor_iou,
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary of average losses and metrics for the epoch
        """
        self.model.train()
        epoch_losses: Dict[str, float] = {}
        epoch_metrics: Dict[str, float] = {
            'positive_acc': 0.0,
            'background_acc': 0.0,
            'positive_anchor_iou': 0.0
        }
        train_evaluator = DetectionEvaluator() if self.compute_train_map else None
        num_batches = 0
        num_metric_batches = 0
        last_metrics: Dict[str, float] = {
            'positive_acc': 0.0,
            'background_acc': 0.0,
            'positive_anchor_iou': 0.0
        }
        
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move data to device
            images = batch['image'].to(self.device)
            anchor_targets = batch['anchor_targets'].to(self.device)
            anchor_ignore = batch.get('anchor_ignore')
            if anchor_ignore is not None:
                anchor_ignore = anchor_ignore.to(self.device)
            gt_boxes_tensor = batch.get('gt_boxes')
            gt_box_counts = batch.get('gt_box_counts')
            if gt_boxes_tensor is not None and gt_box_counts is not None:
                gt_boxes_tensor = gt_boxes_tensor.to(self.device)
                gt_box_counts = gt_box_counts.to(self.device)
            else:
                gt_boxes_tensor = None
                gt_box_counts = None
            
            # Forward pass
            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.use_amp):
                class_logits, anchor_predictions = self._get_training_outputs(images)
                losses = self.loss_fn(
                    class_logits,
                    anchor_predictions,
                    anchor_targets,
                    self.reference_anchors,
                    anchor_ignore=anchor_ignore
                )

            total_loss = losses["total"]
            if self.use_amp:
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            # Compute metrics every 10 batches to reduce overhead
            compute_metrics_this_batch = (batch_idx % 10 == 0 or batch_idx == len(self.train_loader) - 1)
            if compute_metrics_this_batch:
                with torch.no_grad():
                    metrics = self._compute_metrics(
                        class_logits,
                        anchor_targets,
                        anchor_predictions,
                        gt_boxes_tensor,
                        gt_box_counts,
                        threshold=self.metric_threshold,
                        evaluator=train_evaluator,
                        compute_iou_flag=True
                    )
            else:
                # Skip metrics computation for this batch
                metrics = None
            
            # Accumulate losses
            for key, value in losses.items():
                if isinstance(value, torch.Tensor):
                    if key not in epoch_losses:
                        epoch_losses[key] = 0.0
                    epoch_losses[key] += value.item()
            
            # Accumulate metrics only when computed
            if metrics is not None:
                for key, value in metrics.items():
                    epoch_metrics[key] += value
                num_metric_batches += 1
            
            num_batches += 1
            self.global_step += 1
            
            # Log to TensorBoard (only when metrics computed)
            if self.global_step % 10 == 0 and metrics is not None:
                for key, value in losses.items():
                    if isinstance(value, torch.Tensor):
                        self.writer.add_scalar(f'train/{key}', value.item(), self.global_step)
                for key, value in metrics.items():
                    self.writer.add_scalar(f'train/{key}', value, self.global_step)
            
            # Print progress (following vincent1bt style)
            if batch_idx % 20 == 0 or batch_idx == len(self.train_loader) - 1:
                # Use last computed metrics for display
                if metrics is not None:
                    last_metrics = metrics
                print(f'\r  Step {batch_idx}/{len(self.train_loader)} | '
                      f'Loss: {losses["total"].item():.5f} | '
                      f'Pos Acc: {last_metrics["positive_acc"]:.4f} | '
                      f'Bg Acc: {last_metrics["background_acc"]:.4f}'
                    ,end='')
        
        print()  # New line after epoch
        
        # Average losses and metrics
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        # Average metrics only over batches where they were computed
        if num_metric_batches > 0:
            for key in epoch_metrics:
                epoch_metrics[key] /= num_metric_batches
        
        if train_evaluator is not None:
            epoch_metrics.update(train_evaluator.compute())
        else:
            epoch_metrics.setdefault('map_50', 0.0)

        # Combine into single dict
        epoch_losses.update(epoch_metrics)
        
        return epoch_losses
    
    def validate(self, compute_map: bool = True, max_batches: Optional[int] = None) -> Dict[str, float]:
        """
        Run validation over the whole split.

        Detection metrics are pooled across every image into a single PR curve
        rather than averaged per image, so `max_batches` truncation produces a
        number for a different (smaller) dataset and should be used only for
        smoke tests, never for model selection or reporting.

        Args:
            compute_map: Whether to compute pooled detection metrics
            max_batches: Debug-only cap on batches processed (None = whole split)

        Returns:
            Dictionary of average validation losses and pooled metrics
        """
        if self.val_loader is None:
            return {}

        self.model.eval()
        val_losses = {}
        val_metrics = {'positive_acc': 0.0, 'background_acc': 0.0, 'positive_anchor_iou': 0.0}
        evaluator = DetectionEvaluator() if compute_map else None
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                if max_batches is not None and num_batches >= max_batches:
                    break
                images = batch['image'].to(self.device)
                anchor_targets = batch['anchor_targets'].to(self.device)
                anchor_ignore = batch.get('anchor_ignore')
                if anchor_ignore is not None:
                    anchor_ignore = anchor_ignore.to(self.device)
                gt_boxes_tensor = batch.get('gt_boxes')
                gt_box_counts = batch.get('gt_box_counts')
                if gt_boxes_tensor is not None and gt_box_counts is not None:
                    gt_boxes_tensor = gt_boxes_tensor.to(self.device)
                    gt_box_counts = gt_box_counts.to(self.device)
                else:
                    gt_boxes_tensor = None
                    gt_box_counts = None
                
                with autocast(enabled=self.use_amp):
                    class_logits, anchor_predictions = self._get_training_outputs(images)
                    losses = self.loss_fn(
                        class_logits,
                        anchor_predictions,
                        anchor_targets,
                        self.reference_anchors,
                        anchor_ignore=anchor_ignore
                    )
                
                metrics = self._compute_metrics(
                    class_logits,
                    anchor_targets,
                    anchor_predictions,
                    gt_boxes_tensor,
                    gt_box_counts,
                    threshold=self.metric_threshold,
                    evaluator=evaluator,
                    compute_iou_flag=True
                )

                for key, value in losses.items():
                    if isinstance(value, torch.Tensor):
                        if key not in val_losses:
                            val_losses[key] = 0.0
                        val_losses[key] += value.item()

                for key, value in metrics.items():
                    val_metrics[key] += value

                num_batches += 1

        if num_batches == 0:
            return {}

        # Per-batch diagnostics average over batches; detection metrics are
        # pooled once over the whole split.
        for key in val_losses:
            val_losses[key] /= num_batches
            self.writer.add_scalar(f'val/{key}', val_losses[key], self.global_step)
        for key in val_metrics:
            val_metrics[key] /= num_batches
            self.writer.add_scalar(f'val/{key}', val_metrics[key], self.global_step)

        if evaluator is not None:
            for key, value in evaluator.compute().items():
                val_metrics[key] = value
                self.writer.add_scalar(f'val/{key}', value, self.global_step)
        else:
            val_metrics.setdefault('map_50', 0.0)

        val_losses.update(val_metrics)

        return val_losses
    
    def save_checkpoint(self, filename: Optional[str] = None, is_best: bool = False):
        """Save training checkpoint."""
        if filename is None:
            filename = f'{self.model_name}_epoch{self.epoch}.pth'
        
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_map': self.best_val_map
        }
        if self.use_amp:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        print(f'  Saved checkpoint: {path}')
        
        if is_best:
            best_path = self.checkpoint_dir / f'{self.model_name}_best.pth'
            torch.save(checkpoint, best_path)
            print(f'  Saved best model: {best_path}')
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint.

        Refuses a checkpoint written from the other backbone variant. Those
        predate the BatchNorm switch and store `convs.0.weight` per block where
        this model expects `dw_conv.weight` and `bn1.*`; load_state_dict's own
        error is a wall of key names that does not say what to do about it.
        """
        checkpoint = torch.load(path, map_location=self.device)
        state = checkpoint.get('model_state_dict', checkpoint)
        folded_checkpoint = checkpoint_is_folded(state)
        model_is_folded = not any('.bn1.' in k for k in self.model.state_dict())
        if folded_checkpoint != model_is_folded:
            raise RuntimeError(
                f'{path} was written from the '
                f'{"BatchNorm-folded" if folded_checkpoint else "trainable-BatchNorm"} '
                f'backbone, but this model is the '
                f'{"folded" if model_is_folded else "trainable-BatchNorm"} one. '
                'Checkpoints do not carry across that change: start fresh, or '
                'build the model with the matching use_batchnorm setting.'
            )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        try:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except ValueError as exc:
            print("Warning: optimizer state incompatible with current parameter groups.")
            print(f"         {exc}")
            print("         Continuing with freshly initialized optimizer state.")
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_val_map = checkpoint.get('best_val_map', -1.0)

        if self.use_amp and "scaler_state_dict" in checkpoint:
            try:
                self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
            except Exception as exc:
                print(f"Warning: failed to load GradScaler state: {exc}")
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f'Loaded checkpoint from epoch {self.epoch}')
    
    def train(
        self,
        num_epochs: int,
        save_every: int = 10,
        validate_every: int = 1,
        val_compute_map: bool = True
    ):
        """
        Run full training loop.
        
        Args:
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
            validate_every: Run validation every N epochs
            val_compute_map: Whether to compute mAP during validation passes
        """
        print(f'\nStarting training for {num_epochs} epochs')
        print(f'Device: {self.device}')
        print(f'Training samples: {_dataset_length(self.train_loader)}')
        if self.val_loader:
            print(f'Validation samples: {_dataset_length(self.val_loader)}')
        print('-' * 60)
        
        start_epoch = self.epoch
        
        for epoch in range(start_epoch, start_epoch + num_epochs):
            self.epoch = epoch
            
            epoch_str = f'{epoch + 1:03d}/{start_epoch + num_epochs}'
            print(f'\nEpoch {epoch_str}')
            
            # Train
            train_started = time.time()
            train_results = self.train_epoch()
            train_seconds = time.time() - train_started
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
                self.writer.add_scalar('train/lr', current_lr, self.global_step)
            
            # Print epoch summary
            print(f'  Train | Loss: {train_results["total"]:.5f} | '
                f'Pos Acc: {train_results["positive_acc"]:.4f} | '
                f'Bg Acc: {train_results["background_acc"]:.4f} | '
                f'IoU: {train_results["positive_anchor_iou"]:.4f} | '
                f'mAP: {train_results["map_50"]:.4f} | '
                f'{train_seconds:.0f}s')
            
            # Validate over the whole split. Subsetting here previously took an
            # unshuffled prefix, which is ordered by annotation source, so the
            # selection signal came from a single source.
            if self.val_loader and (epoch + 1) % validate_every == 0:
                # Timed separately: per-epoch validation, not the training
                # step, is what dominates wall time on this model.
                val_started = time.time()
                val_results = self.validate(compute_map=val_compute_map)
                val_seconds = time.time() - val_started
                print(f'  Val   | Loss: {val_results["total"]:.5f} | '
                        f'Pos Acc: {val_results["positive_acc"]:.4f} | '
                        f'Bg Acc: {val_results["background_acc"]:.4f} | '
                        f'AnchIoU: {val_results["positive_anchor_iou"]:.4f} | '
                        f'DetIoU: {val_results.get("detection_iou", 0.0):.4f} | '
                        f'mAP50: {val_results["map_50"]:.4f} | '
                        f'mAP50-95: {val_results.get("map_50_95", 0.0):.4f} | '
                        f'{val_seconds:.0f}s'
                      )
                
                # Select on detection quality, not loss. Validation loss is
                # dominated by hard-negative mining, whose magnitude depends on
                # the positive count per batch, so its argmin has no particular
                # relationship to the best detector.
                self.best_val_loss = min(self.best_val_loss, val_results['total'])
                current_map = val_results.get('map_50', 0.0)
                if current_map > self.best_val_map:
                    self.best_val_map = current_map
                    self.save_checkpoint(is_best=True)
                    print(f'  New best mAP@0.5: {current_map:.4f}')
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint()
        
        final_val_metrics = None
        if self.val_loader:
            print('\nRunning final validation over the full split...')
            final_val_metrics = self.validate(compute_map=val_compute_map)
        
        # Save final checkpoint
        self.save_checkpoint(f'{self.model_name}_final.pth')
        self.writer.close()
        
        print('\n' + '=' * 60)
        print('Training complete!')
        if final_val_metrics:
            print('Final Val | '
                  f'Loss: {final_val_metrics["total"]:.5f} | '
                  f'Pos Acc: {final_val_metrics["positive_acc"]:.4f} | '
                  f'Bg Acc: {final_val_metrics["background_acc"]:.4f} | '
                  f'AnchIoU: {final_val_metrics["positive_anchor_iou"]:.4f} | '
                  f'DetIoU: {final_val_metrics.get("detection_iou", 0.0):.4f} | '
                  f'mAP50: {final_val_metrics["map_50"]:.4f} | '
                  f'mAP50-95: {final_val_metrics.get("map_50_95", 0.0):.4f}')
        print(f'Best validation mAP@0.5: {self.best_val_map:.4f}')
        print(f'Checkpoints saved to: {self.checkpoint_dir}')
        print('=' * 60)


def create_model(
    init_weights: str = 'mediapipe',
    weights_path: str = DEFAULT_WEIGHTS_PATH,
    use_batchnorm: bool = True,
    init_checkpoint: str | None = None
) -> BlazeEar:
    """
    Create BlazeEar model with specified weight initialization.

    Args:
        init_weights: Weight initialization strategy:
            - 'scratch': Random initialization
            - 'mediapipe': Load MediaPipe pretrained weights (default)
        weights_path: Path to MediaPipe weights file

    Returns:
        BlazeEar model
    """
    model = BlazeEar(use_batchnorm=use_batchnorm)

    if init_checkpoint:
        # Warm start from an existing BlazeEar rather than from BlazeFace. The
        # features are already ear-adapted, which BlazeFace's are not.
        state = load_checkpoint_state(init_checkpoint)
        folded_ckpt = checkpoint_is_folded(state)
        if folded_ckpt != (not use_batchnorm):
            raise SystemExit(
                f'{init_checkpoint} is the '
                f'{"folded" if folded_ckpt else "BatchNorm"} architecture but the '
                f'model is the {"folded" if not use_batchnorm else "BatchNorm"} one. '
                'Pass --no-batchnorm to match, or pick another checkpoint.'
            )
        model.load_state_dict(state)
        print(f'Initialized from {init_checkpoint}')
        return model

    if init_weights in ('mediapipe', 'mediapipe-backbone'):
        heads = init_weights == 'mediapipe'
        weights_path_obj = Path(weights_path)
        if weights_path_obj.exists():
            print(f'Loading MediaPipe weights from {weights_path_obj}'
                  + ('' if heads else ' (backbone only)'))
            missing, unexpected = load_mediapipe_weights(
                model, str(weights_path_obj), strict=False, load_detection_heads=heads)
            if not heads:
                model.init_detection_heads()
                print('  Detection heads re-initialized with a background prior')
            if missing:
                print(f'  Missing keys: {len(missing)}')
            if unexpected:
                print(f'  Unexpected keys: {len(unexpected)}')
            print('  Successfully loaded MediaPipe weights (backbone + detection heads)')
        else:
            print(f'Warning: MediaPipe weights not found at {weights_path_obj}')
            print('         Using random initialization instead')
            init_weights = 'scratch'

    if init_weights == 'scratch':
        print('Using random weight initialization')

    return model


def main():
    parser = argparse.ArgumentParser(
        description='Train BlazeEar ear detector',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--train-data', type=str, default=DEFAULT_TRAIN_CSV,
                        help='Path to training CSV file')
    parser.add_argument('--val-data', type=str, default=DEFAULT_VAL_CSV,
                        help='Path to validation CSV file')
    parser.add_argument('--data-root', type=str, default=DEFAULT_DATA_ROOT,
                        help='Root directory for image paths (required for CSV)')
    
    # Model arguments
    parser.add_argument('--init-weights', type=str, default='mediapipe-backbone',
                        choices=['scratch', 'mediapipe', 'mediapipe-backbone'],
                        help='Weight initialization: scratch (random) or mediapipe (pretrained)')
    parser.add_argument('--no-batchnorm', dest='use_batchnorm', action='store_false',
                        help='Build the MediaPipe-faithful folded backbone with no '
                             'normalization layers. Trainable BatchNorm renormalises by '
                             'batch statistics, which discards the activation scales the '
                             'pretrained weights were calibrated for: measured, the train-'
                             'mode output differs from the folded model by 97% relative.')
    parser.set_defaults(use_batchnorm=True)
    parser.add_argument('--init-checkpoint', type=str, default=None,
                        help='Warm start from an existing BlazeEar checkpoint instead of '
                             'BlazeFace weights.')
    parser.add_argument('--weights-path', type=str, default=DEFAULT_WEIGHTS_PATH,
                        help='Path to MediaPipe weights file (used with --init-weights=mediapipe)')
    parser.add_argument('--no-freeze-keypoint-heads', action='store_true',
                        help='Allow keypoint regressors to update (default: frozen)')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS,
                        help='Number of epochs (vincent1bt uses 500)')
    parser.add_argument('--lr', type=float, default=DEFAULT_LEARNING_RATE,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=DEFAULT_WEIGHT_DECAY,
                        help='Weight decay')
    parser.add_argument('--metric-threshold', type=float, default=0.5,
                        help='Score threshold used when measuring train-time metrics (balances precision/recall)')
    parser.add_argument('--eval-score-threshold', type=float, default=0.1,
                        help='Minimum score for a detection to be considered during evaluation')
    parser.add_argument('--eval-nms-threshold', type=float,
                        default=NMS_IOU_THRESHOLD,
                        help='IoU threshold for NMS when computing mAP/IoU')
    parser.add_argument('--max-eval-detections', type=int,
                        default=MAX_DETECTIONS,
                        help='Maximum number of candidate detections kept per image before scoring metrics')
    parser.add_argument('--train-map', action='store_true',
                        help='Compute mAP during training (slower)')
    parser.add_argument('--no-val-map', action='store_true',
                        help='Skip computing mAP during validation to save time')
    
    # Loss arguments
    parser.add_argument('--use-focal-loss', dest='use_focal_loss', action='store_true', 
                        help='Use focal loss instead of BCE')
    parser.add_argument('--no-focal-loss', dest='use_focal_loss', action='store_false',
                        help='Disable focal loss and fall back to BCE')
    parser.add_argument('--focal-alpha', type=float, default=0.35,
                        help='Focal loss alpha parameter')
    parser.add_argument('--focal-gamma', type=float, default=1.5,
                        help='Focal loss gamma parameter')
    parser.add_argument('--detection-weight', type=float, default=180.0,
                        help='Weight for detection/regression loss (higher favors box quality)')
    parser.add_argument('--classification-weight', type=float, default=55.0,
                        help='Weight for background classification loss')
    parser.add_argument('--positive-classification-weight', type=float, default=70.0,
                        help='Weight for positive classification loss (encourages confident positives)')
    parser.add_argument('--hard-negative-ratio', type=float, default=1.5,
                        help='Ratio of negatives to positives in hard mining')
    parser.set_defaults(use_focal_loss=True)
    
    # System arguments
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--num-workers', type=int, default=DEFAULT_NUM_WORKERS,
                        help='Number of data loading workers')
    parser.add_argument('--checkpoint-dir', type=str, default=DEFAULT_CHECKPOINT_DIR,
                        help='Checkpoint directory')
    parser.add_argument('--log-dir', type=str, default=DEFAULT_LOG_DIR,
                        help='TensorBoard log directory')
    parser.add_argument('--resume', type=str, 
                        default=DEFAULT_BEST_CHECKPOINT,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--save-every', type=int, default=DEFAULT_SAVE_EVERY,
                        help='Save checkpoint every N epochs')
    
    parser.add_argument('--freeze-thaw', action='store_true', 
                        help='Enable staged freezing/unfreezing of backbone')
    parser.add_argument('--freeze-epochs', type=int, default=2,
                        help='Epochs to train with backbone frozen (phase 1)')
    parser.add_argument('--unfreeze-mid-epochs', type=int, default=3,
                        help='Epochs to train with backbone2 unfrozen (phase 2)')
    parser.add_argument('--freeze-lr-head', type=float, default=1e-3,
                        help='Learning rate when only detection heads are trainable')
    parser.add_argument('--freeze-lr-mid', type=float, default=3e-4,
                        help='Learning rate when backbone2 is unfrozen')

    # Performance flags
    parser.add_argument('--amp', dest='use_amp', action='store_true',
                        help='Enable mixed precision training on CUDA')
    parser.add_argument('--no-amp', dest='use_amp', action='store_false',
                        help='Disable mixed precision')
    parser.set_defaults(use_amp=True)
    parser.add_argument('--compile', dest='use_compile', action='store_true',
                        help='Use torch.compile for training (torch>=2.0)')
    parser.add_argument('--no-compile', dest='use_compile', action='store_false',
                        help='Disable torch.compile')
    parser.set_defaults(use_compile=False)
    parser.add_argument('--prefetch-factor', type=int, default=2,
                        help='DataLoader prefetch_factor (workers only)')

    args = parser.parse_args()
    val_compute_map = not args.no_val_map
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print('CUDA not available, using CPU')
        args.device = 'cpu'

    if str(args.device).startswith("cuda"):
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    
    # Input size (fixed at 128x128 for front model)
    target_size = (DEFAULT_INPUT_SIZE, DEFAULT_INPUT_SIZE)
    scale = DEFAULT_INPUT_SIZE
    
    print('=' * 60)
    print('BlazeEar Ear Detector Training')
    print('=' * 60)
    print(f'Input size: {target_size}')
    print(f'Device: {args.device}')
    print(f'Batch size: {args.batch_size}')
    print(f'Learning rate: {args.lr}')
    print(f'Epochs: {args.epochs}')
    print(f'Loss: {"Focal" if args.use_focal_loss else "BCE"} + Huber')
    print(
        f'Loss weights: detection={args.detection_weight}, '
        f'cls_background={args.classification_weight}, '
        f'cls_positive={args.positive_classification_weight}'
    )
    print(f'Hard negative ratio: {args.hard_negative_ratio}:1 (neg:pos)')
    print(f'Validation mAP logging: {"on" if val_compute_map else "off"}')
    print('=' * 60)
    
    # Create model with requested initialization
    model = create_model(
        init_weights=args.init_weights,
        weights_path=args.weights_path,
        use_batchnorm=args.use_batchnorm,
        init_checkpoint=args.init_checkpoint
    )
    compile_fn = getattr(torch, "compile", None)
    if args.use_compile and callable(compile_fn):
        try:
            model = cast(Callable[[nn.Module], nn.Module], compile_fn)(model)
            print("torch.compile enabled.")
        except Exception as exc:
            print(f"Warning: torch.compile failed, continuing uncompiled: {exc}")
    elif args.use_compile:
        print("Warning: torch.compile not available; skipping.")
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    freeze_kp = not args.no_freeze_keypoint_heads
    freeze_kp_fn = getattr(model, "freeze_keypoint_regressors", None)
    if freeze_kp and callable(freeze_kp_fn):
        cast(Callable[[], None], freeze_kp_fn)()
        print("Keypoint regressors frozen (no grad / weight decay).")

    def apply_freeze_state(backbone1_grad: bool, backbone2_grad: bool, heads_grad: bool) -> None:
        """Enable/disable gradients for different model regions."""
        for name, param in model.named_parameters():
            if name.startswith('backbone1'):
                param.requires_grad_(backbone1_grad)
            elif name.startswith('backbone2'):
                param.requires_grad_(backbone2_grad)
            else:
                param.requires_grad_(heads_grad)
        if freeze_kp and callable(freeze_kp_fn):
            cast(Callable[[], None], freeze_kp_fn)()

    def build_optimizer_for_lr(lr_value: float) -> optim.Optimizer:
        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            raise ValueError("No trainable parameters available for optimizer.")
        return optim.AdamW(params, lr=lr_value, weight_decay=args.weight_decay)

    # Create data loaders (CSV-only pipeline)
    if not args.data_root:
        raise ValueError("--data-root is required for CSV training data")

    persistent_workers = args.num_workers > 0

    train_loader = create_dataloader(
        csv_path=args.train_data,
        root_dir=args.data_root,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        target_size=target_size,
        augment=True,
        persistent_workers=persistent_workers,
        prefetch_factor=args.prefetch_factor
    )
    print(f'Training samples: {_dataset_length(train_loader)}')

    val_loader = None
    if args.val_data:
        val_loader = create_dataloader(
            csv_path=args.val_data,
            root_dir=args.data_root,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            target_size=target_size,
            augment=False,
            persistent_workers=persistent_workers,
            prefetch_factor=args.prefetch_factor
        )
        print(f'Validation samples: {_dataset_length(val_loader)}')
    
    # Create loss function
    loss_fn = BlazeEarDetectionLoss(
        hard_negative_ratio=args.hard_negative_ratio,
        detection_weight=args.detection_weight,
        classification_weight=args.classification_weight,
        positive_classification_weight=args.positive_classification_weight,
        scale=scale,
        use_focal_loss=args.use_focal_loss,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma
    )
    
    # Build freeze-thaw phases
    class TrainPhase(TypedDict):
        name: str
        epochs: int
        lr: float
        grads: tuple[bool, bool, bool]

    remaining_epochs = args.epochs
    phases: List[TrainPhase] = []

    def add_phase(name: str, requested_epochs: int, lr_value: float, grads: tuple[bool, bool, bool]) -> None:
        nonlocal remaining_epochs
        if requested_epochs <= 0 or remaining_epochs <= 0:
            return
        epochs = min(requested_epochs, remaining_epochs)
        if epochs <= 0:
            return
        phases.append({
            "name": name,
            "epochs": epochs,
            "lr": lr_value,
            "grads": grads
        })
        remaining_epochs -= epochs

    if args.freeze_thaw:
        add_phase("Heads", args.freeze_epochs, args.freeze_lr_head, (False, False, True))
        add_phase("Backbone2", args.unfreeze_mid_epochs, args.freeze_lr_mid, (False, True, True))
        add_phase("Full", remaining_epochs, args.lr, (True, True, True))
    else:
        add_phase("Full", remaining_epochs, args.lr, (True, True, True))

    if not phases:
        raise ValueError("No training epochs configured. Increase --epochs.")

    if args.freeze_thaw:
        print("Freeze-thaw schedule:")
        for idx, phase in enumerate(phases, 1):
            print(f"  Phase {idx}: {phase['name']} | epochs={phase['epochs']} | lr={phase['lr']}")
        print('=' * 60)

    # Apply initial phase state and optimizer
    first_phase = phases[0]
    apply_freeze_state(*first_phase["grads"])
    optimizer = build_optimizer_for_lr(first_phase["lr"])
    scheduler: Optional[LRScheduler]
    if args.freeze_thaw:
        scheduler = None
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, args.epochs),
            eta_min=args.lr * 0.01
        )
    
    # Create trainer
    trainer = BlazeEarTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        model_name='BlazeEar',
        scale=scale,
        compute_train_map=args.train_map,
        eval_score_threshold=args.eval_score_threshold,
        nms_iou_threshold=args.eval_nms_threshold,
        max_eval_detections=args.max_eval_detections,
        metric_threshold=args.metric_threshold,
        use_amp=args.use_amp
    )
    
    # Resume from checkpoint if provided
    if args.resume:
        checkpoint_path = Path(args.resume)
        if checkpoint_path.exists():
            print(f'\nFound checkpoint: {checkpoint_path}')
            try:
                trainer.load_checkpoint(str(checkpoint_path))
                print('Resuming training from checkpoint...')
            except RuntimeError as exc:
                # An incompatible checkpoint left over from a previous
                # architecture must not stop a fresh run.
                print(f'Not resuming: {exc}')
                print('Starting from the configured initialization instead.')
        else:
            print(f'Warning: specified checkpoint {checkpoint_path} not found. Starting fresh.')

    # Train
    for phase_idx, phase in enumerate(phases):
        if phase['epochs'] <= 0:
            continue
        if phase_idx > 0:
            apply_freeze_state(*phase["grads"])
            trainer.optimizer = build_optimizer_for_lr(phase["lr"])
            trainer.scheduler = None
        else:
            trainer.scheduler = scheduler

        print(f'\n=== Phase {phase_idx + 1}/{len(phases)}: {phase["name"]} '
              f'(epochs={phase["epochs"]}, lr={phase["lr"]:.2e}) ===')

        trainer.train(
            num_epochs=phase['epochs'],
            save_every=args.save_every,
            val_compute_map=val_compute_map
        )


if __name__ == '__main__':
    main()
