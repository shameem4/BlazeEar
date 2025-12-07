"""
Training script for BlazeFace Face detector.

Complete training pipeline following vincent1bt/blazeface-tensorflow methodology:
- Anchor-based target encoding (from dataloader)
- Hard negative mining loss
- BCE or Focal loss for classification
- Huber loss for box regression

Usage (NPY format):
    python train_blazeface.py --train-data data/preprocessed/train_detector.npy
    python train_blazeface.py --train-data data/preprocessed/train_detector.npy --val-data data/preprocessed/val_detector.npy
    python train_blazeface.py --train-data data/preprocessed/train_detector.npy --epochs 500 --lr 1e-4

Usage (CSV format):
    # Default: MediaPipe weight initialization with auto-resume
    python train_blazeface.py --csv-format --train-data data/splits/train.csv --val-data data/splits/val.csv --data-root data/raw/blazeface

    # Train from scratch (random initialization)
    python train_blazeface.py --csv-format --train-data data/splits/train.csv --val-data data/splits/val.csv --data-root data/raw/blazeface --init-weights scratch

    # Start fresh (disable auto-resume, but use MediaPipe weights)
    python train_blazeface.py --csv-format --train-data data/splits/train.csv --val-data data/splits/val.csv --data-root data/raw/blazeface --no-auto-resume
"""

import argparse
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import time
from pathlib import Path
from collections.abc import Sized
from typing import Dict, List, Optional, cast

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LRScheduler

from blazeface import BlazeFace
from blazebase import generate_reference_anchors, load_mediapipe_weights
from dataloader import create_dataloader
from loss_functions import BlazeFaceDetectionLoss, compute_mean_iou, compute_map
from utils.config import (
    DEFAULT_BEST_CHECKPOINT,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_DATA_ROOT,
    DEFAULT_EPOCHS,
    DEFAULT_INPUT_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LOG_DIR,
    DEFAULT_NMS_IOU_THRESHOLD,
    DEFAULT_NUM_WORKERS,
    DEFAULT_SAVE_EVERY,
    DEFAULT_TRAIN_CSV,
    DEFAULT_VAL_CSV,
    DEFAULT_WEIGHTS_PATH,
    DEFAULT_WEIGHT_DECAY
)


def _dataset_length(loader: DataLoader) -> int:
    """Return the size of a DataLoader's underlying dataset."""
    dataset = loader.dataset
    if not isinstance(dataset, Sized):
        raise TypeError("Dataset must implement __len__() for progress reporting")
    return len(cast(Sized, dataset))


class BlazeFaceTrainer:
    """
    Trainer for BlazeFace Face detector.
    
    Handles training loop, validation, checkpointing, and logging.
    Following vincent1bt methodology for loss computation and metrics.
    """
    
    def __init__(
        self,
        model: BlazeFace,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        loss_fn: Optional[BlazeFaceDetectionLoss] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[LRScheduler] = None,
        device: str = 'cuda',
        checkpoint_dir: str = 'checkpoints',
        log_dir: str = 'logs',
        model_name: str = 'BlazeFace',
        scale: int = 128,
        compute_train_map: bool = False
    ):
        """
        Args:
            model: BlazeFace model
            train_loader: Training data loader
            val_loader: Optional validation data loader
            loss_fn: Loss function (BlazeFaceDetectionLoss if None)
            optimizer: Optimizer (AdamW if None)
            scheduler: Learning rate scheduler
            device: Device to train on
            checkpoint_dir: Directory for saving checkpoints
            log_dir: Directory for TensorBoard logs
            model_name: Name for saving checkpoints
            scale: Image scale for decoding (128 for front, 256 for back)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.model_name = model_name
        self.scale = scale
        self.compute_train_map = compute_train_map
        
        # Generate reference anchors for loss computation
        # generate_reference_anchors returns (reference_anchors, small, big) tuple
        reference_anchors, _, _ = generate_reference_anchors()
        self.reference_anchors = reference_anchors.float().to(device)
        
        # Setup loss function
        self.loss_fn = loss_fn if loss_fn else BlazeFaceDetectionLoss(scale=scale)
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
        
        # Metrics tracking
        self.metrics = {
            'positive_correct': 0,
            'positive_total': 0,
            'background_correct': 0,
            'background_total': 0
        }
        self.eval_score_threshold = 0.1
        self.nms_iou_threshold = DEFAULT_NMS_IOU_THRESHOLD
        self.max_eval_detections = 50
        self.max_map_candidates = 200
        self.metric_threshold = 0.45
    
    def _get_training_outputs(self, images: torch.Tensor) -> tuple:
        """
        Get raw training outputs from BlazeFace model.

        BlazeFace model returns (raw_boxes, raw_scores) from get_training_outputs()
        for training, which bypasses post-processing.

        Args:
            images: [B, 3, H, W] input images

        Returns:
            class_predictions: [B, 896, 1] sigmoid scores (probabilities)
            anchor_predictions: [B, 896, 4] box predictions
        """
        # Use training output method that bypasses NMS
        if hasattr(self.model, 'get_training_outputs'):
            raw_boxes, raw_scores = self.model.get_training_outputs(images)
            # raw_boxes: [B, 896, 16] or [B, 896, 4]
            # raw_scores: [B, 896, 1] - raw logits

            # Extract first 4 coords if model outputs 16 (for keypoints)
            if raw_boxes.shape[-1] > 4:
                raw_boxes = raw_boxes[..., :4]

            # Apply sigmoid to convert logits to probabilities
            # Loss function expects probabilities, not logits
            class_predictions = torch.sigmoid(raw_scores)

            return class_predictions, raw_boxes
        else:
            # Fallback: call model directly
            output = self.model(images)
            if isinstance(output, tuple):
                scores = torch.sigmoid(output[1])  # Apply sigmoid
                return scores, output[0]  # scores, boxes
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
        """
        Basic Non-Maximum Suppression to mimic MediaPipe evaluation pipeline.
        """
        if boxes.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=boxes.device)

        order = torch.argsort(scores, descending=True)
        keep: List[torch.Tensor] = []

        while order.numel() > 0 and len(keep) < self.max_eval_detections:
            current = order[0]
            keep.append(current)

            if order.numel() == 1:
                break

            remaining = order[1:]
            ious = self._pairwise_iou(
                boxes[current].unsqueeze(0),
                boxes[remaining]
            ).squeeze(0)
            mask = ious <= iou_threshold
            order = remaining[mask]

        return torch.stack(keep) if keep else torch.empty(0, dtype=torch.long, device=boxes.device)

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
        class_predictions: torch.Tensor,
        anchor_targets: torch.Tensor,
        anchor_predictions: torch.Tensor,
        gt_boxes_tensor: Optional[torch.Tensor] = None,
        gt_box_counts: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
        compute_map_flag: bool = True,
        compute_iou_flag: bool = True
    ) -> Dict[str, float]:
        """
        Compute training metrics following vincent1bt.

        Metrics:
        - Positive accuracy: % of positive anchors correctly classified
        - Background accuracy: % of background anchors correctly classified
        - Mean IoU: Average IoU for positive predictions
        - mAP@0.5: Mean Average Precision at IoU threshold 0.5

        Args:
            class_predictions: [B, 896, 1] predicted scores
            anchor_targets: [B, 896, 5] targets [class, ymin, xmin, ymax, xmax] (MediaPipe convention)
            anchor_predictions: [B, 896, 4] predicted boxes
            threshold: Classification threshold

        Returns:
            Dictionary of metrics
        """
        true_classes = anchor_targets[:, :, 0]  # [B, 896]
        true_coords = anchor_targets[:, :, 1:]  # [B, 896, 4]
        pred_scores = class_predictions.squeeze(-1)  # [B, 896]

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

        # Mean IoU and mAP for positive predictions
        mean_iou = 0.0
        map_50 = 0.0

        decoded_boxes = None
        if positive_mask.sum() > 0 and (compute_iou_flag or compute_map_flag):
            # Decode predictions
            decoded_boxes = self.loss_fn.decode_boxes(
                anchor_predictions, self.reference_anchors
            )

            # Mean IoU for positive anchors
            if compute_iou_flag:
                pred_coords = decoded_boxes[positive_mask]
                gt_coords = true_coords[positive_mask]
                mean_iou = compute_mean_iou(pred_coords, gt_coords, scale=self.scale).item()

            if compute_map_flag:
                batch_size = class_predictions.shape[0]
                map_scores = []

                fallback_gt = None
                if gt_boxes_tensor is None or gt_box_counts is None:
                    fallback_gt = self._build_gt_from_targets(anchor_targets)

                for b in range(batch_size):
                    if gt_boxes_tensor is not None and gt_box_counts is not None:
                        count = int(gt_box_counts[b].item())
                        if count == 0:
                            continue
                        gt_boxes_batch = gt_boxes_tensor[b, :count]
                    else:
                        if fallback_gt is None or b >= len(fallback_gt):
                            continue
                        gt_boxes_batch = fallback_gt[b]
                        if gt_boxes_batch is None or gt_boxes_batch.numel() == 0:
                            continue

                    batch_scores = pred_scores[b]
                    batch_decoded = decoded_boxes[b]
                    candidate_k = min(self.max_map_candidates, batch_scores.numel())
                    candidate_scores, candidate_indices = torch.topk(batch_scores, k=candidate_k)
                    candidate_boxes = batch_decoded[candidate_indices]

                    score_mask = candidate_scores > self.eval_score_threshold
                    filtered_scores = candidate_scores[score_mask]
                    filtered_boxes = candidate_boxes[score_mask]

                    if filtered_scores.numel() == 0:
                        map_scores.append(0.0)
                        continue

                    keep_indices = self._nms(
                        filtered_boxes,
                        filtered_scores,
                        self.nms_iou_threshold
                    )

                    if keep_indices.numel() == 0:
                        map_scores.append(0.0)
                        continue

                    selected_boxes = filtered_boxes[keep_indices]
                    selected_scores = filtered_scores[keep_indices]

                    ap = compute_map(
                        selected_boxes,
                        selected_scores,
                        gt_boxes_batch,
                        iou_threshold=0.5
                    )
                    map_scores.append(ap.item())

                if map_scores:
                    map_50 = sum(map_scores) / len(map_scores)

        return {
            'positive_acc': positive_acc,
            'background_acc': background_acc,
            'mean_iou': mean_iou,
            'map_50': map_50
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary of average losses and metrics for the epoch
        """
        self.model.train()
        epoch_losses = {}
        epoch_metrics = {'positive_acc': 0.0, 'background_acc': 0.0, 'mean_iou': 0.0, 'map_50': 0.0}
        num_batches = 0
        num_metric_batches = 0
        last_metrics = {'positive_acc': 0.0, 'background_acc': 0.0, 'mean_iou': 0.0, 'map_50': 0.0}
        
        batch_time = time.time()
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move data to device
            images = batch['image'].to(self.device)
            anchor_targets = batch['anchor_targets'].to(self.device)
            gt_boxes_tensor = batch.get('gt_boxes')
            gt_box_counts = batch.get('gt_box_counts')
            if gt_boxes_tensor is not None and gt_box_counts is not None:
                gt_boxes_tensor = gt_boxes_tensor.to(self.device)
                gt_box_counts = gt_box_counts.to(self.device)
            else:
                gt_boxes_tensor = None
                gt_box_counts = None
            
            # Forward pass
            self.optimizer.zero_grad()
            
            class_predictions, anchor_predictions = self._get_training_outputs(images)
            
            # Compute loss
            losses = self.loss_fn(
                class_predictions,
                anchor_predictions,
                anchor_targets,
                self.reference_anchors
            )
            
            # Backward pass
            losses['total'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Compute metrics only every 10 batches to speed up training
            # IoU is skipped during training (computed only during validation)
            compute_metrics_this_batch = (batch_idx % 10 == 0 or batch_idx == len(self.train_loader) - 1)
            if compute_metrics_this_batch:
                with torch.no_grad():
                    metrics = self._compute_metrics(
                        class_predictions,
                        anchor_targets,
                        anchor_predictions,
                        gt_boxes_tensor,
                        gt_box_counts,
                        threshold=self.metric_threshold,
                        compute_map_flag=self.compute_train_map,
                        compute_iou_flag=False  # Skip IoU during training for speed
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
                batch_time = time.time()
        
        print()  # New line after epoch
        
        # Average losses and metrics
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        # Average metrics only over batches where they were computed
        if num_metric_batches > 0:
            for key in epoch_metrics:
                epoch_metrics[key] /= num_metric_batches
        
        # Combine into single dict
        epoch_losses.update(epoch_metrics)
        
        return epoch_losses
    
    def validate(self, compute_map: bool = False, max_batches: Optional[int] = None) -> Dict[str, float]:
        """
        Run validation.
        
        Args:
            compute_map: Whether to compute mAP metric
            max_batches: Maximum number of batches to process (None = all)
        
        Returns:
            Dictionary of average validation losses and metrics
        """
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        val_losses = {}
        val_metrics = {'positive_acc': 0.0, 'background_acc': 0.0, 'mean_iou': 0.0, 'map_50': 0.0}
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                if max_batches is not None and num_batches >= max_batches:
                    break
                images = batch['image'].to(self.device)
                anchor_targets = batch['anchor_targets'].to(self.device)
                gt_boxes_tensor = batch.get('gt_boxes')
                gt_box_counts = batch.get('gt_box_counts')
                if gt_boxes_tensor is not None and gt_box_counts is not None:
                    gt_boxes_tensor = gt_boxes_tensor.to(self.device)
                    gt_box_counts = gt_box_counts.to(self.device)
                else:
                    gt_boxes_tensor = None
                    gt_box_counts = None
                
                class_predictions, anchor_predictions = self._get_training_outputs(images)
                
                losses = self.loss_fn(
                    class_predictions,
                    anchor_predictions,
                    anchor_targets,
                    self.reference_anchors
                )
                
                metrics = self._compute_metrics(
                    class_predictions,
                    anchor_targets,
                    anchor_predictions,
                    gt_boxes_tensor,
                    gt_box_counts,
                    threshold=self.metric_threshold,
                    compute_map_flag=compute_map,
                    compute_iou_flag=False
                )
                
                for key, value in losses.items():
                    if isinstance(value, torch.Tensor):
                        if key not in val_losses:
                            val_losses[key] = 0.0
                        val_losses[key] += value.item()
                
                for key, value in metrics.items():
                    val_metrics[key] += value
                
                num_batches += 1
        
        # Average
        for key in val_losses:
            val_losses[key] /= num_batches
            self.writer.add_scalar(f'val/{key}', val_losses[key], self.global_step)
        for key in val_metrics:
            val_metrics[key] /= num_batches
            self.writer.add_scalar(f'val/{key}', val_metrics[key], self.global_step)
        
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
            'best_val_loss': self.best_val_loss
        }
        
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
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
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
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f'Loaded checkpoint from epoch {self.epoch}')
    
    def train(
        self,
        num_epochs: int,
        save_every: int = 10,
        validate_every: int = 1
    ):
        """
        Run full training loop.
        
        Args:
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
            validate_every: Run validation every N epochs
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
            epoch_start = time.time()
            
            epoch_str = f'{epoch + 1:03d}/{start_epoch + num_epochs}'
            print(f'\nEpoch {epoch_str}')
            
            # Train
            train_results = self.train_epoch()
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
                self.writer.add_scalar('train/lr', current_lr, self.global_step)
            
            # Print epoch summary
            print(f'  Train | Loss: {train_results["total"]:.5f} | '
                  f'Pos Acc: {train_results["positive_acc"]:.4f} | '
                  f'Bg Acc: {train_results["background_acc"]:.4f}')
            
            # Validate (use subset for speed - ~200 samples)
            if self.val_loader and (epoch + 1) % validate_every == 0:
                batch_size = self.val_loader.batch_size or 32
                val_max_batches = max(1, 200 // batch_size)
                val_results = self.validate(compute_map=False, max_batches=val_max_batches)
                print(f'  Val   | Loss: {val_results["total"]:.5f} | '
                      f'Pos Acc: {val_results["positive_acc"]:.4f} | '
                      f'Bg Acc: {val_results["background_acc"]:.4f}'
                      )
                
                # Check for best model
                if val_results['total'] < self.best_val_loss:
                    self.best_val_loss = val_results['total']
                    self.save_checkpoint(is_best=True)
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint()
        
        final_val_metrics = None
        if self.val_loader:
            print('\nRunning final validation for summary (500 samples)...')
            # Use ~500 samples for final validation (500 / batch_size batches)
            batch_size = self.val_loader.batch_size or 32
            max_batches = max(1, 500 // batch_size)
            final_val_metrics = self.validate(compute_map=True, max_batches=max_batches)
        
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
                  f'IoU: {final_val_metrics["mean_iou"]:.4f} | '
                  f'mAP: {final_val_metrics["map_50"]:.4f}')
        print(f'Best validation loss: {self.best_val_loss:.5f}')
        print(f'Checkpoints saved to: {self.checkpoint_dir}')
        print('=' * 60)


def create_model(
    init_weights: str = 'mediapipe',
    weights_path: str = DEFAULT_WEIGHTS_PATH
) -> BlazeFace:
    """
    Create BlazeFace model with specified weight initialization.

    Args:
        init_weights: Weight initialization strategy:
            - 'scratch': Random initialization
            - 'mediapipe': Load MediaPipe pretrained weights (default)
        weights_path: Path to MediaPipe weights file

    Returns:
        BlazeFace model
    """
    model = BlazeFace()

    if init_weights == 'mediapipe':
        weights_path_obj = Path(weights_path)
        if weights_path_obj.exists():
            print(f'Loading MediaPipe weights from {weights_path_obj}')
            missing, unexpected = load_mediapipe_weights(model, str(weights_path_obj), strict=False)
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
        description='Train BlazeFace face detector',
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
    parser.add_argument('--init-weights', type=str, default='mediapipe',
                        choices=['scratch', 'mediapipe'],
                        help='Weight initialization: scratch (random) or mediapipe (pretrained)')
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
    parser.add_argument('--train-map', action='store_true',
                        help='Compute mAP during training (slower)')
    
    # Loss arguments
    parser.add_argument('--use-focal-loss', dest='use_focal_loss', action='store_true', 
                        help='Use focal loss instead of BCE')
    parser.add_argument('--no-focal-loss', dest='use_focal_loss', action='store_false',
                        help='Disable focal loss and fall back to BCE')
    parser.add_argument('--focal-alpha', type=float, default=0.25,
                        help='Focal loss alpha parameter')
    parser.add_argument('--focal-gamma', type=float, default=2.0,
                        help='Focal loss gamma parameter')
    parser.add_argument('--detection-weight', type=float, default=200.0,
                        help='Weight for detection/regression loss')
    parser.add_argument('--classification-weight', type=float, default=60.0,
                        help='Weight for background classification loss')
    parser.add_argument('--positive-classification-weight', type=float, default=80.0,
                        help='Weight for positive classification loss (encourages higher foreground scores)')
    parser.add_argument('--hard-negative-ratio', type=float, default=2,
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
    
    args = parser.parse_args()
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print('CUDA not available, using CPU')
        args.device = 'cpu'
    
    # Input size (fixed at 128x128 for front model)
    target_size = (DEFAULT_INPUT_SIZE, DEFAULT_INPUT_SIZE)
    scale = DEFAULT_INPUT_SIZE
    
    print('=' * 60)
    print('BlazeFace Face Detector Training')
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
    print('=' * 60)
    
    # Create model with requested initialization
    model = create_model(
        init_weights=args.init_weights,
        weights_path=args.weights_path
    )
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    freeze_kp = not args.no_freeze_keypoint_heads
    if freeze_kp and hasattr(model, "freeze_keypoint_regressors"):
        model.freeze_keypoint_regressors()
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
        if freeze_kp and hasattr(model, "freeze_keypoint_regressors"):
            model.freeze_keypoint_regressors()

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
        persistent_workers=persistent_workers
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
            persistent_workers=persistent_workers
        )
        print(f'Validation samples: {_dataset_length(val_loader)}')
    
    # Create loss function
    loss_fn = BlazeFaceDetectionLoss(
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
    remaining_epochs = args.epochs
    phases: List[dict] = []

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
    trainer = BlazeFaceTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        model_name='BlazeFace',
        scale=scale,
        compute_train_map=args.train_map
    )
    
    # Resume from checkpoint if provided
    if args.resume:
        checkpoint_path = Path(args.resume)
        if checkpoint_path.exists():
            print(f'\nFound checkpoint: {checkpoint_path}')
            print('Resuming training from checkpoint...')
            trainer.load_checkpoint(str(checkpoint_path))
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
            save_every=args.save_every
        )


if __name__ == '__main__':
    main()
