"""
Pure PyTorch BlazeEar Inference Pipeline.

This module provides a self-contained inference pipeline that:
- Takes raw images of arbitrary resolution
- Handles all preprocessing (resize, pad, normalize)
- Runs the BlazeEar model
- Applies post-processing (decode boxes, NMS)
- Outputs bounding boxes in original image coordinates

Designed for easy export to ONNX/TorchScript for web deployment (JavaScript).

Usage (Python):
    from blazeear_inference import BlazeEarInference
    
    pipeline = BlazeEarInference(
        weights_path="runs/checkpoints/BlazeEar_best.pth",
        confidence_threshold=0.75,
        iou_threshold=0.3,
        device="cuda"
    )
    
    detections = pipeline(image)  # numpy HWC RGB
    # Returns: (N, 5) with [ymin, xmin, ymax, xmax, confidence]

ONNX Export Options for JavaScript:

1. End-to-End Model (--export-onnx-e2e):
   Recommended for web deployment. Takes preprocessing params as inputs.
   
   Inputs:
     - image: (1, 3, 128, 128) preprocessed float32 in [0, 255]
     - scale: scalar float32 - scale factor from resize
     - pad_y: scalar float32 - y padding in original coords  
     - pad_x: scalar float32 - x padding in original coords
   
   Output:
     - detections: (N, 5) [ymin, xmin, ymax, xmax, conf] in ORIGINAL image coords
   
   JavaScript preprocessing example:
     const max_dim = Math.max(orig_h, orig_w);
     const scale = max_dim / 256.0;
     const new_h = Math.round(orig_h / scale);
     const new_w = Math.round(orig_w / scale);
     const pad_h = 256 - new_h;
     const pad_w = 256 - new_w;
     const pad_y = Math.floor(pad_h / 2) * scale;
     const pad_x = Math.floor(pad_w / 2) * scale;
     // Resize to (new_h, new_w), pad to 256x256 centered, resize to 128x128

2. Simple Postprocessed Model (--export-onnx):
   For cases where you want normalized [0,1] output coordinates.
   
   Input: (1, 3, 128, 128) preprocessed float32 in [0, 255]
   Output: (N, 5) [ymin, xmin, ymax, xmax, conf] in normalized [0, 1] coords
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union, Optional, List, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops


class BlazeEarInference(nn.Module):
    # Type hints for registered buffers
    anchors: torch.Tensor
    """
    Pure PyTorch inference pipeline for BlazeEar ear detection.
    
    This class wraps the BlazeEar model with all necessary preprocessing and
    post-processing in a single, exportable module.
    
    Args:
        weights_path: Path to trained model weights (.pth file)
        confidence_threshold: Minimum score for a detection (default: 0.75)
        iou_threshold: IoU threshold for NMS (default: 0.3)
        input_size: Model input size (default: 128)
        max_detections: Maximum number of detections to return (default: 100)
        device: Device to run on ("cuda", "cpu", or torch.device)
        score_clipping_thresh: Clipping threshold for raw scores (default: 100.0)
    """
    
    # Anchor configuration
    NUM_ANCHORS = 896
    SMALL_GRID = 16
    BIG_GRID = 8
    SMALL_ANCHORS_PER_CELL = 2
    BIG_ANCHORS_PER_CELL = 6
    
    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        confidence_threshold: float = 0.75,
        iou_threshold: float = 0.3,
        input_size: int = 128,
        max_detections: int = 100,
        device: Union[str, torch.device] = "cpu",
        score_clipping_thresh: float = 100.0,
    ):
        super().__init__()
        
        # Store configuration
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.score_clipping_thresh = score_clipping_thresh
        
        # Setup device
        if isinstance(device, str):
            device = torch.device(device)
        self._device_obj = device
        
        # Generate anchors (fixed size, as used in training)
        anchors = self._generate_anchors()
        self.register_buffer("anchors", anchors)
        
        # Load model backbone
        self.model = self._create_model()
        
        if weights_path is not None:
            self._load_weights(weights_path)
        
        self.to(device)
        self.eval()
    
    def _generate_anchors(self) -> torch.Tensor:
        """
        Generate 896 anchors for BlazeEar detector.
        
        Grid layout:
        - 16x16 grid with 2 anchors per cell = 512 anchors
        - 8x8 grid with 6 anchors per cell = 384 anchors
        Total: 896 anchors
        
        Returns:
            Tensor of shape (896, 4) with [x_center, y_center, width, height]
        """
        # Small anchors: 16x16 grid
        small_boxes = torch.linspace(0.03125, 0.96875, self.SMALL_GRID)
        small_x = small_boxes.repeat_interleave(self.SMALL_ANCHORS_PER_CELL).repeat(self.SMALL_GRID)
        small_y = small_boxes.repeat_interleave(self.SMALL_GRID * self.SMALL_ANCHORS_PER_CELL)
        small_w = torch.ones_like(small_x)
        small_h = torch.ones_like(small_x)
        small_anchors = torch.stack([small_x, small_y, small_w, small_h], dim=1)
        
        # Big anchors: 8x8 grid
        big_boxes = torch.linspace(0.0625, 0.9375, self.BIG_GRID)
        big_x = big_boxes.repeat_interleave(self.BIG_ANCHORS_PER_CELL).repeat(self.BIG_GRID)
        big_y = big_boxes.repeat_interleave(self.BIG_GRID * self.BIG_ANCHORS_PER_CELL)
        big_w = torch.ones_like(big_x)
        big_h = torch.ones_like(big_x)
        big_anchors = torch.stack([big_x, big_y, big_w, big_h], dim=1)
        
        return torch.cat([small_anchors, big_anchors], dim=0)
    
    def _create_model(self) -> nn.Module:
        """Create the BlazeEar model backbone."""
        from blazeear import BlazeEar
        return BlazeEar()
    
    def _load_weights(self, weights_path: Union[str, Path]) -> None:
        """Load model weights from checkpoint file."""
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        
        checkpoint = torch.load(str(weights_path), map_location="cpu", weights_only=False)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        
        # Remove "module." prefix if present (from DataParallel)
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
        
        self.model.load_state_dict(state_dict, strict=True)
        print(f"Loaded weights from: {weights_path}")
    
    # =========================================================================
    # Preprocessing
    # =========================================================================
    
    def preprocess(
        self,
        image: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[torch.Tensor, float, Tuple[int, int], Tuple[int, int]]:
        """
        Preprocess an image for model input.
        
        Handles:
        - Format conversion (numpy to tensor, HWC to CHW)
        - Aspect-preserving resize to 256x256 then to 128x128
        - Padding to maintain aspect ratio
        - Normalization to [-1, 1]
        
        Args:
            image: Input image as numpy array (H, W, 3) RGB uint8
                   or torch tensor (3, H, W) or (H, W, 3)
        
        Returns:
            preprocessed: Tensor of shape (1, 3, 128, 128) normalized to [-1, 1]
            scale: Scale factor for denormalizing detections
            pad: (pad_y, pad_x) padding applied in 256x256 space
            orig_size: (orig_h, orig_w) original image dimensions
        """
        # Convert to tensor if numpy
        if isinstance(image, np.ndarray):
            if image.dtype == np.uint8:
                image = torch.from_numpy(image.copy()).float()
            else:
                image = torch.from_numpy(image.copy()).float()
            # Assume HWC format for numpy
            if image.dim() == 3 and image.shape[2] == 3:
                image = image.permute(2, 0, 1)  # HWC -> CHW
        
        # Ensure float and CHW format
        if image.dim() == 3 and image.shape[0] != 3 and image.shape[2] == 3:
            image = image.permute(2, 0, 1)
        
        image = image.float()
        _, orig_h, orig_w = image.shape
        orig_size = (orig_h, orig_w)
        
        # Calculate resize dimensions preserving aspect ratio to fit in 256x256
        if orig_h >= orig_w:
            new_h = 256
            new_w = int(256 * orig_w / orig_h)
            pad_h = 0
            pad_w = 256 - new_w
            scale = orig_w / new_w
        else:
            new_h = int(256 * orig_h / orig_w)
            new_w = 256
            pad_h = 256 - new_h
            pad_w = 0
            scale = orig_h / new_h
        
        # Resize to fit within 256x256
        image = image.unsqueeze(0)  # Add batch dim for interpolate
        resized = F.interpolate(
            image,
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False
        )
        
        # Pad to 256x256 (centered)
        pad_h1 = pad_h // 2
        pad_h2 = pad_h - pad_h1
        pad_w1 = pad_w // 2
        pad_w2 = pad_w - pad_w1
        
        padded = F.pad(resized, (pad_w1, pad_w2, pad_h1, pad_h2), mode="constant", value=0)
        
        # Resize to model input size (128x128)
        preprocessed = F.interpolate(
            padded,
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False
        )
        
        # Normalize to [-1, 1]
        preprocessed = preprocessed / 127.5 - 1.0
        
        # Calculate padding in original image space
        pad = (int(pad_h1 * scale), int(pad_w1 * scale))
        
        return preprocessed, scale, pad, orig_size
    
    # =========================================================================
    # Post-processing
    # =========================================================================
    
    def decode_boxes(
        self,
        raw_boxes: torch.Tensor,
        anchors: torch.Tensor
    ) -> torch.Tensor:
        """
        Decode raw box predictions to normalized coordinates.
        
        Args:
            raw_boxes: Tensor of shape (B, 896, 16) - raw predictions
            anchors: Tensor of shape (896, 4) - anchor boxes [x, y, w, h]
        
        Returns:
            Tensor of shape (B, 896, 4) - decoded boxes [ymin, xmin, ymax, xmax]
        """
        scale = float(self.input_size)
        
        # Decode center and size
        x_center = raw_boxes[..., 0] / scale * anchors[:, 2] + anchors[:, 0]
        y_center = raw_boxes[..., 1] / scale * anchors[:, 3] + anchors[:, 1]
        w = raw_boxes[..., 2] / scale * anchors[:, 2]
        h = raw_boxes[..., 3] / scale * anchors[:, 3]
        
        # Convert to corners [ymin, xmin, ymax, xmax]
        y_min = y_center - h / 2.0
        x_min = x_center - w / 2.0
        y_max = y_center + h / 2.0
        x_max = x_center + w / 2.0
        
        boxes = torch.stack([y_min, x_min, y_max, x_max], dim=-1)
        
        # Ensure proper ordering
        boxes_ordered = torch.stack([
            torch.minimum(boxes[..., 0], boxes[..., 2]),  # ymin
            torch.minimum(boxes[..., 1], boxes[..., 3]),  # xmin
            torch.maximum(boxes[..., 0], boxes[..., 2]),  # ymax
            torch.maximum(boxes[..., 1], boxes[..., 3]),  # xmax
        ], dim=-1)
        
        return boxes_ordered
    
    def nms(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply Non-Maximum Suppression.
        
        Uses torchvision NMS which expects [x1, y1, x2, y2] format.
        
        Args:
            boxes: Tensor of shape (N, 4) - boxes [ymin, xmin, ymax, xmax]
            scores: Tensor of shape (N,) - confidence scores
        
        Returns:
            kept_boxes: Tensor of shape (M, 4)
            kept_scores: Tensor of shape (M,)
        """
        if boxes.shape[0] == 0:
            return boxes, scores
        
        # Convert [ymin, xmin, ymax, xmax] to [xmin, ymin, xmax, ymax] for torchvision
        boxes_xyxy = boxes[:, [1, 0, 3, 2]]
        
        # Apply NMS
        try:
            import torchvision.ops
            keep_idx = torchvision.ops.nms(boxes_xyxy, scores, self.iou_threshold)
        except ImportError:
            # Fallback: simple NMS implementation
            keep_idx = self._simple_nms(boxes_xyxy, scores)
        
        # Limit to max detections
        if len(keep_idx) > self.max_detections:
            keep_idx = keep_idx[:self.max_detections]
        
        return boxes[keep_idx], scores[keep_idx]
    
    def _simple_nms(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor
    ) -> torch.Tensor:
        """Simple NMS implementation as fallback."""
        order = torch.argsort(scores, descending=True)
        keep = []
        
        while len(order) > 0:
            i = order[0].item()
            keep.append(i)
            
            if len(order) == 1:
                break
            
            # Compute IoU with remaining boxes
            remaining = order[1:]
            ious = self._compute_iou(boxes[i], boxes[remaining])
            
            # Keep boxes with IoU below threshold
            mask = ious <= self.iou_threshold
            order = remaining[mask]
        
        return torch.tensor(keep, dtype=torch.long, device=boxes.device)
    
    def _compute_iou(
        self,
        box: torch.Tensor,
        boxes: torch.Tensor
    ) -> torch.Tensor:
        """Compute IoU between one box and multiple boxes."""
        # Intersection
        x1 = torch.maximum(box[0], boxes[:, 0])
        y1 = torch.maximum(box[1], boxes[:, 1])
        x2 = torch.minimum(box[2], boxes[:, 2])
        y2 = torch.minimum(box[3], boxes[:, 3])
        
        inter_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        # Union
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union_area = box_area + boxes_area - inter_area
        
        return inter_area / (union_area + 1e-6)
    
    def denormalize_detections(
        self,
        detections: torch.Tensor,
        scale: float,
        pad: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Convert normalized detections back to original image coordinates.
        
        Args:
            detections: Tensor of shape (N, 5) - [ymin, xmin, ymax, xmax, score]
            scale: Scale factor from preprocessing
            pad: (pad_y, pad_x) padding from preprocessing
        
        Returns:
            Tensor of shape (N, 5) in original image coordinates
        """
        if detections.shape[0] == 0:
            return detections
        
        result = detections.clone()
        
        # Map from normalized [0, 1] to 256 space, then to original
        result[:, 0] = detections[:, 0] * scale * 256 - pad[0]  # ymin
        result[:, 1] = detections[:, 1] * scale * 256 - pad[1]  # xmin
        result[:, 2] = detections[:, 2] * scale * 256 - pad[0]  # ymax
        result[:, 3] = detections[:, 3] * scale * 256 - pad[1]  # xmax
        
        return result
    
    # =========================================================================
    # Main Inference
    # =========================================================================
    
    def forward(
        self,
        image: Union[np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        """
        Run inference on an image.
        
        Args:
            image: Input image as numpy array (H, W, 3) RGB uint8
                   or torch tensor (3, H, W) or (H, W, 3)
        
        Returns:
            Tensor of shape (N, 5) with [ymin, xmin, ymax, xmax, confidence]
            in original image coordinates. Returns empty tensor (0, 5) if
            no detections.
        """
        # Preprocess
        preprocessed, scale, pad, orig_size = self.preprocess(image)
        preprocessed = preprocessed.to(self.anchors.device)
        
        # Run model
        with torch.no_grad():
            raw_boxes, raw_scores = self.model(preprocessed)
        
        # Decode boxes
        boxes = self.decode_boxes(raw_boxes, self.anchors)
        
        # Apply score threshold
        scores = raw_scores.clamp(-self.score_clipping_thresh, self.score_clipping_thresh)
        scores = scores.sigmoid().squeeze(-1)
        
        # Filter by confidence (batch index 0)
        boxes_b = boxes[0]
        scores_b = scores[0]
        
        mask = scores_b >= self.confidence_threshold
        filtered_boxes = boxes_b[mask]
        filtered_scores = scores_b[mask]
        
        if filtered_boxes.shape[0] == 0:
            return torch.zeros((0, 5), dtype=torch.float32, device=self.anchors.device)
        
        # Apply NMS
        kept_boxes, kept_scores = self.nms(filtered_boxes, filtered_scores)
        
        # Combine boxes and scores
        detections = torch.cat([kept_boxes, kept_scores.unsqueeze(-1)], dim=-1)
        
        # Denormalize to original image coordinates
        detections = self.denormalize_detections(detections, scale, pad)
        
        return detections
    
    @torch.no_grad()
    def predict(
        self,
        image: Union[np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        """
        Convenience method that returns numpy array.
        
        Args:
            image: Input image (numpy HWC RGB or torch tensor)
        
        Returns:
            Numpy array of shape (N, 5) with [ymin, xmin, ymax, xmax, confidence]
        """
        detections = self.forward(image)
        return detections.cpu().numpy()
    
    # =========================================================================
    # Export Methods
    # =========================================================================
    
    def to_torchscript(self, output_path: Union[str, Path]) -> None:
        """
        Export model to TorchScript for portable deployment.
        
        Note: This exports a version that takes preprocessed input (128x128 normalized).
        Use BlazeEarInferenceExportable for full pipeline export.
        
        Args:
            output_path: Path to save the .pt file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create exportable wrapper
        exportable = BlazeEarInferenceExportable(
            model=self.model,
            anchors=self.anchors,
            confidence_threshold=self.confidence_threshold,
            iou_threshold=self.iou_threshold,
            input_size=self.input_size,
            max_detections=self.max_detections,
            score_clipping_thresh=self.score_clipping_thresh,
        )
        exportable.eval()
        
        # Trace with example input
        device = self.anchors.device
        dummy = torch.randn(1, 3, self.input_size, self.input_size, device=device)
        traced = torch.jit.trace(exportable, dummy)
        torch.jit.save(traced, str(output_path))
        print(f"Exported TorchScript model to: {output_path}")
    
    def to_onnx(
        self,
        output_path: Union[str, Path],
        opset_version: int = 17,
        include_postprocessing: bool = True,
        end_to_end: bool = False,
        for_web: bool = False,
    ) -> None:
        """
        Export model to ONNX for web deployment.
        
        NOTE: Set for_web=True for ONNX Runtime Web compatibility.
              This avoids int64 operations that aren't supported in the browser.
        
        Args:
            output_path: Path to save the .onnx file
            opset_version: ONNX opset version (default: 17)
            include_postprocessing: If True, includes decode + NMS (ignored if end_to_end=True)
            end_to_end: If True, exports model that takes preprocessing params and outputs
                        detections in original image coordinates.
                        
        Export modes:
        
        1. end_to_end=True: Turnkey model for JavaScript
           Inputs: 
             - image: (1, 3, 128, 128) preprocessed image in [0, 255]
             - scale: scalar - scale factor from preprocessing
             - pad_y: scalar - y padding in original coords
             - pad_x: scalar - x padding in original coords
           Output: (N, 5) detections [ymin, xmin, ymax, xmax, conf] in original coords
           
           JavaScript preprocessing:
             // Given original image of size (orig_h, orig_w):
             // 1. Scale larger dim to 256
             let max_dim = Math.max(orig_h, orig_w);
             let scale = max_dim / 256.0;
             let new_h = Math.round(orig_h / scale);
             let new_w = Math.round(orig_w / scale);
             // 2. Resize image to (new_h, new_w)
             // 3. Pad to 256x256 centered
             let pad_h = 256 - new_h;
             let pad_w = 256 - new_w;
             let pad_y = Math.floor(pad_h / 2) * scale;
             let pad_x = Math.floor(pad_w / 2) * scale;
             // 4. Resize 256x256 to 128x128
             // 5. Pass to model with (image, scale, pad_y, pad_x)
           
        2. include_postprocessing=True, end_to_end=False (default):
           Input: (1, 3, 128, 128) preprocessed image in [0, 255]
           Output: (N, 5) detections in normalized [0, 1] coords
           
        3. include_postprocessing=False:
           Input: (1, 3, 128, 128) preprocessed image in [0, 255]  
           Output: raw_boxes (1, 896, 16), raw_scores (1, 896, 1)
           
        4. for_web=True: Web-optimized export (avoids int64 ops like TopK, NMS)
           Input: (1, 3, 128, 128) preprocessed image in [0, 255]
                  + scale, pad_y, pad_x as preprocessing params
           Output: boxes (896, 4) decoded boxes in original coords
                   scores (896,) confidence scores (after sigmoid)
           JavaScript must do filtering and NMS on these outputs.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        device = self.anchors.device
        
        if for_web:
            # Web-friendly export: no TopK/NMS (these use int64)
            # Returns decoded boxes + scores, JS does NMS
            exportable = BlazeEarWebExportable(
                model=self.model,
                anchors=self.anchors,
                input_size=self.input_size,
                score_clipping_thresh=self.score_clipping_thresh,
            )
            output_names = ["boxes", "scores"]
            input_names = ["image", "scale", "pad_y", "pad_x"]
            dynamic_axes = {}  # Fixed sizes for web
            
            # Dummy inputs
            dummy_image = torch.randn(1, 3, self.input_size, self.input_size, device=device) * 255
            dummy_scale = torch.tensor(2.5, device=device)
            dummy_pad_y = torch.tensor(0.0, device=device)
            dummy_pad_x = torch.tensor(60.0, device=device)
            dummy = (dummy_image, dummy_scale, dummy_pad_y, dummy_pad_x)
            mode_str = "web-optimized"
            
        elif end_to_end:
            # Model with denormalization - takes preprocessing params
            exportable = BlazeEarEndToEndExportable(
                model=self.model,
                anchors=self.anchors,
                confidence_threshold=self.confidence_threshold,
                iou_threshold=self.iou_threshold,
                input_size=self.input_size,
                max_detections=self.max_detections,
                score_clipping_thresh=self.score_clipping_thresh,
            )
            output_names = ["detections"]
            input_names = ["image", "scale", "pad_y", "pad_x"]
            dynamic_axes = {"detections": {0: "num_detections"}}
            
            # Dummy inputs for tracing
            dummy_image = torch.randn(1, 3, self.input_size, self.input_size, device=device) * 255
            dummy_scale = torch.tensor(2.5, device=device)  # Example: 640/256
            dummy_pad_y = torch.tensor(0.0, device=device)
            dummy_pad_x = torch.tensor(60.0, device=device)  # Example padding
            dummy = (dummy_image, dummy_scale, dummy_pad_y, dummy_pad_x)
            mode_str = "end-to-end"
            
        elif include_postprocessing:
            # Just post-processing (expects 128x128 preprocessed input)
            exportable = BlazeEarInferenceExportable(
                model=self.model,
                anchors=self.anchors,
                confidence_threshold=self.confidence_threshold,
                iou_threshold=self.iou_threshold,
                input_size=self.input_size,
                max_detections=self.max_detections,
                score_clipping_thresh=self.score_clipping_thresh,
            )
            output_names = ["detections"]
            input_names = ["image"]
            dynamic_axes = {"image": {0: "batch"}, "detections": {0: "num_detections"}}
            dummy = (torch.randn(1, 3, self.input_size, self.input_size, device=device) * 255,)
            mode_str = "with post-processing"
        else:
            # Raw output only
            exportable = BlazeEarRawExportable(self.model)
            output_names = ["raw_boxes", "raw_scores"]
            input_names = ["image"]
            dynamic_axes = {
                "image": {0: "batch"},
                "raw_boxes": {0: "batch"},
                "raw_scores": {0: "batch"},
            }
            dummy = (torch.randn(1, 3, self.input_size, self.input_size, device=device) * 255,)
            mode_str = "raw"
        
        exportable.eval()
        
        torch.onnx.export(
            exportable,
            dummy,
            str(output_path),
            opset_version=opset_version,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
        )
        
        print(f"Exported ONNX model ({mode_str}) to: {output_path}")


class BlazeEarInferenceExportable(nn.Module):
    """
    Exportable wrapper with post-processing for ONNX/TorchScript.
    
    Takes preprocessed input (normalized 128x128) and outputs detections.
    For web deployment, preprocessing should be done in JavaScript.
    """
    # Type hints for registered buffers
    anchors: torch.Tensor
    
    def __init__(
        self,
        model: nn.Module,
        anchors: torch.Tensor,
        confidence_threshold: float = 0.75,
        iou_threshold: float = 0.3,
        input_size: int = 128,
        max_detections: int = 100,
        score_clipping_thresh: float = 100.0,
    ):
        super().__init__()
        self.model = model
        self.register_buffer("anchors", anchors)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size
        self.max_detections = max_detections
        self.score_clipping_thresh = score_clipping_thresh
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Run inference with post-processing.
        
        Args:
            image: Preprocessed image tensor (B, 3, 128, 128) in range [0, 255]
                   Will be normalized internally to [-1, 1]
        
        Returns:
            Detections tensor (N, 5) with [ymin, xmin, ymax, xmax, confidence]
            in normalized [0, 1] coordinates
        """
        # Always normalize as [0, 255] input (avoid tracing issue with conditionals)
        x = image / 127.5 - 1.0
        
        # Run model
        raw_boxes, raw_scores = self.model(x)
        
        # Decode boxes
        scale = float(self.input_size)
        anchors = self.anchors
        
        x_center = raw_boxes[..., 0] / scale * anchors[:, 2] + anchors[:, 0]
        y_center = raw_boxes[..., 1] / scale * anchors[:, 3] + anchors[:, 1]
        w = raw_boxes[..., 2] / scale * anchors[:, 2]
        h = raw_boxes[..., 3] / scale * anchors[:, 3]
        
        y_min = y_center - h / 2.0
        x_min = x_center - w / 2.0
        y_max = y_center + h / 2.0
        x_max = x_center + w / 2.0
        
        boxes = torch.stack([y_min, x_min, y_max, x_max], dim=-1)
        
        # Ensure proper ordering
        boxes = torch.stack([
            torch.minimum(boxes[..., 0], boxes[..., 2]),
            torch.minimum(boxes[..., 1], boxes[..., 3]),
            torch.maximum(boxes[..., 0], boxes[..., 2]),
            torch.maximum(boxes[..., 1], boxes[..., 3]),
        ], dim=-1)
        
        # Scores
        scores = raw_scores.clamp(-self.score_clipping_thresh, self.score_clipping_thresh)
        scores = scores.sigmoid().squeeze(-1)
        
        # Process batch index 0 (export assumes batch=1)
        boxes = boxes[0]
        scores = scores[0]
        
        # Use topk to get top candidates
        topk_scores, topk_idx = torch.topk(scores, k=self.max_detections, dim=0)
        # Cast indices to int32 for ONNX Runtime Web compatibility
        topk_idx = topk_idx.to(torch.int32)
        topk_boxes = boxes.index_select(0, topk_idx.to(torch.int64))
        
        # Convert to xyxy for NMS [xmin, ymin, xmax, ymax]
        boxes_xyxy = topk_boxes[:, [1, 0, 3, 2]]
        
        # Apply torchvision NMS (imported at module level for tracing)
        keep_idx = torchvision.ops.nms(boxes_xyxy, topk_scores, self.iou_threshold)
        # Cast to int32 for ONNX Runtime Web
        keep_idx = keep_idx.to(torch.int32)
        
        # Gather final detections
        final_boxes = topk_boxes.index_select(0, keep_idx.to(torch.int64))
        final_scores = topk_scores.index_select(0, keep_idx.to(torch.int64))
        
        # Apply confidence threshold mask
        final_mask = final_scores >= self.confidence_threshold
        final_boxes = final_boxes[final_mask]
        final_scores = final_scores[final_mask]
        
        return torch.cat([final_boxes, final_scores.unsqueeze(-1)], dim=-1)


class BlazeEarRawExportable(nn.Module):
    """Exportable wrapper that outputs raw model predictions."""
    
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
    
    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run model and return raw outputs.
        
        Args:
            image: Preprocessed image tensor (B, 3, 128, 128) in range [0, 255]
        
        Returns:
            raw_boxes: (B, 896, 16) raw box predictions
            raw_scores: (B, 896, 1) raw score logits
        """
        x = image / 127.5 - 1.0
        return self.model(x)


class BlazeEarWebExportable(nn.Module):
    """
    Web-optimized exportable model for ONNX Runtime Web.
    
    This version avoids int64-producing operations (TopK, NMS, NonZero)
    that cause "int64 is not supported" errors in the browser.
    
    Returns all 896 decoded boxes + scores. JavaScript should:
    1. Filter by confidence threshold
    2. Apply NMS in JavaScript
    
    Input: 
        image: (1, 3, 128, 128) preprocessed image in [0, 255]
        scale: scalar - the scale factor used during resize
        pad_y: scalar - y-axis padding in original image space
        pad_x: scalar - x-axis padding in original image space
    
    Output:
        boxes: (896, 4) decoded boxes [ymin, xmin, ymax, xmax] in original image coords
        scores: (896,) confidence scores (after sigmoid)
    """
    # Type hints for registered buffers
    anchors: torch.Tensor
    
    def __init__(
        self,
        model: nn.Module,
        anchors: torch.Tensor,
        input_size: int = 128,
        score_clipping_thresh: float = 100.0,
    ):
        super().__init__()
        self.model = model
        self.register_buffer("anchors", anchors)
        self.input_size = input_size
        self.score_clipping_thresh = score_clipping_thresh
    
    def forward(
        self, 
        image: torch.Tensor, 
        scale: torch.Tensor, 
        pad_y: torch.Tensor, 
        pad_x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inference returning decoded boxes + scores.
        
        Args:
            image: Preprocessed image tensor (1, 3, 128, 128) in range [0, 255]
            scale: Scale factor used during resize (scalar tensor)
            pad_y: Y-axis padding in original image space (scalar tensor)
            pad_x: X-axis padding in original image space (scalar tensor)
        
        Returns:
            boxes: (896, 4) with [ymin, xmin, ymax, xmax] in original image coords
            scores: (896,) confidence scores (after sigmoid)
        """
        # Run model
        x = image / 127.5 - 1.0
        raw_boxes, raw_scores = self.model(x)
        
        # Decode boxes
        model_scale = float(self.input_size)
        anchors = self.anchors
        
        x_center = raw_boxes[..., 0] / model_scale * anchors[:, 2] + anchors[:, 0]
        y_center = raw_boxes[..., 1] / model_scale * anchors[:, 3] + anchors[:, 1]
        w = raw_boxes[..., 2] / model_scale * anchors[:, 2]
        h = raw_boxes[..., 3] / model_scale * anchors[:, 3]
        
        y_min = y_center - h / 2.0
        x_min = x_center - w / 2.0
        y_max = y_center + h / 2.0
        x_max = x_center + w / 2.0
        
        boxes = torch.stack([y_min, x_min, y_max, x_max], dim=-1)
        
        # Ensure proper ordering [min, min, max, max]
        boxes = torch.stack([
            torch.minimum(boxes[..., 0], boxes[..., 2]),
            torch.minimum(boxes[..., 1], boxes[..., 3]),
            torch.maximum(boxes[..., 0], boxes[..., 2]),
            torch.maximum(boxes[..., 1], boxes[..., 3]),
        ], dim=-1)
        
        # Scores
        scores = raw_scores.clamp(-self.score_clipping_thresh, self.score_clipping_thresh)
        scores = scores.sigmoid().squeeze(-1)
        
        # Extract batch 0
        boxes = boxes[0]  # (896, 4)
        scores = scores[0]  # (896,)
        
        # Denormalize to original image coordinates
        # Boxes are in normalized [0, 1] coords relative to 256x256 padded image
        denorm_boxes = torch.zeros_like(boxes)
        denorm_boxes[:, 0] = boxes[:, 0] * scale * 256 - pad_y  # ymin
        denorm_boxes[:, 1] = boxes[:, 1] * scale * 256 - pad_x  # xmin
        denorm_boxes[:, 2] = boxes[:, 2] * scale * 256 - pad_y  # ymax
        denorm_boxes[:, 3] = boxes[:, 3] * scale * 256 - pad_x  # xmax
        
        return denorm_boxes, scores


class BlazeEarEndToEndExportable(nn.Module):
    """
    Exportable model with denormalization for ONNX.
    
    This version takes:
    - A preprocessed 128x128 image
    - scale and pad parameters computed during preprocessing
    
    And returns detections in original image coordinates.
    
    For JavaScript, you:
    1. Compute scale/pad while resizing your image to 128x128
    2. Pass the preprocessed image + scale + pad_y + pad_x to this model
    3. Get back detections in your original image coordinates
    
    Input: 
        image: (1, 3, 128, 128) preprocessed image in [0, 255]
        scale: scalar - the scale factor used during resize
        pad_y: scalar - y-axis padding in original image space
        pad_x: scalar - x-axis padding in original image space
    
    Output: (N, 5) detections [ymin, xmin, ymax, xmax, confidence] in original image coords
    """
    # Type hints for registered buffers
    anchors: torch.Tensor
    
    def __init__(
        self,
        model: nn.Module,
        anchors: torch.Tensor,
        confidence_threshold: float = 0.75,
        iou_threshold: float = 0.3,
        input_size: int = 128,
        max_detections: int = 100,
        score_clipping_thresh: float = 100.0,
    ):
        super().__init__()
        self.model = model
        self.register_buffer("anchors", anchors)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size
        self.max_detections = max_detections
        self.score_clipping_thresh = score_clipping_thresh
    
    def forward(
        self, 
        image: torch.Tensor, 
        scale: torch.Tensor, 
        pad_y: torch.Tensor, 
        pad_x: torch.Tensor
    ) -> torch.Tensor:
        """
        Inference with denormalization.
        
        Args:
            image: Preprocessed image tensor (1, 3, 128, 128) in range [0, 255]
            scale: Scale factor used during resize (scalar tensor)
            pad_y: Y-axis padding in original image space (scalar tensor)
            pad_x: X-axis padding in original image space (scalar tensor)
        
        Returns:
            Detections tensor (N, 5) with [ymin, xmin, ymax, xmax, confidence]
            in ORIGINAL image coordinates
        """
        # =====================================================================
        # INFERENCE: Run model (image already preprocessed to 128x128)
        # =====================================================================
        x = image / 127.5 - 1.0
        raw_boxes, raw_scores = self.model(x)
        
        # =====================================================================
        # POST-PROCESSING: Decode boxes
        # =====================================================================
        model_scale = float(self.input_size)
        anchors = self.anchors
        
        x_center = raw_boxes[..., 0] / model_scale * anchors[:, 2] + anchors[:, 0]
        y_center = raw_boxes[..., 1] / model_scale * anchors[:, 3] + anchors[:, 1]
        w = raw_boxes[..., 2] / model_scale * anchors[:, 2]
        h = raw_boxes[..., 3] / model_scale * anchors[:, 3]
        
        y_min = y_center - h / 2.0
        x_min = x_center - w / 2.0
        y_max = y_center + h / 2.0
        x_max = x_center + w / 2.0
        
        boxes = torch.stack([y_min, x_min, y_max, x_max], dim=-1)
        
        # Ensure proper ordering
        boxes = torch.stack([
            torch.minimum(boxes[..., 0], boxes[..., 2]),
            torch.minimum(boxes[..., 1], boxes[..., 3]),
            torch.maximum(boxes[..., 0], boxes[..., 2]),
            torch.maximum(boxes[..., 1], boxes[..., 3]),
        ], dim=-1)
        
        # Scores
        scores = raw_scores.clamp(-self.score_clipping_thresh, self.score_clipping_thresh)
        scores = scores.sigmoid().squeeze(-1)
        
        # Process batch index 0
        boxes = boxes[0]
        scores = scores[0]
        
        # Top-K filtering
        topk_scores, topk_idx = torch.topk(scores, k=self.max_detections, dim=0)
        # Cast indices to int32 for ONNX Runtime Web compatibility
        topk_idx = topk_idx.to(torch.int32)
        topk_boxes = boxes.index_select(0, topk_idx.to(torch.int64))
        
        # NMS
        boxes_xyxy = topk_boxes[:, [1, 0, 3, 2]]
        keep_idx = torchvision.ops.nms(boxes_xyxy, topk_scores, self.iou_threshold)
        # Cast to int32 for ONNX Runtime Web
        keep_idx = keep_idx.to(torch.int32)
        
        final_boxes = topk_boxes.index_select(0, keep_idx.to(torch.int64))
        final_scores = topk_scores.index_select(0, keep_idx.to(torch.int64))
        
        # Confidence threshold
        final_mask = final_scores >= self.confidence_threshold
        final_boxes = final_boxes[final_mask]
        final_scores = final_scores[final_mask]
        
        # =====================================================================
        # DENORMALIZATION: Convert to original image coordinates
        # =====================================================================
        # Boxes are in normalized [0, 1] coords relative to 256x256 padded image
        # Map to original image: coord * scale * 256 - pad
        denorm_boxes = torch.zeros_like(final_boxes)
        denorm_boxes[:, 0] = final_boxes[:, 0] * scale * 256 - pad_y  # ymin
        denorm_boxes[:, 1] = final_boxes[:, 1] * scale * 256 - pad_x  # xmin
        denorm_boxes[:, 2] = final_boxes[:, 2] * scale * 256 - pad_y  # ymax
        denorm_boxes[:, 3] = final_boxes[:, 3] * scale * 256 - pad_x  # xmax
        
        return torch.cat([denorm_boxes, final_scores.unsqueeze(-1)], dim=-1)


# =============================================================================
# Utility Functions
# =============================================================================

def load_inference_pipeline(
    weights_path: str = "runs/checkpoints/BlazeEar_best.pth",
    confidence_threshold: float = 0.75,
    iou_threshold: float = 0.3,
    device: str = "auto",
) -> BlazeEarInference:
    """
    Convenience function to load the inference pipeline.
    
    Args:
        weights_path: Path to model weights
        confidence_threshold: Detection confidence threshold
        iou_threshold: NMS IoU threshold
        device: "auto", "cuda", or "cpu"
    
    Returns:
        BlazeEarInference instance ready for inference
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    return BlazeEarInference(
        weights_path=weights_path,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        device=device,
    )


if __name__ == "__main__":
    # Simple test
    import argparse
    
    parser = argparse.ArgumentParser(description="Test BlazeEar Inference Pipeline")
    parser.add_argument("--weights", type=str, default="runs/checkpoints/BlazeEar_best.pth")
    parser.add_argument("--image", type=str, default=None, help="Test image path")
    parser.add_argument("--export-onnx", type=str, default=None, help="Export to ONNX")
    parser.add_argument("--export-onnx-e2e", type=str, default=None, 
                        help="Export end-to-end ONNX (turnkey: any resolution -> original coords)")
    parser.add_argument("--export-torchscript", type=str, default=None, help="Export to TorchScript")
    parser.add_argument("--confidence", type=float, default=0.75)
    parser.add_argument("--iou", type=float, default=0.3)
    args = parser.parse_args()
    
    # Load pipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    pipeline = BlazeEarInference(
        weights_path=args.weights,
        confidence_threshold=args.confidence,
        iou_threshold=args.iou,
        device=device,
    )
    
    print(f"Pipeline loaded successfully!")
    print(f"  Confidence threshold: {pipeline.confidence_threshold}")
    print(f"  IoU threshold: {pipeline.iou_threshold}")
    print(f"  Input size: {pipeline.input_size}")
    print(f"  Number of anchors: {pipeline.anchors.shape[0]}")
    
    # Test with dummy image
    print("\nTesting with random 640x480 image...")
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    detections = pipeline.predict(dummy_image)
    print(f"  Detections shape: {detections.shape}")
    
    # Test with actual image if provided
    if args.image:
        import cv2
        print(f"\nTesting with image: {args.image}")
        img = cv2.imread(args.image)
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            detections = pipeline.predict(img_rgb)
            print(f"  Image size: {img.shape[:2]}")
            print(f"  Detections: {len(detections)}")
            for i, det in enumerate(detections):
                print(f"    [{i}] box=({det[0]:.1f}, {det[1]:.1f}, {det[2]:.1f}, {det[3]:.1f}), conf={det[4]:.3f}")
        else:
            print(f"  Error: Could not load image")
    
    # Export if requested
    if args.export_onnx:
        print(f"\nExporting to ONNX (128x128 input, normalized output): {args.export_onnx}")
        pipeline.to_onnx(args.export_onnx, include_postprocessing=True, end_to_end=False)
    
    if args.export_onnx_e2e:
        print(f"\nExporting end-to-end ONNX (any resolution, original coords): {args.export_onnx_e2e}")
        pipeline.to_onnx(args.export_onnx_e2e, include_postprocessing=True, end_to_end=True)
    
    if args.export_torchscript:
        print(f"\nExporting to TorchScript: {args.export_torchscript}")
        pipeline.to_torchscript(args.export_torchscript)
    
    print("\nDone!")
