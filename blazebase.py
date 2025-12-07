import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict

# Import from consolidated utils modules
from utils.anchor_utils import (
    generate_reference_anchors,
    encode_boxes_to_anchors,
    flatten_anchor_targets,
    flatten_anchor_targets_torch,
    anchor_options,
)
from utils.iou import compute_iou_np as compute_iou


# =============================================================================
# Weight Conversion: BlazeBlock_WT -> BlazeBlock
# =============================================================================

def unfold_conv_bn(conv_weight: torch.Tensor, conv_bias: torch.Tensor, 
                   num_features: int, eps: float = 1e-5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    "Unfold" a convolution with folded BatchNorm back to separate conv + BN.
    
    When BatchNorm is folded into conv, the transformation is:
        W_folded = W * gamma / sqrt(var + eps)
        b_folded = (b - mean) * gamma / sqrt(var + eps) + beta
    
    To reverse this, we assume the original BN had:
        gamma = 1 (scale)
        beta = 0 (shift)  
        mean = 0
        var = 1
    
    This gives us a valid starting point for fine-tuning where:
        - Conv weights are copied as-is
        - BN is initialized to identity transform
        - The bias is moved to BN beta
    
    Args:
        conv_weight: [out_ch, in_ch, kH, kW] conv weight with folded BN
        conv_bias: [out_ch] conv bias with folded BN
        num_features: number of features for BatchNorm
        eps: BatchNorm epsilon
        
    Returns:
        new_conv_weight: [out_ch, in_ch, kH, kW] 
        bn_weight (gamma): [num_features] initialized to 1
        bn_bias (beta): [num_features] initialized to conv_bias
        bn_running_mean: [num_features] initialized to 0
        bn_running_var: [num_features] initialized to 1
    """
    # Conv weight stays the same
    new_conv_weight = conv_weight.clone()
    
    # BatchNorm parameters: initialize to identity transform
    # gamma (weight) = 1, so output = normalized * 1 = normalized
    bn_weight = torch.ones(num_features)
    
    # beta (bias) absorbs the original conv bias
    # This way: conv(x) + bias ≈ BN(conv(x)) when BN is identity + bias
    bn_bias = conv_bias.clone()
    
    # Running statistics: mean=0, var=1 (identity normalization initially)
    bn_running_mean = torch.zeros(num_features)
    bn_running_var = torch.ones(num_features)
    
    return new_conv_weight, bn_weight, bn_bias, bn_running_mean, bn_running_var


def convert_blazeblock_weights(old_state: Dict[str, torch.Tensor], 
                                prefix: str = ""
) -> Dict[str, torch.Tensor]:
    """
    Convert a single BlazeBlock_WT's weights to BlazeBlock format.
    
    Maps:
        convs.0.weight, convs.0.bias -> dw_conv.weight, bn1.*
        convs.1.weight, convs.1.bias -> pw_conv.weight, bn2.*
        skip_proj.weight, skip_proj.bias -> skip_proj.0.weight, skip_proj.1.*
    
    Args:
        old_state: State dict containing BlazeBlock_WT weights
        prefix: Key prefix (e.g., "backbone1.2.")
        
    Returns:
        New state dict with BlazeBlock weight names
    """
    new_state = {}
    
    # Depthwise conv: convs.0 -> dw_conv + bn1
    dw_weight = old_state[f"{prefix}convs.0.weight"]
    dw_bias = old_state[f"{prefix}convs.0.bias"]
    num_dw_features = dw_weight.shape[0]
    
    new_weight, bn_w, bn_b, bn_mean, bn_var = unfold_conv_bn(
        dw_weight, dw_bias, num_dw_features
    )
    
    new_state[f"{prefix}dw_conv.weight"] = new_weight
    new_state[f"{prefix}bn1.weight"] = bn_w
    new_state[f"{prefix}bn1.bias"] = bn_b
    new_state[f"{prefix}bn1.running_mean"] = bn_mean
    new_state[f"{prefix}bn1.running_var"] = bn_var
    new_state[f"{prefix}bn1.num_batches_tracked"] = torch.tensor(0, dtype=torch.long)
    
    # Pointwise conv: convs.1 -> pw_conv + bn2
    pw_weight = old_state[f"{prefix}convs.1.weight"]
    pw_bias = old_state[f"{prefix}convs.1.bias"]
    num_pw_features = pw_weight.shape[0]
    
    new_weight, bn_w, bn_b, bn_mean, bn_var = unfold_conv_bn(
        pw_weight, pw_bias, num_pw_features
    )
    
    new_state[f"{prefix}pw_conv.weight"] = new_weight
    new_state[f"{prefix}bn2.weight"] = bn_w
    new_state[f"{prefix}bn2.bias"] = bn_b
    new_state[f"{prefix}bn2.running_mean"] = bn_mean
    new_state[f"{prefix}bn2.running_var"] = bn_var
    new_state[f"{prefix}bn2.num_batches_tracked"] = torch.tensor(0, dtype=torch.long)
    
    # Skip projection (if exists): skip_proj -> skip_proj.0 + skip_proj.1
    skip_weight_key = f"{prefix}skip_proj.weight"
    if skip_weight_key in old_state:
        skip_weight = old_state[skip_weight_key]
        skip_bias = old_state[f"{prefix}skip_proj.bias"]
        num_skip_features = skip_weight.shape[0]
        
        new_weight, bn_w, bn_b, bn_mean, bn_var = unfold_conv_bn(
            skip_weight, skip_bias, num_skip_features
        )
        
        new_state[f"{prefix}skip_proj.0.weight"] = new_weight
        new_state[f"{prefix}skip_proj.1.weight"] = bn_w
        new_state[f"{prefix}skip_proj.1.bias"] = bn_b
        new_state[f"{prefix}skip_proj.1.running_mean"] = bn_mean
        new_state[f"{prefix}skip_proj.1.running_var"] = bn_var
        new_state[f"{prefix}skip_proj.1.num_batches_tracked"] = torch.tensor(0, dtype=torch.long)
    
    # PReLU weight (if exists)
    prelu_key = f"{prefix}act.weight"
    if prelu_key in old_state:
        new_state[prelu_key] = old_state[prelu_key].clone()
    
    return new_state


def convert_blazeface_wt_to_trainable(old_state_dict: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    Convert full BlazeFace state dict from BlazeBlock_WT to BlazeBlock format.
    
    This enables loading pretrained MediaPipe weights into a trainable
    BlazeBlock-based model with explicit BatchNorm layers.
    
    Note: Regressor weights are converted to extract only box coordinates (first 4 of 16),
    discarding keypoint regression channels since our trainable model is box-only.
    
    Args:
        old_state_dict: State dict from model using BlazeBlock_WT
        
    Returns:
        New state dict compatible with trainable BlazeBlock model
    """
    new_state_dict = {}
    processed_prefixes = set()
    
    for key in old_state_dict.keys():
        # Check if this is a BlazeBlock_WT conv layer
        if ".convs.0.weight" in key:
            # Extract prefix (e.g., "backbone1.2.")
            prefix = key.replace("convs.0.weight", "")
            
            if prefix not in processed_prefixes:
                processed_prefixes.add(prefix)
                # Convert this block
                block_state = convert_blazeblock_weights(old_state_dict, prefix)
                new_state_dict.update(block_state)
        
        # Copy non-BlazeBlock_WT weights directly (classifiers, regressors, etc.)
        elif ".convs." not in key and ".skip_proj." not in key:
            # Check if it's not already processed as part of a block
            is_block_key = any(key.startswith(p) for p in processed_prefixes)
            if not is_block_key or "act.weight" not in key:
                new_state_dict[key] = old_state_dict[key].clone()
    
    return new_state_dict


def load_mediapipe_weights(model: nn.Module,
                           weights_path: str,
                           strict: bool = False,
                           load_detection_heads: bool = True
) -> Tuple[List[str], List[str]]:
    """
    Load MediaPipe pretrained weights (either BlazeBlock_WT or converted format).

    Attempts to load the state dict directly for BlazeBlock_WT-based models and
    falls back to converting folded BatchNorm weights when targeting BlazeBlock
    architectures with explicit BatchNorm layers.

    Args:
        model: Detector model (BlazeBlock_WT or BlazeBlock)
        weights_path: Path to .pth file with MediaPipe weights
        strict: Whether to require exact match
        load_detection_heads: Whether to load classifier/regressor heads (default: True).
                             Set to False when fine-tuning for a different detection task.

    Returns:
        missing_keys: Keys in model not found in weights
        unexpected_keys: Keys in weights not found in model
    """
    def maybe_strip(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if load_detection_heads:
            return state_dict
        filtered = {
            k: v for k, v in state_dict.items()
            if 'classifier' not in k and 'regressor' not in k
        }
        return filtered
    
    def split_regressor_heads(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Handle legacy single-regressor heads by splitting box/kp channels."""
        mapping = [
            ('regressor_8', 'regressor_8_box', 'regressor_8_kp', 2),
            ('regressor_16', 'regressor_16_box', 'regressor_16_kp', 6),
        ]
        updated = dict(state_dict)
        total_coords = 16
        box_coords = 4
        kp_coords = total_coords - box_coords
        for base, box_key, kp_key, anchors_per_cell in mapping:
            weight_key = f"{base}.weight"
            bias_key = f"{base}.bias"
            if weight_key in updated:
                weight = updated.pop(weight_key)
                out_channels, in_channels, kh, kw = weight.shape
                weight = weight.view(anchors_per_cell, total_coords, in_channels, kh, kw)
                box_weight = weight[:, :box_coords].reshape(-1, in_channels, kh, kw)
                kp_weight = weight[:, box_coords:].reshape(-1, in_channels, kh, kw)
                updated[f"{box_key}.weight"] = box_weight.clone()
                updated[f"{kp_key}.weight"] = kp_weight.clone()
            if bias_key in updated:
                bias = updated.pop(bias_key)
                bias = bias.view(anchors_per_cell, total_coords)
                box_bias = bias[:, :box_coords].reshape(-1)
                kp_bias = bias[:, box_coords:].reshape(-1)
                updated[f"{box_key}.bias"] = box_bias.clone()
                updated[f"{kp_key}.bias"] = kp_bias.clone()
        return updated

    # Load original weights (could be BlazeBlock_WT or previously converted format)
    original_state = torch.load(weights_path, map_location='cpu')

    # First, try loading directly (for BlazeBlock_WT-based models)
    try:
        state_dict = split_regressor_heads(maybe_strip(original_state))
        result = model.load_state_dict(state_dict, strict=strict)
        return result.missing_keys, result.unexpected_keys
    except RuntimeError:
        # Fallback: convert folded-BN weights to BlazeBlock format
        converted = convert_blazeface_wt_to_trainable(original_state)
        converted = split_regressor_heads(maybe_strip(converted))
        result = model.load_state_dict(converted, strict=strict)
        return result.missing_keys, result.unexpected_keys


# =============================================================================
# Model Building Blocks
# =============================================================================

class BlazeBlock(nn.Module):
    """
    BlazeBlock with BatchNormalization for training.
    
    Following vincent1bt/blazeface-tensorflow architecture:
    - DepthwiseConv2D -> BatchNorm -> Conv2D 1x1 -> BatchNorm -> Add -> Activation
    
    This is the default trainable block with explicit BatchNorm layers.
    For loading pretrained MediaPipe weights (which have BatchNorm folded into
    conv weights), use BlazeBlock_WT and the weight conversion utilities.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, act='relu', skip_proj=False):
        super(BlazeBlock, self).__init__()
        
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channel_pad = out_channels - in_channels
        
        if stride == 2:
            self.max_pool = nn.MaxPool2d(kernel_size=stride, stride=stride)
        
        # Depthwise convolution (no bias when using BatchNorm)
        self.dw_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2 if stride == 1 else 0,
            groups=in_channels,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        
        # Pointwise convolution (no bias when using BatchNorm)
        self.pw_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Optional skip projection (matches BlazeBlock interface)
        if skip_proj:
            self.skip_proj = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip_proj = None
        
        # Activation (matches BlazeBlock interface)
        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act == 'prelu':
            self.act = nn.PReLU(out_channels)
        else:
            raise NotImplementedError("unknown activation %s" % act)
    
    def forward(self, x):
        # Handle stride=2 padding (TFLite compatibility, matches BlazeBlock)
        if self.stride == 2:
            if self.kernel_size == 3:
                h = F.pad(x, (0, 2, 0, 2), "constant", 0)
            else:
                h = F.pad(x, (1, 2, 1, 2), "constant", 0)
            x_skip = self.max_pool(x)
        else:
            h = x
            x_skip = x
        
        # Depthwise + Pointwise convolutions with BatchNorm
        h = self.dw_conv(h)
        h = self.bn1(h)
        h = self.pw_conv(h)
        h = self.bn2(h)
        
        # Skip connection (matches BlazeBlock logic)
        if self.skip_proj is not None:
            x_skip = self.skip_proj(x_skip)
        elif self.channel_pad > 0:
            x_skip = F.pad(x_skip, (0, 0, 0, 0, 0, self.channel_pad), "constant", 0)
        
        return self.act(h + x_skip)


class BlazeBlock_WT(nn.Module):
    """
    BlazeBlock for Weight Transfer - used for loading pretrained MediaPipe weights.
    
    This version has BatchNorm folded into conv weights (no explicit BN layers),
    matching the format of pretrained .pth files from MediaPipe conversion.
    
    Use the weight conversion utilities (convert_blazeface_to_bn, load_pretrained_to_bn_model)
    to transfer weights from this format to the trainable BlazeBlock format.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, act='relu', skip_proj=False):
        super(BlazeBlock_WT, self).__init__()

        self.stride = stride
        self.kernel_size = kernel_size
        self.channel_pad = out_channels - in_channels

        # TFLite uses slightly different padding than PyTorch 
        # on the depthwise conv layer when the stride is 2.
        if stride == 2:
            self.max_pool = nn.MaxPool2d(kernel_size=stride, stride=stride)
            padding = 0
        else:
            padding = (kernel_size - 1) // 2

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, 
                      kernel_size=kernel_size, stride=stride, padding=padding, 
                      groups=in_channels, bias=True),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                      kernel_size=1, stride=1, padding=0, bias=True),
        )

        if skip_proj:
            self.skip_proj = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                      kernel_size=1, stride=1, padding=0, bias=True)
        else:
            self.skip_proj = None

        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act == 'prelu':
            self.act = nn.PReLU(out_channels)
        else:
            raise NotImplementedError("unknown activation %s"%act)

    def forward(self, x):
        if self.stride == 2:
            if self.kernel_size==3:
                h = F.pad(x, (0, 2, 0, 2), "constant", 0)
            else:
                h = F.pad(x, (1, 2, 1, 2), "constant", 0)
            x = self.max_pool(x)
        else:
            h = x

        if self.skip_proj is not None:
            x = self.skip_proj(x)
        elif self.channel_pad > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.channel_pad), "constant", 0)
        

        return self.act(self.convs(h) + x)


class FinalBlazeBlock(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super(FinalBlazeBlock, self).__init__()

        # TFLite uses slightly different padding than PyTorch
        # on the depthwise conv layer when the stride is 2.
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels,
                      kernel_size=kernel_size, stride=2, padding=0,
                      groups=channels, bias=True),
            nn.Conv2d(in_channels=channels, out_channels=channels,
                      kernel_size=1, stride=1, padding=0, bias=True),
        )

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = F.pad(x, (0, 2, 0, 2), "constant", 0)

        return self.act(self.convs(h))


class BlazeBase(nn.Module):
    """ Base class for media pipe models. """
    
    # Type annotations for class attributes
    detection2roi_method: str
    kp1: int
    kp2: int
    dy: float
    dscale: float
    theta0: float
    
    # Training-related attributes
    reference_anchors: Optional[torch.Tensor] = None
    small_anchors: Optional[torch.Tensor] = None
    big_anchors: Optional[torch.Tensor] = None


    def _device(self):
        """Which device (CPU or GPU) is being used by this model?"""
        return next(self.parameters()).device
    
    def load_weights(self, path: str) -> None:
        self.load_state_dict(torch.load(path))
        self.eval()
        if hasattr(self, "generate_anchors"):
            self.generate_anchors(anchor_options)  # type: ignore[attr-defined]
    
    def init_anchors(self, input_size: int = 128) -> None:
        """
        Initialize reference anchors for training/inference.
        
        Args:
            input_size: Input image size (128 for front model)
        """
        self.reference_anchors, self.small_anchors, self.big_anchors = \
            generate_reference_anchors(input_size)
        
        # Move to device
        device = self._device()
        self.reference_anchors = self.reference_anchors.to(device)
        self.small_anchors = self.small_anchors.to(device)
        self.big_anchors = self.big_anchors.to(device)

    def detection2roi(
        self,
        detection: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Convert detections from detector to an oriented bounding box.

        Adapted from:
        # mediapipe/modules/face_landmark/face_detection_front_detection_to_roi.pbtxt

        The center and size of the box is calculated from the center 
        of the detected box. Rotation is calcualted from the vector
        between kp1 and kp2 relative to theta0. The box is scaled
        and shifted by dscale and dy.

        """


        if self.detection2roi_method == 'box':
            # compute box center and scale
            # use mediapipe/calculators/util/detections_to_rects_calculator.cc
            xc = (detection[:,1] + detection[:,3]) / 2
            yc = (detection[:,0] + detection[:,2]) / 2
            scale = (detection[:,3] - detection[:,1]) # assumes square boxes

        elif self.detection2roi_method == 'alignment':
            # compute box center and scale
            # use mediapipe/calculators/util/alignment_points_to_rects_calculator.cc
            xc = detection[:,4+2*self.kp1]
            yc = detection[:,4+2*self.kp1+1]
            x1 = detection[:,4+2*self.kp2]
            y1 = detection[:,4+2*self.kp2+1]
            scale = ((xc-x1)**2 + (yc-y1)**2).sqrt() * 2
        else:
            raise NotImplementedError(
                "detection2roi_method [%s] not supported"%self.detection2roi_method)

        yc += self.dy * scale
        scale *= self.dscale

        # compute box rotation
        x0 = detection[:,4+2*self.kp1]
        y0 = detection[:,4+2*self.kp1+1]
        x1 = detection[:,4+2*self.kp2]
        y1 = detection[:,4+2*self.kp2+1]
        theta = torch.atan2(y0-y1, x0-x1) - self.theta0
        return xc, yc, scale, theta

