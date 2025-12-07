"""
Model loading and device setup utilities.
"""
import torch
from blazeface import BlazeFace
from utils.config import DEFAULT_NMS_IOU_THRESHOLD


def setup_device() -> torch.device:
    """Setup device for inference/training.

    Returns:
        torch.device: CUDA device if available, otherwise CPU
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def load_model(
    weights_path: str,
    device: torch.device,
    threshold: float | None = None,
    grad_enabled: bool = False
) -> BlazeFace:
    """Load BlazeFace model from either MediaPipe weights or training checkpoint.

    Args:
        weights_path: Path to .pth (MediaPipe) or .ckpt (retrained) file
        device: Device to load model on
        threshold: Optional detection threshold to override model default
        grad_enabled: Whether to enable gradients (default: False for inference)

    Returns:
        Loaded BlazeFace model in eval mode
    """
    from blazebase import anchor_options, load_mediapipe_weights

    prev_grad_state = torch.is_grad_enabled()
    torch.set_grad_enabled(grad_enabled)
    try:
        model = BlazeFace().to(device)

        # Check if this is a training checkpoint or MediaPipe weights
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Training checkpoint format - already in BlazeBlock format
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint.get('epoch', '?')
            val_loss = checkpoint.get('val_loss', None)
            print(f"Loaded training checkpoint (epoch {epoch})", end="")
            if val_loss is not None:
                print(f" - val_loss: {val_loss:.4f}")
            else:
                print()
        else:
            # MediaPipe weights or converted checkpoints (auto-detect format)
            missing, unexpected = load_mediapipe_weights(model, weights_path, strict=False)
            if missing:
                print(f"Warning: Missing keys: {missing}")
            if unexpected:
                print(f"Warning: Unexpected keys: {unexpected}")
            print("Loaded MediaPipe weights")

        # Common setup for both formats
        model.eval()
        if hasattr(model, "generate_anchors"):
            model.generate_anchors(anchor_options)

        if hasattr(model, "min_suppression_threshold"):
            model.min_suppression_threshold = DEFAULT_NMS_IOU_THRESHOLD

        # Override detection threshold if specified
        if threshold is not None:
            model.min_score_thresh = threshold
            print(f"Detection threshold: {threshold}")

        return model
    finally:
        torch.set_grad_enabled(prev_grad_state)
