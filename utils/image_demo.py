"""                                                                                                    
Image demo for BlazeFace detection.
Loads images from train.csv and compares detector output with ground truth annotations.

Navigation:
- 'a' or Left Arrow: Previous image
- 'd' or Right Arrow: Next image
- 'q' or ESC: Quit

Supports loading:
- MediaPipe weights: blazeface.pth (raw state_dict)
- Retrained checkpoints: *.ckpt (dict with 'model_state_dict' key)
"""
import argparse
import cv2
import numpy as np
import sys
from pathlib import Path

# Allow running as `python utils/image_demo.py` from repo root
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils import model_utils, drawing, metrics, config
from utils.data_utils import load_image_boxes_from_csv
def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Image demo for BlazeFace detection with CSV comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--weights", "-w",
        type=str,
        default=config.DEFAULT_WEIGHTS_PATH,
        help="Path to weights file (.pth for MediaPipe, .ckpt for retrained)"
    )
    parser.add_argument(
        "--csv", "-c",
        type=str,
        default=config.DEFAULT_TRAIN_CSV,
        help="Path to CSV file with image annotations"
    )
    parser.add_argument(
        "--data-root", "-d",
        type=str,
        default=config.DEFAULT_DATA_ROOT,
        help="Root directory for image paths (prepended to CSV image_path)"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=config.DEFAULT_DETECTOR_THRESHOLD_DEBUG,
        help="Detection threshold (overrides model default)"
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Start from this image index"
    )
    detection_mode = parser.add_mutually_exclusive_group()
    detection_mode.add_argument(
        "--detection-only",
        dest="detection_only",
        action="store_true",
        help="Show only detections without ground truth comparison (disables IoU matching)"
    )
    detection_mode.add_argument(
        "--no-detection-only",
        dest="detection_only",
        action="store_false",
        help="Enable ground truth overlays and IoU matching"
    )
    parser.set_defaults(detection_only=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    SCRIPT_DIR = Path(__file__).parent

    # Setup device
    gpu = model_utils.setup_device()

    # Load model with threshold
    print(f"Loading weights: {args.weights}")
    detector = model_utils.load_model(args.weights, gpu, threshold=args.threshold)

    print("Model loaded")

    # Load CSV and sort by image path
    csv_path = SCRIPT_DIR / args.csv
    print(f"Loading CSV: {csv_path}")
    image_paths, image_to_boxes = load_image_boxes_from_csv(str(csv_path))
    print(f"Loaded {len(image_paths)} unique images with annotations")

    # Setup window
    WINDOW = "BlazeFace Image Demo"
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    # Navigation state
    current_idx = min(args.start_idx, len(image_paths) - 1)

    print("\nControls:")
    print("  A / Left Arrow  - Previous image")
    print("  D / Right Arrow - Next image")
    print("  Q / ESC         - Quit")
    print()

    while True:
        # Get current image info
        image_path = image_paths[current_idx]
        gt_boxes = image_to_boxes[image_path]

        # Construct full path
        full_image_path = SCRIPT_DIR / args.data_root / image_path

        # Load image
        if not full_image_path.exists():
            print(f"Warning: Image not found: {full_image_path}")
            current_idx = (current_idx + 1) % len(image_paths)
            continue

        img = cv2.imread(str(full_image_path))
        if img is None:
            print(f"Warning: Failed to load image: {full_image_path}")
            current_idx = (current_idx + 1) % len(image_paths)
            continue

        # Convert BGR to RGB for detection
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Run detection
        detections = detector.process(img_rgb)

        if detections is None:
            detections_np = np.empty((0, 4), dtype=np.float32)
        else:
            detections_np = (
                detections.cpu().numpy() if hasattr(detections, 'cpu') else np.asarray(detections)
            )
        detections_np = np.asarray(detections_np)

        # Create display image
        display_img = img.copy()

        if args.detection_only:
            # Detection-only mode: show all detections, no ground truth
            if len(detections_np) > 0:
                drawing.draw_detections(
                    display_img, detections_np, indices=None,
                    color=(0, 255, 0), thickness=2, label="Det"
                )

            # Draw info text (detection-only mode)
            drawing.draw_info_text(
                display_img,
                current_idx,
                len(image_paths),
                num_gt_boxes=0,
                num_detections=len(detections_np),
                num_matched=0,
                avg_iou=0.0,
                image_path=image_path,
                detection_only=True
            )
        else:
            # Comparison mode: match detections to ground truth
            matched_indices, matched_ious = metrics.match_detections_to_ground_truth(
                gt_boxes, detections_np, iou_threshold=0.3
            )

            # Calculate stats
            num_matched = sum(1 for idx in matched_indices if idx != -1)
            avg_iou = float(np.mean([iou for iou in matched_ious if iou > 0])) if num_matched > 0 else 0.0

            # Draw ground truth boxes (blue) with IoU values
            drawing.draw_ground_truth_boxes(
                display_img, gt_boxes, ious=matched_ious,
                color=(255, 0, 0), thickness=2, label="GT"
            )

            # Draw only matched detections (green)
            if len(detections_np) > 0:
                drawing.draw_detections(
                    display_img, detections_np, indices=matched_indices,
                    color=(0, 255, 0), thickness=2, label="Det"
                )

            # Draw info text (comparison mode)
            drawing.draw_info_text(
                display_img,
                current_idx,
                len(image_paths),
                len(gt_boxes),
                len(detections_np),
                num_matched,
                avg_iou,
                image_path,
                detection_only=False
            )

        # Display
        cv2.imshow(WINDOW, display_img)

        # Handle keyboard input
        key = cv2.waitKey(0) & 0xFF

        if key == 27 or key == ord('q'):  # ESC or 'q'
            break
        elif key == ord('a') or key == 81:  # 'a' or Left arrow
            current_idx = (current_idx - 1) % len(image_paths)
            print(f"Previous: {current_idx + 1}/{len(image_paths)}")
        elif key == ord('d') or key == 83:  # 'd' or Right arrow
            current_idx = (current_idx + 1) % len(image_paths)
            print(f"Next: {current_idx + 1}/{len(image_paths)}")

    cv2.destroyAllWindows()
    sys.exit(0)
