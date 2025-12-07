"""                                                                                                    
Webcam demo for BlazeFace detection.
Detection only - no landmarks.

Supports loading:
- MediaPipe weights: blazeface.pth (raw state_dict)
- Retrained checkpoints: *.ckpt (dict with 'model_state_dict' key)
"""
import argparse
import cv2
import torch
import numpy as np
import sys
import time
from pathlib import Path

# Allow running as `python utils/webcam_demo.py` from repo root
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils import model_utils, drawing, video_utils, config
def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Webcam demo for BlazeFace ear detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--weights", "-w",
        type=str,
        default=config.DEFAULT_BEST_CHECKPOINT,
        help="Path to weights file (.pth for MediaPipe, .ckpt for retrained). "
             "If not specified, uses model_weights/blazeface.pth"
    )
    parser.add_argument(
        "--camera", "-c",
        type=int,
        default=0,
        help="Camera device index"
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Disable mirror mode (default: mirror enabled)"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=config.DEFAULT_DETECTOR_THRESHOLD_DEBUG,
        help="Detection threshold (overrides model default)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Get the directory where this script is located for relative model paths
    SCRIPT_DIR = Path(__file__).parent

    # Setup device
    gpu = model_utils.setup_device()

    # Determine weights path
    if args.weights:
        weights_path = args.weights
    else:
        weights_path = str(SCRIPT_DIR / config.DEFAULT_WEIGHTS_PATH)

    print(f"Loading weights: {weights_path}")

    # Load detector with threshold
    detector = model_utils.load_model(weights_path, gpu, threshold=args.threshold)

    print("Model loaded")

    WINDOW = "BlazeFace Detection"
    cv2.namedWindow(WINDOW)

    # Use threaded webcam capture
    vs = video_utils.WebcamVideoStream(src=args.camera).start()
    time.sleep(1.0)  # Allow camera to warm up

    mirror_img = not args.no_mirror
    fps_counter = video_utils.FPSCounter(avg_frames=30)

    print("Press 'q' or ESC to quit")

    while True:
        frame = vs.read()
        if frame is None:
            continue

        # Convert BGR to RGB and optionally mirror
        if mirror_img:
            frame = np.ascontiguousarray(frame[:, ::-1, ::-1])
        else:
            frame = np.ascontiguousarray(frame[:, :, ::-1])

        # Run detection
        detections = detector.process(frame)

        # Draw detections
        drawing.draw_detections(frame, detections, color=(0, 255, 0), thickness=2)

        # Update and draw FPS
        fps = fps_counter.tick()
        drawing.draw_fps(frame, fps)

        # Display (convert RGB back to BGR for OpenCV)
        cv2.imshow(WINDOW, frame[:, :, ::-1])

        # Check for quit
        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            break

    vs.stop()
    cv2.destroyAllWindows()
    sys.exit(0)