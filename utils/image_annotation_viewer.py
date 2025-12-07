import os
import sys
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Allow running as `python utils/image_annotation_viewer.py` from repo root
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Import unified decoder and image drawing utility
from data_decoder import find_annotation, decode_annotation, get_annotation_color


def visualize_annotations(image_path, annotations, bbox_color='red', keypoint_color='blue',
                          bbox_width=2, keypoint_radius=3, figsize=(10, 10)):
    """
    Visualize annotations (bounding boxes and keypoints) on an image.

    Args:
        image_path: Path to the image file
        annotations: List of annotation dicts with 'bbox' and/or 'keypoints' keys
                    bbox format: [x, y, width, height]
                    keypoints format: [x1, y1, v1, x2, y2, v2, ...] (COCO format)
        bbox_color: Color for bounding boxes (default: 'red')
        keypoint_color: Color for keypoints (default: 'blue')
        bbox_width: Line width for bounding boxes (default: 2)
        keypoint_radius: Radius for keypoint markers (default: 3)
        figsize: Figure size tuple (default: (10, 10))
    """
    # Load image
    img = Image.open(image_path)
    
    # Create figure and axis
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(img)
    
    # Draw each annotation
    for ann in annotations:
        # Draw bounding box if present
        bbox = ann.get('bbox')
        if bbox is not None:
            x, y, w, h = bbox
            rect = patches.Rectangle(
                (x, y), w, h,
                linewidth=bbox_width,
                edgecolor=bbox_color,
                facecolor='none'
            )
            ax.add_patch(rect)
        
        # Draw keypoints if present
        keypoints = ann.get('keypoints')
        if keypoints is not None:
            # COCO format: [x1, y1, v1, x2, y2, v2, ...]
            # v = visibility (0: not labeled, 1: labeled but not visible, 2: labeled and visible)
            for i in range(0, len(keypoints), 3):
                kp_x = keypoints[i]
                kp_y = keypoints[i + 1]
                visibility = keypoints[i + 2] if i + 2 < len(keypoints) else 2
                
                # Only draw visible keypoints (visibility > 0)
                if visibility > 0:
                    circle = patches.Circle(
                        (kp_x, kp_y),
                        radius=keypoint_radius,
                        color=keypoint_color,
                        fill=True
                    )
                    ax.add_patch(circle)
    
    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Set title
    ax.set_title(os.path.basename(image_path))
    
    plt.tight_layout()
    plt.show()






# Draw annotations on image (COCO, CSV, or PTS)
def draw_annotations(image_path, annotation_path, annotation_type):
    # Use unified decoder
    decoded = decode_annotation(annotation_path, image_path, annotation_type)
    bbox_color = get_annotation_color(annotation_type)

    if not decoded:
        print('Image not found in annotation file.')
        return

    # Use annotation drawer utility to visualize
    visualize_annotations(
        image_path,
        decoded,
        bbox_color=bbox_color,
        keypoint_color='blue',
        bbox_width=2,
        keypoint_radius=2,
        figsize=(8, 8)
    )


def main():
    root = tk.Tk()
    root.withdraw()
    image_path = filedialog.askopenfilename(title='Select image file', filetypes=[('Image Files', '*.jpg *.jpeg *.png')])
    if not image_path:
        print('No image selected.')
        return
    annotation_path, annotation_type = find_annotation(image_path)
    print(annotation_path,"\n", annotation_type,"\n", image_path,"\n")
    if not annotation_path:
        print('No annotation file found (COCO, CSV, or PTS).')
        return
    draw_annotations(image_path, annotation_path, annotation_type)

if __name__ == '__main__':
    main()
