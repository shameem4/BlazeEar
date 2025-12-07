"""
Unified decoder that automatically selects the appropriate annotation decoder
based on available annotation files.
"""
import json
import csv
import os
from collections.abc import MutableMapping
from typing import Any, Dict


def _read_xy_pairs_from_txt(txt_path):
    """Parse whitespace or comma separated XY pairs from a text file."""
    coords = []
    try:
        with open(txt_path, 'r') as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith('#'):
                    continue
                # Allow either whitespace or comma separated values
                line = line.replace(',', ' ')
                parts = [p for p in line.split() if p]
                if len(parts) < 2:
                    continue
                try:
                    x_val = float(parts[0])
                    y_val = float(parts[1])
                except ValueError:
                    # Non numeric token indicates this is not a coordinate file
                    return []
                coords.append((x_val, y_val))
    except OSError:
        return []

    return coords

def find_coco_annotation(image_path):
    folder = os.path.dirname(image_path)
    for file in os.listdir(folder):
        if file.endswith('_annotations.coco.json'):
            return os.path.join(folder, file)
    return None

def decode_coco_annotation(annotation_path, image_filename):
    with open(annotation_path, 'r') as f:
        annotation = json.load(f)
    img_id = None
    for img in annotation.get('images', []):
        if os.path.basename(img['file_name']) == os.path.basename(image_filename):
            img_id = img['id']
            break
    if img_id is None:
        return []
    decoded = []
    for ann in annotation.get('annotations', []):
        if ann['image_id'] == img_id:
            item = {'bbox': ann.get('bbox'), 'keypoints': ann.get('keypoints')}
            decoded.append(item)
    return decoded

def find_csv_annotation(image_path):
    folder = os.path.dirname(image_path)
    for file in os.listdir(folder):
        if file.endswith('_annotations.csv'):
            return os.path.join(folder, file)
    return None

def decode_csv_annotation(annotation_path, image_filename):
    decoded = []
    with open(annotation_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Support both 'filename' and 'image_path' columns
            csv_filename = row.get('filename') or row.get('image_path')
            if csv_filename and os.path.basename(csv_filename) == os.path.basename(image_filename):
                try:
                    # Support both (x, y, w, h) and (xmin, ymin, xmax, ymax) formats
                    if 'xmin' in row and 'ymin' in row and 'xmax' in row and 'ymax' in row:
                        xmin, ymin, xmax, ymax = float(row['xmin']), float(row['ymin']), float(row['xmax']), float(row['ymax'])
                        bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
                    elif 'x' in row and 'y' in row and 'w' in row and 'h' in row:
                        bbox = [float(row['x']), float(row['y']), float(row['w']), float(row['h'])]
                    else:
                        continue
                    decoded.append({'bbox': bbox})
                except Exception:
                    continue
    return decoded


def find_pts_annotation(image_path):
    """Find corresponding .pts file for an image."""
    base_path = os.path.splitext(image_path)[0]
    pts_path = base_path + '.pts'
    if os.path.exists(pts_path):
        return pts_path
    return None


def find_lfpw_txt_annotation(image_path):
    """Find Dataset LFPW style .txt annotation for an image."""
    base_path = os.path.splitext(image_path)[0]
    txt_path = base_path + '.txt'
    if os.path.exists(txt_path):
        coords = _read_xy_pairs_from_txt(txt_path)
        if len(coords) >= 2:
            return txt_path
    return None

def decode_pts_annotation(annotation_path, image_filename):
    """
    Decode .pts annotation file.

    PTS file format:
    version: 1
    n_points: N
    {
    x1 y1
    x2 y2
    ...
    }

    Returns list with single dict containing keypoints in COCO format [x, y, visibility, ...]
    """
    decoded = []
    keypoints = []

    try:
        with open(annotation_path, 'r') as f:
            lines = f.readlines()

        in_points = False
        for line in lines:
            line = line.strip()

            if line == '{':
                in_points = True
                continue
            elif line == '}':
                in_points = False
                break

            if in_points:
                parts = line.split()
                if len(parts) == 2:
                    x, y = float(parts[0]), float(parts[1])
                    # COCO format: [x, y, visibility, ...]
                    # visibility: 0=not labeled, 1=labeled but not visible, 2=labeled and visible
                    keypoints.extend([x, y, 2])

        if keypoints:
            decoded.append({'keypoints': keypoints, 'bbox': None})

    except Exception as e:
        print(f"Error decoding PTS file: {e}")
        return []

    return decoded


def decode_lfpw_txt_annotation(annotation_path, image_filename):
    """Decode Dataset LFPW bounding box stored as XY pairs in a .txt file."""
    coords = _read_xy_pairs_from_txt(annotation_path)
    if len(coords) < 2:
        return []

    xs = [pt[0] for pt in coords]
    ys = [pt[1] for pt in coords]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    bbox = [xmin, ymin, xmax - xmin, ymax - ymin]

    return [{'bbox': bbox}]



def find_annotation(image_path):
    """
    Find corresponding annotation file for an image.
    Checks for COCO, CSV, PTS, and Dataset LFPW TXT formats in order.

    Args:
        image_path: Path to the image file

    Returns:
        tuple: (annotation_path, annotation_type) or (None, None) if not found
               annotation_type can be 'coco', 'csv', 'pts', or 'lfpw_txt'
    """
    coco_path = find_coco_annotation(image_path)
    csv_path = find_csv_annotation(image_path)
    pts_path = find_pts_annotation(image_path)
    lfpw_txt_path = find_lfpw_txt_annotation(image_path)

    if coco_path:
        return coco_path, 'coco'
    elif csv_path:
        return csv_path, 'csv'
    elif pts_path:
        return pts_path, 'pts'
    elif lfpw_txt_path:
        return lfpw_txt_path, 'lfpw_txt'

    return None, None


def decode_annotation(annotation_path, image_path, annotation_type):
    """
    Decode annotation file using the appropriate decoder.

    Args:
        annotation_path: Path to the annotation file
        image_path: Path to the image file
        annotation_type: Type of annotation ('coco', 'csv', 'pts', or 'lfpw_txt')

    Returns:
        list: List of annotation dictionaries with 'bbox' and/or 'keypoints' keys
              Returns empty list if decoding fails or image not found
    """
    if annotation_type == 'coco':
        return decode_coco_annotation(annotation_path, image_path)
    elif annotation_type == 'csv':
        return decode_csv_annotation(annotation_path, image_path)
    elif annotation_type == 'pts':
        return decode_pts_annotation(annotation_path, image_path)
    elif annotation_type == 'lfpw_txt':
        return decode_lfpw_txt_annotation(annotation_path, image_path)
    else:
        print(f"Unknown annotation type: {annotation_type}")
        return []


def get_annotation_color(annotation_type):
    """
    Get the color to use for drawing annotations based on type.

    Args:
        annotation_type: Type of annotation ('coco', 'csv', or 'pts')

    Returns:
        str: Color name for drawing
    """
    color_map = {
        'coco': 'red',
        'csv': 'green',
        'pts': 'purple',
        'lfpw_txt': 'orange'
    }
    return color_map.get(annotation_type, 'blue')


def find_all_annotations(directory):
    """
    Find all annotation files in a directory and its subdirectories.
    Also detects image-only folders (no annotations) for teacher data.

    Args:
        directory: Path to directory to search

    Returns:
        list: List of tuples (annotation_file_path, annotation_type, image_directory)
              annotation_type can be 'coco', 'csv', 'pts', or 'images_only'
    """
    from pathlib import Path

    directory = Path(directory)
    annotations = []
    annotated_dirs = set()

    # Find COCO annotation files
    for coco_file in directory.glob('**/_annotations.coco.json'):
        annotations.append((coco_file, 'coco', coco_file.parent))
        annotated_dirs.add(coco_file.parent)

    # Find CSV annotation files
    for csv_file in directory.glob('**/*_annotations.csv'):
        annotations.append((csv_file, 'csv', csv_file.parent))
        annotated_dirs.add(csv_file.parent)

    # Find PTS files (group by directory since they're per-image)
    pts_dirs = set()
    for pts_file in directory.glob('**/*.pts'):
        pts_dirs.add(pts_file.parent)

    for pts_dir in pts_dirs:
        annotations.append((None, 'pts', pts_dir))
        annotated_dirs.add(pts_dir)

    # Find Dataset LFPW style TXT annotations (per-image text files)
    txt_dirs = set()
    image_exts = ['.jpg', '.jpeg', '.png', '.bmp']
    for txt_file in directory.glob('**/*.txt'):
        # Require at least two coordinate pairs and a matching image file
        coords = _read_xy_pairs_from_txt(txt_file)
        if len(coords) < 2:
            continue
        has_image = any(txt_file.with_suffix(ext).exists() for ext in image_exts)
        if not has_image:
            continue
        txt_dirs.add(txt_file.parent)

    for txt_dir in txt_dirs:
        if txt_dir in annotated_dirs:
            continue
        annotations.append((None, 'lfpw_txt', txt_dir))
        annotated_dirs.add(txt_dir)

    # Find image-only folders (for teacher data without annotations)
    # Look for directories with images but no annotations
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    for img_file in directory.glob('**/*'):
        if img_file.suffix.lower() in image_extensions:
            img_dir = img_file.parent
            if img_dir not in annotated_dirs:
                # Check if this directory has images but no annotation files
                has_images = any(img_dir.glob('*'))
                has_coco = (img_dir / '_annotations.coco.json').exists()
                has_csv = any(img_dir.glob('*_annotations.csv'))
                has_pts = any(img_dir.glob('*.pts'))

                if has_images and not (has_coco or has_csv or has_pts):
                    annotations.append((None, 'images_only', img_dir))
                    annotated_dirs.add(img_dir)

    return annotations


def decode_all_annotations(annotation_file, annotation_type, image_dir, progress_callback=None):
    """
    Decode all annotations from a file or directory.

    Args:
        annotation_file: Path to annotation file (or None for PTS or images_only)
        annotation_type: Type of annotation ('coco', 'csv', 'pts', 'lfpw_txt', or 'images_only')
        image_dir: Directory containing images
        progress_callback: Optional callback function(current, total) for progress tracking

    Returns:
        list: List of dicts with keys:
              - 'image_path': str - Full path to image
              - 'bbox': list (for detector/teacher data)
              - 'keypoints': array (for landmarker data)

    Example:
        >>> annotations = decode_all_annotations('annotations.coco.json', 'coco', 'images/')
        >>> for ann in annotations:
        ...     print(f"Image: {ann['image_path']}, bbox: {ann.get('bbox')}")
    """
    import json
    import csv as csv_module
    from pathlib import Path

    annotation_file = Path(annotation_file) if annotation_file else None
    image_dir = Path(image_dir)
    results = []

    if annotation_type == 'coco':
        # COCO format - process all images in JSON
        if annotation_file is None:
            return results
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)

        images = coco_data['images']
        total = len(images)

        for i, img_info in enumerate(images, 1):
            if progress_callback:
                progress_callback(i, total)

            image_path = image_dir / img_info['file_name']
            if not image_path.exists():
                continue

            try:
                annotations = decode_coco_annotation(str(annotation_file), img_info['file_name'])
                if annotations:
                    first_annotation = annotations[0]
                    if isinstance(first_annotation, MutableMapping):
                        annotation: Dict[str, Any] = dict(first_annotation)
                        annotation['image_path'] = str(image_path)
                        results.append(annotation)
            except Exception:
                # Skip failed annotations silently (caller can log if needed)
                pass

    elif annotation_type == 'csv':
        # CSV format - process all unique images
        if annotation_file is None:
            return results
        with open(annotation_file, 'r') as f:
            rows = list(csv_module.DictReader(f))

        # Get unique image names
        image_groups = {}
        for row in rows:
            img_name = row.get('image_path') or row.get('filename')
            if img_name and img_name not in image_groups:
                image_groups[img_name] = True

        image_names = list(image_groups.keys())
        total = len(image_names)

        for i, img_name in enumerate(image_names, 1):
            if progress_callback:
                progress_callback(i, total)

            image_path = image_dir / img_name
            if not image_path.exists():
                continue

            try:
                annotations = decode_csv_annotation(str(annotation_file), img_name)
                if annotations:
                    first_annotation = annotations[0]
                    if isinstance(first_annotation, MutableMapping):
                        annotation: Dict[str, Any] = dict(first_annotation)
                        annotation['image_path'] = str(image_path)
                        results.append(annotation)
            except Exception:
                pass

    elif annotation_type == 'pts':
        # PTS format - process all .pts files in directory
        pts_files = list(image_dir.glob('**/*.pts'))
        total = len(pts_files)

        for i, pts_file in enumerate(pts_files, 1):
            if progress_callback:
                progress_callback(i, total)

            # Find corresponding image
            image_path = None
            for ext in ['.png', '.jpg', '.jpeg']:
                potential_path = pts_file.with_suffix(ext)
                if potential_path.exists():
                    image_path = potential_path
                    break

            if not image_path:
                continue

            try:
                annotations = decode_pts_annotation(str(pts_file), image_path.name)
                if annotations:
                    first_annotation = annotations[0]
                    if isinstance(first_annotation, MutableMapping):
                        annotation: Dict[str, Any] = dict(first_annotation)
                        annotation['image_path'] = str(image_path)
                        results.append(annotation)
            except Exception:
                pass

    elif annotation_type == 'lfpw_txt':
        # LFPW-style TXT format - per image text files with coordinate pairs
        txt_files = []
        for txt_file in image_dir.glob('**/*.txt'):
            if len(_read_xy_pairs_from_txt(txt_file)) >= 2:
                txt_files.append(txt_file)

        total = len(txt_files)
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

        for i, txt_file in enumerate(txt_files, 1):
            if progress_callback:
                progress_callback(i, total)

            image_path = None
            for ext in image_extensions:
                candidate = txt_file.with_suffix(ext)
                if candidate.exists():
                    image_path = candidate
                    break

            if not image_path:
                continue

            try:
                annotations = decode_lfpw_txt_annotation(str(txt_file), image_path.name)
                if annotations:
                    first_annotation = annotations[0]
                    if isinstance(first_annotation, MutableMapping):
                        annotation: Dict[str, Any] = dict(first_annotation)
                        annotation['image_path'] = str(image_path)
                        results.append(annotation)
            except Exception:
                pass

    elif annotation_type == 'images_only':
        # Images without annotations - return just image paths
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = []
        for ext in image_extensions:
            image_files.extend(image_dir.glob(f'*{ext}'))
            image_files.extend(image_dir.glob(f'*{ext.upper()}'))

        total = len(image_files)

        for i, image_path in enumerate(image_files, 1):
            if progress_callback:
                progress_callback(i, total)

            # Return just the image path, no annotations
            results.append({
                'image_path': str(image_path)
            })

    return results
