"""
Image augmentation utilities for training.
"""
import cv2
import numpy as np


def augment_saturation(image: np.ndarray, factor_range: tuple[float, float] = (0.5, 1.5)) -> np.ndarray:
    """Apply random saturation adjustment.

    Args:
        image: RGB image
        factor_range: (min, max) saturation multiplication factor

    Returns:
        Augmented RGB image
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    saturation_factor = np.random.uniform(*factor_range)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def augment_brightness(image: np.ndarray, delta_range: tuple[float, float] = (-0.2, 0.2)) -> np.ndarray:
    """Apply random brightness adjustment.

    Args:
        image: RGB image
        delta_range: (min, max) brightness delta as fraction of 255

    Returns:
        Augmented RGB image
    """
    brightness_delta = np.random.uniform(*delta_range) * 255
    return np.clip(image.astype(np.float32) + brightness_delta, 0, 255).astype(np.uint8)


def augment_horizontal_flip(
    image: np.ndarray,
    bboxes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Apply horizontal flip to image and bounding boxes.

    Args:
        image: RGB image
        bboxes: Bounding boxes in [ymin, xmin, ymax, xmax] format (normalized)

    Returns:
        (flipped_image, flipped_bboxes)
    """
    image = np.fliplr(image)

    # Flip x coordinates
    xmin_old = bboxes[:, 1].copy()
    xmax_old = bboxes[:, 3].copy()
    bboxes[:, 1] = 1.0 - xmax_old
    bboxes[:, 3] = 1.0 - xmin_old

    return image, bboxes


def augment_synthetic_occlusion(
    image: np.ndarray,
    num_occlusions: int = 1,
    occlusion_size_range: tuple[int, int] = (10, 50)
) -> np.ndarray:
    """Add synthetic occlusions (black rectangles) to image.

    Args:
        image: RGB image
        num_occlusions: Number of occlusions to add
        occlusion_size_range: (min, max) size of occlusion rectangles

    Returns:
        Augmented image
    """
    h, w = image.shape[:2]

    for _ in range(num_occlusions):
        occ_h = np.random.randint(*occlusion_size_range)
        occ_w = np.random.randint(*occlusion_size_range)
        occ_y = np.random.randint(0, max(1, h - occ_h))
        occ_x = np.random.randint(0, max(1, w - occ_w))

        image[occ_y:occ_y + occ_h, occ_x:occ_x + occ_w] = 0

    return image


def augment_scale(
    image: np.ndarray,
    bboxes: np.ndarray,
    scale_range: tuple[float, float] = (0.8, 1.2)
) -> tuple[np.ndarray, np.ndarray]:
    """Apply random scale/zoom augmentation.

    Args:
        image: RGB image
        bboxes: Bounding boxes in [ymin, xmin, ymax, xmax] format (normalized)
        scale_range: (min, max) scale factor

    Returns:
        (scaled_image, scaled_bboxes)
    """
    h, w = image.shape[:2]
    scale = np.random.uniform(*scale_range)
    
    new_h, new_w = int(h * scale), int(w * scale)
    scaled = cv2.resize(image, (new_w, new_h))
    
    if scale > 1.0:
        # Crop to original size (random crop)
        crop_y = np.random.randint(0, new_h - h + 1)
        crop_x = np.random.randint(0, new_w - w + 1)
        image = scaled[crop_y:crop_y + h, crop_x:crop_x + w]
        
        # Adjust bboxes
        if len(bboxes) > 0:
            bboxes = bboxes.copy()
            # Convert to pixel coords, adjust, convert back
            bboxes[:, 0] = (bboxes[:, 0] * new_h - crop_y) / h  # ymin
            bboxes[:, 1] = (bboxes[:, 1] * new_w - crop_x) / w  # xmin
            bboxes[:, 2] = (bboxes[:, 2] * new_h - crop_y) / h  # ymax
            bboxes[:, 3] = (bboxes[:, 3] * new_w - crop_x) / w  # xmax
            bboxes = np.clip(bboxes, 0, 1)
            
            # Filter out boxes that are mostly cropped
            valid = (bboxes[:, 2] - bboxes[:, 0]) > 0.02
            valid &= (bboxes[:, 3] - bboxes[:, 1]) > 0.02
            bboxes = bboxes[valid]
    else:
        # Pad to original size
        pad_y = (h - new_h) // 2
        pad_x = (w - new_w) // 2
        image = np.zeros((h, w, 3), dtype=np.uint8)
        image[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = scaled
        
        # Adjust bboxes
        if len(bboxes) > 0:
            bboxes = bboxes.copy()
            bboxes[:, 0] = bboxes[:, 0] * scale + pad_y / h  # ymin
            bboxes[:, 1] = bboxes[:, 1] * scale + pad_x / w  # xmin
            bboxes[:, 2] = bboxes[:, 2] * scale + pad_y / h  # ymax
            bboxes[:, 3] = bboxes[:, 3] * scale + pad_x / w  # xmax
            bboxes = np.clip(bboxes, 0, 1)
    
    return image, bboxes


def augment_rotation(
    image: np.ndarray,
    bboxes: np.ndarray,
    angle_range: tuple[float, float] = (-15, 15)
) -> tuple[np.ndarray, np.ndarray]:
    """Apply random rotation augmentation.

    Args:
        image: RGB image
        bboxes: Bounding boxes in [ymin, xmin, ymax, xmax] format (normalized)
        angle_range: (min, max) rotation angle in degrees

    Returns:
        (rotated_image, rotated_bboxes)
    """
    h, w = image.shape[:2]
    angle = np.random.uniform(*angle_range)
    
    # Rotation matrix
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Rotate image
    rotated = cv2.warpAffine(image, M, (w, h), borderValue=(0, 0, 0))
    
    # Rotate bounding boxes
    if len(bboxes) > 0:
        new_bboxes = []
        for box in bboxes:
            ymin, xmin, ymax, xmax = box
            # Convert to pixel corners
            corners = np.array([
                [xmin * w, ymin * h],
                [xmax * w, ymin * h],
                [xmax * w, ymax * h],
                [xmin * w, ymax * h]
            ])
            
            # Apply rotation
            ones = np.ones((4, 1))
            corners_h = np.hstack([corners, ones])
            rotated_corners = (M @ corners_h.T).T
            
            # Get new bounding box
            new_xmin = np.min(rotated_corners[:, 0]) / w
            new_xmax = np.max(rotated_corners[:, 0]) / w
            new_ymin = np.min(rotated_corners[:, 1]) / h
            new_ymax = np.max(rotated_corners[:, 1]) / h
            
            # Clip and validate
            new_box = np.clip([new_ymin, new_xmin, new_ymax, new_xmax], 0, 1)
            if (new_box[2] - new_box[0]) > 0.02 and (new_box[3] - new_box[1]) > 0.02:
                new_bboxes.append(new_box)
        
        bboxes = np.array(new_bboxes) if new_bboxes else np.zeros((0, 4), dtype=np.float32)
    
    return rotated, bboxes


def augment_color_jitter(
    image: np.ndarray,
    hue_range: tuple[float, float] = (-0.1, 0.1),
    contrast_range: tuple[float, float] = (0.8, 1.2)
) -> np.ndarray:
    """Apply color jittering (hue shift and contrast adjustment).

    Args:
        image: RGB image
        hue_range: (min, max) hue shift as fraction of 180
        contrast_range: (min, max) contrast factor

    Returns:
        Augmented RGB image
    """
    # Hue shift
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    hue_shift = np.random.uniform(*hue_range) * 180
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
    image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    # Contrast
    contrast = np.random.uniform(*contrast_range)
    mean = np.mean(image, axis=(0, 1), keepdims=True)
    image = np.clip((image - mean) * contrast + mean, 0, 255).astype(np.uint8)
    
    return image


def augment_cutout(
    image: np.ndarray,
    num_holes: int = 1,
    hole_size_range: tuple[int, int] = (20, 60),
    fill_value: int = 128
) -> np.ndarray:
    """Apply cutout/random erasing augmentation.

    Args:
        image: RGB image
        num_holes: Number of cutout holes
        hole_size_range: (min, max) size of cutout squares
        fill_value: Fill value for cutout (0=black, 128=gray)

    Returns:
        Augmented image
    """
    h, w = image.shape[:2]
    image = image.copy()
    
    for _ in range(num_holes):
        hole_h = np.random.randint(*hole_size_range)
        hole_w = np.random.randint(*hole_size_range)
        hole_y = np.random.randint(0, max(1, h - hole_h))
        hole_x = np.random.randint(0, max(1, w - hole_w))
        
        image[hole_y:hole_y + hole_h, hole_x:hole_x + hole_w] = fill_value
    
    return image
