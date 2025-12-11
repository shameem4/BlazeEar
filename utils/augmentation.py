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


def augment_photometric_jitter(
    image: np.ndarray,
    brightness_range: tuple[float, float] = (-0.25, 0.25),
    contrast_range: tuple[float, float] = (0.75, 1.25),
    saturation_range: tuple[float, float] = (0.7, 1.3),
    hue_range: tuple[float, float] = (-0.08, 0.08),
    noise_std_range: tuple[float, float] = (0.0, 8.0)
) -> np.ndarray:
    """Comprehensive photometric jitter that mimics varied lighting.

    Randomly applies brightness/contrast shifts, hue and saturation tweaks,
    and light gaussian noise so fine-tuned weights experience failure-case
    illumination similar to the provided samples.
    """
    jittered = image.astype(np.float32)

    brightness_delta = np.random.uniform(*brightness_range) * 255
    jittered = np.clip(jittered + brightness_delta, 0, 255)

    mean = np.mean(jittered, axis=(0, 1), keepdims=True)
    contrast = np.random.uniform(*contrast_range)
    jittered = np.clip((jittered - mean) * contrast + mean, 0, 255)

    hsv = cv2.cvtColor(jittered.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hue_shift = np.random.uniform(*hue_range) * 180
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
    saturation_factor = np.random.uniform(*saturation_range)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_factor, 0, 255)
    jittered = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)

    noise_std = np.random.uniform(*noise_std_range)
    if noise_std > 0:
        noise = np.random.normal(0.0, noise_std, size=jittered.shape)
        jittered = np.clip(jittered + noise, 0, 255)

    return jittered.astype(np.uint8)


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


def augment_face_cutout(
    image: np.ndarray,
    bboxes: np.ndarray,
    context_scale_range: tuple[float, float] = (1.3, 1.9),
    drop_probability: float = 0.6,
    max_regions: int = 2
) -> np.ndarray:
    """Apply cutout over expanded bbox regions to emulate facial occlusions."""
    if len(bboxes) == 0:
        return image

    h, w = image.shape[:2]
    output = image.copy()
    regions = min(max_regions, len(bboxes))
    if regions <= 0:
        return image
    selected = np.random.choice(len(bboxes), size=regions, replace=False)

    for idx in selected:
        if np.random.random() > drop_probability:
            continue
        ymin, xmin, ymax, xmax = bboxes[idx]
        cy = (ymin + ymax) * 0.5 * h
        cx = (xmin + xmax) * 0.5 * w
        box_h = max(1.0, (ymax - ymin) * h)
        box_w = max(1.0, (xmax - xmin) * w)
        scale = np.random.uniform(*context_scale_range)
        half_h = box_h * scale * 0.5
        half_w = box_w * scale * 0.5

        y1 = int(np.clip(cy - half_h, 0, h))
        y2 = int(np.clip(cy + half_h, 0, h))
        x1 = int(np.clip(cx - half_w, 0, w))
        x2 = int(np.clip(cx + half_w, 0, w))
        if y2 <= y1 or x2 <= x1:
            continue

        fill_color = np.random.randint(0, 256, size=(1, 1, 3), dtype=np.uint8)
        output[y1:y2, x1:x2] = fill_color

    return output


def augment_targeted_ear_occlusion(
    image: np.ndarray,
    bboxes: np.ndarray,
    occlusion_fraction: tuple[float, float] = (0.25, 0.6),
    max_regions: int = 3
) -> np.ndarray:
    """Hide random slices inside ear boxes to mimic hair/hands covering the ear."""
    if len(bboxes) == 0:
        return image

    h, w = image.shape[:2]
    output = image.copy()
    regions = min(max_regions, len(bboxes))
    if regions <= 0:
        return image
    selected = np.random.choice(len(bboxes), size=regions, replace=False)

    for idx in selected:
        ymin, xmin, ymax, xmax = bboxes[idx]
        y1 = int(np.clip(ymin * h, 0, h - 1))
        y2 = int(np.clip(ymax * h, y1 + 1, h))
        x1 = int(np.clip(xmin * w, 0, w - 1))
        x2 = int(np.clip(xmax * w, x1 + 1, w))
        ear_h = max(1, y2 - y1)
        ear_w = max(1, x2 - x1)

        frac = np.random.uniform(*occlusion_fraction)
        occ_h = max(1, int(ear_h * frac))
        occ_w = max(1, int(ear_w * frac * np.random.uniform(0.4, 1.0)))
        if occ_h >= ear_h:
            occ_h = ear_h - 1 if ear_h > 1 else ear_h
        if occ_w >= ear_w:
            occ_w = ear_w - 1 if ear_w > 1 else ear_w
        if occ_h <= 0 or occ_w <= 0:
            continue

        max_y = max(y1 + 1, y2 - occ_h + 1)
        max_x = max(x1 + 1, x2 - occ_w + 1)
        start_y = np.random.randint(y1, max_y)
        start_x = np.random.randint(x1, max_x)
        fill_color = np.random.randint(0, 80, size=(1, 1, 3), dtype=np.uint8)
        output[start_y:start_y + occ_h, start_x:start_x + occ_w] = fill_color

    return output
