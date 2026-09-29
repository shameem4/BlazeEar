"""
Image augmentation utilities for training.

Geometric and occlusion augmentations run at the image's native resolution,
before the letterbox resize to the model input size. Occlusion sizes are
therefore expressed as fractions of the image rather than absolute pixels, so
they mean the same thing regardless of source resolution.

Occlusion augmentations never remove a labelled ear. An occlusion that would
cover more than `max_ear_coverage` of any labelled box is rejected rather than
applied, because covering an ear while keeping its positive label trains the
classifier to fire on flat colour patches.
"""
import cv2
import numpy as np

# An occlusion may hide at most this fraction of any labelled ear.
DEFAULT_MAX_EAR_COVERAGE = 0.5

# How many random placements to try before giving up on an occlusion.
_PLACEMENT_ATTEMPTS = 8


class _EarCoverage:
    """
    Tracks how much of each labelled ear the occlusions applied so far cover.

    Coverage per rectangle is accumulated independently, which over-counts when
    two rectangles overlap inside the same box. That errs towards rejecting an
    occlusion, which is the safe direction: the failure we are preventing is an
    ear that is fully hidden but still labelled positive.
    """

    def __init__(self, bboxes, image_h, image_w, max_coverage=DEFAULT_MAX_EAR_COVERAGE):
        self.max_coverage = float(max_coverage)
        if bboxes is None or len(bboxes) == 0:
            self.boxes = np.zeros((0, 4), dtype=np.float32)
            self.areas = np.zeros((0,), dtype=np.float32)
            self.covered = np.zeros((0,), dtype=np.float32)
            return

        boxes = np.asarray(bboxes, dtype=np.float32)[:, :4].copy()
        boxes[:, [0, 2]] *= image_h
        boxes[:, [1, 3]] *= image_w
        self.boxes = boxes
        self.areas = np.clip(
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), 1e-6, None
        )
        self.covered = np.zeros(len(boxes), dtype=np.float32)

    def _fractions(self, y1, x1, y2, x2):
        """Fraction of each labelled box covered by this pixel rectangle."""
        if len(self.boxes) == 0:
            return np.zeros((0,), dtype=np.float32)
        inter_h = np.clip(
            np.minimum(self.boxes[:, 2], y2) - np.maximum(self.boxes[:, 0], y1), 0, None
        )
        inter_w = np.clip(
            np.minimum(self.boxes[:, 3], x2) - np.maximum(self.boxes[:, 1], x1), 0, None
        )
        return (inter_h * inter_w) / self.areas

    def would_exceed(self, y1, x1, y2, x2) -> bool:
        """True when applying this rectangle would over-occlude a labelled ear."""
        if len(self.boxes) == 0:
            return False
        return bool(np.any(self.covered + self._fractions(y1, x1, y2, x2) > self.max_coverage))

    def commit(self, y1, x1, y2, x2) -> None:
        if len(self.boxes) > 0:
            self.covered += self._fractions(y1, x1, y2, x2)


def _random_rect(h, w, size_frac_range, rng=np.random):
    """A random axis-aligned rectangle sized as a fraction of the image."""
    frac_h = rng.uniform(*size_frac_range)
    frac_w = rng.uniform(*size_frac_range)
    rect_h = max(1, int(round(h * frac_h)))
    rect_w = max(1, int(round(w * frac_w)))
    y1 = rng.randint(0, max(1, h - rect_h))
    x1 = rng.randint(0, max(1, w - rect_w))
    return y1, x1, min(h, y1 + rect_h), min(w, x1 + rect_w)


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
    # np.fliplr returns a negative-stride view. Later augmentations write into
    # the array and hand it to OpenCV, so materialise a real contiguous copy.
    image = np.ascontiguousarray(np.fliplr(image))

    # Flip x coordinates
    bboxes = bboxes.copy()
    xmin_old = bboxes[:, 1].copy()
    xmax_old = bboxes[:, 3].copy()
    bboxes[:, 1] = 1.0 - xmax_old
    bboxes[:, 3] = 1.0 - xmin_old

    return image, bboxes


def augment_synthetic_occlusion(
    image: np.ndarray,
    bboxes: np.ndarray | None = None,
    num_occlusions: int = 1,
    size_frac_range: tuple[float, float] = (0.04, 0.18),
    max_ear_coverage: float = DEFAULT_MAX_EAR_COVERAGE
) -> np.ndarray:
    """Add synthetic occlusions (black rectangles) to image.

    Args:
        image: RGB image
        bboxes: Labelled boxes [ymin, xmin, ymax, xmax] normalized, used to keep
                occlusions from hiding a labelled ear
        num_occlusions: Number of occlusions to add
        size_frac_range: (min, max) occlusion size as a fraction of the image
        max_ear_coverage: Maximum fraction of any labelled ear that may be hidden

    Returns:
        Augmented image
    """
    h, w = image.shape[:2]
    image = np.ascontiguousarray(image)
    coverage = _EarCoverage(bboxes, h, w, max_ear_coverage)

    for _ in range(num_occlusions):
        for _ in range(_PLACEMENT_ATTEMPTS):
            y1, x1, y2, x2 = _random_rect(h, w, size_frac_range)
            if coverage.would_exceed(y1, x1, y2, x2):
                continue
            image[y1:y2, x1:x2] = 0
            coverage.commit(y1, x1, y2, x2)
            break

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
    bboxes: np.ndarray | None = None,
    num_holes: int = 1,
    hole_size_frac_range: tuple[float, float] = (0.06, 0.20),
    fill_value: int = 128,
    max_ear_coverage: float = DEFAULT_MAX_EAR_COVERAGE
) -> np.ndarray:
    """Apply cutout/random erasing augmentation.

    Args:
        image: RGB image
        bboxes: Labelled boxes [ymin, xmin, ymax, xmax] normalized, used to keep
                holes from hiding a labelled ear
        num_holes: Number of cutout holes
        hole_size_frac_range: (min, max) hole size as a fraction of the image
        fill_value: Fill value for cutout (0=black, 128=gray)
        max_ear_coverage: Maximum fraction of any labelled ear that may be hidden

    Returns:
        Augmented image
    """
    h, w = image.shape[:2]
    image = image.copy()
    coverage = _EarCoverage(bboxes, h, w, max_ear_coverage)

    for _ in range(num_holes):
        for _ in range(_PLACEMENT_ATTEMPTS):
            y1, x1, y2, x2 = _random_rect(h, w, hole_size_frac_range)
            if coverage.would_exceed(y1, x1, y2, x2):
                continue
            image[y1:y2, x1:x2] = fill_value
            coverage.commit(y1, x1, y2, x2)
            break

    return image


def augment_face_cutout(
    image: np.ndarray,
    bboxes: np.ndarray,
    context_scale_range: tuple[float, float] = (1.3, 1.9),
    occlusion_probability: float = 0.6,
    max_regions: int = 2,
    max_ear_coverage: float = DEFAULT_MAX_EAR_COVERAGE
) -> np.ndarray:
    """Occlude part of the region around an ear, to emulate facial occlusions.

    This used to fill the whole expanded box with a solid colour, which erased
    the ear while its positive label was kept, teaching the classifier that a
    flat colour patch is an ear. The occluder is now a sub-rectangle of the
    expanded region, placed so it hides at most `max_ear_coverage` of any
    labelled ear; if no such placement is found the region is skipped.

    Args:
        image: RGB image
        bboxes: Labelled boxes [ymin, xmin, ymax, xmax] normalized
        context_scale_range: How far around the ear the occluder may reach
        occlusion_probability: Chance of occluding each selected region
        max_regions: Maximum number of regions to occlude
        max_ear_coverage: Maximum fraction of any labelled ear that may be hidden
    """
    if len(bboxes) == 0:
        return image

    h, w = image.shape[:2]
    output = image.copy()
    regions = min(max_regions, len(bboxes))
    if regions <= 0:
        return image
    selected = np.random.choice(len(bboxes), size=regions, replace=False)
    coverage = _EarCoverage(bboxes, h, w, max_ear_coverage)

    for idx in selected:
        if np.random.random() > occlusion_probability:
            continue
        ymin, xmin, ymax, xmax = bboxes[idx]
        cy = (ymin + ymax) * 0.5 * h
        cx = (xmin + xmax) * 0.5 * w
        box_h = max(1.0, (ymax - ymin) * h)
        box_w = max(1.0, (xmax - xmin) * w)
        scale = np.random.uniform(*context_scale_range)

        region_y1 = int(np.clip(cy - box_h * scale * 0.5, 0, h))
        region_y2 = int(np.clip(cy + box_h * scale * 0.5, 0, h))
        region_x1 = int(np.clip(cx - box_w * scale * 0.5, 0, w))
        region_x2 = int(np.clip(cx + box_w * scale * 0.5, 0, w))
        if region_y2 <= region_y1 or region_x2 <= region_x1:
            continue

        # Try random sub-rectangles of the expanded region until one leaves
        # enough of every labelled ear visible.
        region_h = region_y2 - region_y1
        region_w = region_x2 - region_x1
        for _ in range(_PLACEMENT_ATTEMPTS):
            occ_h = max(1, int(region_h * np.random.uniform(0.3, 0.9)))
            occ_w = max(1, int(region_w * np.random.uniform(0.3, 0.9)))
            y1 = np.random.randint(region_y1, max(region_y1 + 1, region_y2 - occ_h + 1))
            x1 = np.random.randint(region_x1, max(region_x1 + 1, region_x2 - occ_w + 1))
            y2 = min(h, y1 + occ_h)
            x2 = min(w, x1 + occ_w)
            if coverage.would_exceed(y1, x1, y2, x2):
                continue
            fill_color = np.random.randint(0, 256, size=(1, 1, 3), dtype=np.uint8)
            output[y1:y2, x1:x2] = fill_color
            coverage.commit(y1, x1, y2, x2)
            break

    return output


def augment_targeted_ear_occlusion(
    image: np.ndarray,
    bboxes: np.ndarray,
    occlusion_fraction: tuple[float, float] = (0.25, 0.6),
    max_regions: int = 3,
    max_ear_coverage: float = DEFAULT_MAX_EAR_COVERAGE
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
    coverage = _EarCoverage(bboxes, h, w, max_ear_coverage)

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
        for _ in range(_PLACEMENT_ATTEMPTS):
            start_y = np.random.randint(y1, max_y)
            start_x = np.random.randint(x1, max_x)
            end_y = min(h, start_y + occ_h)
            end_x = min(w, start_x + occ_w)
            if coverage.would_exceed(start_y, start_x, end_y, end_x):
                continue
            fill_color = np.random.randint(0, 80, size=(1, 1, 3), dtype=np.uint8)
            output[start_y:end_y, start_x:end_x] = fill_color
            coverage.commit(start_y, start_x, end_y, end_x)
            break

    return output
