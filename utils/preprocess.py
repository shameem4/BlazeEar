"""
The one image preprocessing implementation.

Every inference path must resize and pad identically, and identically to
training, or the model sees a different pixel distribution than it was fitted
on. There were three implementations: `cv2.resize` in `BlazeDetector.resize_pad`
and in the training dataloader, `F.interpolate(mode="bilinear")` in
`BlazeEarInference.preprocess`, and canvas `drawImage` in the browser demo.

The measured skew between the cv2 and torch paths was 0.0072 max (0.0013 mean)
per pixel on a [-1, 1] scale, moving raw outputs by 0.003. Small, but it is a
train/serve difference that nothing would ever have surfaced. cv2 is the
reference because it is what the dataloader uses, so it is what the weights
were fitted against.

The browser cannot call cv2, so `docs/blazeear_inference.js` necessarily
approximates this with canvas resampling; that residual is unavoidable without
moving preprocessing into the exported graph.
"""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def resize_pad(
    image: np.ndarray,
    output_size: int = 128,
    intermediate_size: int = 256
) -> Tuple[np.ndarray, np.ndarray, float, Tuple[int, int]]:
    """
    Aspect-preserving resize and centre pad, matching MediaPipe's convention.

    The image is fitted into `intermediate_size` square with zero padding, then
    resized to `output_size`. The two-step path is kept because it is what the
    detector has always done and what the coordinate denormalization assumes.

    Args:
        image: (H, W, 3) uint8

    Returns:
        img_intermediate: (intermediate_size, intermediate_size, 3)
        img_output: (output_size, output_size, 3)
        scale: multiply a normalized coordinate by `scale * intermediate_size`
               to reach original image pixels
        pad: (pad_y, pad_x) already expressed in original image pixels
    """
    height, width = image.shape[:2]

    if height >= width:
        resized_h = intermediate_size
        resized_w = intermediate_size * width // height
        pad_h, pad_w = 0, intermediate_size - resized_w
        scale = width / resized_w
    else:
        resized_h = intermediate_size * height // width
        resized_w = intermediate_size
        pad_h, pad_w = intermediate_size - resized_h, 0
        scale = height / resized_h

    pad_top = pad_h // 2
    pad_bottom = pad_h // 2 + pad_h % 2
    pad_left = pad_w // 2
    pad_right = pad_w // 2 + pad_w % 2

    img_intermediate = cv2.resize(image, (resized_w, resized_h))
    img_intermediate = np.pad(
        img_intermediate, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
    )
    img_output = cv2.resize(img_intermediate, (output_size, output_size))

    pad = (int(pad_top * scale), int(pad_left * scale))
    return img_intermediate, img_output, scale, pad
