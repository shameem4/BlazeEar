#!/usr/bin/env python3
"""Do the face stage's predicted ear tragions land on the ears that escape
every crop window?

The README records 8.7% of ears falling outside every `FACE_CROP_EXPAND`
window as the larger share of the recall ceiling, and that widening the crop
is measured dead at every setting -- so a different crop PLACEMENT is the only
lever left for them. The face stage already predicts both ear tragions, which
makes a tragion-centred second crop the obvious proposal.

It does not work, and this is what rules it out: the tragion lands on the
reachable ears and misses the escaping ones, because the two failure modes are
correlated. The geometry that puts an ear beyond the crop window is the
geometry that occludes or extrapolates that side's tragion.

    python measure_tragion_coverage.py

Reads a split's human-annotated ear boxes, runs the shipped face stage exactly
as `make_face_crops` does, and reports the distance from the nearer predicted
tragion to each ear centre, split by whether the ear is reachable at all. The
reachability counts it prints double as a check on the harness: they reproduce
the README's 315 / 31 / 9 table on `common_heldout`.

Keypoints are present because the model emits 16 coords per anchor (4 box plus
6x2 keypoints) and `load_mediapipe_weights` splits MediaPipe's combined
regressor into the box and keypoint heads. `BlazeEarInference.decode_boxes`
reads `raw_boxes[..., 0:4]` and drops 4:16, which is the single place they are
lost. Note that `blazeear.py` calls the keypoints untrained: true of the ear
model, which has no keypoint labels, and NOT of the face stage, which is
MediaPipe's weights unchanged.
"""
import argparse
import csv
import os
import pathlib

import cv2
import numpy as np
import torch

from make_face_crops import crop_window, load_face_detector
from utils.anchor_utils import generate_reference_anchors
from utils.config import (FACE_CROP_EXPAND, FACE_CROP_MAX_FACES,
                          FACE_INPUT_SIZE, HUMAN_ANNOTATION_SOURCES)
from utils.preprocess import resize_pad

# resize_pad fits the frame into this square before resizing to the model's
# input, and denormalize_detections assumes the same constant.
INTERMEDIATE_SIZE = 256
# MediaPipe BlazeFace keypoint order: 0 right eye, 1 left eye, 2 nose,
# 3 mouth, 4 right ear tragion, 5 left ear tragion.
TRAGION_KEYPOINTS = (4, 5)


def iou_to(box, others):
    ix0 = np.maximum(box[0], others[:, 0]); iy0 = np.maximum(box[1], others[:, 1])
    ix1 = np.minimum(box[2], others[:, 2]); iy1 = np.minimum(box[3], others[:, 3])
    inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
    union = ((box[2] - box[0]) * (box[3] - box[1])
             + (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
             - inter)
    return inter / np.maximum(union, 1e-12)


def ears_by_image(csv_path):
    """Human-annotated ear boxes as (centre_x, centre_y, size), per image."""
    ears = {}
    for row in csv.DictReader(open(csv_path)):
        if row['annotation_source'] not in HUMAN_ANNOTATION_SOURCES:
            continue
        x, y, w, h = (float(row[k]) for k in ('x1', 'y1', 'w', 'h'))
        if w <= 0 or h <= 0:
            continue
        ears.setdefault(row['image_path'], []).append((x + w / 2, y + h / 2,
                                                       max(w, h)))
    return ears


def tragions_and_windows(detector, anchors, image, rgb, expand, max_faces):
    """Crop windows, and the tragion keypoints of every detected face.

    `process` returns boxes and scores only, so the keypoints are read from a
    second pass over the raw output and matched back to each detection by the
    anchor whose decoded box fits it best.
    """
    with torch.no_grad():
        faces = detector.process(rgb).cpu().numpy()
    if not len(faces):
        return [], []
    faces = faces[:max_faces]

    _, img_out, scale, pad = resize_pad(rgb, output_size=FACE_INPUT_SIZE,
                                        intermediate_size=INTERMEDIATE_SIZE)
    tensor = (torch.from_numpy(img_out.copy()).permute(2, 0, 1)
              .float().div(127.5).sub(1.0).unsqueeze(0))
    with torch.no_grad():
        raw, _ = detector(tensor.to(anchors.device))
    raw = raw[0].cpu().numpy()
    priors = anchors.cpu().numpy()

    def to_pixels(norm_x, norm_y):
        # denormalize_detections' convention: pad is (pad_y, pad_x), already
        # in original-image pixels.
        return (norm_x * scale * INTERMEDIATE_SIZE - pad[1],
                norm_y * scale * INTERMEDIATE_SIZE - pad[0])

    cx = raw[:, 0] / FACE_INPUT_SIZE * priors[:, 2] + priors[:, 0]
    cy = raw[:, 1] / FACE_INPUT_SIZE * priors[:, 3] + priors[:, 1]
    bw = raw[:, 2] / FACE_INPUT_SIZE * priors[:, 2]
    bh = raw[:, 3] / FACE_INPUT_SIZE * priors[:, 3]
    lo = to_pixels(cx - bw / 2, cy - bh / 2)
    hi = to_pixels(cx + bw / 2, cy + bh / 2)
    decoded = np.stack([lo[0], lo[1], hi[0], hi[1]], 1)

    windows, tragions = [], []
    for face in faces:
        windows.append(crop_window(face, expand, image.shape))
        # process() emits [ymin, xmin, ymax, xmax]; decoded is xyxy.
        as_xyxy = np.array([face[1], face[0], face[3], face[2]])
        best = int(np.argmax(iou_to(as_xyxy, decoded)))
        for j in TRAGION_KEYPOINTS:
            kx = raw[best, 4 + 2 * j] / FACE_INPUT_SIZE * priors[best, 2] + priors[best, 0]
            ky = raw[best, 5 + 2 * j] / FACE_INPUT_SIZE * priors[best, 3] + priors[best, 1]
            tragions.append(to_pixels(kx, ky))
    return windows, tragions


def measure(args):
    detector = load_face_detector(device=args.device)
    anchors = generate_reference_anchors(
        input_size=FACE_INPUT_SIZE, fixed_anchor_size=True)[0].to(args.device)

    escaping, reachable, no_face = [], [], 0
    for rel, ears in ears_by_image(args.csv).items():
        image = cv2.imread(os.path.join(args.data_root, rel))
        if image is None:
            continue
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        windows, tragions = tragions_and_windows(
            detector, anchors, image, rgb, args.expand, args.max_faces)
        if not windows:
            no_face += len(ears)
            continue
        for ex, ey, size in ears:
            inside = any(x0 <= ex < x0 + side and y0 <= ey < y0 + side
                         for x0, y0, side in windows)
            nearest = min(float(np.hypot(tx - ex, ty - ey))
                          for tx, ty in tragions)
            (reachable if inside else escaping).append(nearest / max(size, 1e-6))
    return np.array(escaping), np.array(reachable), no_face


def report(escaping, reachable, no_face):
    total = len(escaping) + len(reachable) + no_face
    if total == 0:
        print('no ears measured')
        return
    print(f'{total} human ear boxes\n')
    print(f'  reachable (centre inside some crop window) : '
          f'{len(reachable):4d}  {100 * len(reachable) / total:5.1f}%')
    print(f'  escaping every crop window                 : '
          f'{len(escaping):4d}  {100 * len(escaping) / total:5.1f}%')
    print(f'  no face detected at all                    : '
          f'{no_face:4d}  {100 * no_face / total:5.1f}%')
    print('\ndistance from the nearer predicted tragion to the ear centre, '
          'in ear-sizes:')
    print(f"  {'group':>10s} {'n':>5s} {'p25':>6s} {'median':>7s} {'p75':>6s} "
          f"{'<=0.5x':>7s} {'<=1.0x':>7s} {'<=2.0x':>7s}")
    for label, values in (('reachable', reachable), ('escaping', escaping)):
        if not len(values):
            continue
        print(f"  {label:>10s} {len(values):5d} "
              f"{np.percentile(values, 25):6.2f} {np.median(values):7.2f} "
              f"{np.percentile(values, 75):6.2f} "
              f"{100 * (values <= 0.5).mean():6.1f}% "
              f"{100 * (values <= 1.0).mean():6.1f}% "
              f"{100 * (values <= 2.0).mean():6.1f}%")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--csv', default='data/splits/common_heldout.csv',
                        help='split whose human ear boxes to measure against')
    parser.add_argument('--data-root', default='data/raw')
    parser.add_argument('--expand', type=float, default=FACE_CROP_EXPAND,
                        help='crop side in multiples of the face box')
    parser.add_argument('--max-faces', type=int, default=FACE_CROP_MAX_FACES)
    parser.add_argument('--device', default='cuda')
    return parser


def main():
    args = build_parser().parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
    os.chdir(pathlib.Path(__file__).resolve().parent)
    report(*measure(args))


if __name__ == '__main__':
    main()
