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
regressor into the box and keypoint heads. They reach this script through
`blazedetector.emit_keypoints`, which is off by default -- the ear model has
no keypoint labels, so for it the columns are noise, while the face stage runs
MediaPipe's weights unchanged and its six are trained. `blazeear.py` calls
them untrained, which is true of the one and not the other.
"""
import argparse
import csv
import os
import pathlib

import cv2
import numpy as np
import torch

from make_face_crops import crop_window, load_face_detector
from utils.config import (FACE_CROP_EXPAND, FACE_CROP_MAX_FACES,
                          HUMAN_ANNOTATION_SOURCES)

# MediaPipe BlazeFace keypoint order: 0 right eye, 1 left eye, 2 nose,
# 3 mouth, 4 right ear tragion, 5 left ear tragion.
TRAGION_KEYPOINTS = (4, 5)


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


def tragions_and_windows(detector, image, rgb, expand, max_faces):
    """Crop windows, and the tragion keypoints of every detected face.

    `blazedetector.emit_keypoints` widens `process`'s output from
    [box, score] to [box, 6 keypoints, score], already denormalised to
    original-image pixels, so the keypoints come straight off the detection
    they belong to.

    This used to take a second raw pass over the model and match each
    keypoint row back to its detection by best-IoU, because `process`
    discarded the keypoint columns. That is what `emit_keypoints` replaced;
    both give the same numbers, and the flag removes the matching step, which
    was the part that could silently pair a face with another face's
    keypoints.
    """
    detector.emit_keypoints = True
    try:
        with torch.no_grad():
            faces = detector.process(rgb).cpu().numpy()
    finally:
        detector.emit_keypoints = False
    if not len(faces):
        return [], []

    windows, tragions = [], []
    for face in faces[:max_faces]:
        windows.append(crop_window(face, expand, image.shape))
        for j in TRAGION_KEYPOINTS:
            tragions.append((float(face[4 + 2 * j]), float(face[5 + 2 * j])))
    return windows, tragions


def measure(args):
    detector = load_face_detector(device=args.device)

    escaping, reachable, no_face = [], [], 0
    for rel, ears in ears_by_image(args.csv).items():
        image = cv2.imread(os.path.join(args.data_root, rel))
        if image is None:
            continue
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        windows, tragions = tragions_and_windows(
            detector, image, rgb, args.expand, args.max_faces)
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
