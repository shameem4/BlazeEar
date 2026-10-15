#!/usr/bin/env python3
"""How far would a SAME-SIZE crop have to move to catch the ears that escape?

The `FACE_CROP_EXPAND` sweep came back null at every setting, and the README
explains why: a concentric square buys periphery in every direction while
paying resolution everywhere, so coverage and resolution trade directly and
1.5 is already the optimum. That argument is specific to *scaling*. A window
of the same side, translated, pays no resolution at all -- and the sweep never
varied placement, only size, so nothing in it covers this.

    python measure_crop_shift.py

For each human ear, this measures the offset of the ear centre from its
nearest crop window's centre, in units of that window's side. 0.5 is the
window edge, so an ear at 0.62 escapes the current crop but sits inside a
same-size window shifted by 0.12 of a side. It reports what fraction of the
escaping ears each shift budget would recover, how much room the frame leaves
to shift into, and where the reachable ears already sit -- which is the check
that a shifted window lands in-distribution rather than off it.

What this does NOT answer is which way to shift. The numbers here are a
ceiling on a direction oracle, not a result. See the README's lateral-shift
entry; `measure_tragion_coverage.py` rules out the tragion as that oracle's
*position*, which is a stronger requirement than its direction.

The reachability counts printed first double as a check on the harness: they
reproduce the README's 315 / 31 / 9 split on `common_heldout`.
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

# Shift budgets to report, in units of the crop side.
SHIFT_BUDGETS = (0.1, 0.2, 0.3, 0.5)


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


def offsets_for_image(windows, ears, image_shape):
    """One record per ear: its offset from the nearest window, and the room
    the frame leaves to shift that window.

    "Nearest" is by Chebyshev offset, because a square window contains a point
    exactly when max(|dx|, |dy|) < 0.5 -- the same test `remap_boxes` makes,
    expressed in window-relative units so it is comparable across face sizes.
    """
    height, width = image_shape[:2]
    out = []
    for ear_x, ear_y, ear_size in ears:
        inside = any(x0 <= ear_x < x0 + side and y0 <= ear_y < y0 + side
                     for x0, y0, side in windows)

        def chebyshev(window):
            x0, y0, side = window
            return max(abs(ear_x - (x0 + side / 2)) / side,
                       abs(ear_y - (y0 + side / 2)) / side)

        x0, y0, side = min(windows, key=chebyshev)
        out.append({
            'inside': inside,
            'dx': abs(ear_x - (x0 + side / 2)) / side,
            'dy': abs(ear_y - (y0 + side / 2)) / side,
            'chebyshev': chebyshev((x0, y0, side)),
            # How far the window can travel before leaving the frame, taking
            # the roomier of the two directions on each axis.
            'room_x': max(x0, width - (x0 + side)) / side,
            'room_y': max(y0, height - (y0 + side)) / side,
            'ear_frac': ear_size / side,
        })
    return out


def measure(args):
    detector = load_face_detector(device=args.device)
    reachable, escaping, no_face = [], [], 0

    for rel_path, ears in ears_by_image(args.csv).items():
        image = cv2.imread(os.path.join(args.data_root, str(rel_path)))
        if image is None:
            continue
        with torch.no_grad():
            faces = detector.process(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).cpu().numpy()
        if not len(faces):
            no_face += len(ears)
            continue
        windows = [crop_window(face, args.expand, image.shape)
                   for face in faces[:args.max_faces]]
        for record in offsets_for_image(windows, ears, image.shape):
            (reachable if record['inside'] else escaping).append(record)

    return reachable, escaping, no_face


def summarise(label, rows):
    if not rows:
        print(f'\n{label}: none')
        return
    column = lambda key: np.array([r[key] for r in rows])
    dx, dy = column('dx'), column('dy')
    cheb, room_x, room_y = column('chebyshev'), column('room_x'), column('room_y')

    print(f'\n{label}  n={len(rows)}')
    print('  offset from crop centre, in crop sides (0.5 = the edge):')
    for name, values in (('|dx|', dx), ('|dy|', dy),
                         ('max(|dx|,|dy|)', cheb)):
        print(f'    {name:<16s} median {np.median(values):.3f}  '
              f'p90 {np.percentile(values, 90):.3f}  max {values.max():.3f}')
    print('  shift needed to bring the ear inside a SAME-SIZE window:')
    for budget in SHIFT_BUDGETS:
        print(f'    captured by a shift of <= {budget:.1f} sides: '
              f'{100 * ((cheb - 0.5) <= budget).mean():5.1f}%')
    print('  image room to shift (the roomier direction), in sides:')
    print(f'    x median {np.median(room_x):.3f}   '
          f'y median {np.median(room_y):.3f}')
    outside = cheb > 0.5
    if outside.any():
        feasible = outside & ((cheb - 0.5) <= np.maximum(room_x, room_y))
        print(f'    of those outside, a shift fits within the image: '
              f'{100 * feasible.sum() / outside.sum():.1f}%')
    print(f'  ear as a fraction of the crop side: '
          f'median {np.median(column("ear_frac")):.3f}')


def report(reachable, escaping, no_face):
    total = len(reachable) + len(escaping) + no_face
    print(f'{total} ears: {len(reachable)} reachable, {len(escaping)} '
          f'escaping, {no_face} no-face')
    summarise('REACHABLE', reachable)
    summarise('ESCAPING', escaping)


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
