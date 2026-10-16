#!/usr/bin/env python3
"""Does crop PLACEMENT carry ear information, or is it just avoiding sky?

An ablation elsewhere (Ear_Landmarker, on this corpus) randomised where the
face stage's crops land while holding their count, sizes and images fixed,
and face-stage misses rose ninefold. That establishes that placement carries
information about where ears are. It does not separate two explanations, as
`trainable_blazeface` pointed out:

  head-ness   -- the windows seek heads, which is where ears are
  composition -- the frames are head-filling, so "lands on a head" and
                 "lands on image content rather than empty background" are
                 nearly the same event, and a tail that merely avoids
                 background would ablate the same way

The discriminator is face size. On frames where the head fills the image the
two explanations coincide; on frames where the head is small they come apart,
because a random window then usually lands on background. So the lift of real
placement over a size-matched random null, measured SEPARATELY PER FACE-SIZE
BAND, separates them. Composition predicts the lift collapses toward 1.0 as
heads fill more of the frame and is carried entirely by the small-face band
being easy to miss at random. Head-ness predicts a lift above 1.0 that
survives in the small-face bands, where random placement has no chance.

    python measure_placement_lift.py --csv data/splits/val_v2.csv

The null preserves, per image, the number of windows and each window's side,
and places each uniformly at random subject to fitting inside the frame --
so count, scale and frame are held fixed and only position is destroyed.
That is the same ablation, conditioned on the confound.
"""
import argparse
import collections
import csv
import os
import pathlib

import cv2
import numpy as np
import torch

from make_face_crops import crop_window, load_face_detector
from utils.config import (FACE_CROP_EXPAND, FACE_CROP_MAX_FACES,
                          FACE_CROP_THRESHOLD, HUMAN_ANNOTATION_SOURCES)

BANDS = ((0.00, 0.20), (0.20, 0.30), (0.30, 0.45), (0.45, 10.0))


def human_ears(path):
    out = collections.defaultdict(list)
    for row in csv.DictReader(open(path)):
        if row['annotation_source'] not in HUMAN_ANNOTATION_SOURCES:
            continue
        w, h = float(row['w']), float(row['h'])
        if w > 0 and h > 0:
            out[row['image_path']].append(
                (float(row['x1']) + w / 2, float(row['y1']) + h / 2))
    return out


def covered(windows, ears):
    return sum(1 for ex, ey in ears
               if any(x0 <= ex < x0 + s and y0 <= ey < y0 + s
                      for x0, y0, s in windows))


def random_windows(windows, shape, rng):
    """Same count, same sides, positions destroyed."""
    height, width = shape[:2]
    out = []
    for _, _, side in windows:
        side = min(side, height, width)
        out.append((rng.integers(0, max(1, width - side + 1)),
                    rng.integers(0, max(1, height - side + 1)), side))
    return out


def measure(args):
    det = load_face_detector(args.face_threshold, args.device)
    rng = np.random.default_rng(args.seed)
    rows = []
    for rel, ears in list(human_ears(args.csv).items())[:args.limit]:
        image = cv2.imread(os.path.join(args.data_root, str(rel)))
        if image is None:
            continue
        with torch.no_grad():
            faces = det.process(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).cpu().numpy()
        if not len(faces):
            continue
        windows = [crop_window(f, FACE_CROP_EXPAND, image.shape)
                   for f in faces[:args.max_faces]]
        # How much of the frame the head occupies: the confound's axis.
        face_frac = max(f[2] - f[0] for f in faces[:args.max_faces]) \
            / image.shape[0]
        null = np.mean([covered(random_windows(windows, image.shape, rng),
                                ears) for _ in range(args.draws)])
        rows.append({'face_frac': face_frac, 'n_ears': len(ears),
                     'real': covered(windows, ears), 'null': null,
                     'n_win': len(windows)})
    return rows


def report(rows, args):
    print(f'\n{len(rows)} images with a face, '
          f'{sum(r["n_ears"] for r in rows)} ears, '
          f'{args.draws} random draws per image, expand {FACE_CROP_EXPAND}')
    print('\nband is the largest face box side as a fraction of frame height')
    print('  face/frame     images   ears   real cover   null cover   lift')
    print('  ' + '-' * 62)
    for lo, hi in BANDS:
        g = [r for r in rows if lo <= r['face_frac'] < hi]
        if not g:
            continue
        ears = sum(r['n_ears'] for r in g)
        real = sum(r['real'] for r in g) / ears
        null = sum(r['null'] for r in g) / ears
        lab = f'{lo:.2f}-{hi:.2f}' if hi < 10 else f'{lo:.2f}+'
        print(f'  {lab:<12} {len(g):7d} {ears:6d}   {100*real:9.1f}%   '
              f'{100*null:9.1f}%   {real/null if null else float("inf"):5.2f}x')
    ears = sum(r['n_ears'] for r in rows)
    real = sum(r['real'] for r in rows) / ears
    null = sum(r['null'] for r in rows) / ears
    print('  ' + '-' * 62)
    print(f'  {"all":<12} {len(rows):7d} {ears:6d}   {100*real:9.1f}%   '
          f'{100*null:9.1f}%   {real/null:5.2f}x')
    print(f'\n  mean windows per image: '
          f'{np.mean([r["n_win"] for r in rows]):.2f}')
    print('\nReading: composition predicts the lift collapses toward 1.0 in '
          'the\nhead-filling bands and is carried by small faces being easy '
          'to miss at\nrandom. Head-ness predicts a lift well above 1.0 that '
          'survives where\nheads are small and a random window lands on '
          'background.')


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--csv', default='data/splits/val_v2.csv')
    p.add_argument('--data-root', default='data/raw')
    p.add_argument('--face-threshold', type=float, default=FACE_CROP_THRESHOLD)
    p.add_argument('--max-faces', type=int, default=FACE_CROP_MAX_FACES)
    p.add_argument('--draws', type=int, default=20)
    p.add_argument('--limit', type=int, default=1200)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda')
    return p


def main():
    args = build_parser().parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
    os.chdir(pathlib.Path(__file__).resolve().parent)
    report(measure(args), args)


if __name__ == '__main__':
    main()
