#!/usr/bin/env python3
"""Can stage 2 score the ears a better stage 1 would deliver?

The stage-1 handoff (Ear_Landmarker / trainable_blazeface) establishes that a
fine-tuned face stage covers ears the shipped crop misses, and that the extra
coverage converts to nothing at matched precision. The open question is
whether the rescued ears are scoreable AT ALL by stage 2, independent of how
they are delivered -- because if they are not, no stage-1 improvement can
convert and the roadmap's gate is smaller than the roadmap assumes.

This is deliberately an UPPER BOUND. Each rescued ear is handed to stage 2 in
the most favourable crop a face-box-derived stage 1 could produce: the stock
face box nearest the ear, expanded by the minimum factor that brings the ear
inside, which is what a better-placed or larger box would have done. If even
that crop cannot score the ear, a real stage 1 certainly cannot.

    python score_rescued_ears.py --csv <rescued_union.csv>

Prediction recorded before running, from this repository's own measurements:
the 0-32 px share is effectively unconvertible, because the median ear in
these crops is 24 real pixels, 70.9% are under 32, a scale-augmentation lever
aimed at that band was null, and both reverted levers produced a uniform
confidence shift with no small-ear-specific component.
"""
import argparse
import collections
import csv
import os
import pathlib

import cv2
import numpy as np
import torch

from evaluate_two_stage import detect_single_stage, load_ear_model
from make_face_crops import load_face_detector
from utils.config import (DEFAULT_INPUT_SIZE, DETECTION_SCORE_THRESHOLD,
                          FACE_CROP_MAX_FACES, FACE_CROP_THRESHOLD)

IOU_HIT = 0.5
BAND_ORDER = ['0-32', '32-48', '48-64', '64-80', '80+']
MAX_EXPAND = 6.0


def covering_window(face, ear, shape):
    """Smallest concentric square on this face box that contains the ear.

    A face-box-derived stage 1 can only move or grow the window; this is the
    best such window for this ear, so the score it yields is a ceiling on
    what any face-derived crop could achieve.
    """
    ymin, xmin, ymax, xmax = face[:4]
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    base = max(xmax - xmin, ymax - ymin)
    ex0, ey0, ex1, ey1 = ear
    need = 2 * max(abs(ex0 - cx), abs(ex1 - cx), abs(ey0 - cy), abs(ey1 - cy))
    side = int(min(max(need * 1.05, base * 1.5), MAX_EXPAND * base,
                   max(shape[0], shape[1])))
    x0 = int(np.clip(cx - side / 2, 0, max(0, shape[1] - side)))
    y0 = int(np.clip(cy - side / 2, 0, max(0, shape[0] - side)))
    return x0, y0, side, side / base


def iou_xyxy(a, b):
    if not len(b):
        return np.zeros(0)
    x0 = np.maximum(a[0], b[:, 0]); y0 = np.maximum(a[1], b[:, 1])
    x1 = np.minimum(a[2], b[:, 2]); y1 = np.minimum(a[3], b[:, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    return inter / np.maximum((a[2] - a[0]) * (a[3] - a[1])
                              + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
                              - inter, 1e-9)


def run(args):
    face = load_face_detector(FACE_CROP_THRESHOLD, args.device)
    ear = load_ear_model(args.crop_checkpoint, 0.01, args.device,
                         input_size=args.input_size)
    rows = []
    for r in csv.DictReader(open(args.csv)):
        path = r['image_path']
        image = cv2.imread(path)
        if image is None:
            rows.append({'band': r['band_px'], 'status': 'image missing'})
            continue
        x1, y1, w, h = (float(r[k]) for k in ('x1', 'y1', 'w', 'h'))
        ear_box = (x1, y1, x1 + w, y1 + h)
        with torch.no_grad():
            faces = face.process(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).cpu().numpy()
        if not len(faces):
            rows.append({'band': r['band_px'], 'status': 'no face'})
            continue
        cx, cy = x1 + w / 2, y1 + h / 2
        nearest = min(faces[:args.max_faces],
                      key=lambda f: (cx - (f[1] + f[3]) / 2) ** 2
                      + (cy - (f[0] + f[2]) / 2) ** 2)
        x0, y0, side, expand = covering_window(nearest, ear_box, image.shape)
        patch = image[y0:y0 + side, x0:x0 + side]
        if patch.size == 0:
            rows.append({'band': r['band_px'], 'status': 'empty crop'})
            continue
        boxes, scores = detect_single_stage(ear, patch)
        if not len(scores):
            rows.append({'band': r['band_px'], 'status': 'scored',
                         'iou': 0.0, 'score': 0.0, 'expand': expand,
                         'ear_px_in_crop': max(w, h) * DEFAULT_INPUT_SIZE / side})
            continue
        pr = np.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], 1)
        pr = pr + np.array([x0, y0, x0, y0], dtype=np.float32)
        ious = iou_xyxy(np.array(ear_box), pr)
        j = int(np.argmax(ious))
        rows.append({'band': r['band_px'], 'status': 'scored',
                     'iou': float(ious[j]),
                     'score': float(scores[j]) if ious[j] >= IOU_HIT else 0.0,
                     'expand': expand,
                     'ear_px_in_crop': max(w, h) * DEFAULT_INPUT_SIZE / side})
    return rows


def report(rows, gate):
    print(f'\n{len(rows)} rescued ears, handed to stage 2 in the BEST '
          f'face-derived crop\n(stock face box nearest the ear, grown to the '
          f'minimum expand that covers it)')
    bad = collections.Counter(r['status'] for r in rows if r['status'] != 'scored')
    for k, v in bad.items():
        print(f'  {k}: {v}')
    scored = [r for r in rows if r['status'] == 'scored']
    print(f'\n  band    n   localised   scored >= {gate}   median score   '
          f'ear px in crop   median expand')
    print('  ' + '-' * 86)
    for band in BAND_ORDER + [b for b in {r['band'] for r in scored}
                              if b not in BAND_ORDER]:
        g = [r for r in scored if r['band'] == band]
        if not g:
            continue
        loc = np.array([r['iou'] >= IOU_HIT for r in g])
        sc = np.array([r['score'] for r in g])
        hit = sc >= gate
        med = np.median(sc[loc]) if loc.any() else float('nan')
        print(f'  {band:<6} {len(g):4d}   {100*loc.mean():8.1f}%   '
              f'{100*hit.mean():12.1f}%   {med:12.3f}   '
              f'{np.median([r["ear_px_in_crop"] for r in g]):14.1f}   '
              f'{np.median([r["expand"] for r in g]):13.2f}')
    loc = np.array([r['iou'] >= IOU_HIT for r in scored])
    sc = np.array([r['score'] for r in scored])
    print('  ' + '-' * 86)
    print(f'  {"all":<6} {len(scored):4d}   {100*loc.mean():8.1f}%   '
          f'{100*(sc >= gate).mean():12.1f}%   '
          f'{np.median(sc[loc]):12.3f}')
    print('\nThis is a CEILING: the crop is chosen per ear to contain it. '
          f'A real stage 1\ncannot do better, so a band that fails here '
          f'cannot be rescued by stage 1 at all.')


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--csv', required=True)
    p.add_argument('--crop-checkpoint',
                   default='runs/checkpoints_crop/BlazeEar_best.pth')
    p.add_argument('--input-size', type=int, default=DEFAULT_INPUT_SIZE)
    p.add_argument('--max-faces', type=int, default=FACE_CROP_MAX_FACES)
    p.add_argument('--gate', type=float, default=DETECTION_SCORE_THRESHOLD)
    p.add_argument('--device', default='cuda')
    return p


def main():
    args = build_parser().parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
    os.chdir(pathlib.Path(__file__).resolve().parent)
    report(run(args), args.gate)


if __name__ == '__main__':
    main()
