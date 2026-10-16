#!/usr/bin/env python3
"""Is the 0-32 px band unresolvable, or are my targets just not covering it?

`trainable_blazeface` flagged this as the confusion that cost them most of a
day: they concluded their data was the limit (39% of faces under 8 px, below
the anchor stride) when in fact every ground-truth box was being assigned to
both anchor grids, so most positive targets demanded a detection at a scale
the model could not answer. The faces were resolvable; the TARGETS were not.
The two produce identical symptoms and have opposite fixes, and inspection
cannot separate them. What separated them was scoring a model they trusted
against their own objective and finding it "failed" too.

This repo has the same claim at risk. "The 0-32 px band is information
limited" rests on stage 2 failing to localise those ears. That is consistent
with the ears being unresolvable AND with the anchors not representing them:
`anchor_coverage` measured best-anchor IoU median 0.425 in the 0-32 band,
below the 0.5 matching threshold, with 57.2% of 0-24 px ears unable to reach
IoU 0.5 against ANY prior -- which is suspiciously close to the 56.9% of
rescued ears that produced no box at all.

So: hand the SAME crops to an independent detector that shares no anchors,
no targets and no training recipe with stage 2. If it finds ears stage 2
cannot, the band is resolvable and the limit is representational. If it fails
in the same band, the information argument survives a real test.

    python check_resolvability.py --limit 1500

Prediction recorded before running: if the anchor numbers mean what they look
like, the referee should localise materially more of the 0-32 band than stage
2 does, while the two should converge above 48 px where anchor fit is fine.
If instead the referee tracks stage 2 down through the small bands, the
information limit is real and this closes the question rather than opening it.
"""
import argparse
import collections
import csv

import cv2
import numpy as np
import torch

from evaluate_two_stage import detect_single_stage, load_ear_model
from utils.config import DEFAULT_INPUT_SIZE, HUMAN_ANNOTATION_SOURCES

IOU_HIT = 0.5
BANDS = [(0, 24), (24, 32), (32, 48), (48, 64), (64, 88), (88, 10 ** 6)]


def band_of(px):
    for lo, hi in BANDS:
        if lo <= px < hi:
            return f'{lo}-{hi}px' if hi < 10 ** 6 else f'{lo}+px'
    return 'other'


def iou_one(a, boxes):
    if not len(boxes):
        return 0.0
    boxes = np.asarray(boxes, float)
    x0 = np.maximum(a[0], boxes[:, 0]); y0 = np.maximum(a[1], boxes[:, 1])
    x1 = np.minimum(a[2], boxes[:, 2]); y1 = np.minimum(a[3], boxes[:, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    union = ((a[2] - a[0]) * (a[3] - a[1])
             + (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) - inter)
    return float(np.max(inter / np.maximum(union, 1e-9)))


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--csv', default='data/splits/crops_val.csv')
    p.add_argument('--data-root', default='data/face_crops/')
    p.add_argument('--crop-checkpoint',
                   default='runs/checkpoints_crop/BlazeEar_best.pth')
    p.add_argument('--referee', default='model_weights/yolov11_ear_detector.pt')
    p.add_argument('--referee-conf', type=float, default=0.25)
    p.add_argument('--input-size', type=int, default=DEFAULT_INPUT_SIZE)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()

    from ultralytics import YOLO
    ref = YOLO(args.referee)
    ear = load_ear_model(args.crop_checkpoint, 0.01, args.device,
                         input_size=args.input_size)

    # Group GT ears by crop image, so each image is read and run once.
    per_image = collections.defaultdict(list)
    for r in csv.DictReader(open(args.csv)):
        if r['annotation_source'] not in HUMAN_ANNOTATION_SOURCES:
            continue
        per_image[r['image_path']].append(r)
    paths = sorted(per_image)
    if args.limit:
        paths = paths[:args.limit]

    rows = []
    for i, rel in enumerate(paths):
        img = cv2.imread(args.data_root + rel)
        if img is None:
            continue
        mine_boxes, mine_scores = detect_single_stage(ear, img)
        mb = np.asarray(mine_boxes)
        mine = (np.stack([mb[:, 1], mb[:, 0], mb[:, 3], mb[:, 2]], 1)
                if len(mb) else np.zeros((0, 4)))
        rb = ref.predict(img, conf=args.referee_conf, verbose=False)[0]
        refb = rb.boxes.xyxy.cpu().numpy() if rb.boxes is not None else np.zeros((0, 4))
        side = float(per_image[rel][0]['crop_side'])
        for r in per_image[rel]:
            x1, y1, w, h = (float(r[k]) for k in ('x1', 'y1', 'w', 'h'))
            gt = (x1, y1, x1 + w, y1 + h)
            rows.append({
                'band': band_of(max(w, h) * args.input_size / side),
                'mine': iou_one(gt, mine) >= IOU_HIT,
                'ref': iou_one(gt, refb) >= IOU_HIT,
            })
        if (i + 1) % 200 == 0:
            print(f'  {i+1}/{len(paths)} images, {len(rows)} ears', flush=True)

    print(f'\n{len(rows)} human-labelled ears in {len(paths)} val crops')
    print(f'localised = IoU >= {IOU_HIT} against the GT box, same crop, '
          'same image, both arms\n')
    print(f'  {"band":<10} {"n":>5}  {"stage 2":>8}  {"referee":>8}  '
          f'{"ref only":>9}  {"mine only":>9}')
    print('  ' + '-' * 60)
    order = [band_of((lo + hi) / 2 if hi < 10 ** 6 else lo + 1)
             for lo, hi in BANDS]
    for band in order:
        g = [r for r in rows if r['band'] == band]
        if not g:
            continue
        m = np.array([r['mine'] for r in g])
        f = np.array([r['ref'] for r in g])
        print(f'  {band:<10} {len(g):5d}  {100*m.mean():7.1f}%  '
              f'{100*f.mean():7.1f}%  {100*(f & ~m).mean():8.1f}%  '
              f'{100*(m & ~f).mean():8.1f}%')
    m = np.array([r['mine'] for r in rows])
    f = np.array([r['ref'] for r in rows])
    print('  ' + '-' * 60)
    print(f'  {"all":<10} {len(rows):5d}  {100*m.mean():7.1f}%  '
          f'{100*f.mean():7.1f}%  {100*(f & ~m).mean():8.1f}%  '
          f'{100*(m & ~f).mean():8.1f}%')
    print('\nThe referee is NOT ground truth and shares human GT labels with '
          'stage 2.\nIt shares no anchors, no target assignment and no '
          'training recipe, which is\nwhat this asks about. Read "ref only" '
          'in the small bands as the headline:\nears that are there, and that '
          'a different representation can find.')


if __name__ == '__main__':
    main()
