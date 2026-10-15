#!/usr/bin/env python3
"""The pre-registered read of the three arms: recall per ear-size band, the
mode band as a guard, and false positives per image.

Headline mAP is deliberately not the comparison. The hypothesis was about the
0-32 and 32-48 px bands, which are 5.9% and 10.5% of the data and can move a
long way without moving a headline at all. 0-32 and 32-48 are never pooled.
"""
import csv
import os
import pathlib
import collections

import cv2
import numpy as np
import torch

os.chdir(pathlib.Path(__file__).resolve().parent)
from blazeear import BlazeEar
from utils.anchor_utils import anchor_options
from utils.config import (DEFAULT_INPUT_SIZE as S, DETECTION_SCORE_THRESHOLD,
                          NMS_IOU_THRESHOLD, HUMAN_ANNOTATION_SOURCES)

BANDS = [(0, 32), (32, 48), (48, 64), (64, 80), (80, 96), (96, 120), (120, 10**6)]
ARMS = ['baseline', 'cap_only', 'treatment']
IOU_HIT = 0.5


def band_of(px):
    for lo, hi in BANDS:
        if lo <= px < hi:
            return f'{lo}-{hi}' if hi < 10**6 else '120+'


def load(arm):
    m = BlazeEar(); m.generate_anchors(anchor_options)
    sd = torch.load(f'runs/checkpoints_{arm}/BlazeEar_best.pth',
                    map_location='cpu', weights_only=False)
    m.load_state_dict(sd.get('model_state_dict', sd), strict=False)
    m.min_score_thresh = DETECTION_SCORE_THRESHOLD
    m.min_suppression_threshold = NMS_IOU_THRESHOLD
    m = m.eval().to('cuda')
    # generate_anchors assigns a plain attribute rather than a registered
    # buffer, so .to() does not move it and decode gets a CPU anchor tensor.
    m.anchors = m.anchors.to('cuda')
    return m


def gt_by_image():
    out = collections.defaultdict(list)
    for r in csv.DictReader(open('data/splits/crops_val.csv')):
        if r['annotation_source'] not in HUMAN_ANNOTATION_SOURCES:
            continue
        w, h = float(r['w']), float(r['h'])
        if w <= 0 or h <= 0:
            continue
        out[r['image_path']].append((float(r['x1']), float(r['y1']), w, h))
    return out


def iou_xyxy(a, b):
    if not len(b):
        return np.zeros(0)
    x0 = np.maximum(a[0], b[:, 0]); y0 = np.maximum(a[1], b[:, 1])
    x1 = np.minimum(a[2], b[:, 2]); y1 = np.minimum(a[3], b[:, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[:,2]-b[:,0])*(b[:,3]-b[:,1]) - inter
    return inter / np.maximum(ua, 1e-9)


def score(arm, gt):
    m = load(arm)
    hit = collections.Counter(); tot = collections.Counter()
    fp_small = fp_large = fp_all = images = 0
    for rel, boxes in gt.items():
        img = cv2.imread(os.path.join('data/face_crops', rel))
        if img is None:
            continue
        images += 1
        side = max(img.shape[:2])
        with torch.no_grad():
            det = m.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).cpu().numpy()
        # process emits [ymin,xmin,ymax,xmax,...,score]; to xyxy
        if len(det):
            pred = np.stack([det[:,1], det[:,0], det[:,3], det[:,2]], 1)
        else:
            pred = np.zeros((0, 4))
        gt_xyxy = np.array([[x, y, x+w, y+h] for x, y, w, h in boxes])
        matched = set()
        for gi, (x, y, w, h) in enumerate(boxes):
            b = band_of(max(w, h) / side * S)
            tot[b] += 1
            if len(pred):
                ious = iou_xyxy(gt_xyxy[gi], pred)
                j = int(np.argmax(ious))
                if ious[j] >= IOU_HIT:
                    hit[b] += 1
                    matched.add(j)
        for j in range(len(pred)):
            if j in matched:
                continue
            fp_all += 1
            pw = max(pred[j,2]-pred[j,0], pred[j,3]-pred[j,1]) / side * S
            if pw < 48: fp_small += 1
            elif pw >= 96: fp_large += 1
    return hit, tot, fp_all, fp_small, fp_large, images


gt = gt_by_image()
print(f'{sum(len(v) for v in gt.values())} human ears in '
      f'{len(gt)} crops_val images, hit = IoU>={IOU_HIT} at the '
      f'{DETECTION_SCORE_THRESHOLD} gate\n')
res = {a: score(a, gt) for a in ARMS}

names = [f'{lo}-{hi}' if hi < 10**6 else '120+' for lo, hi in BANDS]
print(f"{'band':>8s} {'n':>5s} " + ''.join(f'{a:>12s}' for a in ARMS)
      + f"{'cap-base':>10s}{'B1':>8s}")
for b in names:
    n = res['baseline'][1][b]
    if not n:
        continue
    r = {a: 100*res[a][0][b]/n for a in ARMS}
    flag = '  <-- MODE' if b == '64-80' else ''
    print(f'{b:>8s} {n:5d} ' + ''.join(f'{r[a]:11.1f}%' for a in ARMS)
          + f"{r['cap_only']-r['baseline']:+9.1f}{r['treatment']-r['cap_only']:+8.1f}{flag}")

print(f"\n{'arm':>12s} {'recall all':>11s} {'FP/image':>9s} "
      f"{'FP<48px':>8s} {'FP>=96px':>9s}")
for a in ARMS:
    hit, tot, fa, fs, fl, im = res[a]
    print(f'{a:>12s} {100*sum(hit.values())/sum(tot.values()):10.1f}% '
          f'{fa/im:9.2f} {fs/im:8.2f} {fl/im:9.2f}')
