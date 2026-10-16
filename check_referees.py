#!/usr/bin/env python3
"""Do independent detectors agree with the midline/lateral position proxy?

The clips measurement splits candidates by position -- |u| < 0.15 from the
crop centre is the nose and brow, |u| > 0.25 is where ears are -- and treats
midline as mostly-false and lateral as mostly-true. Everything concluded about
`minHits` and the confirm score rests on that proxy, and it has never been
checked against anything that knows what an ear is.

Two referees, both run on frames neither they nor BlazeEar have trained on:

  ear detector  `model_weights/yolov11_ear_detector.pt`, a fine-tuned YOLOv11.
                Shares human GT labels with the crop model but not
                pseudo-labels: `crops_train.csv` is GT / GT+EAR / GT+REVIEW /
                NEGATIVE / IGNORE with no POSE or raw YOLO rows, and
                GT+REVIEW -- YOLO-proposed, human-confirmed -- is 0.9% of it.
                Same corpus, different architecture.

  pose ears     COCO-pretrained `yolo11x-pose.pt`, keypoints 3 and 4. A
                different objective on a different corpus with different
                annotators, and `evaluate.py` records that POSE regions are
                ignored rather than used as labels. The most independent
                signal available here, and it gives a POINT rather than a box,
                which is enough to ask "is there an ear about here".

Neither is ground truth. Agreement is evidence; disagreement marks the frames
a human should look at, which is the point -- this is meant to cut labelling
work, not to replace the labels.

    python check_referees.py --videos data/clips --stride 10
"""
import argparse

import glob
import os
import pathlib

import cv2
import numpy as np
import torch

from evaluate_two_stage import detect_two_stage, load_ear_model
from make_face_crops import crop_window, load_face_detector
from utils.config import (DEFAULT_INPUT_SIZE, FACE_CROP_EXPAND,
                          FACE_CROP_MAX_FACES, FACE_CROP_THRESHOLD)
from utils.tta import VIEWS

MIDLINE, LATERAL = 0.15, 0.25
MATCH_IOU = 0.3
KPT_CONF = 0.5


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


def near_point(box, points, slack=0.5):
    """Is an ear keypoint inside this box, or within `slack` of a side?"""
    w, h = box[2] - box[0], box[3] - box[1]
    for px, py in points:
        if (box[0] - slack * w <= px <= box[2] + slack * w
                and box[1] - slack * h <= py <= box[3] + slack * h):
            return True
    return False


def classify(u):
    if np.isnan(u):
        return 'unknown'
    return 'midline' if abs(u) < MIDLINE else (
        'lateral' if abs(u) > LATERAL else 'between')


def collect(args):
    from ultralytics import YOLO
    face = load_face_detector(FACE_CROP_THRESHOLD, args.device)
    ear = load_ear_model(args.crop_checkpoint, args.floor, args.device,
                         input_size=args.input_size)
    yolo = YOLO(args.ear_weights)
    pose = YOLO(args.pose_weights)
    views = VIEWS[:args.views] if args.views else None

    rows = []
    for path in sorted(glob.glob(os.path.join(args.videos, '*.mp4'))):
        cap = cv2.VideoCapture(path)
        k = 0
        used = 0
        while used < args.max_frames:
            ok, image = cap.read()
            if not ok:
                break
            k += 1
            if k % args.stride:
                continue
            used += 1
            boxes, scores, _ = detect_two_stage(
                face, ear, image, FACE_CROP_EXPAND, args.max_faces,
                views=views)
            if not len(boxes):
                continue
            with torch.no_grad():
                faces = face.process(
                    cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).cpu().numpy()
            windows = [crop_window(f, FACE_CROP_EXPAND, image.shape)
                       for f in faces[:args.max_faces]]

            r = yolo.predict(image, verbose=False, conf=args.ref_conf)[0]
            ref_boxes = r.boxes.xyxy.cpu().numpy() if len(r.boxes) else []
            rp = pose.predict(image, verbose=False, conf=args.ref_conf)[0]
            pts = []
            if rp.keypoints is not None and len(rp.keypoints.xy):
                xy = rp.keypoints.xy.cpu().numpy()
                cf = (rp.keypoints.conf.cpu().numpy()
                      if rp.keypoints.conf is not None
                      else np.ones(xy.shape[:2]))
                for pi in range(xy.shape[0]):
                    for ki in (3, 4):
                        if cf[pi, ki] >= KPT_CONF:
                            pts.append(tuple(xy[pi, ki]))

            for i in range(len(boxes)):
                ymin, xmin, ymax, xmax = boxes[i].tolist()
                cy, cx = (ymin + ymax) / 2, (xmin + xmax) / 2
                u = np.nan
                if windows:
                    x0, y0, side = min(
                        windows, key=lambda w: max(abs(cx - (w[0] + w[2] / 2)),
                                                   abs(cy - (w[1] + w[2] / 2))))
                    u = (cx - (x0 + side / 2)) / side
                xyxy = (xmin, ymin, xmax, ymax)
                rows.append({
                    'cls': classify(u), 'score': float(scores[i]),
                    'yolo': iou_one(xyxy, ref_boxes) >= MATCH_IOU,
                    'pose': near_point(xyxy, pts),
                })
        cap.release()
    return rows


def report(rows, gate):
    print(f'\n{len(rows)} BlazeEar candidates over the sampled frames')
    for lab, sel in (('all candidates', lambda r: True),
                     (f'candidates >= {gate} (would confirm a track)',
                      lambda r: r['score'] >= gate)):
        g = [r for r in rows if sel(r)]
        print(f'\n{lab}: n={len(g)}')
        print('  class      n    ear detector says ear    pose says ear    '
              'either')
        print('  ' + '-' * 70)
        for cls in ('midline', 'between', 'lateral', 'unknown'):
            h = [r for r in g if r['cls'] == cls]
            if not h:
                continue
            y = 100 * np.mean([r['yolo'] for r in h])
            p = 100 * np.mean([r['pose'] for r in h])
            e = 100 * np.mean([r['yolo'] or r['pose'] for r in h])
            print(f'  {cls:<9} {len(h):5d}   {y:18.1f}%   {p:14.1f}%   '
                  f'{e:6.1f}%')
    mid = [r for r in rows if r['cls'] == 'midline']
    lat = [r for r in rows if r['cls'] == 'lateral']
    if mid and lat:
        ym, yl = np.mean([r['yolo'] for r in mid]), np.mean([r['yolo'] for r in lat])
        print(f'\nThe proxy assumes midline is mostly false and lateral mostly '
              f'true.\nEar detector confirms {100*yl:.1f}% of lateral against '
              f'{100*ym:.1f}% of midline'
              f' -- a {yl/ym if ym else float("inf"):.1f}x separation.')


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--videos', default='data/clips')
    p.add_argument('--crop-checkpoint',
                   default='runs/checkpoints_crop/BlazeEar_best.pth')
    p.add_argument('--ear-weights',
                   default='model_weights/yolov11_ear_detector.pt')
    p.add_argument('--pose-weights', default='model_weights/yolo11x-pose.pt')
    p.add_argument('--input-size', type=int, default=DEFAULT_INPUT_SIZE)
    p.add_argument('--views', type=int, default=len(VIEWS))
    p.add_argument('--max-faces', type=int, default=FACE_CROP_MAX_FACES)
    p.add_argument('--stride', type=int, default=10)
    p.add_argument('--max-frames', type=int, default=30)
    p.add_argument('--floor', type=float, default=0.50)
    p.add_argument('--ref-conf', type=float, default=0.25)
    p.add_argument('--gate', type=float, default=0.80)
    p.add_argument('--device', default='cuda')
    return p


def main():
    args = build_parser().parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
    os.chdir(pathlib.Path(__file__).resolve().parent)
    report(collect(args), args.gate)


if __name__ == '__main__':
    main()
