"""Score the two-stage face->crop->ear pipeline end to end.

Both stages are scored in ORIGINAL image pixels against every human ear in
the image, so an ear whose face BlazeFace missed counts as a miss rather than
quietly leaving the denominator. That is the only comparison that means
anything: crop-space numbers flatter the pipeline by hiding its own recall
ceiling.

Pass --checkpoint alone for the single-stage baseline, --crop-checkpoint
alone for the pipeline, or both to print them side by side through an
identical detection path.
"""
import argparse
import os

import cv2
import pandas as pd
import torch

from blazeear import BlazeEar
from blazebase import checkpoint_is_folded, load_checkpoint_state
from make_face_crops import crop_window, load_face_detector
from utils.config import (
    DEFAULT_DATA_ROOT,
    FACE_CROP_EXPAND,
    FACE_CROP_MAX_FACES,
    FACE_CROP_THRESHOLD,
    HUMAN_ANNOTATION_SOURCES,
    NEGATIVE_ANNOTATION_SOURCE,
)
from utils.detection_eval import DetectionEvaluator
from utils.nms import suppress_overlapping


def load_ear_model(path, score_threshold, device):
    state = load_checkpoint_state(path)
    model = BlazeEar(use_batchnorm=not checkpoint_is_folded(state))
    model.load_state_dict(state)
    model.eval().to(device)
    model.min_score_thresh = score_threshold
    # The fitted ear priors, the same set the dataloader and loss use.
    model.generate_anchors({})
    return model


def detect_single_stage(model, image):
    with torch.no_grad():
        det = model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).cpu()
    if not len(det):
        return torch.zeros((0, 4)), torch.zeros((0,))
    return det[:, :4].float(), det[:, 4].float()


def detect_two_stage(face_detector, ear_model, image, expand, max_faces):
    """Ear detections in original-frame coordinates, via one crop per face.

    Returns the face count as well, because an image with no face is where the
    pipeline's recall ceiling lives and the caller decides what to do about it.
    """
    with torch.no_grad():
        faces = face_detector.process(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).cpu().numpy()
    all_boxes, all_scores = [], []
    for face in faces[:max_faces]:
        x0, y0, side = crop_window(face, expand, image.shape)
        patch = image[y0:y0 + side, x0:x0 + side]
        if patch.size == 0:
            continue
        boxes, scores = detect_single_stage(ear_model, patch)
        if not len(scores):
            continue
        # Crop coordinates back to the original frame.
        boxes = boxes + torch.tensor([y0, x0, y0, x0], dtype=torch.float32)
        all_boxes.append(boxes)
        all_scores.append(scores)
    if not all_boxes:
        return torch.zeros((0, 4)), torch.zeros((0,)), len(faces)
    # One crop per face means overlapping crops can each report the same ear.
    boxes, scores = suppress_overlapping(
        torch.cat(all_boxes), torch.cat(all_scores))
    return boxes, scores, len(faces)


def ground_truth(rows):
    gt, ignore = [], []
    for row in rows.itertuples(index=False):
        source = str(row.annotation_source)
        # A placeholder row marks a background image; it is not a box.
        if source == NEGATIVE_ANNOTATION_SOURCE:
            continue
        box = [float(row.y1), float(row.x1),
               float(row.y1) + float(row.h), float(row.x1) + float(row.w)]
        if source in HUMAN_ANNOTATION_SOURCES:
            gt.append(box)
        else:
            ignore.append(box)
    to_tensor = lambda b: (torch.tensor(b, dtype=torch.float32) if b
                           else torch.zeros((0, 4)))
    return to_tensor(gt), to_tensor(ignore)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', default='data/splits/val_v2.csv')
    parser.add_argument('--data-root', default=DEFAULT_DATA_ROOT)
    parser.add_argument('--checkpoint', default=None,
                        help='ear model trained on full frames')
    parser.add_argument('--crop-checkpoint', default=None,
                        help='ear model trained on face crops')
    parser.add_argument('--fallback', action='store_true',
                        help='on images with no detected face, fall back to '
                             'the full-frame model from --checkpoint')
    parser.add_argument('--expand', type=float, default=FACE_CROP_EXPAND)
    parser.add_argument('--face-threshold', type=float,
                        default=FACE_CROP_THRESHOLD)
    parser.add_argument('--max-faces', type=int, default=FACE_CROP_MAX_FACES)
    parser.add_argument('--score-threshold', type=float, default=0.01)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not args.checkpoint and not args.crop_checkpoint:
        parser.error('pass --checkpoint, --crop-checkpoint, or both')

    frame = pd.read_csv(args.csv)
    groups = list(frame.groupby('image_path', sort=False))
    if args.limit:
        groups = groups[:args.limit]

    single = (load_ear_model(args.checkpoint, args.score_threshold, args.device)
              if args.checkpoint else None)
    crop_model = (load_ear_model(args.crop_checkpoint, args.score_threshold,
                                 args.device)
                  if args.crop_checkpoint else None)
    face_detector = (load_face_detector(args.face_threshold, args.device)
                     if crop_model else None)

    evaluators = {}
    if single:
        evaluators['single-stage'] = DetectionEvaluator()
    if crop_model:
        evaluators['two-stage'] = DetectionEvaluator()
    fallback = single if (args.fallback and single) else None
    if args.fallback and not single:
        parser.error('--fallback needs --checkpoint to fall back to')
    if fallback:
        evaluators['two-stage+fallback'] = DetectionEvaluator()
    no_face = 0
    images = 0
    human_ears = 0

    for image_path, rows in groups:
        image = cv2.imread(os.path.join(args.data_root, str(image_path)))
        if image is None:
            continue
        images += 1
        gt, ignore = ground_truth(rows)
        human_ears += len(gt)
        if single:
            boxes, scores = detect_single_stage(single, image)
            evaluators['single-stage'].add_image(boxes, scores, gt,
                                                 ignore_boxes=ignore)
        if crop_model:
            boxes, scores, n_faces = detect_two_stage(
                face_detector, crop_model, image, args.expand, args.max_faces)
            no_face += (n_faces == 0)
            evaluators['two-stage'].add_image(boxes, scores, gt,
                                              ignore_boxes=ignore)
            if fallback:
                if n_faces == 0:
                    boxes, scores = detect_single_stage(fallback, image)
                evaluators['two-stage+fallback'].add_image(
                    boxes, scores, gt, ignore_boxes=ignore)
        if images % 200 == 0:
            print(f'  {images} images', flush=True)

    print(f'\n{images} images, {human_ears} human ears')
    if crop_model:
        print(f'face detector found nothing in {no_face} '
              f'({100 * no_face / max(images, 1):.1f}%) -- a hard recall ceiling')
    header = f'{"pipeline":20s} {"mAP@0.5":>9s} {"mAP@[.5:.95]":>13s} {"det IoU":>9s}'
    print()
    print(header)
    print('-' * len(header))
    for name, evaluator in evaluators.items():
        m = evaluator.compute()
        print(f'{name:20s} {m["map_50"]:9.4f} {m["map_50_95"]:13.4f} '
              f'{m["detection_iou"]:9.4f}')


if __name__ == '__main__':
    main()
