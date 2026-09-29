"""
Standalone evaluation for a BlazeEar checkpoint.

Reports pooled detection metrics over a whole split, broken out by annotation
provenance. The breakdown matters because `data/splits/val.csv` mixes
human-verified boxes with pose-model pseudo-labels, and averaging them into one
headline number reports partly on agreement with the pseudo-labeler rather than
on accuracy.

Three views are reported:

  all         every box treated as ground truth (the historical number)
  human       GT and GT+EAR as ground truth, POSE regions ignored
  pose        POSE as ground truth, human regions ignored (agreement with the
              pose model, not accuracy)

"Ignored" means a detection landing there is neither rewarded nor punished, so
excluding untrusted labels does not turn real ears into false positives.

Usage:
    python evaluate.py --checkpoint runs/checkpoints/BlazeEar_best.pth
    python evaluate.py --checkpoint runs/checkpoints/BlazeEar_best.pth \
        --csv data/splits/val.csv --data-root data/raw
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch

from blazeear import BlazeEar
from dataloader import create_dataloader
from utils.config import (
    DEFAULT_BEST_CHECKPOINT,
    DEFAULT_DATA_ROOT,
    DEFAULT_VAL_CSV,
    HUMAN_ANNOTATION_SOURCES,
)
from utils.detection_eval import DetectionEvaluator

# Views to report: name -> set of annotation_source values treated as ground
# truth. Everything else present in the image becomes an ignore region.
VIEWS: Dict[str, object] = {
    'all': None,                                # None = every source is ground truth
    'human': set(HUMAN_ANNOTATION_SOURCES),
    'pose': {'POSE'},
}


def load_sources_by_image(csv_path: str) -> Dict[str, List[str]]:
    """Map image_path -> annotation_source per box, in CSV row order.

    The dataset groups rows the same way (`groupby(..., sort=False)`) and, with
    augmentation disabled, returns gt_boxes one-to-one with those rows.
    """
    df = pd.read_csv(csv_path)
    if 'annotation_source' not in df.columns:
        return {}
    return {
        str(image_path): [str(v) for v in group['annotation_source'].tolist()]
        for image_path, group in df.groupby('image_path', sort=False)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', default=DEFAULT_BEST_CHECKPOINT)
    parser.add_argument('--csv', default=DEFAULT_VAL_CSV)
    parser.add_argument('--data-root', default=DEFAULT_DATA_ROOT)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--score-threshold', type=float, default=0.1,
                        help='Detections below this are discarded before scoring. '
                             'Keep it low: average precision needs the full curve.')
    parser.add_argument('--nms-threshold', type=float, default=0.5)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu'

    loader = create_dataloader(
        csv_path=args.csv, root_dir=args.data_root, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, augment=False, pin_memory=True,
    )
    dataset = loader.dataset
    sources_by_image = load_sources_by_image(args.csv)
    if not sources_by_image:
        print('No annotation_source column found; reporting the "all" view only.')

    model = BlazeEar()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state)

    # Reuse the trainer's detection path so evaluation cannot drift from the
    # numbers reported during training. (P5 will unify post-processing and this
    # can then construct the shared path directly.)
    from train_blazeear import BlazeEarTrainer
    trainer = BlazeEarTrainer(
        model=model, train_loader=loader, val_loader=loader, device=device,
        checkpoint_dir='runs/checkpoints', log_dir='runs/logs/evaluate',
        eval_score_threshold=args.score_threshold, nms_iou_threshold=args.nms_threshold,
        use_amp=False,
    )
    trainer.model.eval()

    active_views = {k: v for k, v in VIEWS.items() if k == 'all' or sources_by_image}
    evaluators = {name: DetectionEvaluator() for name in active_views}

    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            gt_boxes = batch['gt_boxes'].to(device)
            gt_counts = batch['gt_box_counts'].to(device)
            sample_indices = batch['sample_index']

            class_pred, anchor_pred = trainer._get_training_outputs(images)
            scores = class_pred.squeeze(-1)
            decoded = trainer.loss_fn.decode_boxes(anchor_pred, trainer.reference_anchors)

            for b in range(images.shape[0]):
                count = int(gt_counts[b].item())
                boxes = gt_boxes[b, :count]
                det_boxes, det_scores = trainer._detections_for_image(scores[b], decoded[b])

                idx = int(sample_indices[b].item())
                sources: List[str] = []
                if idx >= 0 and sources_by_image:
                    image_path = dataset.samples[idx]['image_path']  # type: ignore[attr-defined]
                    sources = sources_by_image.get(image_path, [])

                for name, keep_sources in active_views.items():
                    if keep_sources is None or len(sources) != count:
                        # No provenance for this image: score every box, ignore nothing.
                        evaluators[name].add_image(det_boxes, det_scores, boxes)
                        continue
                    mask = torch.tensor(
                        [s in keep_sources for s in sources],
                        dtype=torch.bool, device=boxes.device
                    )
                    evaluators[name].add_image(
                        det_boxes, det_scores, boxes[mask], ignore_boxes=boxes[~mask]
                    )

    print()
    print(f'checkpoint : {args.checkpoint}')
    print(f'split      : {args.csv}  ({int(evaluators["all"].compute()["num_images"])} images)')
    print()
    header = f'{"view":8s} {"GT":>7s} {"dets":>8s} {"ignored":>8s} {"mAP@0.5":>9s} {"mAP@[.5:.95]":>13s} {"det IoU":>9s}'
    print(header)
    print('-' * len(header))
    for name in active_views:
        m = evaluators[name].compute()
        print(f'{name:8s} {int(m["num_ground_truth"]):7d} {int(m["num_detections"]):8d} '
              f'{int(m["num_ignored"]):8d} {m["map_50"]:9.4f} {m["map_50_95"]:13.4f} '
              f'{m["detection_iou"]:9.4f}')
    print()


if __name__ == '__main__':
    main()
