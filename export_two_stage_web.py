"""Export the two-stage pipeline for the browser demo.

The demo used to run one ear detector over the whole frame, which is the
design the measurements rejected: the median ear is 14 px once a full frame is
squeezed into the 128 px input, and that model scores mAP@0.5 0.3142 against
the two-stage pipeline's 0.5760 on ground neither trained on.

So the browser needs both graphs, and they do not share anchors:

  face  MediaPipe BlazeFace, folded weights, the original w=h=1.0 squares
  ear   the crop-trained checkpoint, the fitted ear priors

Decoding either through the other's anchors yields boxes at the wrong scale
rather than an error, which is exactly the failure this repo keeps re-learning,
so each graph carries its own anchors baked in.

Both are exported web-style: all 896 decoded boxes plus scores, leaving
thresholding and NMS to JavaScript, because TopK and NMS emit int64 and ONNX
Runtime Web rejects it.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from blazebase import checkpoint_is_folded, load_checkpoint_state
from blazeear import BlazeEar
from blazeear_inference import BlazeEarWebExportable
from make_face_crops import load_face_detector
from utils.anchor_utils import get_anchors

OPSET = 17
INPUT_SIZE = 128
# Boxes come out in original-image pixels, so judge them in pixels: a
# hundredth of a pixel cannot move a detection. Scores are probabilities
# and get a tighter bound.
BOX_TOLERANCE_PX = 0.01
SCORE_TOLERANCE = 1e-4


def export(wrapper, path, opset=OPSET):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Seeded: the export check compares against this same input, and a probe
    # that passes or fails depending on the draw is not a check.
    generator = torch.Generator().manual_seed(0)
    dummy = (
        torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, generator=generator) * 255,
        torch.tensor(2.5),
        torch.tensor(0.0),
        torch.tensor(60.0),
    )
    torch.onnx.export(
        wrapper, dummy, str(path),
        input_names=['image', 'scale', 'pad_y', 'pad_x'],
        output_names=['boxes', 'scores'],
        opset_version=opset, dynamo=False,
    )
    return dummy


def check(wrapper, path, dummy):
    """The graph has to agree with the model it came from."""
    import onnxruntime as ort

    with torch.no_grad():
        want_boxes, want_scores = wrapper(*dummy)
    session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    got_boxes, got_scores = session.run(None, {
        'image': dummy[0].numpy(),
        'scale': np.array(dummy[1].item(), dtype=np.float32),
        'pad_y': np.array(dummy[2].item(), dtype=np.float32),
        'pad_x': np.array(dummy[3].item(), dtype=np.float32),
    })
    box_err = float(np.abs(got_boxes - want_boxes.numpy()).max())
    score_err = float(np.abs(got_scores - want_scores.numpy()).max())
    ok = box_err < BOX_TOLERANCE_PX and score_err < SCORE_TOLERANCE
    label = 'ok' if ok else 'MISMATCH'
    print(f'  {path.name}: boxes max|d| {box_err:.2e} px, '
          f'scores max|d| {score_err:.2e}  [{label}]')
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ear-checkpoint',
                        default='runs/checkpoints_crop/BlazeEar_best.pth')
    parser.add_argument('--out-dir', default='docs')
    args = parser.parse_args()
    out_dir = Path(args.out_dir)

    # Stage one: the genuine MediaPipe face detector, at its own anchors.
    face = load_face_detector(device='cpu')
    face_wrapper = BlazeEarWebExportable(
        model=face, anchors=face.anchors.cpu(), input_size=INPUT_SIZE).eval()

    # Stage two: the crop-trained ear model, at the fitted priors.
    state = load_checkpoint_state(args.ear_checkpoint)
    ear = BlazeEar(use_batchnorm=not checkpoint_is_folded(state))
    ear.load_state_dict(state)
    ear.eval()
    ear_wrapper = BlazeEarWebExportable(
        model=ear, anchors=get_anchors(), input_size=INPUT_SIZE).eval()

    print('Exporting, then checking each graph against its source model:')
    ok = True
    for wrapper, name in ((face_wrapper, 'BlazeFace_web.onnx'),
                          (ear_wrapper, 'BlazeEar_web.onnx')):
        path = out_dir / name
        dummy = export(wrapper, path)
        ok &= check(wrapper, path, dummy)
        print(f'    -> {path} ({path.stat().st_size / 1024:.0f} KB)')

    if not ok:
        raise SystemExit('a graph disagrees with its model; not shipping it')
    print('\nBoth graphs match their models.')


if __name__ == '__main__':
    main()
