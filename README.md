# BlazeEar

![BlazeEar cover](assets/linkedin_cover_blazeear.png)

**[Live demo](https://shameem4.github.io/BlazeEar/)** &mdash; runs entirely in
your browser, on your device. No image leaves the page.

> ### Where this ended up, and how it got there
>
> An ear detector on MediaPipe BlazeFace, two stages: find a face, crop around
> it, find ears in the crop. It works -- 0.78 crop-space mAP@0.5, 72% recall
> end to end -- and most of what is written below is the record of finding out
> *why* it does not work better.
>
> The short version of that record: **the second stage localises about 88% of
> ears and reports 72% of them.** The gap is almost all scoring, not finding.
> Chasing it produced a long run of plausible mechanisms that measurement
> refuted -- a score-head fix, wider scale augmentation, a per-prior mining
> ceiling, three separate geometric vetoes, a better upstream face detector
> from a sibling project. Every one was reasonable. Every one came back null
> or negative, and the nulls are the useful part, because each replaced a story
> with a number.
>
> What actually moved the needle was never a model change. It was calibration
> (giving the fused and single-frame paths their own measured thresholds,
> +0.11 F1), resolution (256 px input over 128), and -- most recently --
> **using time**: the browser now tracks ears across frames with hysteresis
> instead of re-deciding every frame, which recovers detections a lower
> threshold could not buy without tripling false positives.
>
> Three things turned out to be structural rather than fixable. The 0-32 px
> band is information-limited: the median ear is 24 real pixels, and handed
> 767 ears that a better face stage rescued, in the best crop that stage could
> possibly produce, **1 of 332 in that band scored above the gate** -- and 57%
> of them produced no box at all. Stage-1 coverage is not the bottleneck;
> neither is stage-2 calibration, for those ears. And every cue derived from
> the face box -- crop geometry, face keypoints, a lower face threshold --
> fails in exactly the conditions that create the problem, because it fails
> together with the box it came from.
>
> The habit worth stealing is in the commit log rather than the code. Claims
> here carry the measurement that produced them, retractions stay in the text
> next to what they retract, and the count of mechanisms that sounded right
> and were wrong is kept deliberately. It is about two in three.

**Write-ups:**
[Measuring the whole head, privately, in the browser](https://www.linkedin.com/m/pulse/measuring-whole-head-privately-browser-shameem-hameed-ep0mc)
&middot;
[BlazeEar: extending trainable BlazeFace from faces to ears](https://www.linkedin.com/pulse/blazeear-extending-trainable-blazeface-from-faces-ears-shameem-hameed-dalic/)

> Note on the earlier article: it reports a peak validation mAP@0.5 of about
> 0.46 declining to 0.25. Both figures are withdrawn -- they were an artifact
> of an unshuffled validation prefix ordered by annotation source. See
> [v2: what changed](#v2-what-changed-and-what-it-was-worth) for the
> corrected numbers and how they were measured.

BlazeEar is a lightweight ear detector built on the MediaPipe BlazeFace family of models. The goal is practical ear localization in unconstrained images (profiles, partial occlusions, mixed lighting) with a compact architecture that can run in real time on consumer GPUs and edge devices.

The project started as an experiment to reuse BlazeFace’s anchor layout and training recipe for a different small target (ears). Early versions relied on external detectors to generate pseudo‑labels; the current codebase supports a full pipeline from heterogeneous annotations to a trained BlazeEar model, with optional staged fine‑tuning and tooling for qualitative inspection.

---

## Motivation and Evolution

### Initial phase

- Many open ear datasets are small, biased, or inconsistently annotated.  
- BlazeFace provides a proven, low‑latency detector with a fixed 896‑anchor layout.  
- The first iteration focused on “can BlazeFace anchors + loss learn ears at all?”, using automatic boxes from a YOLO ear detector and pose keypoints to bootstrap training data.

### Current state

- A unified data preparation pipeline merges multiple annotation sources into a single CSV with provenance.  
- The BlazeEar model is a BlazeFace‑style front detector (256×256 input) with ear‑specific training, hard negative mining, and focal loss (on by default). The architecture is MediaPipe's; only the input resolution differs, and the parameter count is identical at either size.  
- Debug and visualization scripts help compare BlazeEar to baseline detectors and export sample predictions.

---

## Lineage: Trainable BlazeFace → BlazeEar

This repository is a direct descendant of the sibling project `trainable_blazeface/` (one folder above this repo). That earlier work solved a specific technical problem: MediaPipe’s published BlazeFace weights are “inference‑only”. BatchNorm statistics are folded into convolution weights and the TensorFlow Lite graph cannot backpropagate. `trainable_blazeface` reintroduced trainable BatchNorm into the BlazeBlocks, unfolded the pretrained parameters into those layers, and verified that the PyTorch model produces the same outputs as MediaPipe at initialization.

Once BlazeFace was trainable end‑to‑end, it became a stable platform for adapting the detector to other small, roughly face‑adjacent targets. Ears are similar enough to faces to benefit from the same anchor pyramid and receptive fields, but different enough to expose the limitations of existing ear datasets and off‑the‑shelf detectors.

> **Correction (v2).** Until the v2 branch, `BlazeEar` built its entire backbone
> from `BlazeBlock_WT` — the *folded* block, with BatchNorm baked into the conv
> weights. `BlazeBlock`, the trainable-BatchNorm block this lineage exists to
> provide, was never instantiated, so the shipped model had **no normalization
> layers at all** and the central claim below was not true of it. Worse, the
> conversion path was unreachable: `load_mediapipe_weights` chose it inside an
> `except RuntimeError`, but `load_state_dict(strict=False)` does not raise on
> missing keys, so loading into a BatchNorm backbone reported success while
> leaving 160 keys missing. Both are fixed; eval-mode parity with the folded
> model at initialization is now 1.2e-4 relative.

What carried over unchanged:

- The front‑detector backbone, two‑scale heads, and 896‑anchor layout.  
- The training recipe from `vincent1bt/blazeface‑tensorflow`: anchor target encoding, hard negative mining, and SmoothL1 regression.  
- Weight initialization via `load_mediapipe_weights()` and the same preprocessing/NMS conventions.

What changed to become BlazeEar:

- `blazeface.py` became `blazeear.py` with identical topology but a new task label.  
- `data_prep.py` expanded from “single‑format face boxes” to a multi‑source ear annotation merger with provenance and optional pseudo‑labeling.  
- Training defaults and debug tooling shifted to ear‑specific thresholds and failure modes.

In short: `trainable_blazeface` made BlazeFace trainable; BlazeEar uses that foundation to address ear localization with a purpose‑built dataset pipeline and training run.

---

## Repository Structure

- `blazebase.py`  
  Core BlazeBase utilities: preprocessing, anchor generation, TFLite‑compatible padding, and MediaPipe weight loader.
- `blazedetector.py`  
  Base detector class with MediaPipe preprocessing, batched inference, NMS, and a `get_training_outputs()` path that bypasses post‑processing.
- `blazeear.py`  
  BlazeEar architecture. Anchor count follows the input size: 3584 at the shipped 256 px (2048 from 32×32×2 + 1536 from 16×16×6), 896 at 128 px (512 from 16×16×2 + 384 from 8×8×6). Regressor outputs 16 coords per anchor (4 box + 6 keypoints ×2), though only box coords are supervised by default.
- `train_blazeear.py`  
  Full training pipeline: CSV/NPY dataloading, anchor target encoding, loss, metrics, mixed precision, checkpointing, and TensorBoard logging.
- `data_prep.py`  
  Annotation unification and split generation. Merges COCO/CSV/PTS/LFPW and optional automatic sources (YOLO ear detector, pose ears). Produces `data/splits/train.csv` and `data/splits/val.csv`.
- `dataloader.py`  
  CSV dataset with resize‑pad to square, strong augmentations, and encoding into BlazeFace anchors via `utils/anchor_utils.py`.
- `loss_functions.py`  
  `BlazeEarDetectionLoss`: hard negative mining + BCE or focal classification + SmoothL1 regression.
- `queue_augment_scale_ab.sh`  
  The queued A/B for the augmentation lever, not started. Two sequential
  arms, control at the old scale range and treatment at the new one, under
  one checked-out tree via `BLAZEEAR_AUGMENT_SCALE_{LOW,HIGH}` -- because
  `train_blazeear.py` now has `--seed`, defaulting to None so prior runs are
  described correctly as unseeded. A single unseeded run cannot be compared
  against the shipped checkpoint. Writes an as-run snapshot and checksums
  before launching anything, and runs under `systemd-run --user` so the
  output lands in the journal and survives a reboot.
- `measure_tragion_coverage.py`, `measure_crop_shift.py`,
  `measure_anchor_budget.py`  
  Standalone measurements behind claims in this README, kept runnable so the
  numbers can be rechecked rather than trusted: whether the face stage's
  predicted tragions land on the ears that escape the crop (they do not), what
  a same-size shifted crop could recover (5.1pp, given a direction), and where
  the anchor budget goes against where the ears actually are.
- `utils/`  
  Anchor helpers, augmentations, visualization, metric utilities, debug scripts, and tests.
- `runs/`  
  Training artifacts. `runs/checkpoints/` contains saved `.pth` weights, and `runs/logs/` holds TensorBoard events and debug screenshots.

---

## Architecture

BlazeEar matches the MediaPipe BlazeFace front detector topology. The network is a small two‑stage BlazeBlock backbone followed by two detection heads operating at different strides.

```text
Input RGB image (H×W×3)
        │
        ▼
Resize + pad to square
Resize to 256×256
        │
        ▼
Conv 5×5 s=2, 24ch
        │
        ▼
Backbone1: BlazeBlock stack
  → feature map 16×16×88
        │
   ┌────┴─────────────┐
   │                  │
   ▼                  ▼
Classifier_8          Regressor_8
16×16 grid            16×16 grid
2 anchors/cell        2 anchors/cell
512 scores            512 box+kp preds
   │                  │
   └──────┬───────────┘
          │
          ▼
Backbone2: BlazeBlock stack
  → feature map 8×8×96
          │
   ┌────┴─────────────┐
   │                  │
   ▼                  ▼
Classifier_16         Regressor_16
8×8 grid              8×8 grid
6 anchors/cell        6 anchors/cell
384 scores            384 box+kp preds
   │                  │
   └──────┬───────────┘
          │
          ▼
Concat across scales
  scores: (896, 1) logits
  boxes:  (896, 16) raw coords
          │
          ▼
Decode + sigmoid + NMS
→ final ear detections
```

Anchor pyramid:

- Small scale: 16×16 grid with 2 anchors per cell (512 anchors).  
- Large scale: 8×8 grid with 6 anchors per cell (384 anchors).  
- Total: 896 anchors, each predicting 1 class score and 16 regression values (4 box + 12 keypoint coords).

Anchor priors are fitted to the ear box statistics rather than inherited from
MediaPipe. The original anchors were all `w = h = 1.0`, so the 896 anchors
covered only 320 distinct positions with no scale or aspect prior at all;
measured over 13477 human boxes, the closest anchor reached IoU 0.5 for 0.1% of
ears. MediaPipe's own variable-size anchors reach 18.0%, and priors fitted by
k-means reach 23.8%. Even fitted, most ears never reach IoU 0.5, because the
limit is spatial — stride 8 spaces anchor centres 0.0625 apart for objects
0.053 wide — so targets are assigned best-match per box rather than by an IoU
threshold. A stride-4 head would raise that to 57.6% and is the largest
remaining architectural gain.

---

## Pipeline Dataflow

End‑to‑end flow from raw data to a trained model:

```text
Raw datasets under data/raw/
  COCO / CSV / PTS / LFPW
        │
        ▼
data_prep.py
  - decode annotations
  - optional YOLO ear boxes
  - optional pose ear keypoints
  - merge + de‑duplicate
        │
        ▼
master.csv (with annotation_source)
        │
        ├───────────────┐
        ▼               ▼
train.csv           val.csv
        │               │
        ▼               ▼
CSVDetectorDataset (dataloader.py)
  - load image
  - resize/pad to 256×256
  - augment (color + geometry + occlusion)
  - normalize boxes
  - encode to anchors (utils/anchor_utils.py)
        │
        ▼
train_blazeear.py
  - get_training_outputs()
  - BlazeEarDetectionLoss
  - hard negative mining
  - AMP / freeze‑thaw optional
        │
        ▼
runs/checkpoints/*.pth
runs/logs/BlazeEar/*
        │
        ▼
utils/debug_training.py
  - qualitative screenshots
  - metric sweeps / comparisons
```

---

## Installation

1. Create an environment (Python 3.10+ recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

Notes:

- Training and debug scripts expect PyTorch and Ultralytics.  
- CUDA is optional but strongly recommended for training.

---

## Data Preparation

Raw data should live under `data/raw/` in any of the supported formats. The prep script walks subfolders, decodes annotations, and produces a single metadata table.

```bash
python data_prep.py \
  --data-root data/raw \
  --output-csv data/splits/master.csv \
  --train-csv data/splits/train.csv \
  --val-csv data/splits/val.csv
```

Key behaviors:

- Supports COCO JSON, CSV, PTS, and LFPW TXT formats via `utils/data_decoder.py`.  
- Optional pseudo‑labeling with Ultralytics YOLO and ear keypoints from a pose model.  
- Each box records an `annotation_source` value (e.g., `GT`, `EAR`, `POSE`, `GT+EAR`) to preserve provenance.

---

## Training BlazeEar

The primary path is CSV training that mirrors the MediaPipe front detector setup:

```bash
python train_blazeear.py \
  --train-data data/splits/train.csv \
  --val-data data/splits/val.csv \
  --data-root data/raw
```

There is no `--csv-format` flag, though this block documented one until it was
run: the format is inferred from the extension. Nor is there an
`--input-size` -- that comes from `DEFAULT_INPUT_SIZE`.

Useful flags:

- `--init-weights mediapipe-backbone|mediapipe|scratch`  
  Default `mediapipe-backbone` loads the backbone and re-initializes the
  detection heads with a background prior. MediaPipe's head is calibrated for
  the folded backbone's activation scale, and trainable BatchNorm renormalizes
  the features under it: inherited, it starts at a median logit of -339 on
  anchors holding an ear. Re-initialized, -4.8.
- `--freeze-thaw`  
  Staged training: heads only → backbone2 → full model (see script defaults).
- `--no-focal-loss`  
  Fall back to BCE. **Focal loss is the default**, not an opt-in:
  `train_blazeear.py` calls `parser.set_defaults(use_focal_loss=True)`, so
  every run in this repository's history trained with it unless this flag was
  passed. The CLI defaults are `--focal-alpha 0.35 --focal-gamma 1.5`; the
  `0.25 / 2.0` pair is `BlazeEarDetectionLoss`'s own signature default, which
  the trainer overrides and which therefore never reaches a training run.
  An earlier version of this section documented `--use-focal-loss` as the way
  to switch focal loss on, with the `0.25 / 2.0` figures -- both wrong.
- `--amp / --no-amp`  
  Mixed precision toggle.
- `--resume runs/checkpoints/BlazeEar_epochXX.pth`  
  Resume from an explicit checkpoint.

Outputs:

- Checkpoints in `runs/checkpoints/` (best model at `runs/checkpoints/BlazeEar_best.pth`).  
- TensorBoard logs in `runs/logs/BlazeEar/`.

To view logs:

```bash
tensorboard --logdir runs/logs
```

---

## Training Results

> **These figures were withdrawn.** The v2 branch found the measurement that
> produced them to be wrong in several independent ways, and the model is being
> retrained. Nothing below should be cited until that finishes.

The previously reported "peak validation mAP@0.5 ≈ 0.46, declining to ≈ 0.25"
was **a measurement artifact, not a training dynamic**. Per-epoch validation
scored `200 // batch_size` batches while the final summary scored
`500 // batch_size`, and `val.csv` preserved master-CSV row order, which is
source-by-source — so a short prefix sampled one easy dataset. Holding the
epoch-94 checkpoint completely fixed and varying only how many validation
images are scored reproduces the whole "decline":

| val images scored | mAP@0.5 |
| --- | --- |
| 96 | 0.5508 |
| 192 | 0.4286 |
| 480 | 0.2497 |
| 2022 (all) | 0.2801 |

The metric itself was also non-standard: VOC07 11-point AP computed per image
and averaged, on images holding one or two boxes, which measures quantization
more than detection. Pooled properly over the whole split, that same checkpoint
scores **mAP@0.5 = 0.149**, or **0.213** against human-verified labels only.
Around 38% of validation boxes were pose-model pseudo-labels that no detector
reproduces — 97.8% of them have zero overlap with any human box.

Corrected numbers will be published here after the retrain.

Qualitative samples are exported to `runs/logs/debug_images/`. These PNGs show predicted boxes over diverse poses and occlusions and are useful for spotting failure modes (small ears, heavy occlusion, extreme profiles).

Examples (from this run):

![Qualitative example](assets/frontal.png)

![Qualitative example](assets/profile.png)

![Qualitative example](assets/multi_face.png)

![Qualitative example](assets/crowd_small.png)

![Qualitative example](assets/qual_example_1.png)

![Qualitative example](assets/qual_example_2.png)

---

## Inference and Debugging

The simplest programmatic usage:

```python
from blazeear_inference import BlazeEarInference

pipeline = BlazeEarInference(
    weights_path="runs/checkpoints/BlazeEar_best.pth",
    device="cuda",
)

# img: numpy HxWx3 RGB
dets = pipeline.predict(img)  # (N, 5) [ymin, xmin, ymax, xmax, score] in image space
```

`BlazeEarInference` accepts both checkpoint layouts (a bare state dict, or a
training checkpoint with the weights under `model_state_dict`). To drive the
bare model instead, unwrap the checkpoint yourself and build the anchors after
moving the model to its device:

```python
import torch
from blazeear import BlazeEar
from utils.anchor_utils import anchor_options

model = BlazeEar()
checkpoint = torch.load("runs/checkpoints/BlazeEar_best.pth", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.to("cuda").eval()
model.generate_anchors(anchor_options)

dets = model.process(img)  # returns boxes in original image space
```

Note: `load_weights()` only handles a bare state dict, so it cannot load the
`.pth` files written by `train_blazeear.py` directly.

Utilities:

- `utils/image_demo.py` and `utils/webcam_demo.py`  
  Run BlazeEar on images or webcam streams.
- `utils/debug_training.py`  
  Compare BlazeEar vs MediaPipe weights, dump debug screenshots, and run quick eval loops.

Example screenshot export:

```bash
python utils/debug_training.py \
  --weights runs/checkpoints/BlazeEar_best.pth \
  --data-root data/raw \
  --csv data/splits/val.csv \
  --screenshot-count 20
```

---

## Optional: YOLO auto-labelling (not distributed)

A fine-tuned YOLOv11 detector was used to propose ear labels while building
`data/splits/*.csv`. Neither the datasets nor the splits are distributed here
(`/data` is gitignored), so none of this is needed to run or evaluate the
shipped model.

The tooling itself is **not in this repository**: `finetune_yolov11.py`,
`utils/yolo11_demo.py` and `model_weights/yolov11_ear_detector.pt` depend on
[Ultralytics](https://github.com/ultralytics/ultralytics), which is
**AGPL-3.0** and cannot be redistributed under Apache-2.0. See
[NOTICE](NOTICE).

`data_prep.py` and `relabel.py` ship with optional YOLO paths that import
Ultralytics lazily, so both files work without it installed. To re-run label
proposal you would install Ultralytics yourself and accept its terms:

```bash
pip install "ultralytics>=8.2.0"
python relabel.py propose --weights <your-yolo-weights.pt> ...
```

---

## Testing

Unit tests live under `utils/tests/`.

```bash
pytest
```

---

## Export to ONNX

This repo includes a small helper script that exports the training checkpoint format
(`model_state_dict` inside the `.pth`) to an ONNX graph.

```bash
# Needed for the torch.export-based exporter:
pip install onnx onnxscript
# Needed to check the exported graph against the model:
pip install onnxruntime

python export_onnx.py \
  --checkpoint runs/checkpoints/BlazeEar_best.pth \
  --output runs/checkpoints/BlazeEar_best.onnx
```

Every export is then run through ONNX Runtime and compared against the PyTorch
model it came from; the script refuses a graph that disagrees. `--no-check`
skips the comparison, and it is skipped with a warning if `onnxruntime` is not
installed. Exporting without checking is how a graph that is quietly wrong gets
shipped, so prefer leaving the check on.

Note on opsets: PyTorch 2.9 made the `torch.export`-based exporter the default
and dropped the `fallback` argument that used to reach the legacy
TorchScript exporter, so the old `--no-fallback` flag is gone. That exporter
implements opset 18 upwards and keeps 18 when it cannot down-convert a node --
this model's `Pad` has no opset 17 adapter -- so `--opset` defaults to 18.
Asking for 17 produces a noisy conversion failure and an opset 18 graph anyway.

(`export_two_stage_web.py`, which writes the `docs/*.onnx` graphs the browser
demo loads, deliberately still calls the legacy exporter at opset 17, because
ONNX Runtime Web is the consumer there.)

The exported model takes `image: (N, 3, 256, 256)` and returns:

- `raw_boxes: (N, 3584, 16)`
- `raw_scores: (N, 3584, 1)` (logits)

To export an ONNX model that includes post-processing (decode + NMS) and returns final detections:

```bash
python export_onnx.py \
  --postprocess \
  --checkpoint runs/checkpoints/BlazeEar_best.pth \
  --output runs/checkpoints/BlazeEar_best_post.onnx
```

`detections` is `(num_detections, 5)` in `[ymin, xmin, ymax, xmax, score]`
(normalized coords), and as of now that is what it actually returns.

**It used to return the zeroed tail as well.** Sub-threshold candidates are
zeroed rather than removed, because `topk` needs a fixed length -- but the
zeros were then passed through NMS and returned. On a real crop the graph
emitted 56 to 72 rows where 1 or 2 cleared the gate, the rest at score
exactly 0.0, so a caller that did not re-filter got dozens of phantom boxes
per image. Nothing shipped consumed it: `docs/blazeear_inference.js` requires
outputs named `boxes` and `scores` and refuses any other graph, and this one
emits `detections`.

The tail was known -- the self-check filtered `score > 0.0` on both sides
before comparing, with a comment saying why -- so the graph was never
numerically wrong where it was checked. It was a contract defect, and it had
one real consequence: **the graph could not be compared against its own
wrapper at all.** The tail's order is NMS tie-break among equal zero scores,
which torch and onnxruntime resolve differently, so row counts disagreed (165
against 56) while every real detection matched exactly. Dropping the tail
makes the comparison exact. Measured over 300 real crops: 0 shape mismatches
against 300 before, 197 detections rather than ~17,000 rows containing 197
real ones, agreement to 1.43e-06, and the same 197 that `BlazeEarInference`
produces on the same inputs.

**Use `--check-image` with `--postprocess`.** The default probe is
`torch.randn`, and the wrapper maps its input through `x / 127.5 - 1.0`, so
that probe lands in roughly [-1.03, -0.97] -- near-uniform grey, scoring
nowhere near the 0.8 gate. Decode, NMS and the gate then go unverified, and
the script now says that in those words rather than implying a pass. On a real
image the same check reports `max|d| 5.96e-08`:

```bash
python export_onnx.py \
  --postprocess --check-image data/face_crops/<some>/<crop>.jpg \
  --checkpoint runs/checkpoints/BlazeEar_best.pth \
  --output runs/checkpoints/BlazeEar_best_post.onnx
```

`export_onnx.py` inspects the checkpoint and builds the matching architecture,
so pre-BatchNorm checkpoints still export.

### Exporting the two-stage pipeline for the browser

```bash
python export_two_stage_web.py
```

This writes `docs/BlazeFace_web.onnx` and `docs/BlazeEar_web.onnx`, then runs
each graph against the PyTorch model it came from and refuses to ship one that
disagrees. The two graphs carry different anchors -- MediaPipe's squares and
the fitted ear priors -- each baked into its own graph, because decoding either
through the other's anchors gives boxes at the wrong scale rather than an
error. See `docs/README.md` for the JavaScript side.

---

## Evaluation

`evaluate.py` scores a checkpoint over a whole split, pooling every detection
into one precision/recall curve rather than averaging per-image AP, and breaks
the result out by annotation provenance so pseudo-label agreement is never
folded into the headline number.

```bash
python evaluate.py --checkpoint runs/checkpoints/BlazeEar_best.pth
```

Boxes from an untrusted source are scored as ignore regions in the other views,
so excluding them does not turn a correctly-detected ear into a false positive.
Checkpoints predating the v2 anchor change need `--legacy-anchors`.

## Test-time augmentation

The two-stage pipeline can detect each ear through several nuisance-perturbed
crops and fuse the agreeing boxes. The ear never moves between views, so the
spread between those detections is the model's own noise, and fusing cancels
part of it. Views are `(expand, flip)` pairs in `utils/tta.py`, mirrored in
`docs/blazeear_inference.js` and pinned against each other by
`utils/tests/tests/test_tta.py`.

```bash
python evaluate_two_stage.py --csv data/splits/common_heldout.csv \
  --data-root data/raw \
  --crop-checkpoint model_weights/BlazeEar_best_256px.pth --tta 4
```

**The tables in this section were measured on the 128 px model**, which is what
TTA was developed against; they are kept because the conclusions about fusion
(the rule, the suppression, the view count) are what they establish, and those
carry over. For how TTA interacts with the shipped 256 px model -- the short
version is that resolution absorbs most of the gain -- see
[the 256 px section](#the-256-px-input-shipped).

Measured on the 279-image common held-out set at 128 px, against the same
checkpoint, with a paired bootstrap over 2000 resamples of the images:

| views | mAP@0.5 | mAP@[.5:.95] | det IoU | paired gain, mAP@0.5 |
|---|---|---|---|---|
| 1 (no TTA) | 0.5809 | 0.2445 | 0.7193 | --- |
| 2 | 0.6078 | 0.2565 | 0.7258 | +0.0240 [-0.0018, +0.0533] |
| 4 | 0.6299 | 0.2776 | 0.7284 | +0.0467 [+0.0191, +0.0788] |

Only the four-view interval excludes zero, so four views is the configuration
worth running; at two the gain is real-looking but not separable from noise on
a set this size. It costs one full ear-model pass per view per face, and the
face stage still runs once per frame regardless: 2.55 ms per view at 128 and
7.72 ms at 256, on one CPU thread.

Six and eight views were then measured against the shipped four, extending the
pattern with flips of the two scale views and a wider scale pair:

| views | mAP@0.5 | mAP@[.5:.95] | det IoU | paired gain vs 4 views, mAP@0.5 |
|---|---|---|---|---|
| 6 | 0.6402 | 0.2826 | 0.7325 | +0.0102 [-0.0076, +0.0282] |
| 8 | 0.6346 | 0.2868 | 0.7338 | +0.0039 [-0.0172, +0.0242] |

Both intervals straddle zero and mAP@0.5 is not even monotonic -- six views
beats eight -- so four stands. `VIEWS` therefore holds four entries, and
`--tta N` above that is now an error: `VIEWS[:N]` silently returns the whole
list, so `--tta 8` had been running four views and reporting success.

Two findings are worth repeating because both cost a measurement to find:

- A fused box takes the **highest** score among its members. Scoring it as
  sum-over-view-count instead demotes real ears that only one view found; at
  two views that rule scored *below* no TTA at all (0.5530 against 0.5809).
- Fusion clusters at a looser overlap than the pipeline's NMS, so the fused set
  must go through NMS afterwards as well. Without it, more views simply emit
  more boxes: 7 / 12 / 17 detections for 1 / 2 / 4 views on a two-face image in
  the browser, against 7 / 7 / 11 once suppression was added.

Both of those were found by running the browser pipeline against the real ONNX
graphs, and so was the anchor-count bug that the move to 256 exposed: the
browser read a hardcoded `numAnchors = 896` from a graph now emitting 3584, so
it saw only the top fourteen rows of the grid. Nothing threw -- the tensor is
simply longer than the loop -- and the whole Python suite passed. It showed up
as detections at confidence 0.32 where Python got 0.96. The count now comes
from the output tensor. **A JS change is not verified by the Python tests**;
the parity worth checking is one image through both paths, which should agree
to a fraction of a pixel (canvas and cv2 resample slightly differently).

The browser pipeline runs four views by default; pass `{views: 1}` to
`createTwoStageDetector` for the single-crop behaviour.

### TTA needs its own score threshold

mAP sweeps the entire score range, so it says nothing about the single
threshold a deployment actually sits at -- and fusion moves the score
distribution, because a fused box takes its best member's score. Sweeping the
operating point on the same held-out set, at IoU 0.5:

| threshold | plain F1 | TTA 4 F1 |
|---|---|---|
| 0.70 (the old shared value) | 0.6212 | 0.5824 |
| 0.80 | **0.6646** | 0.6533 |
| 0.85 | 0.6575 | **0.6864** |
| 0.90 | 0.5790 | 0.6884 |

So the inherited 0.70 was wrong for both paths, and **at 0.70 turning TTA on
was a measured regression**: -0.0389 F1 against no TTA (95% CI [-0.0649,
-0.0123]), while still gaining +0.0467 mAP@0.5. A ranking gain and an
operating-point loss are not in conflict; they are different questions, and
only the first had been asked. Recalibrating TTA to its own threshold is worth
+0.1054 F1 (95% CI [+0.0715, +0.1404]) -- the largest confirmed gain here, and
it is a constant, not a model.

`DETECTION_SCORE_THRESHOLD` was set to 0.80 here and `TTA_SCORE_THRESHOLD` to
0.85, mirrored
in the browser pipeline and pinned by `test_tta.py`. **The plain value is now
0.70**, re-picked on a 2,193-ear crop-space sweep -- see "Why the gate was
0.80" below; the 0.70 row in the table above is the superseded 355-ear
end-to-end measurement, and the TTA value is unchanged.
TTA takes 0.85 rather than
its argmax 0.90 because F1 is flat over 0.85-0.90 and then falls off a cliff
(0.666 at 0.91); the plateau centre is the safer place to sit. With both paths
at their own best, TTA's F1 edge is +0.0234 [-0.0028, +0.0496] -- not
distinguishable. TTA buys ranking quality, not a better operating point.

## Relabelling

Most images here carry exactly one human-labelled ear (11084 of 13482), so the
second ear is usually present but unlabelled — and hard negative mining selects
the highest-scoring background anchors, which is precisely that ear.
`relabel.py` closes the gap with human review rather than more pseudo-labels:

```bash
python relabel.py propose --weights <labeller>/best.pt
python relabel.py review
python relabel.py apply --output data/splits/master_relabelled.csv
```

A reviewer can also mark a real ear whose box is wrong (queued for correction,
never merged) or one too degraded to learn from (merged as an ignore region, so
it is neither a target nor a hard negative).

---

## v2: what changed, and what it was worth

The v2 branch set out to fix correctness problems and ended up changing the
architecture, because the measurements kept pointing at the same thing.

### The headline

Every checkpoint below is scored through one identical detection path, on the
**279 images (355 ears, 12 sources) that no model in this repo's history ever
trained on**. That set matters: the splits changed during v2, and 85.3% of the
current validation images sit inside the *original* model's training set, so
the obvious comparison would have measured memorisation.
`make_common_heldout.py` builds it, using content-hash photo identity rather
than filenames, because one photo appears under several Roboflow hashes.

| model | mAP@0.5 | mAP@[.5:.95] | detection IoU |
|---|---|---|---|
| original (pre-v2) | 0.1790 | 0.0421 | 0.6189 |
| v2 single-stage | 0.3142 | 0.0938 | 0.6558 |
| **v2 two-stage** | **0.5809** | **0.2445** | **0.7193** |

**3.2x on mAP@0.5, 5.8x on mAP@[.5:.95].** On the full 2021-image validation
split the two v2 models read 0.3561 and 0.6447.

All two-stage figures are at the shipped face threshold of 0.2. That value was
chosen by sweeping the validation split, so it is worth saying that the
held-out set above is independent of that sweep and agrees: 0.5760 at 0.3,
0.5809 at 0.2.

Two honest caveats. 279 images is a small set -- treat a few points as noise,
not the 3x. And the gain is not purely architectural: v2 also relabelled the
data, and the evaluation ground truth is the relabelled set the original never
trained against. Separating those would need the original design retrained on
the new labels.

### The architecture change

The single biggest finding was a structural one. The median ear is **6.8 x 14.0
px** once a full frame is squeezed into the 128x128 input, and **65.7% of ears
are smaller than BlazeFace's smallest anchor**. The model was being asked to do
something its anchor geometry could not express.

MediaPipe does not work this way. Its pipelines run a coarse detector on the
full frame and a fine model on a crop around each hit. Restoring that pattern
-- BlazeFace, then a square crop at 1.5x the face box, then the ear model --
raises the median ear to 32.5 px and accounts for most of the gain above.

Stage one is MediaPipe's published BlazeFace weights, unmodified. Nothing about
this deviates from MediaPipe; the original design had simply collapsed a
two-stage pattern into one stage.

The cost is a recall ceiling: an ear the pipeline never shows the second stage
cannot be found. Measured per ear, 11.3% are unreachable -- 2.5% because no
face was detected, and 8.7% because the ear fell outside every crop window
despite a face being found (see
[the ceiling section](#the-recall-ceiling-is-crop-geometry-not-face-detection)).
All of that is charged honestly in every number above -- those ears count as
misses rather than leaving the denominator. A full-frame fallback on those images was
measured and is a wash, because the ears it would have to find there are a
median 4.4 px.

See `make_face_crops.py`, `evaluate_two_stage.py`, and the P7 section of
`V2_PLAN.md`.

### Correctness fixes that came first

None of this would have been measurable without fixing the measurement. In
rough order of how much they mattered:

- **The published "mAP 0.46 peak, declining to 0.25" was an artifact.** The
  same fixed checkpoint reads 0.5508 / 0.4286 / 0.2497 / 0.2801 as the
  validation prefix grows through 96 / 192 / 480 / 2022 images, because the
  prefix was unshuffled and ordered by annotation source. Both figures are
  withdrawn.
- **A validation loop that scored one image.** Overlapping edits left the loop
  body only reassigning variables, so everything after it ran once on the final
  batch. Every validation number for a stretch of v2 was wrong.
- **`load_mediapipe_weights` had a conversion path that never executed** --
  160 missing keys, reported as success.
- **73.5% of positive anchors sat inside `torch.clamp`'s exact-zero-gradient
  dead zone** (median logit -1047).
- **`augment_face_cutout` erased the ear entirely in 120 of 200 seeds** while
  keeping the positive label.
- **The geometry filter rejected 25.86% of real ears**, and ran in one of four
  inference paths.
- **Duplicate photos straddled the split.** Content hashing moved 121 photos
  to one side; 4.1% of the corpus is duplicate files.
- **POSE pseudo-labels are junk**, not a rescalable convention: 97.8% have zero
  overlap with human boxes. They are ignore regions now.
- **Hard-negative mining was starved** by a default of 1.5, giving 10 negatives
  out of 894.

### Consolidation

- **Five NMS implementations became one.** They disagreed on the boundary case
  and ran at three different IoU thresholds, so every reported mAP described
  post-processing nothing deployed. Replacing the trainer's Python loop over
  CUDA tensors also cut a validation pass from **50.4s to 8.9s at identical
  mAP**.
- **The test runner passed by not looking.** It drove `unittest` discovery over
  a mostly pytest-style suite, reporting `Ran 51 tests ... OK` while `pytest`
  ran 336. It delegates to pytest now.
- Detection thresholds, the anchor count and grid sizes have one definition
  each in `utils/config.py`, with a raise if the built anchor count ever
  disagrees.

Tests went from 41 to 344.

### Reproducing the comparison

```bash
python make_common_heldout.py

python evaluate_two_stage.py \
  --csv data/splits/common_heldout.csv \
  --checkpoint runs/checkpoints/BlazeEar_best.pth --legacy-anchors

python evaluate_two_stage.py \
  --csv data/splits/common_heldout.csv \
  --checkpoint runs/checkpoints_v2/BlazeEar_best.pth \
  --crop-checkpoint model_weights/BlazeEar_best_256px.pth
```

The two `runs/` checkpoints above are local training output and are not
distributed (`/runs` is excluded from version control); only the two crop
models in `model_weights/` ship.

`--legacy-anchors` is needed for the pre-v2 checkpoint: it was trained against
the original `w=h=1.0` squares, and decoding it through the fitted ear priors
gives boxes at the wrong scale rather than an error. Those checkpoints also
predate the 256 px input, so they need `--input-size 128`.

### Building the face-crop dataset

```bash
python make_face_crops.py --keep-empty-crops
python split_face_crops.py
python train_blazeear.py \
  --train-data data/splits/crops_train.csv \
  --val-data data/splits/crops_val.csv \
  --data-root data/face_crops/ \
  --checkpoint-dir runs/checkpoints_crop --log-dir runs/logs_crop
```

`--keep-empty-crops` matters: 29.6% of face crops contain no visible ear, and
training without them leaves the model never having seen a face whose ears are
hidden while being handed exactly that a third of the time at inference.

## License

Apache License 2.0 &mdash; see [LICENSE](LICENSE) and [NOTICE](NOTICE).
This replaces the earlier CC BY-NC 4.0 terms, so commercial use is now
permitted.

**The code, the architecture and the trained weights are Apache-2.0.**

The sample photographs are not, and are not meant to be. They are
illustrations of model output: dataset photographs with detection boxes drawn
over them, used to show what the detector does. They keep their original
authors' Creative Commons terms, credited in [NOTICE](NOTICE):

- Seven are from [Open Images](https://storage.googleapis.com/openimages/web/index.html),
  licensed by their photographers under
  [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/), each listed by
  its Open Images ID.
- Two are from
  [*ear annotations.*](https://universe.roboflow.com/annotations-rzsmp/ear-annotations)
  by Roboflow user `annotations-rzsmp`, licensed
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

All nine are **modified** &mdash; bounding boxes and score labels were drawn
on &mdash; which CC BY requires be stated, and [NOTICE](NOTICE) states it.

The training data is not distributed: `/data` is excluded from version
control, so no dataset images or annotations are redistributed. The YOLO
auto-labelling tooling is AGPL-3.0 and is not distributed either (see below).

BlazeEar derives from [MediaPipe](https://github.com/google-ai-edge/mediapipe)
BlazeFace, which is also Apache-2.0, and the two-stage pipeline runs
MediaPipe's published BlazeFace weights unmodified as its first stage.

The YOLO auto-labelling tooling that depends on
[Ultralytics](https://github.com/ultralytics/ultralytics) is **AGPL-3.0**, so
it is not distributed here: `finetune_yolov11.py`, `utils/yolo11_demo.py` and
`model_weights/yolov11_ear_detector.pt` are excluded from version control, and
`ultralytics` is not in `requirements.txt`. The optional YOLO paths that do
ship, in `data_prep.py` and `relabel.py`, import it lazily and are not
reachable without installing it yourself. Nothing in the BlazeEar model, its
training, its evaluation, or the browser demo needs it. See [NOTICE](NOTICE).

## Next Directions

Areas that naturally follow from the current pipeline:

- Add explicit ear keypoint supervision if labels become available (regressor heads already exist).  
- Calibrate detection thresholds per operating point. The shared defaults
  are now fitted rather than inherited (see above); what remains is a
  front-facing vs profile split, which needs those subsets labelled. Note
  that the *size* axis of this is already closed -- a per-size gate was
  swept and has no headroom, because the ranking inside each band is fine
  and the base rate is not. See "The score head discards ears it has
  already localised".
- Widen `augment_scale`'s low end from 0.75 to about 0.35, so the training
  distribution reaches the size band the model is worst at. One line, 1-3
  GPU-hours, untested; the reasoning is in the score-head section below.
- Cap hard-negative mining per feature level. 57% of the anchor budget
  currently takes global mining pressure while receiving 7.7% of the
  positives. Same section.
- Export to ONNX/TFLite for edge deployment.
- Price 320 px. 256 paid, the grid argument keeps scaling, and the
  anchor machinery is now size-parameterised, so the run is a constant
  change -- but CPU grows as the square and 3.0x is already the headline
  cost of 256.

### The 256 px input (shipped)

The network is fully convolutional, so the MediaPipe weights load unchanged at
any input size and only the spatial grids grow. At 256 px the stride-8 head
becomes 32x32 and the stride-16 head 16x16, so the anchor set goes 896 -> 3584
and anchor centres sit 0.03125 apart instead of 0.0625. The priors need no
refit: they are normalized to the input, so the same prior simply lands on a
finer grid. **Nothing about the architecture, the parameter count or the
two-stage pipeline changes.**

This is the shipped configuration. The face stage stays at 128 -- those are
MediaPipe's weights at their own input size -- so both sizes exist in one
process, and input size is therefore a `BlazeEar` constructor argument rather
than only a module constant. `FACE_INPUT_SIZE` pins the face stage explicitly;
inheriting `DEFAULT_INPUT_SIZE` there would hand the face weights a 256 px
input and a 3584-anchor decode.

Retrained at 256 with the same data, initialisation, hyperparameters and 100
epochs. Both checkpoints ship, so every figure below is reproducible:
`model_weights/BlazeEar_best_128px.pth` and
`model_weights/BlazeEar_best_256px.pth`.

#### 128 against 256, every way it was measured

Three measurements, because they answer different questions and do not agree in
magnitude. **Read the path label before comparing any two numbers here** -- the
two crop-space paths differ by 0.03 mAP@0.5 on the *same* checkpoint, and
mixing them cancelled this entire effect on the first attempt.

| measurement | 128 | 256 | delta |
|---|---|---|---|
| **crop-space, deployed path** (`crops_val`, 2193 ears) | | | |
| &nbsp;&nbsp;mAP@0.5 | 0.7426 | **0.7783** | +0.0357 |
| &nbsp;&nbsp;mAP@[.5:.95] | 0.3405 | **0.3596** | +0.0191 |
| &nbsp;&nbsp;det IoU | 0.7338 | **0.7390** | +0.0052 |
| **crop-space, dataloader path** (`evaluate.py`, same split) | | | |
| &nbsp;&nbsp;mAP@0.5 | 0.7106 | **0.7456** | +0.0350 |
| &nbsp;&nbsp;mAP@[.5:.95] | 0.3246 | **0.3446** | +0.0200 |
| &nbsp;&nbsp;det IoU | 0.7273 | **0.7334** | +0.0061 |
| **end to end, 1 view** (`common_heldout`, 355 ears) | | | |
| &nbsp;&nbsp;mAP@0.5 | 0.5809 | **0.6766** | +0.0957 |
| &nbsp;&nbsp;mAP@[.5:.95] | 0.2445 | **0.2962** | +0.0517 |
| &nbsp;&nbsp;det IoU | 0.7193 | **0.7314** | +0.0121 |
| **end to end, 4 views** (TTA, the browser default) | | | |
| &nbsp;&nbsp;mAP@0.5 | 0.6299 | **0.6961** | +0.0662 |
| &nbsp;&nbsp;mAP@[.5:.95] | 0.2776 | **0.3188** | +0.0412 |
| &nbsp;&nbsp;det IoU | 0.7284 | **0.7412** | +0.0128 |

Paired bootstrap on the deployed crop-space path, 1000 resamples over images:

| paired difference | 95% CI | |
|---|---|---|
| mAP@0.5 +0.0353 | [+0.0210, +0.0499] | real |
| mAP@[.5:.95] +0.0188 | [+0.0089, +0.0281] | real |
| det IoU +0.0052 | [+0.0001, +0.0099] | real, but only just |

Three things in that table are worth more than the headline.

**The end-to-end gain is nearly three times the crop-space gain** -- paired,
+0.0926 mAP@0.5 end to end (95% CI [+0.0557, +0.1306]) against +0.0353
crop-space (95% CI [+0.0210, +0.0499]). The intervals are disjoint, so the
difference between the two measurements is real and not sampling noise.

The mechanism first offered for it was wrong, and the test that refuted it is
below. The guess was that end-to-end crops are smaller, so more of them are
upscaled into the input and crop-space under-samples that. The crop-size
distributions are in fact nearly identical -- median side 237 px for
`crops_val` against 262 px live, with 96.6% against 96.5% of crops above
128 px -- so that is not the explanation. What remains is that `crops_val` and
`common_heldout` are different image sets of very different difficulty (0.7426
against 0.5809 mAP@0.5 at 128 px), and a harder set leaves more for resolution
to recover. That is consistent with the numbers but has not been isolated, and
no mechanism specific to the two-stage pipeline survives.

#### Stratified by face crop size

The crop side is `FACE_CROP_EXPAND` x the face box, and that crop is resized
to the input, so the crop side decides whether the input is discarding real
pixels. The prediction was sharp: a crop below 128 px is upscaled by both
models and 256 cannot recover detail that was never captured, so the gain
should vanish there and concentrate in the large-crop bins.

**It does the opposite.** Each ear is assigned to the smallest crop window
containing its centre -- the best-resolution view the pipeline had of it --
and scored COCO-style, with out-of-bin ears as ignore regions so a detection
on one is neither rewarded nor punished. One view, no TTA, so this is
resolution alone:

| bin | ears | 128 | 256 | delta | 95% CI | |
|---|---|---|---|---|---|---|
| crop <=192 px | 85 | 0.5507 | **0.7364** | **+0.1856** | [+0.0776, +0.2753] | 256 better |
| crop 192-256 px | 68 | 0.7004 | 0.7649 | +0.0645 | [-0.0098, +0.1371] | no difference |
| crop 256-384 px | 86 | 0.6138 | 0.6771 | +0.0633 | [-0.0168, +0.1468] | no difference |
| crop >384 px | 76 | 0.8142 | **0.9060** | +0.0918 | [+0.0113, +0.1809] | 256 better |
| unreachable | 40 | 0.0000 | 0.0000 | --- | --- | neither model, ever |

The gain is positive in every bin and **largest in the smallest-crop bin**,
which is where no gain was predicted. There is no monotonic relationship with
crop size, so preserving real detail is not what 256 is buying -- that would
have produced the opposite ordering.

The standard explanation for this in object detection fits: the heads have a
fixed stride in *model* pixels (8 and 16), so what matters is the object's size
relative to that stride, not its size in real pixels. Upscaling the input makes
a small object cover more cells and overlap more anchors even though it adds no
information. The crop-size table supports that reading -- the ear is a
near-constant ~32 px of model input in every bin, because the crop is
proportional to the face, so the only thing resolution changes is how many
cells that ~32 px becomes. It is not isolated, though: the bins also differ in
baseline difficulty (0.55 to 0.81 at 128 px), and the clean test would vary the
stride rather than the input size.

The `unreachable` row scoring exactly zero in both arms is the check that the
binning is doing what it claims.

**Localisation moves least** -- det IoU gains +0.0052 crop-space, the only
interval here that barely clears zero. Consistent with everything else in this
README: det IoU has been the metric least responsive to every intervention
tried, which is the standing reason to think box tightness needs a bigger model
rather than more work on this one.

**Resolution and TTA overlap**, because both reduce the same localisation
noise. TTA's marginal gain falls from +0.0467 at 128 to +0.0194 at 256
(95% CI [+0.0009, +0.0394] -- still real, but only just on mAP@0.5):

| configuration | mAP@0.5 | mAP@[.5:.95] | det IoU | ear CPU per face |
|---|---|---|---|---|
| 128, 1 view | 0.5809 | 0.2445 | 0.7193 | 2.55 ms |
| 128, 2 views | 0.6078 | 0.2565 | 0.7258 | 5.1 ms |
| 128, 4 views | 0.6299 | 0.2776 | 0.7284 | 10.2 ms |
| **256, 1 view** | **0.6766** | **0.2962** | **0.7314** | **7.72 ms** |
| 256, 2 views | 0.6799 | 0.3104 | 0.7350 | 15.4 ms |
| 256, 4 views | 0.6961 | 0.3188 | 0.7412 | 30.9 ms |

**Single-view 256 beats four-view 128 on every metric and costs less than it**
(7.72 ms against 10.2 ms). That is the row to copy if the frame budget matters,
and it is a `{views: 1}` option away in the browser. Four views stay the
default because they are still the most accurate configuration, but the 256
model is doing most of the work the fusion used to.

Cost: 3.0x the ear model's CPU time per view at identical capacity -- 103,418
parameters either way, since only the spatial grids change. The face stage is
unaffected; it still runs once per frame at 128.

**A checkpoint does not record the input size it was trained for.** The 128 px
and 256 px files have byte-identical state dict shapes -- 103,418 parameters
plus 4,088 BatchNorm buffers, 107,506 state-dict elements -- and differ only in
their values. Loading one at the wrong size does not fail, it degrades: the
128 px model scores 0.7426 mAP@0.5 at 128 and **0.4577** at 256 on the deployed
path. `evaluate.py` and `evaluate_two_stage.py` take `--input-size` for this
reason, and the value has to come from whoever knows the provenance of the
file. `evaluate.py` asserts that its dataloader, its model and its decode scale
all agree on it rather than trusting three defaults to line up.

```bash
# the shipped model (256 is the default, so --input-size is optional here)
python evaluate.py --checkpoint model_weights/BlazeEar_best_256px.pth \
  --csv data/splits/crops_val.csv --data-root data/face_crops/

# the superseded baseline, which MUST be told its size
python evaluate.py --checkpoint model_weights/BlazeEar_best_128px.pth \
  --csv data/splits/crops_val.csv --data-root data/face_crops/ --input-size 128
```

Those print 0.7456 and 0.7106 -- the dataloader-path rows above, not the
deployed-path ones. Both are correct; they are not interchangeable.

Worth recording that this was predicted to FAIL. The case for 256 rested on
raising the anchor-match ceiling -- the same argument that justified refitting
the priors, which measured worse on every metric. The arguments are not
actually the same: better priors only re-describe boxes the grid already
cannot place, whereas more pixels supply feature detail as well as a finer
grid. Mechanism matters, and a ceiling argument alone does not distinguish the
two cases.

### The recall ceiling is crop geometry, not face detection

The face detector finds nothing in 2.9% of held-out images, and that number was
previously reported as the pipeline's recall ceiling. Counting per **ear**
instead of per image, and calling an ear reachable when its centre falls inside
some crop window:

| | ears | share |
|---|---|---|
| reachable | 315 | 88.7% |
| no face found in the image | 9 | 2.5% |
| **face found, ear outside every crop** | **31** | **8.7%** |
| total unreachable | 40 | 11.3% |

So the structural ceiling is **88.7%, not 97.5%**, and the dominant cause is
`FACE_CROP_EXPAND` rather than the face detector -- by 3.4x. Ears sit at the
side of the head, and 1.5x a frontal face box does not reliably contain them.
Those 40 ears score exactly 0.0000 mAP@0.5 at both input sizes, in every
stratum, which is how this was confirmed rather than inferred.

This is why the back-camera face model removed the no-face rate completely and
bought nothing measurable: it was fixing the 2.5%.

Two things followed. The first has since been measured and came back null;
the second is still open:

- **Re-sweeping `FACE_CROP_EXPAND` at 256 px does not pay. 1.5 is already the
  optimum.** This was the recommendation above, and it was wrong about where
  the optimum would move. Measured externally (Ear_Landmarker, 2026-10-15) on
  1,200 full frames / 1,400 human-labelled ears, two-stage recall at IoU>=0.3:

  | expand | recall | vs 1.5 | p | paired counts |
  |---|---|---|---|---|
  | 1.20 | 71.93% | **-7.93pp** | 6.8e-19 | lost 139, gained 28 |
  | **1.50** | **79.86%** | optimal | | |
  | 1.90 | 75.57% | **-4.29pp** | 2.6e-09 | lost 82, gained 22 |
  | 2.40 | 70.86% | **-9.00pp** | 4.0e-24 | lost 148, gained 22 |

  The predicted mechanism is in the paired counts and it does not break even
  at any setting: every widening gains a couple of dozen ears by covering more
  of them and loses three to five times that many by shrinking the ear in the
  crop. Coverage and resolution trade directly, and at 256 px resolution still
  wins -- the reasoning above assumed there was resolution to spare, and there
  is not.

  These figures are final: the same ranking and the same mechanism appeared on
  three independent ground-truth bases, and 1.5 scored 79.86% here against
  79.55% on a separate human-basis run. One caveat remains, and it does not
  rescue the recommendation: the statistic is not the same one as the 8.7%
  (that is per-ear reachability on `common_heldout`; this is two-stage recall
  on full frames), so the two are compatible rather than directly comparable.

  What this changes: the 8.7% is still the larger share of the recall ceiling,
  and `FACE_CROP_EXPAND` is still what causes it -- but it is **not recoverable
  by widening the crop**. Ears sit lateral and posterior to a frontal face box,
  so a bigger concentric square buys periphery in every direction while paying
  resolution everywhere. Anything aimed at those 31 ears needs a different crop
  *placement or shape*, not a larger scalar.
- **A same-size lateral shift has a ceiling of 5.1pp, and no way to aim it.**
  It was the one crop change the sweep never tested, and it is now measured
  on both halves: the geometry is favourable and the direction is not
  predictable. The expand sweep varied one
  scalar on a *concentric* square, which is why coverage and resolution
  traded against each other: growing the window to reach one ear shrinks
  every ear in it. Translating a window of the *same* side pays no resolution
  at all, and nothing in the sweep covers it. On the 355 ears of
  `common_heldout` (`python measure_crop_shift.py`), offsets measured in crop
  sides (0.5 is the window edge):

  | | n | median max offset | shift <=0.1 | <=0.2 | <=0.3 | <=0.5 |
  |---|---|---|---|---|---|---|
  | reachable | 315 | 0.333 | 100% | 100% | 100% | 100% |
  | escaping | 31 | **0.888** | 32.3% | 41.9% | 48.4% | **58.1%** |

  A shift of half a window recovers 18 of the 31, which is **5.1% of all
  ears** -- and it is geometrically possible inside the frame for 100% of
  them, with a median 1.67 sides of room in both axes. It also lands
  in-distribution rather than off it: the 12,595 human training ears already
  sit at a median radial offset of 0.361 sides, p90 0.443, so a shifted
  window puts its ear where the model has always seen ears.

  **The direction oracle has since been tested and there is not one, so the
  5.1pp stays a ceiling and this is closed as a post-hoc rule.** The open
  question was which way to shift -- a strictly weaker requirement than the
  tragion crop below, which needed a predicted *position*. Measured
  externally (Ear_Landmarker, 2026-10-15) on 3,000 images / 3,525 ears, share
  of ears whose true shift direction a predictor gets right:

  | predictor | escaping (n=290) | delivered (n=3235) |
  |---|---|---|
  | eye midpoint minus nose | 57.6% | **76.4%** |
  | far tragion | 57.6% | 78.6% |
  | eye midpoint minus crop centre | 41.7% | 24.0% |
  | always shift right | 30.3% | 51.4% |
  | *majority class* | *69.7%* | *51.4%* |

  The keypoints do carry real direction information -- 76.4% against a 51.4%
  baseline on delivered ears, p=3e-188 -- which is what makes the escaping
  column decisive rather than merely negative. There it falls to 57.6%
  [51.7, 63.3], and **worse than always shifting the same way** (69.7%,
  p=1.3e-05). A keypoint rule is beaten by a constant on the only ears it
  would ever be applied to, and that constant is just this corpus's
  handedness skew, so it is not a rule either.

  What could *not* be established: whether the 31 split into a geometry group
  and a detector group. The offsets are a near-continuum (0.89, 0.90, 0.96,
  1.04, 1.08 ...), the largest gap is a lone 3.41 outlier, and the ordering
  of any two-group story flips between a cut at 1.0 and one at 1.25. So they
  are one population as far as this measurement can tell.
- **The general lesson, which is worth more than any of the three results:
  face keypoints cannot rescue a face-box crop, because they fail together
  with it.** Three separate attempts now -- the tragion as a position
  predictor (9.7% within one ear-size), the tragion-centred crop, and
  eye-nose shift direction -- each show strong signal on the ears the crop
  already delivers and none on the ears that escape it. That is one property
  of the problem rather than three findings. Whatever puts an ear outside a
  face-centred window is a turned or partly out-of-frame head, and that is
  the same condition that degrades every cue derived from the face box. So
  the 8.7% is not addressable by any post-hoc rule built on face-stage
  output, however the rule is shaped. What remains is re-targeting the first
  stage so the box is right to begin with, which is a different project.
- **Measure per-ear reachability on `crops_val` too.** The 11.3% is from 355
  ears on one 279-image set. The direction is not in doubt -- the 31 ears were
  counted directly -- but the magnitude deserves the larger split. Still open.

### The score head discards ears it has already localised

The second stage finds an ear and then fails to be confident about it. The
symptom was measured externally (Ear_Landmarker, 2026-10-15) on rendered
cohorts at fixed photo identity, varying only how much of the frame the ear
fills, so size is the only thing moving. "Localised" is an ear with a
candidate at IoU >= 0.3 anywhere in the output; "passes gate" is that
candidate surviving `DETECTION_SCORE_THRESHOLD`, 0.80 when this was measured:

| ear / frame | px in a 256 crop | localised | passes gate | gap | median score |
|---|---|---|---|---|---|
| *above the mode, n=160* | | | | | |
| 0.774 | 198 | 61.9% | 25.0% | **36.9pp** | 0.580 |
| 0.60 | 154 | 68.1% | 38.1% | 30.0pp | 0.659 |
| 0.45 | 115 | 80.0% | 61.3% | 18.7pp | 0.809 |
| 0.33 | 84 | 86.2% | 70.0% | 16.2pp | 0.860 |
| 0.24 | 61 | 86.2% | 75.0% | 11.2pp | 0.850 |
| *below the mode, n=60* | | | | | |
| 0.33 | 84 | 71.7% | 50.0% | 21.7pp | 0.711 |
| 0.24 | 61 | 75.0% | 55.0% | 20.0pp | 0.753 |
| 0.17 | 44 | 73.3% | 53.3% | 20.0pp | 0.745 |
| 0.12 | 31 | 58.3% | 38.3% | 20.0pp | 0.656 |

**The scoring gap is scale-dependent in both directions, and the recoverable
mass is small.** Two statistics say different things here and both are true.
Read down the gap column above: below the mode it is a flat ~20pp at every
size, and above the mode it *grows* to 36.9pp at 198 px, so the worst
discard *rate* is at the large end. (The two cohorts are different photo sets
and share only the 0.33 framing, so compare within a block, not across.)
But the ears are not at the large end. Counting where recall is actually
lost, on the held-out bands rather than rendered cohorts:

| band | ears | recall | localised but gated |
|---|---|---|---|
| 0-32 px | 203 | 18.2% | 148 |
| 32-48 px | 426 | 58.0% | -- |
| the mode | | 95.7% | -- |
| 120-160 px | 34 | | 3 |

So the large bands hold 17 to 34 ears against 203 and 426, and the high
discard rate there is a rate over almost nothing. The original framing,
"under-scores small ears", is right about where the recoverable ears are and
wrong about where the score head behaves worst. For targeting a fix the
recall column is the operative one. (Band recalls and counts measured
externally, Ear_Landmarker, 2026-10-15.)

Raising or shaping the threshold does not recover it. A size-dependent gate
was swept and has no headroom: every per-band gate is dominated by simply
moving the global one, because ranking inside a band is already good (AUC
0.926-0.978 at every band) while the *base rate* is not -- false positives
outnumber true ones about 2700:1 at 0-32 px against 1.2:1 at the mode. A
threshold cannot separate what a ranking already separates; the problem is
upstream, in training.

**Ruled out: positive-supervision starvation.** The intuitive cause is that
small ears get fewer matched anchors. They do not. `ANCHOR_TOP_K = 3` makes
every box claim its three best anchors regardless of prior fit, so over the
12,595 human training ears the mean is exactly 3.00 positives per ear in
every size band from 0-32 px to 120+. That is the design working.

**Found: one prior absorbs over half the negative budget and has almost no
positives.** What varies by size is not how many anchors an ear gets but how
much *negative* pressure sits on the prior serving it
(`python measure_anchor_budget.py`, and `--mine` for the second table):

| prior (px) | anchors | positives | pos per anchor | share of positives |
|---|---|---|---|---|
| 6.9 x 11.3 | 1024 | 299 | **0.29** | 0.8% |
| 12.5 x 24.3 | 1024 | 2,596 | 2.54 | 6.9% |
| 17.7 x 41.2 | 256 | 2,955 | 11.54 | 7.8% |
| 34.8 x 39.9 | 256 | 7,088 | 27.69 | 18.8% |
| 29.2 x 62.7 | 256 | 17,589 | **68.71** | 46.6% |
| 58.4 x 101.1 | 256 | 6,760 | 26.41 | 17.9% |
| 129.8 x 178.2 | 256 | 405 | 1.58 | 1.1% |
| 191.5 x 260.4 | 256 | 93 | 0.36 | 0.2% |

2048 of 3584 slots -- 57% of the budget -- receive 7.7% of all positives, a
27x gap in positives per slot against the prior carrying the mode. **An
earlier version of this section stopped there and concluded that those slots
therefore absorb the negative budget. That was an inference from slot counts,
and measuring it shows it is wrong.** `HARD_NEGATIVE_RATIO = 50.0` mines a
global top-k of background logits, so the budget follows how *confusing* a
prior is, not how many slots it owns. Measured on the shipped weights over
1,280 augmented training images:

| prior (px) | slots | % mined neg | % positives | neg:pos |
|---|---|---|---|---|
| 6.9 x 11.3 | 1024 | 9.6% | 1.0% | 9.4x |
| 12.5 x 24.3 | 1024 | 16.8% | 12.6% | 1.3x |
| 17.7 x 41.2 | 256 | 0.2% | 9.2% | 0.0x |
| 34.8 x 39.9 | 256 | 0.1% | 18.3% | 0.0x |
| 29.2 x 62.7 | 256 | 0.3% | 35.0% | 0.0x |
| 58.4 x 101.1 | 256 | 19.3% | 22.3% | 0.9x |
| **129.8 x 178.2** | 256 | **53.5%** | **1.0%** | **52x** |
| 191.5 x 260.4 | 256 | 0.1% | 0.6% | 0.2x |

The pathology is at the **large** end, not the small one. One prior, 7.1% of
slots and 1.0% of positives, takes over half the entire negative budget. The
three priors carrying 62.5% of positives take 0.6% between them, which is why
the mode reaches 95.7% recall. The 32x32 priors a small ear lands on are
close to balanced -- 12.5 x 24.3 sits at 1.3x -- so mining is *not* what
holds the small bands back. And the concentration is learned: at
initialisation that same large prior takes 2.1% and 58.4 x 101.1 takes 87%,
so it drifts there over the run.

This independently predicts the thing that looked anomalous above: **the
score head's worst discard is at the largest ears, 36.9pp at 198 px.** The
prior serving them is trained roughly 52:1 against. Two measurements taken
for different reasons agree, which is the only reason either is quoted here.

Two secondary gradients run with size as well:
positive match quality, median IoU 0.425 at 0-32 px against 0.535 at the
mode and 0.648 at 120+; and ignore-band protection, 0.94 anchors parked in
the ignore band per small ear against 8.84 at the mode and 67.63 at 120+, so
a near-miss next to a small ear is scored as background where the same
near-miss next to a large one is excused.

Three levers follow. They do not target the same ears, and the measurement
above reorders them against the obvious guess:

- **Widen `augment_scale`'s low end.** `dataloader.py:197` augments with
  `scale_range=(0.75, 1.25)`. The median training ear is 64.9 px in a 256
  crop, so 0.75x is 48.7 px -- the augmentation *cannot manufacture a 0-32 px
  ear*, and the band the model is worst at is the band it was never shown at
  volume (5.9% of boxes). Taking the low end to about 0.35 is a one-line
  change that moves mass onto the starved priors without touching the loss.
  It makes the easy bands slightly harder, which is the point. This is the
  highest-mass lever of the three and the only one aimed at where the ears
  actually are. Queued as a two-arm A/B, not run -- see
  `queue_augment_scale_ab.sh`.
- **Accept that the 0-32 px band is information-limited, and aim at 32-48
  instead.** The band the model is worst at is small because the ears are
  genuinely small *in the source*, not because the pipeline shrank them: over
  the 741 training ears in it, the median ear is **24 real pixels**, 70.9% are
  under 32 real pixels, and the crop resamples at 0.94x -- essentially 1:1.
  A 24 px ear carries a few hundred pixels of signal, and no loss
  re-weighting, anchor rebalance or augmentation manufactures information that
  was never captured. 32-48 px is the band worth attacking: 1,322 training
  ears, a median 38 real pixels, only 36.5% under 32 real, and 58.0% recall
  against the mode's 95.7%. There the information exists and is not being
  used.
- **Put a ceiling on any one prior's share of the negative budget.** The
  concentration above is a property of global top-k mining, and the direct
  fix is roughly fifteen lines in `loss_functions.py`: replace the single
  `torch.sort(...)[:, :background_num]` with the same selection applied under
  a per-prior cap.

  It must be a **ceiling, not a quota.** A uniform per-prior quota would hand
  each of the eight priors an eighth of the budget, which means giving the
  three mode priors -- currently at 0.1 to 0.3% -- roughly a hundred times
  more background pressure than they have now, on the priors carrying 62.5%
  of all positives and reaching 95.7% recall. That is how a rebalance turns
  into a regression. Capping the maximum any single prior may take, without
  imposing a floor, removes the pathology and adds pressure nowhere.

  Be clear about the payoff, though: it targets the large-ear discard, and
  large ears are 17 to 34 of 3,790 held-out ears. It should fix a real and
  measured 36.9pp defect on about 1% of the data.

Focal loss is **not** a third lever -- it has been on by default for every run
in this repository's history (see the training flags above), so it is already
priced into these numbers. What remains there is only retuning `--focal-gamma`
away from its 1.5 default, which is a weaker and less specific change than
either lever above.

**Cost, measured rather than inferred.** On an RTX 5060 Ti 16 GB at batch 32
with AMP and 4 workers: 29.9 ms/step and 508 steps/epoch is 15.2 s/epoch of
training, plus 11.4 s/epoch of validation with val mAP on (2.7 s with
`--no-val-map`). At the 100-epoch default that is **0.74 GPU-hours per arm**,
1.48 for two and 2.22 for three. An earlier version of this line said 1-3
GPU-hours *per arm* from checkpoint timestamps, which was wrong by 2-4x --
mtime gaps include whatever else the machine was doing. Timing 60 real steps
costs a minute and is not an estimate.

The ceiling's own compute cost is **negligible**: `_mine_negatives` goes from
21 to 141 microseconds, because eight sliced `topk` calls replace one sort,
but that is 0.9% of a 22 ms training step -- about 10 seconds over a
100-epoch run. At inference it costs exactly nothing, since the loss does not
run there.

### The three-arm A/B: both levers are null, and one costs the mode

Run 2026-10-15, 100 epochs per arm on `crops_train`, scored on `crops_val`
(2,193 human ears in 2,033 crops) at the then-deployed 0.80 gate, hit =
IoU >= 0.5, which is also the deployed gate -- it was briefly moved to 0.70
and reverted; see the gate section below for that excursion and for this same
comparison re-read at 0.70.
`python score_arm_bands.py`. Arms differ only in the two constants:

| band | n | baseline | cap_only | treatment | cap - base | B1 |
|---|---|---|---|---|---|---|
| 0-32 | 123 | 2.4% | 2.4% | 4.9% | +0.0 | **+2.4** |
| 32-48 | 250 | 38.4% | 35.6% | 37.6% | -2.8 | +2.0 |
| 48-64 | 698 | 77.2% | 76.5% | 76.5% | -0.7 | +0.0 |
| **64-80 (mode)** | **758** | **89.2%** | **88.0%** | **86.8%** | **-1.2** | **-1.2** |
| 80-96 | 230 | 84.3% | 86.1% | 83.0% | +1.7 | -3.0 |
| 96-120 | 88 | 71.6% | 73.9% | 70.5% | +2.3 | -3.4 |
| 120+ | 46 | 78.3% | 69.6% | 76.1% | -8.7 | +6.5 |
| **all** | **2193** | **73.3%** | **72.4%** | **72.0%** | **-0.9** | **-0.4** |

False positives per image are flat: 0.12 / 0.12 / 0.11, and FP at >= 96 px is
0.00 in all three arms, so the cap's specific risk -- fewer negatives on the
prior that suppresses large false positives -- did not materialise. Neither
did its benefit.

**Read against the standard pre-registered in `queue_augment_scale_ab.sh`
before the run, this is a non-result, and a worse shape than the one that
standard rejects.** It says a single band improved by a couple of points with
the mode flat is not a result. Here the one band that improved is 0-32 at
+2.4pp, which on n=123 is **three ears becoming six**; and the mode is not
flat, it is down 1.2pp on 758 ears, about nine ears lost. B1 trades nine ears
at the mode for three at the bottom. The 120+ swings (-8.7, +6.5) are four
and three ears on n=46 and carry no information.

**Why B1 failing was predictable from the scoping measurement, which was
taken too late to stop the run.** The 0-32 px band is information-limited:
median ear 24 *real* pixels, 70.9% under 32 real px, crops resampling at
0.94x. Augmentation can put a 24 px ear in front of the model, but it cannot
put detail there that the camera never captured. The band's recall is 2.4%,
not 18.2% as measured elsewhere on a different basis -- at the deployed gate
on this split it is almost entirely unreachable.

**The cap engaged and did not pay.** Verified rather than assumed, with
`measure_anchor_budget.py --mine --prior-cap`: the worst single-prior share
is 25.2% in `cap_only` against 47.6% in `baseline`, so the ceiling bound
exactly as designed and the rebalance happened. It bought -0.9pp.

**A third correction to the mechanism, and the most important thing here.**
The concentration is real and reproducible; *which prior* concentrates is
not. On the shipped checkpoint, 129.8 x 178.2 took 53.5% of the budget. On
`baseline` -- same code, same data, a fresh 100-epoch run -- the worst prior
is **12.5 x 24.3 at 47.6%**, and 129.8 x 178.2 takes only 14.4%. So "one
prior absorbs half the negative budget" survives as a phenomenon and "the
large-ear prior does" does not. The prediction built on it -- that capping
would relieve the large-ear discard -- was therefore never really under test
in this run, and the 120+ band is too small to have tested it anyway.

**Both constants are reverted, because "inside noise" was wrong.** That was
asserted and then tested, paired per ear with McNemar over all 2,193:

| comparison | lost | gained | delta | p |
|---|---|---|---|---|
| cap vs baseline | 99 | 80 | -0.87pp | 0.178 |
| B1 vs cap_only | 92 | 84 | -0.36pp | 0.598 |
| **both vs baseline** | **101** | **74** | **-1.23pp** | **0.049** |
| **both, 64-80 mode band** | **31** | **13** | **-2.37pp** | **0.0096** |

Neither lever reaches significance alone, and together they do: -1.23pp
overall and -2.37pp on the mode band, which is the largest band by mass. The
mode result is the solid one -- p=0.0096 on 758 ears, 31 lost against 13
gained -- and the overall p=0.049 is marginal and would not survive a
correction for the three comparisons. Either way nothing here is a gain, so
the values go back: `AUGMENT_SCALE_RANGE = (0.75, 1.25)` and
`HARD_NEGATIVE_PRIOR_CAP = 1.0`.

The cap's implementation and tests stay in place, defaulted off. They are
correct and cheap, 1.0 restores the exact previous arithmetic, and a future
run that finds a *reproducible* concentration has the lever ready.

**What the lost ears actually are, which reframes all of the above.**
`python diagnose_arm_losses.py` records every ear's best candidate *before*
the score gate, so a loss can be attributed. **About 80% of every loss is a
scoring loss, not a localisation loss**: the ear is still found at IoU >= 0.5
and its score has merely fallen below 0.80. For the full stack, 83 of 102 lost
ears are still localised, their median score is 0.752 against a 0.80 gate,
half are within 0.05 of it, and the median drop from baseline is 0.125. The
gained ears are the same phenomenon with the sign flipped, so the net -27 is
the residue of ~177 ears churning across one decision boundary.

Sweeping that boundary separates capability from calibration:

| gate | baseline | cap_only | treatment | treat - base | p |
|---|---|---|---|---|---|
| 0.00 (no gate) | 87.9% | 87.5% | 87.3% | -0.59 | 0.341 |
| 0.50 | 81.8% | 81.7% | 82.3% | +0.46 | 0.440 |
| 0.60 | 80.3% | 79.3% | 80.4% | +0.05 | 1.000 |
| **0.80 (deployed then)** | **73.1%** | **72.4%** | **71.9%** | **-1.23** | **0.050** |
| 0.85 | 68.0% | 66.9% | 65.8% | -2.14 | 0.0005 |
| 0.90 | 58.3% | 56.4% | 55.1% | -3.24 | <0.0001 |

The gap **grows with the gate and vanishes without one**, which is the
signature of a calibration shift rather than a capability loss. Two more
measurements agree: with the score ignored the three arms localise 87.9 /
87.5 / 87.3% of ears (not distinguishable, p=0.34), and the score's ranking
quality is if anything *best* in the treatment arm, AUC 0.8860 against the
baseline's 0.8784. The treatment model orders candidates marginally better and
trusts the true ones marginally less; at a gate as high as 0.80 the second
effect dominates.

**So "both changes made the model worse at finding ears" was the wrong
reading, and it was mine.** All three arms find the same ears. The revert
still stands, for a narrower reason: 0.80 was the threshold that shipped, no arm
gains at any gate, and each arm's own best achievable recall still orders
baseline first (87.87 / 87.46 / 87.28%).

### Why the gate is 0.80, what it costs, and the 0.70 excursion

`DETECTION_SCORE_THRESHOLD` was 0.80, fitted on a different measurement
from the one above: **F1**, end to end, on `common_heldout` (355 ears), where
F1 peaks at 0.80 (0.665) against 0.646 at 0.75 and 0.658 at 0.85. Two things
follow from that provenance, and neither says the value is wrong.

First, **F1 is a choice, not a fact.** It weights a missed ear and a false ear
equally, and nothing about this pipeline says they are equal: a false ear box
hands the consumer landmarks for something that is not an ear, a missed ear
hands it nothing. Which is worse belongs to whatever consumes the detections.

Second, that sweep had 355 ears end to end. The crop-space curve on
`crops_val` has 2,193 and is the one that governs stage 2. Shipped 256 px
model, IoU 0.5, 203,300 candidates over 2,033 images:

| gate | precision | recall | F1 | FP/image |
|---|---|---|---|---|
| 0.50 | 0.660 | 0.824 | 0.733 | 0.459 |
| 0.60 | 0.760 | 0.803 | 0.781 | 0.273 |
| 0.65 | 0.789 | 0.789 | 0.789 | 0.228 |
| **0.70 (now shipped)** | 0.813 | **0.775** | 0.794 | 0.192 |
| **0.75** | 0.839 | 0.755 | **0.795** | 0.156 |
| 0.80 (was shipped) | 0.869 | 0.724 | 0.790 | 0.118 |
| 0.85 | 0.901 | 0.665 | 0.765 | 0.079 |
| 0.90 | 0.934 | 0.550 | 0.692 | 0.042 |

**The F1 peak is at 0.75 here, and it is nearly flat from 0.65 to 0.80** --
0.789, 0.794, 0.795, 0.790, a spread of 0.006. Recall over that same span is
not flat: 0.789 down to 0.724. So **6.5pp of recall, 143 ears, is being
exchanged for precision at no measurable F1 cost**, and taking it back costs
0.11 extra false positives per image -- one per nine images. Below 0.60 the
trade stops being graceful: 0.50 buys 2pp more recall for double the false
positives, and 0.30 reaches 85.6% recall at 2.3 FP/image.

The config comment anticipated this, keeping its sweep so that "a deployment
that wants precision over recall can re-pick deliberately rather than by
inheritance". What was missing was a crop-space curve at full sample size to
re-pick against.

**The gate was briefly moved to 0.70 on both paths, and is back at 0.80 /
0.85.** The move was argued from the crop-space table above; the regression
was caught by eye on the demo page as a flickering false box on the nose
bridge, and then measured. `sweep_tta_gate.py` sweeps the whole two-stage
path on `val_v2` -- 2,414 human ears in 2,021 images, the shipped
`runs/checkpoints_crop` model, IoU 0.5:

| path | gate | precision | recall | F1 | FP/image | midline FP/image |
|---|---|---|---|---|---|---|
| plain | 0.70 | 0.723 | 0.694 | 0.7081 | 0.318 | 0.058 |
| plain | 0.75 | 0.770 | 0.682 | 0.7234 | 0.243 | 0.035 |
| **plain** | **0.80** | 0.821 | 0.653 | **0.7273** | 0.170 | 0.022 |
| plain | 0.85 | 0.874 | 0.604 | 0.7147 | 0.104 | 0.013 |
| TTA 4 | 0.70 | 0.644 | 0.740 | 0.6886 | 0.488 | 0.104 |
| TTA 4 | 0.80 | 0.774 | 0.710 | 0.7410 | 0.247 | 0.040 |
| **TTA 4** | **0.85** | 0.841 | 0.676 | **0.7497** | 0.152 | 0.023 |
| TTA 4 | 0.90 | 0.903 | 0.600 | 0.7208 | 0.077 | 0.011 |

**There is no plateau on either path end to end.** F1 rises to a clear argmax
-- 0.80 plain, 0.85 fused -- so 0.70 cost 0.019 F1 on one and 0.061 on the
other while roughly doubling and tripling false positives. The 355-ear
end-to-end sweep further up this file had said exactly this. The crop-space
curve was the outlier among three measurements, and it was the one acted on.

"Midline FP" is a false positive whose centre is within 0.15 face-box sides of
the crop centre -- the nose and brow, since the crop is centred on the face
and ears are lateral. It is the column that explains the symptom. At 0.70 the
fused path puts a box on the facial midline in **one frame in ten** against
one in forty-three at 0.85, and it flickers because those scores sit on the
decision boundary. The lower gate admits midline junk preferentially: 24.0% of
the false positives in the 0.70-0.85 band are midline, against 15.3% above
0.85.

**The mistake was the choice of measurement, not the arithmetic.** Crop space
scores stage 2 on pre-made crops, so it excludes the face stage's recall
ceiling and every false positive the pipeline can place outside a crop. That
makes it the right context for comparing two stage-2 checkpoints -- which is
what it was built for, and where the three-arm A/B above still stands -- and
the wrong one for picking a deployed gate. A deployed default should have been
swept on the deployed path before it moved, not after a human noticed.

What survives from the crop-space table is narrower: between two stage-2
checkpoints, gates 0.65-0.80 do not separate them on F1. A deployment that
genuinely wants recall over precision can still re-pick, against the
end-to-end table here, expecting midline false positives to be the first
thing that degrades.

**A geometric veto was the obvious alternative, and it is measurably worse
than the gate.** If the crop is centred on the face then |u| ~ 0 is the nose
and brow, an ear is lateral by construction, and rejecting the midline should
buy precision on a *different* axis from the score -- which is the only way
out of a pure recall/precision trade. `sweep_tta_gate.py --tta 4` prints the
cost. It does not work:

| | p1 | p5 | p10 | p25 | median |
|---|---|---|---|---|---|
| true ears, \|u\| | 0.016 | 0.080 | **0.157** | 0.265 | 0.335 |
| false positives, \|u\| | 0.010 | 0.029 | 0.053 | 0.150 | 0.268 |

**10% of real ears sit within 0.157 of the crop centre**, against the 3.0%
within 0.15 radially that the training-set geometry suggested. At gate 0.70 a
veto on |u| < 0.15 loses 145 true ears to remove 210 false ones -- 1.4 false
positives per ear sacrificed -- while raising the gate from 0.70 to 0.85
loses 154 ears and removes 679, a rate of 4.4. The gate is three times the
better instrument. At 0.85 the veto is simply harmful: 117 ears for 47 false
positives.

The reason is that the crop centre is a poor proxy for the facial midline.
The face box is noisy, and under head rotation or a profile pose a genuine
ear lands centrally. So this rules out the cheap geometric form; it does not
rule out a veto defined against the face *keypoints*, which locate the nose
and eyes directly rather than assuming the crop is centred, and which the
pipeline already carries (see the keypoint pass-through). That remains
untested, with a mildly negative prior now that the proxy has failed.

**Positional midline negatives are a dead end too, and for a reason that
generalises.** The natural next thought is to mine the midline as an explicit
negative during training rather than veto it at inference. The objection is
the turned head: a profile face puts its one visible ear near the crop
centre, so a positional negative would train against exactly that case.
Measured on `val_v2`'s 2,385 human ears that have a face, using annotated
ears per image as a yaw proxy:

| group | n | median \|u\| | p10 | within 0.15 |
|---|---|---|---|---|
| 1 ear (profile / heavy yaw) | 1628 | 0.339 | 0.131 | 11.6% |
| 2 ears (roughly frontal) | 696 | 0.317 | 0.150 | 10.1% |
| 3+ ears (crowd) | 61 | 0.304 | 0.086 | 23.0% |
| all | 2385 | 0.333 | 0.133 | **11.4%** |

So the concern is right about the magnitude and wrong about the mechanism.
**11.4% of true ears -- 273 of them -- sit inside |u| < 0.15**, which a
positional negative would train against. But yaw is not what puts them
there: 11.6% against 10.1% is not a difference, and 69% of midline ears come
from 1-ear images only because 1-ear images are 68% of the sample. The more
likely cause is face-box noise, the crop not being centred where the method
assumes. The 23% for crowds is probably an artefact of this measurement
rather than a finding: each ear is assigned to its *nearest* crop window,
which is unreliable when faces overlap.

The verdict survives either explanation -- training against 11% of real ears
to suppress a population already measured at 1.4 false positives per ear
lost is a bad trade. But the stronger objection is that **there is no new
information to add.** Those locations are already labelled background in
every training image without an ear there, and top-k hard-negative mining
already selects the highest-scoring negatives, which is exactly what a nose
bridge scoring 0.8 is. Mining the midline "specifically" would reweight
negatives the loss already picks.

**The keypoint veto is the fourth casualty of a lesson already recorded in
this file, and it was proposed here against it.** The argument offered was
that under yaw the nose and eye keypoints move off centre while the ear moves
toward it, so keypoints would separate the case crop-centre geometry fails
on. That is backwards, and the escaping-ears section above says so: *face
keypoints cannot rescue a face-box crop, because they fail together with it.*
Two specifics make it concrete. MediaPipe's BlazeFace regresses all six
keypoints unconditionally with no per-keypoint confidence, so under yaw the
nose is never *absent* -- it is silently extrapolated to a plausible wrong
place, which for a veto is worse than missing, because nothing marks it
unreliable. And the predicted tragion sits a median 0.31 ear-sizes from the
ear it belongs to on ears the crop reaches, against 6.61 on ears that escape,
with 0.0% inside half an ear-size; the eye-nose direction rule scores 76.4%
on delivered ears and 57.6% on escaping ones, losing to a constant.

In fairness to the idea the regime is milder: those measurements are about
ears *outside* the crop and keypoints used to predict a *position*, while a
midline veto would act on ears inside the crop where the keypoints do carry
real signal. But the route back is short. Midline true ears are not
concentrated in profile images, which left face-box displacement as the
likely cause -- and keypoints derived from a displaced box are displaced
with it. They fail together by a second path.

So every inference-time geometric fix on this axis is now ruled out: the
crop-centre veto by measurement, positional negatives by the ear population
they would destroy and by adding nothing mining does not already do, and the
keypoint veto by the correlated-failure property. What remains is not
geometry. It is the tracker, which uses time rather than position, and the
per-view spread measurement that would tell us how much the tracker actually
buys.

**What the symptom actually indicates is a missing component, not a wrong
constant.** `docs/index.html` drives a `requestAnimationFrame` webcam loop
with no temporal state: every frame is detected independently, so a candidate
sitting near the gate re-decides thirty times a second and the box strobes.
Per-frame thresholding cannot fix that at any value -- 0.85 only makes the
strobing rarer by making borderline candidates rarer. Detection-then-tracking
with hysteresis is the standard answer and is orthogonal to the gate: require
a high score to *start* a track and a lower one to *continue* it, and drop a
track after N consecutive misses.

**The tracker now exists**, in `docs/index.html` as `EarTracker`. A new track
needs the confirm score (the slider, default `TTA_SCORE_THRESHOLD` = 0.85)
*and* two consecutive frames; a running track then survives on 0.70 and
coasts through up to three dropped frames before it is dropped. Matching is
IoU >= 0.3 against the track's own smoothed box, and the box is EMA-smoothed
at 0.6 for display. The detector is asked for candidates down to the continue
score, because a track needs something to survive on; the confirm decision
moves out of the detector and into the tracker.

Five behaviours, checked before it shipped:

| frames in | drawn |
|---|---|
| one 0.86 then nothing | never -- `minHits` is 2 |
| 0.86 then 0.72 x4 | confirms on frame 2, survives all four |
| 0.72 x5, never confirmed | never |
| confirmed, then 5 empty frames | coasts 3, then drops |
| two ears 100 px apart | two tracks, neither absorbs the other |

The second row is the point: **0.72 is drawn when it belongs to a confirmed
track and discarded when it does not**, which no single threshold can
express.

**On whether this licenses a lower confirm score, this section has now been
wrong in both directions.** It first claimed the hysteresis works "because a
real ear persists across frames and a midline artefact does not". That was
retracted on the reasoning that a nose bridge is a *stationary* structure, so
a midline candidate at 0.72 would match itself frame after frame, confirm,
and coast -- making it steadier rather than rarer. **Measured on 7,500 frames
of 25 conversational clips, the retraction was the wrong call and the
original claim was closer to right.** Midline candidates are *less*
persistent than lateral ones: p90 lifetime 4 frames against 8, reaching 0.85
in 3.1% of tracks against 6.8%, and crossing the 0.85 boundary *more* often
per frame, 0.080 against 0.055. They are flickerier than real ears, so
`minHits` suppresses them preferentially. The stationary-structure argument
was asserted from the geometry and never tested against a frame sequence; it
is wrong by every statistic that bears on it. See the clips section below.

So the two mechanisms are not interchangeable. The gate decides what may
enter, and is picked on the end-to-end sweep. The tracker decides what may
*stay*, and buys two things the gate cannot: a box that has earned its place
stops strobing, and an ear that crosses 0.85 intermittently is held at 0.72
between crossings.

**Measured** by `measure_view_spread.py`. The four TTA views are four looks
at one ear -- expands 1.35 / 1.50 / 1.65 plus a flip -- and fusion takes the
max, so the spread across views stands in for the spread across frames. Each
ear is queried in every view at its own ground-truth location, so a view that
finds nothing scores 0 rather than leaving the sample. On `val_v2`, 2,385
human ears of which 2,083 are found by at least one view:

| | share of found | ears |
|---|---|---|
| every view >= 0.85, tracker irrelevant | 56.5% | 1176 |
| no view reaches 0.85, tracker cannot help | 20.3% | 422 |
| **straddles 0.85, the tracker's case** | **23.3%** | **485** |
| of those, worst view still >= 0.70 and so held | 45.4% | **220** |

Median per-ear spread is 0.062, so most ears are stable; the tail is not
(p90 0.857, and 5.6% of ears are found by only one view). **The 220 held ears
are 10.6% of found ears and 9.2pp of all of them** -- more recall than the
6.4pp the fused gate bought going 0.85 to 0.70, and without the tripled false
positives, because nothing changed at entry. Read it as an upper bound: the
proxy shows the score distribution straddling 0.85, while in video the
ordering matters -- the ear has to clear 0.85 on some frame while the track is
alive. And the proxy itself is partial. Varying the crop expand perturbs the
face box, a fair analogue of webcam box jitter, but says nothing about
lighting, motion blur or a genuinely moving head, and the flip is not a
temporal transformation at all.

`CONTINUE_MARGIN = 0.15` turns out to sit on the knee, which was luck rather
than foresight -- it was picked from principle before this ran:

| margin | keep score | straddling ears held |
|---|---|---|
| 0.05 | 0.80 | 130 |
| 0.10 | 0.75 | 196 |
| **0.15** | **0.70** | **220** |
| 0.20 | 0.65 | 233 |
| 0.25 | 0.60 | 245 |
| 0.35 | 0.50 | 250 |

Past 0.15 the curve flattens: widening to 0.35 adds 30 ears while dropping
the keep score to 0.50, which would hold far more junk for them.

**A >=2-view agreement rule is a good trade, and the prediction here was that
it would be a bad one.** The guess was roughly one ear lost per false positive
removed, by analogy with the geometric veto. The cross-tab says otherwise.
Ears found by a single view whose lone look would confirm: 29, out of 1,661
confirmable. Confirmable false-positive locations seen by a single view: 112,
out of 652. That is **3.9 false positives per ear sacrificed**, against the
score gate's 4.4 and the geometric veto's 1.4 -- so it removes 17.2% of
confirmable false positives for 1.7% of confirmable ears, on an axis
independent of both score and time.

Units caveat before anyone acts on the 17.2%: these are per-view clusters at
a 0.10 floor, counted before fusion and NMS, so they are not the same objects
as the 0.152 FP/image in the gate table. The ratio should carry; the absolute
rate will not.

The structural finding underneath is the one that explains the rest of this
section. **The false positives that matter agree across views**: 1.6% of
4-view locations reach the confirm score against 0.0% of 1-view ones, and 540
of the 652 confirmable ones are seen by two or more views. They are
consistent artefacts, not sampling noise. That is why the entry bar has to
stay at the measured argmax, why temporal persistence cannot filter them --
a stationary structure persists as well as an ear does -- and why view
agreement, which catches only the transient 17%, is a supplement rather than
a replacement.

Implementing it needs `merge_views` to return each cluster's distinct view
count, which it computes and discards today, and the pipeline to drop
single-view clusters. Not done. `CONTINUE_MARGIN` is measured below, and
`minHits` and `maxMisses` -- which are about frame *ordering*, so no
still-image proxy reaches them -- are measured on `data/clips` further down.

The first version of this filtered drawn boxes on `misses === 0` and so
dropped the box on any single missed frame -- reintroducing the strobe for
exactly the ears it was meant to stabilise. The gap row above is what caught
it, which is the argument for writing the table before believing the code.

Scope: deliberately not in `blazeear_inference.js`. That file is mirrored
against the Python pipeline and pinned by `test_tta.py`; a temporal component
has no Python counterpart, so putting it there would break that parity. The
cost is that a consumer wanting hysteresis has to lift it from the demo page.
`test_tta.py` does pin the arithmetic -- confirm minus `CONTINUE_MARGIN` must
equal 0.70 and must sit below the single-frame gate -- so a later edit to
either constant cannot silently move the continue score somewhere unmeasured.
The still-image path is untouched: one frame carries no temporal evidence.

One trap worth recording, since it nearly produced a confident wrong answer:
`export_onnx.py` defaults `--checkpoint` to `runs/checkpoints/BlazeEar_best.pth`,
which is a December single-stage relic, not the shipped crop model. Pointed at
it, this sweep reported 0.004 precision and 2.8% recall -- and
`evaluate_two_stage.py` independently reported 0.0052 mAP@0.5 on the same
file, which is what identified the checkpoint rather than the harness as the
fault. The shipped model is `runs/checkpoints_crop/BlazeEar_best.pth`, dated
one minute before `docs/BlazeEar_web.onnx`.

**What the levers actually do, paired.** Per-arm median score on small ears
reads like a mechanism -- 0.748 baseline, 0.696 cap_only, 0.761 treatment,
i.e. B1 lifting small-ear confidence and the cap depressing it. That reading
is wrong, and it is composition: each arm localises a different set of ears,
so the three medians are over three different samples. Paired on the ears
localised by *both* arms:

| band | arm | n | median shift | mean | p |
|---|---|---|---|---|---|
| small (<48 px) | cap_only | 208 | -0.0004 | +0.0002 | 0.74 |
| small (<48 px) | treatment | 195 | +0.0004 | +0.0008 | 0.81 |
| mode (64-80) | cap_only | 718 | -0.0045 | -0.0055 | 5.8e-08 |
| mode (64-80) | treatment | 718 | -0.0064 | -0.0084 | 3.8e-14 |
| all | cap_only | 1856 | -0.0038 | -0.0047 | 6.5e-09 |
| all | treatment | 1841 | -0.0058 | -0.0086 | 1.7e-17 |

So **the mechanism both levers were adopted for is absent on the ears that
can test it.** B1 does not raise small-ear confidence (+0.0004, p=0.81) and
the cap does not help the starved small priors' score heads. What they do
produce is a uniform downward confidence shift on everything, 0.005 to 0.009
of score, which is the most significant effect in this entire dataset
(p = 1.7e-17) and is a side effect rather than the goal. A shift that small
is invisible except where it crosses a decision boundary -- which is exactly
the ~1pp of gated recall measured above, and exactly why the deficit tracks
the gate.

That also settles what adoption would be buying. Against it: the browser path
still gates at 0.85, where the treatment is -2.14pp (p=0.0005); the cap alone
is negative at every gate from 0.60 to 0.80; and B1 was never run as its own
arm, so every observation of it is confounded with the cap. For it: ranking
quality is consistently ahead (AUC 0.8784 / 0.8830 / 0.8860), the compute
cost is nil, the cap removes a real and run-dependent budget asymmetry as
hygiene, and B1's intended generalisation benefit cannot appear in a
validation split that shares the training scale distribution. Neither is
adopted.

**The measurement that should come before any further arm** is two baseline
seeds, 1.5 GPU-hours. Every figure here, the -1.23pp that drove the revert
included, is read against a run-to-run noise floor nobody has measured, and
`train_blazeear.py` still has no `--seed`. Something seed-sensitive is
already known to be in here: which prior the negative budget concentrates on
is not reproducible across runs. If baseline-to-baseline spread is +/-1.5pp
then this A/B establishes nothing and no further arm is worth running; if it
is +/-0.3pp then the cap's -1.09pp is real and a B1-only arm becomes worth
its 0.74 GPU-hours.

Read those p-values as a sweep, not as five independent tests: they are
uncorrected over the gate grid, and a 1.09pp difference is 24 ears. The
precision side is not in this re-read -- the cached dump holds only each GT
ear's best pre-gate candidate, so FP/image at 0.70 per arm would need a
re-run, and no decision here turns on it.

What this leaves established: the budget concentration is real but is not the
binding constraint on recall; the prior it lands on is not stable across runs;
the 0-32 px band is information-limited rather than exposure-limited, so it is
not reachable from the loss or the augmentation at all; and at the deployed
gate the second stage localises 87.9% of ears while reporting 72.4%, of which
about 6.5pp is available for 0.11 FP/image and the rest is not cheap.

### The flicker is partly upstream: 4.2% transient face dropout

`measure_video_dynamics.py`, label-free, on 5,378 frames of 60 hand-held
videos. No annotation is needed because the two constants in question --
`minHits` and `maxMisses` -- are about candidate *persistence* rather than
correctness, and because geometry splits the population for free: midline
candidates (|u| < 0.15, the nose and brow) are mostly false positives and
lateral ones (|u| > 0.25) mostly ears. Noisy, since 11.4% of real ears are
midline, but enough for the question asked.

| face stage | |
|---|---|
| frames with **no face at all** | **19.2%** (1,035 of 5,378) |
| faces per frame, mean | 1.31 |
| face-box IoU between adjacent frames, median | 0.914 |
| ... p10 | 0.565 |
| ... below 0.5 | 6.8% of transitions |

**19.2% of frames lose the face entirely** -- but only about a fifth of that
is a face the detector should have found, as the run-length split below
shows, so 4.2% is the figure that transfers and the 19.2% is mostly frames
with no face in them. A frame with no face has no
crop, so it has no ear detection whatever the ear model or the gate does.
That single number explains the ear-side result: median candidate lifetime is
1 frame for *every* position class, because a track cannot survive a frame
that produced no crop. The flicker originates one stage above everything this
part of the README has been tuning.

| candidate persistence | tracks | median life | p90 | ever >= 0.85 | mean peak |
|---|---|---|---|---|---|
| midline | 1112 | 1.0 | 4.0 | 4.0% | 0.610 |
| between | 715 | 1.0 | 4.0 | 5.5% | 0.620 |
| lateral | 1448 | 1.0 | 5.0 | 5.2% | 0.618 |

**Most of that 19.2% is not a detector failure, and the corpus is the wrong
instrument.** Two follow-ups, prompted by the objection that whole-body
exercise framing makes faces and ears tiny. The size explanation does *not*
hold: the largest face per frame is a median 0.284 of frame height here
against 0.327 in `val_v2` stills, only 1.2x, with near-identical p90s (0.599
against 0.597). A 1.2x size gap cannot produce a 21x dropout gap (19.2%
against 0.9%). What does explain it is visibility, which gap run-lengths
separate:

| gap length | frames | share of no-face frames | reading |
|---|---|---|---|
| 1 frame | 72 | 7.0% | present but missed |
| 2-3 | 155 | 15.0% | present but missed |
| 4-10 | 298 | 28.8% | ambiguous |
| 11-30 | 422 | 40.8% | no face to find |
| 31+ | 88 | 8.5% | no face to find |

**49.3% of no-face frames sit in gaps of 11 or more**, and 19.1% are in a gap
starting at frame 0 -- intro cards, cutaways, and people bent over or facing
the floor mid-repetition. None of that is the detector missing a face, and
none of it is the deployment condition, where a webcam or video-call face is
continuously present and roughly frontal.

**The number that transfers is the transient rate: 227 frames, 4.2% of all
frames, in gaps of 1 to 3.** That is a face the detector should have found and
did not -- one frame in 24, enough to flicker, and the only part a face
tracker can legitimately repair. Two consequences. `maxMisses = 3` covers
exactly that class, which was picked from principle and is now measured. And
a face tracker has to keep the same bound: coasting until a face returns
would draw ears over a twenty-frame cutaway, so the rule is reuse the last
crop for at most three frames, not reuse it until detection recovers.

The still-image rate on `val_v2` is 1.2%, so the honest range for transient
face dropout on selfie or video-call footage is somewhere between that and
4.2%, and this corpus cannot narrow it. And **this corpus cannot settle whether midline candidates persist
like ears do.** They are indistinguishable here, but that is because nothing
persists, not because they are alike -- so the claim that a stationary nose
bridge defeats temporal filtering remains consistent with the data and
untested by it.

**`minHits` is a weak instrument, which is a result rather than a gap.**

| minHits | maxMisses | midline confirmed | lateral confirmed | ratio |
|---|---|---|---|---|
| 1 | 3 | 37 | 67 | 1.81 |
| 2 | 3 | 27 | 49 | 1.81 |
| 3 | 3 | 22 | 42 | 1.91 |
| 4 | 3 | 19 | 40 | 2.11 |

Going from 1 to 4 moves the lateral-to-midline ratio only 1.81 -> 2.11 while
cutting lateral confirmations 67 -> 40. It discards likely-real detections
about as fast as it removes midline ones, so `minHits = 2` stands and should
not be raised. The sweep is run through a Python mirror of the shipped
`EarTracker`, checked against the same five behaviour cases the JS passes --
which caught a one-frame disagreement on the first attempt, where `drawn` was
incremented before the `maxMisses` filter rather than after it.

**So the next change is probably a face tracker rather than a better ear
gate, on a smaller claim than this section first made.** The hysteresis now
shipping tracks ears, while the instability that remains is in the face box:
reusing the previous crop for up to three frames when detection drops, and
smoothing the box when it jitters, would stabilise the ear model's *input*
instead of repairing its output a stage later. That addresses the 4.2%
transient rate, not the 19.2% headline. This supersedes nothing already
measured -- the gates and the ear tracker keep their justifications.

**What is still needed is the right corpus**: selfie or video-call footage
with head movement, where the face is continuously present so that structural
gaps vanish and the transient rate can be measured on its own. Exercise video
cannot provide it, and that is now demonstrated rather than assumed.

### A constant buys stage-1 coverage, and the coverage does not convert

Prompted by a handoff from `trainable_blazeface` (via Ear_Landmarker,
`handoff/t4_stage1/` at commit 3d17c65), which measured a fine-tuned stage 1
end to end on this corpus: ears no crop covers fall 1,301 -> 530 of 15,359,
and converting that into delivered ears at matched precision yields roughly
nothing, because the extra crops carry extra non-ear scene. The handoff's
conclusion is that the cheapest stage-1 win is `FACE_CROP_THRESHOLD` rather
than a new model. That framing is Ear_Landmarker's rather than
`trainable_blazeface`'s -- worth stating, because three sessions passing one
package made attribution drift in both directions -- and when measured it
turns out to be wrong.

Their nine configurations swept the face threshold only on their own
checkpoint, holding stock at its fitted 0.2. On stock MediaPipe, 900 `val_v2`
images and 1,148 ears at expand 1.5, counting ears whose centre no crop
contains:

| face threshold | crops/image | ears no crop covers |
|---|---|---|
| 0.30 | 1.51 | 12.63% |
| **0.20 (shipped)** | **1.95** | **9.41%** |
| 0.15 | 2.50 | 7.75% |
| **0.10** | 3.69 | **4.62%** |
| 0.05 | 6.15 | 2.61% |

**Stock halves its own uncovered ears by lowering a constant**, 9.41% to
4.62%, which is the same order of effect as their 8.5% -> 3.5% and needs no
new weights. So "a fine-tuned stage 1 finds faces stock misses" and "a lower
threshold finds faces the shipped constant discards" are not separated by
anything in that handoff. The missing arm is stock at 0.10 and 0.05 through
the same paired harness; their `--ckpt` and `--face-thresh-*` flags already
support it.

**And it does not convert, so the constant is not a win either.** Run end to
end on `val_v2`'s 2,414 ears, plain path, stock at face threshold 0.20
against 0.10, with 0.10 interpolated onto 0.20's own precision points:

| gate | matched precision | recall @ 0.20 | @ 0.10 | delta |
|---|---|---|---|---|
| 0.60 | 0.623 | 0.717 | 0.721 | +0.45 pp |
| 0.65 | 0.681 | 0.708 | 0.707 | -0.06 pp |
| 0.70 | 0.723 | 0.694 | 0.697 | +0.27 pp |
| 0.75 | 0.770 | 0.682 | 0.675 | -0.74 pp |
| **0.80 (deployed)** | 0.821 | 0.653 | 0.639 | **-1.40 pp** |
| 0.85 | 0.874 | 0.604 | 0.580 | **-2.40 pp** |

Coverage improves about as far as it can -- ears in images where the face
stage found nothing go **29 to 1** -- and raw recall rises 0.653 to 0.669 at
the deployed gate. At matched precision that is a wash at loose operating
points and a 1.4 to 2.4pp *loss* at the ones this pipeline actually ships.
Lowering `FACE_CROP_THRESHOLD` buys crops, and the crops cost more precision
than the ears inside them are worth. **So the "cheapest stage-1 win is a
constant" framing is wrong in the same way, and for the same reason, as the
fine-tuned detector it was contrasted with.** `FACE_CROP_THRESHOLD` stays at
0.2.

One side effect worth connecting to the flicker section above: at the
deployed gate, midline false positives per image go 0.022 to 0.056, two and a
half times, because more crops means more nose and brow passing under the ear
model. A lower face threshold makes the nose-bridge artefact worse.

`trainable_blazeface` then measured the half of this I could not. Stock's
*face* recall is saturated and is already there at its shipped 0.20 -- 0.4860
at 0.20, 0.4884 at 0.10, 0.4870 at 0.05 on 4,848 GT faces -- and its 8-16 px
face recall never leaves 0.085-0.116 against their 0.474-0.578. No threshold
closes a 4-5x gap. So the coverage this constant buys is **not** face recall:
of 39,822 of their detections between 0.10 and 0.20 that miss face ground
truth, an independent third detector confirms 19 as faces, 0.0%. The gain is
non-face crops that happen to contain ears, which is why coverage moves while
face recall does not -- and why it does not survive a precision match. Their
face-detection component is a separate thing that a constant cannot reach,
and nothing measured here bears on it.

This is coverage, not delivered recall at matched precision, and that
distinction is the whole lesson of their result: 3.69 crops per image instead
of 1.95 means more non-ear scene, so a lower stock threshold should pay the
same 4-7pp precision cost theirs did. The cost looks like a property of
lowering the threshold rather than of anyone's weights -- which is why the
constant is not a free win either, and why it is only worth taking against
the downstream objective.

Two of their constants are worth recording here regardless. Their boxes are
linearly **1.0998x** larger than stock's on this corpus (bootstrap CI
1.094-1.105), so `make_face_crops.py`'s `side = expand * max(box_w, box_h)`
turns an expand of 1.5 into an effective 1.65 for their detector. And their
measurement that restoring 1.364 makes *both* arms worse -- stock 79.09% ->
77.52% -- is the coverage/resolution trade this README already records for
the 1.5 optimum, with coverage dominating below it. They read that as the
recorded mechanism being incomplete; it reads here as confirmation, since the
claim was always that the two trade and 1.5 is where they balance.

Their `FACE_CROP_THRESHOLD` lesson generalises past this pipeline and is
worth keeping verbatim in substance: a face detector's own precision metric
and this pipeline's objective disagree by construction, because a profile or
partial face that the face GT scores as a false positive is exactly the crop
that contains an ear. Every constant fitted against stock is in stock's
units, and the optimum for a swapped detector cannot be derived from either
model's own metrics -- it has to be swept against delivered ears.

### The upgraded face stage is not swapped in, and both sessions agree why

`trainable_blazeface` retrained its face detector and I asked whether to
replace stock MediaPipe with it. The determination is **no**, reached
independently on both sides and for the same reason, so it is worth recording
with the numbers rather than as a preference.

The upgrade is retrained weights and nothing else: same architecture, same
128 px input, same 896 anchors, same 101,390 parameters, all 78 state_dict
tensors identical in key and shape. It loads into this pipeline with
`strict=True`. What changed is how ground-truth boxes were assigned to
anchors during training. On their full val split, 8,081 images and 20,505
boxes at IoU >= 0.5:

| | precision | recall | FP | 8-16 px | 16-32 px | 32+ px |
|---|---|---|---|---|---|---|
| stock | 0.9380 | 0.4372 | 592 | 0.078 | 0.862 | 0.973 |
| tuned | 0.9649 | **0.5011** | 374 | **0.402** | 0.934 | 0.980 |

At matched precision that is 0.433 -> 0.503, three seeds, spread 0.0022, and
it holds against YuNet detections rather than their own labels (+0.071
recall, larger than against the teacher, which is the check that rules out
mimicry). Their split is per-image disjoint with zero path overlap, but all
62 WIDER categories appear on both sides and is not perceptually
deduplicated, so it is "held out" rather than independent -- their
characterisation, offered unprompted.

**The detector is better. It is better in the one band this pipeline has
measured as unable to receive it.** The 8-16 px recall going 0.078 -> 0.402
is the largest single number they have, and `score_rescued_ears.py` found
1 of 332 ears in the 0-32 px band scoreable *in an oracle crop* -- the stock
face box nearest the ear, expanded by the minimum factor that brings the ear
inside -- with 56.9% producing no box at all. So the strongest thing on offer
is delivery into the band that cannot score. That is the whole argument, and
they made it themselves.

**The geometry is the second reason and it is the expensive one.** On 1,739
faces both models detect, theirs are 11.7% larger linearly (area 1.248x,
width 1.115x, height 1.120x) with the centre ~4.8% of box height higher. The
1.0998x [1.0936, 1.1048] measured here on this corpus *excludes* their 1.117,
so the ratio is real and does not transfer between corpora. Every constant
denominated in face-box units moves: `expand` 1.5 becomes effectively ~1.675,
and the gate and tracker thresholds sit downstream of a crop that is no
longer the same crop. Rescaling does not rescue it, because expand 1.364 was
already measured worse than 1.5 for *both* detectors here. A swap costs a
full re-sweep of expand, gate and tracker constants.

Cost is identical (101,390 params, 2.24 vs 2.26 ms CPU forward) and there is
an ONNX export at 128 px verified to 1e-7, so neither cost nor the browser
path is the obstacle.

**The axis that would decide it has no data on either side.** Stage 1 is not
this pipeline's complaint on video -- stock gives 3 no-face frames in 7,500
and adjacent-frame box IoU median 0.972 -- but nobody has run the tuned
detector on video at all. Their nearest proxy is horizontal-flip agreement on
stills (0.9140 -> 0.9364), a different axis, which they declined to
substitute for the one asked about. The open measurement is adjacent-frame
IoU and no-face-frame rate for both detectors on the 25 clips, a few hours,
and it is the only thing that could make the swap worth its re-sweep before
stage-2 scoring is fixed.

**Re-ask when the 0-32 px band becomes convertible.** The gain is real and
sitting there; it is gated on stage 2, not on stage 1. Their generalisable
warning stands and is why neither model's own metrics could have settled
this: a face detector's precision metric and this pipeline's objective
disagree by construction, because a profile or partial face that face GT
scores as a false positive is exactly the crop that contains an ear.

### The 0-32 px limit survives an independent representation

`trainable_blazeface` flagged the confusion that cost them most of a day, and
it is the same shape as this repo's load-bearing claim. They concluded their
data was the limit -- 39% of faces under 8 px, below the anchor stride, six
configurations failing to beat stock -- when in fact every ground-truth box
was being assigned to *both* anchor grids, so most positive targets demanded
a detection at a scale the model could not answer. The faces were resolvable;
the **targets** were not. Identical symptoms, opposite fixes, and inspection
cannot separate them. What separated them was scoring a model they trusted
against their own objective and watching a known-good detector "fail" it.

The parallel here is sharp, because this repo's own anchor measurements point
the same way. In the 0-32 px band the median best-anchor IoU is **0.425**,
below the 0.5 matching threshold, and **57.2% of 0-24 px ears cannot reach
IoU 0.5 against any prior** -- against 56.9% of oracle-crop rescued ears that
produced no box at all. Two numbers that close together invite exactly the
wrong conclusion in either direction.

So `check_resolvability.py` hands the *same* crops to the fine-tuned YOLOv11
ear detector, which shares this repo's human GT labels but none of its
anchors, target assignment or training recipe. 2,193 human-labelled ears in
2,033 `crops_val` images, localised meaning IoU >= 0.5 against the GT box:

| band (px in a 256 crop) | n | stage 2 | referee | ref only | mine only | McNemar |
|---|---|---|---|---|---|---|
| 0-24 | 59 | 28.8% | 37.3% | 22.0% | 13.6% | p=0.38 |
| 24-32 | 64 | 46.9% | 45.3% | 14.1% | 15.6% | p=1.00 |
| 32-48 | 250 | 77.6% | 74.8% | 13.2% | 16.0% | p=0.48 |
| 48-64 | 698 | 91.3% | 92.1% | 5.6% | 4.7% | |
| 64-88 | 909 | 96.4% | 94.8% | 2.1% | 3.6% | |
| 88+ | 213 | 87.3% | 85.0% | 5.6% | 8.0% | |
| **all** | **2193** | **88.5%** | **87.7%** | 5.7% | 6.4% | p=0.39 |

**No significant difference in any band, and the referee fails the small
bands too.** It localises 37.3% of the 0-24 px band and misses 62.7% of it.
The prediction recorded in the script -- that the referee should localise
materially more of 0-32 px if the anchor numbers mean what they look like --
did not happen.

The referee's failure is the stronger half of this, because it was
*advantaged*. Ultralytics runs at 640 by default, so it saw each crop at 640
where stage 2 sees 256: an ear measured at 0-24 px in this table was 0-60 px
in the referee's frame, two and a half times the linear resolution, on an
architecture with its own anchor-free head. It still could not find two
thirds of them. An information limit is the reading that survives.

**One process note, because it nearly went the other way.** A 400-crop
subsample run first showed the 0-24 px band at 27.3% stage 2 against 9.1%
referee -- the opposite direction, and it would have read as "the referee
cannot see small ears either, decisively". The full set reversed the sign.
n=11 in that band was noise, and had the subsample pointed the convenient way
it is the number that would have been quoted.

**What this does and does not settle.** It does not vindicate the anchor
geometry: best-anchor IoU 0.425 in that band is a real representational
weakness and the per-prior mining cap that addresses it measured -0.87 to
-1.09pp at p=0.178, which is unresolved pending the noise floor. What it
settles is narrower and was the question asked: the 0-32 px band is not
merely mis-targeted, because a different representation at higher effective
resolution fails on the same ears. `score_rescued_ears.py`'s result is not
an artefact of this repo's anchors.

### Crop placement carries ear information, and it is not just composition

`trainable_blazeface` raised the sharpest objection in the stage-1 exchange.
A placement ablation on this corpus (Ear_Landmarker) randomised where the
face stage's crops land while holding count, sizes and images fixed, and
face-stage misses rose ninefold -- which establishes that placement carries
information about where ears are, but cannot separate two explanations:

* **head-ness** -- the windows seek heads, which is where ears are
* **composition** -- these frames are head-filling, so "lands on a head" and
  "lands on image content rather than background" are nearly the same event,
  and a tail that merely avoids background would ablate identically

Their proposed discriminator is face size, and it is a subset selection on
data already in hand rather than a new measurement. `measure_placement_lift.py`
runs it: real crop placement against a null that preserves each image's window
count and each window's side and destroys only position, 20 draws per image,
reported per face-size band.

| largest face / frame height | images | ears | real cover | random null | lift |
|---|---|---|---|---|---|
| 0.00-0.20 | 138 | 176 | 81.8% | 15.9% | **5.15x** |
| 0.20-0.30 | 358 | 483 | 87.0% | 33.7% | 2.58x |
| 0.30-0.45 | 354 | 425 | 93.6% | 58.1% | 1.61x |
| 0.45+ | 337 | 412 | 98.3% | 89.6% | **1.10x** |
| all | 1187 | 1496 | 91.4% | 53.9% | 1.69x |

**The objection is correct as method and the conclusion survives it.** In
head-filling frames the lift is 1.10x against an 89.6% null; in small-head
frames it is 5.15x against a 15.9% null, where a random window lands on
background five times out of six. The effect therefore strengthens in
precisely the regime where the two explanations come apart, which is
head-ness's prediction and not composition's. (Read the lift column with the
correction below: it exaggerates how band-dependent this is, because a lift
is bounded by 1/null. The absolute gap is the fairer statistic and tells the
same story less dramatically.)

The methodological half matters independently: the aggregate 1.69x is a
mixture dominated by head-filling frames, so any placement ablation reported
in aggregate on this corpus is mostly measuring composition. The small-face
band is where the answer lives, and that is where it should be reported.

**Band on occupancy, not face size.** `trainable_blazeface` refined this
after seeing the table above, and the refinement is right. What saturates the
null is the fraction of frame the windows cover, which is size TIMES count --
their own corpus is all small heads, my 0.00-0.20 band by construction, and
its null still sits at 52.5% because crowd frames carry many small heads
instead of one large one. Size is a proxy that fails on crowds; occupancy is
the thing. Re-banded on it, with each band's power ceiling (1/null) printed
so a reader can see which rows could have shown an effect at all:

| window occupancy | images | ears | real | null | lift | ceiling | **gap** | headroom kept |
|---|---|---|---|---|---|---|---|---|
| 0.00-0.10 | 66 | 78 | 78.2% | 9.6% | 8.13x | 10.4x | **+68.6** | 75.9% |
| 0.10-0.25 | 272 | 350 | 86.6% | 25.8% | 3.36x | 3.9x | +60.8 | 81.9% |
| 0.25-0.50 | 427 | 541 | 90.9% | 50.7% | 1.79x | 2.0x | +40.2 | 81.5% |
| 0.50+ | 422 | 527 | 97.0% | 82.5% | 1.18x | 1.2x | +14.5 | 82.9% |

**Compare the gap, not the lift -- and this corrects a claim made above.**
A lift cannot exceed 1/null, so lifts from bands with different nulls are not
comparable: the 0.50+ band reads 1.18x against a *ceiling* of 1.21x, which is
placement capturing nearly all the room it had, not placement failing. In
points it is **+14.5 on 527 ears**, which is not "almost nothing". So the
earlier statement that in head-filling frames composition explains almost all
of it is **too strong**: placement carries real information in every band,
and the lift metric exaggerates the band dependence because its ceiling
collapses with the null. What survives, and is what head-ness actually
predicts, is that the gap widens monotonically as scenes get sparser --
+14.5 to +68.6 points.

Ear_Landmarker reached the same correction independently on their own band
table (+6.4 points on head-filling frames against +47.8 on small-head ones,
with per-row ceilings of 2.42 / 1.75 / 1.38 / 1.07) after being handed the
ceiling argument, and added the statistic that does *not* flatter the
conclusion: headroom captured runs mildly the **other** way, 75.9% in the
sparsest band here against 82.9% in the densest. That is what "sparse scenes
are harder to detect in" predicts and it is not evidence for composition,
since the null being divided out *is* composition. They also discarded a
"share of ceiling achieved" column on finding it algebraically identical to
coverage -- a tautology dressed as an analysis, and worth recording as a
near-miss.

**And the tail specifically is only weakly head-seeking.** The population the
ablation used is the low-score tail, not the ordinary crop, and that gap was
the one real hole in the argument. Measured on stock's own 0.10-0.20 tail --
the boxes `FACE_CROP_THRESHOLD` admits when lowered, so the population that
governs the decision above:

| window occupancy | images | ears | real | null | lift | ceiling |
|---|---|---|---|---|---|---|
| 0.00-0.10 | 71 | 85 | 22.4% | 9.9% | **2.26x** | 10.1x |
| 0.10-0.25 | 247 | 311 | 32.2% | 25.8% | 1.24x | 3.9x |
| 0.25-0.50 | 279 | 366 | 50.5% | 39.6% | 1.28x | 2.5x |
| 0.50+ | 247 | 311 | 80.4% | 69.0% | 1.17x | none |

Compared at equal power -- both sparsest bands have a ceiling near 10x --
the tail reaches 2.26x where ordinary crops reach 8.13x, and covers 22.4% of
ears where they cover 78.2%. So the tail is **not** a shotgun: 2.26x is well
above chance and the lift is above unity in every band. But it is roughly a
quarter as well-placed as a confident face box. That sits between the two
prior results: above `trainable_blazeface`'s 0.69-0.98x, which their own
analysis shows was underpowered at a 52.5% null, and far below the aggregate
ninefold, which this section's mixture finding says to distrust.

**This is also the mechanism behind the matched-precision loss above**, which
had been a bare result. A lower face threshold admits boxes that are about
four times worse at finding ears per box. More crops, each much less likely
to contain an ear, is exactly a precision cost exceeding the recall gain --
the -1.40pp and the 2.26x-against-8.13x are the same fact measured twice.

Two scope limits. This is **stock's** tail, not the fine-tuned detector's; a
model trained to find more faces may well have a better-placed tail, so the
head-ness question stays open for their checkpoint and should be settled by
training the objective rather than by mining anyone's tail. And the sparse
band is 85 ears, so 2.26x is suggestive rather than tight.

### Tracker constants, measured on 25 conversational clips

`data/clips` is 25 people, 30s each, 750s at 720p and 24-60 fps -- the selfie
and video-call framing the earlier video work lacked, and the deployment
distribution: median face side 0.473 of frame height against the exercise
corpus's 0.284, and 0.8% of sampled frames with no face against 19.2%.
`measure_video_dynamics.py` on 7,500 consecutive frames at the deployed
4-view setting.

**The face stage is not a flicker source here, which retires a claim this
file made three times.** 3 of 7,500 frames have no face -- 0.0%. Adjacent
frame face-box IoU is median 0.972, p10 0.908, and under 0.5 in 0.2% of
transitions. The "flicker is upstream" finding went 19.2% on exercise
footage, to 4.2% once structural gaps were separated out, to essentially zero
on the distribution that ships. `maxMisses = 3` therefore protects very
little: coasting a dropped frame is free and still prevents a one-frame
blink, but the recall it defends is near nil, and the earlier claim that the
constant was "now measured" was measured on the wrong corpus.

**Midline candidates are flickerier than ears, not steadier**, which reverses
the retraction recorded in the tracker section above.

| class | tracks | median life | p90 life | ever >= 0.85 | crossings/frame |
|---|---|---|---|---|---|
| midline | 544 | 1.0 | 4.0 | 3.1% | **0.080** |
| lateral | 2634 | 1.0 | 8.0 | 6.8% | 0.055 |

**So `minHits` discriminates, and a lower confirm score becomes affordable.**
Both swept over one detection pass, confirmed tracks counted by position:

| confirm | keep | minHits | midline | lateral | lateral vs shipped |
|---|---|---|---|---|---|
| **0.85** | **0.70** | **2 (shipped)** | **9** | **164** | -- |
| 0.85 | 0.70 | 3 | 6 | 150 | 91.5% |
| 0.80 | 0.65 | 2 | 17 | 210 | 128.0% |
| **0.80** | **0.65** | **3** | **9** | **189** | **115.2%** |
| 0.80 | 0.65 | 4 | 7 | 171 | 104.3% |
| 0.75 | 0.60 | 3 | 15 | 227 | 138.4% |
| 0.70 | 0.55 | 3 | 27 | 302 | 184.1% |

Bootstrapped over the 25 clips, 20,000 draws, **0.80 / 0.65 / minHits 3**
against the shipped **0.85 / 0.70 / minHits 2**: lateral confirmations
**+25, 95% CI [+6, +50], P(gain) = 99.8%**; midline confirmations **0, 95% CI
[-5, +4], P(worse) = 44.9%**. The gain is solid; the cost is
*indistinguishable* rather than zero -- that test has power only to about
+/-5 events on a base of 9, because midline false positives are concentrated
in 5 of the 25 clips and absent from the other 20. `minHits = 3` costs one
extra frame of latency, 33-42 ms here.

**What this is not.** "Lateral" is not "true positive": lateral candidates
include false positives that happen to sit laterally, and 11.4% of real ears
fall inside the midline band, so this is a position proxy rather than a
precision/recall measurement. Converting it needs labelled frames, which is
the deliberately deferred Phase 2. The direction is strong enough to justify
that labelling and not strong enough on its own to be called a recall gain.

**Shipped as `TRACK_CONFIRM = 0.80`, `CONTINUE_MARGIN = 0.15`,
`MIN_HITS = 3`**, with the proxy caveat standing. The confirm score is
deliberately *below* `TTA_SCORE_THRESHOLD`: that constant is the single-frame
argmax, and a track confirming over three frames carries evidence a single
frame does not, so the two quantities are no longer the same thing and are no
longer tied together in code.

**Two defects found while shipping it, both mine, both worse than the change
itself.**

The demo's slider carried `value="0.70"` in its HTML, and `loadModel()` feeds
that attribute straight into `applyThresholds`. **So the page was confirming
tracks at 0.70 with a continue score of 0.55** -- not the 0.85 / 0.70 the
constants said and `test_tta.py` pinned. On the clips corpus that is the
worst configuration in the table above, 40 midline confirmations against 9.
The test checked the constants and never checked the value that overrides
them at load, which is the same defect as a retraction that does not reach
the claim's most prominent form. `test_tta.py` now pins the slider's `value`
attribute and its displayed label against `TRACK_CONFIRM`.

And the still-image upload path had no business running at the tracker's
continue score. `applyThresholds` sets the ear stage's threshold to `keep` so
a track has something to survive on, but a single uploaded image has no
tracker to confirm anything, so it was drawing unconfirmed candidates at 0.70
where before the tracker it gated at 0.85. That path now raises the threshold
to `TTA_SCORE_THRESHOLD` around its one call and restores it afterwards. The
tracker commit introduced this and the five behaviour cases could not have
caught it, because they exercise the tracker and this path does not use it.

### The position proxy, checked against two independent detectors

Everything the clips section concludes rests on splitting candidates by
position and treating midline as mostly-false and lateral as mostly-true. That
proxy had never been checked against anything that knows what an ear is.
`check_referees.py` checks it on frames neither referee nor BlazeEar has
trained on:

* **`yolov11_ear_detector.pt`**, a fine-tuned YOLOv11. It shares human GT
  labels with the crop model but not pseudo-labels -- `crops_train.csv` is
  GT / GT+EAR / GT+REVIEW / NEGATIVE / IGNORE, with no POSE and no raw YOLO
  rows, and GT+REVIEW (YOLO-proposed, human-confirmed) is 0.9% of it. Same
  corpus, different architecture. An earlier note in this file called this
  detector too contaminated to use, on the basis that it had proposed 29% of
  the rows in *a* split; that split was not the shipped crop model's.
* **COCO-pretrained `yolo11x-pose.pt`**, keypoints 3 and 4. Different
  objective, different corpus, different annotators, and POSE regions are
  ignored rather than used as labels here. It returns a point rather than a
  box, which is enough to ask whether there is an ear about here.

1,734 candidates over sampled frames of the 25 clips:

| | n | ear detector | pose | either |
|---|---|---|---|---|
| **all candidates** | | | | |
| midline | 153 | 10.5% | 13.7% | 14.4% |
| between | 105 | 9.5% | 20.0% | 21.0% |
| lateral | 1476 | 46.0% | 69.3% | 69.4% |
| **score >= 0.80, i.e. would confirm a track** | | | | |
| midline | 33 | 48.5% | 51.5% | **54.5%** |
| between | 20 | 45.0% | 50.0% | 55.0% |
| lateral | 798 | 78.6% | **94.2%** | 94.4% |

**The proxy holds directionally and breaks where it was leaned on hardest.**
Lateral candidates are 4.4x more likely to be confirmed than midline ones by
the ear detector and 5.1x by pose, so splitting on position does separate the
populations. And **94.2% of the lateral tracks that would confirm are
corroborated**, which is the strongest available evidence that the +25
confirmations shipped with `TRACK_CONFIRM = 0.80` are real ears rather than
noise.

But at the confirm threshold **54.5% of midline candidates are corroborated
too.** So "midline confirmations" was never a clean false-positive count --
roughly half are real ears, which is consistent with 11.4% of all ears
falling inside that band and with real ears scoring higher. Raising `minHits`
to 3 cut midline confirmations 9 to 6, and about half of that cut was
genuine ears. The cost side of that argument is weaker than the table in the
clips section makes it look; the gain side is stronger.

Neither referee is ground truth, and they disagree with each other in a
useful way: pose confirms more than the ear detector almost everywhere, and
for lateral candidates the ear detector's confirmations are nearly a subset
of pose's (69.3% against a 69.4% union). That bounds how far either can be
trusted and points at the right use, which is **triage rather than
labelling**: auto-propose, have a human check only the disagreements plus a
random audit sample to estimate the referee's own error rate. The audit
sample is what stops a referee's bias quietly becoming the ground truth.

### Where this can go next, and what it is worth

Written at the close of the gate/tracker work, so the next person does not
re-derive the ranking. One number frames all of it. Of `val_v2`'s 2,385 human
ears:

| | ears | share |
|---|---|---|
| no face detected, so stage 2 never sees it | 29 | 1.2% |
| no TTA view finds it at all | 302 | 12.7% |
| found, but no view ever reaches 0.85 | 422 | 17.7% |
| **beyond any gate, tracker or fusion change** | **724** | **30.4%** |

**Calibration work is essentially exhausted.** Everything in this section of
the README -- the gate sweep, the arm A/B, the geometric vetoes, the tracker,
view agreement -- operates on the ~70% of ears the model can already score
above a useful threshold in at least one look. The remaining 30% is a
capability problem: the model does not produce a confident candidate for
those ears under any decision rule. No threshold, hysteresis or fusion trick
reaches them.

In rough order of value per unit of cost:

1. **Two baseline seeds, and a `--seed` flag to make them possible.** ~1.5
   GPU-hours. `train_blazeear.py` has no seed control, so every training
   result here is one run against one run, and the effects in play are about
   a point. This blocks more than it looks: whether the negative-mining cap's
   -1.09pp is real, whether a B1-only arm is worth running, and whether *any*
   future training change could be detected at all. Cheapest thing on this
   list and it unblocks the whole of item 4. Something seed-sensitive is
   already known to be present -- which prior the negative budget
   concentrates on does not reproduce across runs.

2. **The >=2-view agreement rule.** Measured at 3.9 false positives removed
   per ear sacrificed (29 ears against 112 confirmable false positives),
   which beats the geometric veto's 1.4 and approaches the score gate's 4.4,
   on an axis independent of both score and time -- so it stacks with the
   tracker. `merge_views` already computes each cluster's membership and
   discards it; the work is returning the distinct view count, dropping
   single-view clusters, mirroring it in `docs/blazeear_inference.js`, and a
   test. Verify the ratio in deployed units first: the 17.2% is per-view
   clusters at a 0.10 floor, counted before fusion and NMS.

3. **A short labelled video clip.** Thirty to sixty seconds of webcam footage
   with ears boxed. This is the only data this repository has never had, and
   three things depend on it. `minHits` and `maxMisses` are about frame
   *ordering*, which no still-image proxy reaches, so both are still picked
   from principle. The tracker's 9.2pp is an upper bound that only video can
   convert into a number. And false-positive temporal persistence -- the
   property reasoned about wrongly twice in this work -- would be measured
   directly rather than argued. Semi-automatic labelling with the current
   detector plus hand correction makes this cheap in wall-clock terms; the
   cost is attention, not compute.

4. **Capability work on the 30%.** Where the remaining recall actually is,
   and the only item that can move the headline. The learning curve is flat
   past 25% of the existing crops, so more of the same data is not it; the
   candidates are profile and yaw coverage specifically, a larger backbone, or
   a different second stage. All of them are expensive and none is evaluable
   until item 1 exists, which is the argument for doing item 1 regardless of
   whether anything here is ever attempted.

**Stopping is also a defensible option.** The reported symptom is fixed, both
gates sit on their measured end-to-end argmax, and the tracker delivers more
recall than the lower gate did without its false positives. The residual
midline false-positive rate at 0.85 is 0.023 per image before hysteresis,
and items 2 and 3 are refinements of a pipeline that now behaves.

What should *not* be revisited, with the reasons recorded above so they are
not relitigated: the crop-centre geometric veto (measured, three times worse
than the gate), positional midline hard negatives (would train against 11.4%
of real ears, and add nothing top-k mining does not already do), the keypoint
veto (keypoints fail together with the face box, the fourth demonstration of
one property), and a single shared gate across both paths (the fused argmax
is genuinely higher than the plain one, because `max` fusion shifts the score
distribution up).

### How the measurements here were checked

Worth recording because the numbers above are only as good as this, and
because the answer was not what either of us doing the work expected.

Across both sides of this investigation about eight measurements were wrong in
a way that would have produced a *confident false mechanism* -- not a crash, a
story. On this side: the wrong box format to `assign_anchor_targets` (a
starved-supervision story), `anchors_to_corners` read as xyxy (median positive
IoU 0.000), a hand-rolled letterbox instead of `resize_pad` (tragion figures
that understated the gap), and two different seeds divided as one measurement
(a 6.82% face-stage ceiling).

**Not one was caught by reading the code.** Every one was caught by a clash
with a quantity already known: 54% against a documented 78.6%, a ceiling that
would not reconcile, exactly 1.000 ears per image across 13,482 files with no
exceptions. That is the useful asymmetry -- a wrong measurement has no syntax
error and no failed assertion, but it cannot agree with an independent
quantity. So the cheap detector is to always have one, and the discipline is
to *name it before trusting the result* rather than after being contradicted.

The scope of that claim is narrow and worth stating precisely, because the
loose version is wrong. Reading code works, and found most of what was fixed
in this round: focal loss documented as opt-in when the trainer has always
defaulted it on, a `--csv-format` flag documented as the primary training
command that no parser has ever accepted, a parity docstring claiming to pin
the exported graphs, two inference paths with different candidate filters, and
a self-check whose probe could not reach the code path it claimed to check.
Every one of those came from reading. What reading cannot see is a *measurement*
error, because a wrong measurement is not a defect in the code that produced
it -- the script is correct and the question it answers is not the one asked.
Eight cases with none caught by review supports that much, and not more.

Two traps were hit twice each, from different directions, which is what makes
them structural rather than careless:

- **Slicing a sorted or grouped file and calling it a sample.** A CSV ordered
  by source gives a "sample" that is one dataset.
- **Combining figures from runs with different draws.** `train_blazeear.py`
  has no `--seed`, so two runs are two draws; a ratio across them measures
  nothing. This is also why `queue_augment_scale_ab.sh` runs a control arm
  instead of comparing against the shipped checkpoint.

A third, specific to this repository and already warned about elsewhere in
it: `decode_boxes`, `anchors_to_corners` and `assign_anchor_targets` all speak
`[ymin, xmin, ymax, xmax]`, and every one of them returns plausible-looking
numbers when handed xyxy or `[x1, y1, w, h]`.

### What was measured and ruled out

Several plausible directions were tested and did not pay, which is worth
recording so they are not retried blind:

- **Tragion-centred crops, for the 8.7% that escape the window.** The face
  stage carries MediaPipe's six trained keypoints -- two of which are the ear
  tragions -- and `BlazeEarInference.decode_boxes` reads `raw_boxes[..., 0:4]`
  and drops coords 4:16, so nothing downstream sees them. Placing a second crop
  on the predicted tragion is the obvious use for them, and it fails on exactly
  the ears it would need to serve. Distance from the nearer predicted tragion
  to the ear centre, in multiples of that ear's own size, over the 355 ears of
  `common_heldout` (`python measure_tragion_coverage.py`, which also prints the
  reachability split above as a check on itself):

  | | n | median | <=0.5x | <=1.0x |
  |---|---|---|---|---|
  | reachable ears | 315 | 0.31x | 68.3% | 86.3% |
  | ears escaping every window | 31 | **6.61x** | **0.0%** | **9.7%** |

  Not one of the 31 has a tragion within half an ear-size. The failure modes
  are correlated rather than independent: the geometry that puts an ear beyond
  `FACE_CROP_EXPAND` x the face box is the geometry that occludes or
  extrapolates that side's tragion. So the prior is strong where the crop
  already works and absent where it fails, which rules it out as a union, a
  confidence gate and a fallback alike -- there is no configuration in which it
  supplies information about the ears that need it. Note when reading the
  keypoints at all: `blazeear.py:89` calls them untrained, which is true of the
  ear model -- the crop training data has no keypoint labels -- but not of the
  face stage, which is MediaPipe's weights unchanged and loads its keypoint
  heads with zero missing keys.
- **The `--postprocess` ONNX graph returned a tail of zero-score boxes
  (fixed).** Not a numerical defect -- it was checked, and the check filtered
  the tail deliberately -- but `detections` did not mean what the README said,
  and the tail made the graph uncheckable against its own wrapper because its
  order is NMS tie-break among equal zeros, which torch and onnxruntime
  resolve differently. The instructive half is the self-check: it probed with
  `torch.randn`, which the wrapper's own `x / 127.5 - 1.0` turns into
  near-uniform grey, so nothing cleared the 0.8 gate, decode and NMS saw no
  box, and the script returned success. A check that passed loudest when it
  had verified nothing. There is now a `--check-image`, and the no-detection
  case says it is unverified rather than implying a pass. See the export
  section for the numbers.
- **A latent disagreement between `process` and the export path (fixed).**
  Found while adding the emitted-detections parity test.
  `_tensors_to_detections` dropped decoded boxes with no area and non-finite
  scores before NMS; `BlazeEarInference` dropped neither, so the two could
  hand NMS different candidate sets on identical weights -- and a different
  candidate set is a different suppression outcome, not a rounding
  difference. On untrained weights they did: 562 of 3,584 candidates
  degenerate, and the top-100 sets differing by a row. On the shipped model
  there are **zero degenerate boxes in 1,075,200 candidates from 300 real
  crops**, so it never showed in any measurement here.
  `box_utils.sort_corners` and `usable_box_mask` are now the single
  definition, used by both eager paths and by the three exportable graphs
  (which zero the score instead of masking rows, keeping their fixed shapes).
  Untrained weights agree across seeds 0-4 now and there is a regression test
  at those weights, since that is where the defect was reachable at all.

  Worth keeping the reasoning: the useful lesson is not the bug but *when* it
  was reachable. The paths agreed on every trained model anyone measured and
  disagreed on an untrained one -- which is exactly the case someone hits
  comparing the paths to check a model part-way through a run.
- **Treating the crop periphery as off-distribution.** The obvious reading of
  the escaping ears is that the model is weak near the window edge, so an
  ear that lands there is doubly penalised. It is the wrong way round: ears
  sit at the periphery *by default*, because the window is centred on the
  face and ears are lateral to it. Over the 12,595 human training ears, the
  radial offset from the crop centre is median 0.361 of a side, 89.4% are
  beyond 0.25, and only 3.0% fall within 0.15. The centre of a face crop is
  the one place an ear almost never is, so the periphery is the training
  distribution and there is no edge penalty to recover. Nor are far ears
  confounded with small ones: corr(offset, size) = -0.164, and median ear
  size is flat at 58-67 px across every offset band.
- **More training data.** A learning curve at a fixed 50,800-step budget over
  12.5 / 25 / 50 / 100% of the crops is flat past 25%: 0.6681, 0.7086, 0.7233,
  0.7173 crop-space mAP@0.5. The last doubling -- 6,300 more human-labelled ear
  boxes -- bought nothing measurable.
- **More ears from an auto-labeller.** The fine-tuned YOLO detector finds
  exactly one confident ear without a human box across 279 held-out images.
  `relabel.py` already harvested that reservoir; 29% of the rows in that split
  came from it.
- **Refitting the anchor priors.** The shipped priors reproduce k-means on
  *full frames*, so they predate the crop pipeline and are about half the size
  the model now sees; refitting on crops lifts the anchor-match ceiling from
  0.469 to 0.571 median IoU. Retrained, it is worse on every crop-space metric
  (0.7136 against 0.7426). Background accuracy falls 0.9949 -> 0.9920, because
  larger priors push more anchors past `ANCHOR_IGNORE_IOU` into the ignore band
  and starve the negative pool. The priors cannot be refit in isolation.
- **A wider backbone.** Doubling every channel (357,608 parameters against
  103,160) is worth +0.085 crop-space mAP@0.5 against its own from-scratch
  control -- but MediaPipe initialisation is worth +0.127, and a widened
  backbone cannot load those weights. The wide scratch model lands below the
  shipped narrow pretrained one. Collecting both effects needs a
  function-preserving widening (Net2Net) from the MediaPipe weights rather than
  a wider model trained from scratch.
- **A bigger face stage.** The front BlazeFace finds no face in 2.9% of
  held-out images, and an ear whose face is missed never reaches the second
  stage. MediaPipe's
  back-camera model (136,240 parameters at 256 px, against 101,390 at 128)
  removes the ceiling completely -- no-face goes to 0.0% at its own swept
  threshold of 0.10 -- and buys nothing: -0.0078 mAP@0.5, 95% CI [-0.0557,
  +0.0395], not distinguishable, and the same verdict on mAP@[.5:.95] and det
  IoU. Perfect FACE DETECTION was simply not worth much: those 2.9% of images
  hold **9 of 355 ears (2.5%)**, and the back model recovers 3 of the 9. A
  plausible reason the rest of the frame does not improve either:
  `make_face_crops.py` built the training crops with the *front* detector, so
  the ear model's input distribution is front-model crops and a
  differently-calibrated face box is off-distribution even when it is correct.
  That is untested -- all the end-to-end intervals here straddle zero, so there
  is no established regression to explain, only an absent gain.

  **This entry originally generalised from 2.5% to "the recall ceiling", which
  was wrong.** Counting per ear rather than per image, 11.3% of ears never
  reach the second stage, and the larger part is not the detector's fault --
  see below.
For scale: on the same images and labels, YOLOv11m at 640 px reaches 0.8829
mAP@0.5 against this pipeline's 0.5809, so the headroom is real -- but it
differs in both capacity and input resolution, and nothing here has separated
those two.
