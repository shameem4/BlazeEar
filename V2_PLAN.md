# BlazeEar v2 — fix plan

Working branch: `v2`. Ordered by dependency, not by severity: each phase makes the
next one measurable. Nothing in P2–P4 can be evaluated until P0 lands, and the
retrain in P4 should happen once, after P1–P3 are all in.

Status legend: `[ ]` todo · `[~]` in progress · `[x]` done

---

## P0 — Measurement

Until these land, no experiment is interpretable. None of them require a retrain;
they change what the numbers mean, not what the model is.

- [x] **Dataset-level mAP.** `utils/metrics.py:compute_map_torch` computes VOC07
      11-point AP *per image*, and `_compute_metrics` averages those. With 1–2 GT
      boxes per image the per-image AP quantizes to a few discrete values and the
      mean is dominated by that quantization. Replace with a single PR curve pooled
      over the whole split (all-point interpolation), and report mAP@0.5 plus
      mAP@[.5:.95].
- [x] **Fix checkpoint selection.** `train()` selects `is_best` on validation
      *loss* over `200 // batch_size` batches of an unshuffled loader, and
      `split_dataframe_by_images` preserves master-CSV row order, which is
      source-by-source. The selection slice is therefore the first ~192 images of
      one annotation source. Select on mAP, evaluate on the full val split.
- [x] **Stop reporting `mean_iou` as detection quality.** It is computed only over
      target-positive anchors, so false positives and misses cannot affect it.
      Rename to `positive_anchor_iou` and add a real post-NMS detection IoU.
- [x] **Break metrics out by `annotation_source`**, so GT and pseudo-label
      performance are never averaged into one headline number again. Shipped as
      `evaluate.py`, using ignore regions so excluding untrusted labels does not
      turn real ears into false positives.

### Measured impact of the P0 fixes

`runs/checkpoints/BlazeEar_best.pth` (epoch 94) on the full 2022-image val split,
old metric vs new, same checkpoint and same detection path throughout:

| metric | value |
| --- | --- |
| old per-image average, first 192 images (what was published) | 0.4286 |
| old per-image average, full val | 0.2801 |
| **pooled mAP@0.5** | **0.1358** |
| pooled mAP@[.5:.95] | 0.0328 |
| post-NMS IoU on matched detections | 0.6242 |
| detections kept vs 4009 GT boxes | 78834 |

The published number was roughly 3x the real one. The detector emits ~39 detections
per image at the 0.1 eval threshold; per-image AP averaging hid that, because a
flood of false positives spread across images barely moves a per-image mean but
dominates a pooled ranking. Localization is not the problem — matched boxes sit at
0.62 IoU. Precision is.

**The documented "peak 0.46 -> final 0.25 collapse" is a measurement artifact, not a
training dynamic.** On this one unchanging checkpoint the old metric reads:

| val images evaluated | old mAP@0.5 |
| --- | --- |
| 96 | 0.5508 |
| 192 | 0.4286 |
| 288 | 0.3312 |
| 480 | 0.2497 |
| 2022 | 0.2801 |

Per-epoch validation used `200 // batch_size` batches and the final summary used
`500 // batch_size`, so the reported decline from ~0.46 to ~0.25 is reproduced
exactly by changing only the subset size, with the model held fixed. `val.csv`
preserves master-CSV order, which is source-by-source, so short prefixes sample one
easy source.

### POSE labels are a different annotation convention, not noise

`python evaluate.py --checkpoint runs/checkpoints/BlazeEar_best.pth`:

| view | GT boxes | mAP@0.5 | mAP@[.5:.95] |
| --- | --- | --- | --- |
| all sources | 4009 | 0.1358 | 0.0328 |
| human only (POSE ignored) | 2495 | **0.1935** | 0.0476 |
| POSE only (human ignored) | 1514 | **0.0067** | 0.0014 |

The model scores ~0 against POSE boxes *that it was trained on*. Box statistics
from `master.csv` explain why: POSE boxes have the same aspect ratio as human
boxes (median 0.52 vs 0.51) but a median area of 900 px vs 2383 px — about 2.6x
smaller. Two concentric boxes differing 2.6x in area cannot reach IoU 0.5, so
POSE is not noisy labelling of the same convention, it is a *different*
convention: a tight box around the ear keypoint rather than the ear's extent.

This makes the pseudo-labels actively harmful rather than merely unreliable. The
same visual object is labelled at two incompatible scales depending on source, so
box regression receives contradictory targets it cannot satisfy — which is
consistent with mAP@[.5:.95] of 0.033 while matched detections sit at 0.62 IoU.
Relabeling (P1) should therefore also normalise POSE boxes to the human
convention, not just fill in missing ears.

### The shipped geometry filter discards 22% of real ears

`EAR_MIN_ASPECT_RATIO = 0.35` / `EAR_MAX_ASPECT_RATIO = 1.4` against the human
labels in `master.csv`: 22.0% of GT boxes and 26.3% of GT+EAR boxes fall outside
that gate. The filter that `BlazeEar.process` and the JS demo apply is rejecting
more than a fifth of true ears on aspect ratio alone, before the close-up size
gate is even considered. See P5.

## P1 — Data and labels (long pole; blocks the retrain)

Evidence from `data/splits/master.csv`: 13482 images, 26567 boxes. 82% of images
(11084) carry exactly one human-labeled ear; only 4821 images have POSE fill-in.
37.8% of *validation* boxes are POSE. Most frontal images therefore contain a real
but unlabeled second ear, and hard-negative mining selects the highest-scoring
background anchors — precisely that ear.

- [~] **Relabel the missing second ears.** An auto-labeller is viable: measured
      against human-verified boxes on val, `model_weights/yolov11_ear_detector.pt`
      scores mAP@0.5 **0.8554**, mAP@[.5:.95] 0.6104, matched-box IoU 0.8621 —
      against BlazeEar's 0.1935 / 0.0476 / 0.6288. Do *not* use the pose model.
      Step 1 (running): retrain the labeller on human-only, geometry-filtered
      boxes via `finetune_yolov11.py --sources human`, since the shipped one was
      trained on POSE boxes too (`convert_split` applied no source filter).
      Step 2: propose on every image, queue disagreements with existing labels
      for human accept/reject through `utils/image_annotation_viewer.py`.
      The review pass is human work and is the schedule driver.

### POSE pseudo-labels are junk, not a rescalable convention

*Correction to the earlier entry in this plan.* From box statistics alone POSE
looked like a tighter annotation convention (2.6x smaller median area). Checked
properly against the human labels in the same image:

- **97.8% of POSE boxes have zero overlap with any human box**; 0.0% reach IoU 0.5.
- Median centre distance to the nearest human box: **5.34 human-box diagonals**.
- Rendering them confirms it: POSE boxes land on eyebrows, hair above the ear,
  and in one sampled image a blurry background object.

Two independent detectors corroborate it. Both BlazeEar (0.0067) and the YOLO
ear detector (0.0073) score ~zero against POSE boxes, *and the YOLO detector was
trained on them* — they were inconsistent enough to wash out as noise rather
than be learned. So they are misplaced, not mis-scaled: drop them, do not
normalise them.

### Human labels need a light sanity filter

3.84% of human boxes (613 of 15972) are degenerate: under 6 px on a side, under
100 px2 in area, or aspect outside 0.15-2.0. They concentrate in one source,
*Ear Detection from Full Face image.v1i.coco*, at 6.8%. `finetune_yolov11.py`
now drops these by default.
- [x] **Ignore band in anchor assignment.** Even with complete labels, anchors that
      overlap a GT box but lose assignment are currently trained as background.
      Exclude a middle IoU band from both the positive set and hard-negative mining.
- [ ] **Stratify the split** by `annotation_source`, and by subject identity
      wherever the source dataset exposes it — a random per-image split leaks the
      same person across train and val.

## P2 — Augmentation correctness

- [x] **Order: geometric augs before resize/pad.** `dataloader.py:__getitem__` calls
      `_resize_and_pad` then `_augment_image`, so `augment_scale` letterboxes an
      already-letterboxed image and `augment_rotation` rotates the padding bars into
      diagonal wedges that never occur at inference — all at 128px, on ears of
      roughly 10×15px. Augment at native resolution, resize once.
- [x] **Stop erasing the ear while keeping the label.** `augment_face_cutout` fills
      1.3–1.9× the ear box with a solid random color and returns the GT box
      untouched; ~24% of boxes per epoch. `augment_targeted_ear_occlusion` adds
      25–60% solid fill, also label-preserving. Either drop the box when occlusion
      passes a visibility threshold, or cap occlusion so the ear stays partly
      visible. (Correction to an earlier note in this plan: `drop_probability`
      was *not* inverted — it is genuinely the probability of dropping the
      region. Renamed to `occlusion_probability` for clarity only.)
- [x] **Contiguity.** `np.fliplr` returns a negative-stride view that is then written
      through by the occlusion augs and handed to `cv2.resize` / `cv2.warpAffine`.
      Fixed by materialising a contiguous copy; the boxes array is also no
      longer mutated in place.

### Measured impact of the P2 fixes

Old `augment_face_cutout` over 200 seeds on a labelled ear: mean ear visibility
0.400, and the ear **fully erased in 120 of 200 calls** while its positive label
was kept. With the dataloader's 40% gate that is ~24% of boxes per epoch trained
as "this flat colour patch is an ear". After the fix: 0 erasures, mean visibility
0.792, and 110 tests assert the invariant across seeds.

Augmenting before the letterbox costs throughput, since the warps now run on a
real image rather than a 128x128 thumbnail. Sources here are uniformly 640x640:

| augmentation working size | samples/s/worker |
| --- | --- |
| native 640 | 44.7 |
| capped at 256 (default) | 204.8 |
| old, on the finished 128px tensor | ~900 |

256 is 2x the model input, so nothing that survives the final resize is lost, and
it aligns training with inference, which already resamples source -> 256 -> 128 in
`BlazeDetector.resize_pad`.

## P3 — Loss and numerics

- [x] **Use `binary_cross_entropy_with_logits`.** Training pre-applies `sigmoid`
      inside `autocast`, then the loss hand-rolls BCE with `clamp(1e-7, 1-1e-7)`.
      Under fp16 the sigmoid saturates long before the clamp, capping per-sample
      loss at ~16.1 and truncating gradients on confidently-wrong anchors — exactly
      the samples hard-negative mining just selected. Same for the focal path.
- [x] **Box regression scale.** *Correction to the original review claim:* the
      SmoothL1 argument was **not** always inside the quadratic regime. Measured
      on the epoch-94 checkpoint, 51.7% of positive-anchor errors exceed 1.0, so
      the loss was already largely linear. The real defect is the opposite end:
      beta=1.0 is the width of the entire image, so a 6 px error received a
      gradient of 0.05 while a gross error received one 20x larger, and fine
      localization went untrained. beta is now 0.1 (~13 px at the 128 px input):
      quadratic for fine localization, linear and outlier-robust above.

### Measured impact of the P3 fixes

The old classification loss applied `sigmoid` and then `torch.log` behind a
`clamp(1e-7, 1 - 1e-7)`. `torch.clamp` has **exactly zero gradient outside its
range**, so any positive anchor whose logit fell below about -16.1 received no
gradient at all, permanently. On the epoch-94 checkpoint:

| statistic over positive-target anchors | value |
| --- | --- |
| median logit | **-1047** |
| 25th / 75th percentile logit | -4990 / -13.3 |
| **fraction below -16.1 (the dead zone)** | **73.5%** |
| background anchors above +16.1 | 0.001% |

Three quarters of the positive supervision signal was switched off and could not
recover. Mean positive loss sat at 13.69 against the clamp's 16.1 ceiling. In
logit space the same batch scores 4411, i.e. 322x — that ratio measures how far
the checkpoint had drifted into the dead zone, not what a fresh run will see,
since MediaPipe-initialised logits start near 0.

This is very likely the dominant cause of the detector's behaviour: anchors that
should fire are pinned at hugely negative logits, their box heads are therefore
unconstrained (median regression error 6.09 in normalized units, where 1.0 is the
whole image), and the surviving detections flood the image with low-confidence
boxes. It also explains why the geometry and duplicate filters were needed at
inference.

`binary_cross_entropy_with_logits` has bounded gradient regardless of how wrong a
prediction is, so no clamping is needed and nothing destabilises.

## P4 — Architecture (single retrain)

- [x] **Put normalization into the model.** `BlazeEar._define_layers` uses
      `BlazeBlock_WT` (BN folded into conv) throughout; `BlazeBlock`, the trainable-BN
      block the `trainable_blazeface` lineage exists to provide, is never
      instantiated, and `convert_blazeear_wt_to_trainable` is only reachable from a
      fallback that never fires. Switch to `BlazeBlock`, load MediaPipe weights via
      `unfold_conv_bn`, and add a test asserting eval-mode output parity with the
      folded model at initialization.
- [x] **Variable-size anchors.** *Not a one-line flag.* `encode_boxes_to_anchors`
      ignores the anchor tensor entirely: it hardcodes square anchors of 0.0625 /
      0.125 and emits targets per grid *cell* (`[16,16,5]`), which
      `flatten_anchor_targets` repeats ×2 and ×6 — so all anchors in a cell share one
      target by construction. Enabling `fixed_anchor_size=False` in decode alone
      would desync training targets from inference decode. Rewrite assignment to be
      driven by the real `[896,4]` anchor tensor with per-anchor targets, then enable
      the variable-size path.
- [ ] **Retrain**, once P1–P3 are in. Invalidates every checkpoint in
      `runs/checkpoints/` and all five ONNX files in `docs/`.

### Anchor priors: measured, and the choice revised

Best-IoU between each of 13477 human ground-truth ears and its closest anchor:

| anchors | median best-IoU | >=0.5 | >=0.35 |
| --- | --- | --- | --- |
| fixed w=h=1.0 (original) | 0.006 | 0.1% | 0.2% |
| MediaPipe variable-size | 0.234 | 18.0% | 36.4% |
| **fitted to ear statistics (now used)** | **0.372** | 23.8% | 54.7% |
| all 8 fitted priors on the 16x16 grid | 0.441 | 40.5% | 62.1% |
| 4 fitted priors on a stride-4 grid | 0.530 | 57.6% | 81.8% |

The selected option was "enable the existing variable-size path". Measured, that
reaches only 18% of ears at IoU 0.5, because MediaPipe's priors are square and
span 0.148-0.866 while ears here are small and tall (median 0.053 x 0.108) --
the smallest square prior is wider than a median ear is tall. Priors fitted by
k-means to the actual box statistics dominate it on every column at identical
cost and identical exported shape, so those are the default. Switch back with
`generate_reference_anchors(fixed_anchor_size=False)`.

Two things follow from the table:

1. **IoU-thresholded assignment is not viable on this architecture.** Even with
   fitted priors, 76% of ears never reach IoU 0.5 with any anchor, so a 0.5
   threshold would discard them. Assignment is therefore best-match per box
   (top-k), which supervises every box regardless of prior quality.
2. **The ceiling is spatial, not prior quality.** Stride 8 spaces anchor centres
   0.0625 apart for objects 0.053 wide. A stride-4 head roughly doubles the
   fraction of ears reaching IoU 0.5. That changes the exported graph, so it is
   not done here, but it is the single highest-value architecture change left.

`ANCHOR_TOP_K` is 3, against the old encoder's effective 12 positives per box
(one per cell, repeated 2x and 6x). Worth a sweep during the retrain.

### The unfold path had never once executed

`BlazeEar` now builds from `BlazeBlock` (32 BatchNorm layers, 103418 params
against 101390) with `use_batchnorm=False` retained for loading folded weights
and for parity checks.

Switching it exposed why the dead code was dead. `load_mediapipe_weights` chose
between the direct and converted paths with `try: load_state_dict(...) except
RuntimeError`, but `load_state_dict(strict=False)` does not raise on missing or
unexpected keys, only on shape mismatch. The direct load therefore always
"succeeded" and the conversion never ran. Loading MediaPipe weights into a
BatchNorm backbone reported success while leaving **160 keys missing and 64
unexpected** -- the entire backbone randomly initialized, silently.

The choice is now made by inspecting the target model's keys, and a backbone
that fails to load raises instead of returning a list nobody checks. With that
fixed, both variants load 0 missing / 0 unexpected, and eval-mode parity between
the folded and BatchNorm models is **1.2e-4 relative** -- the residual being
BatchNorm's eps compounding over 32 layers. The lineage claim is true for the
first time.

## P5 — Inference parity

- [x] **One implementation each of anchors, decode, and NMS.** Anchors are generated
      in 3 places, box decode exists in 5, NMS in 5, the geometry filter in 2.
      Commit `efea1f3` touched 15 files to change one conceptual threshold.
- [ ] **Geometry filter policy.** `EAR_MAX_SIZE_FRAC = 0.55` rejects any ear filling
      more than 55% of the frame — i.e. the earbud-fitting close-up that motivates
      the project. Make it configurable and default it off once precision no longer
      depends on it.
- [ ] **Make the four paths agree.** `BlazeEar.process` filters by geometry;
      `BlazeEarInference.forward` does not; the e2e ONNX does not; the web ONNX plus
      JS does. Add a parity test pinning all four to the same output on fixed inputs.
- [ ] **Re-export ONNX artifacts** after the retrain.

### Unification found two live divergences

Changing the priors in P4a immediately desynced the paths, which is the hazard
this item exists to remove: the dataloader matched against fitted priors while
the trainer's decode, `BlazeEar.process` and `BlazeEarInference` all still used
three separately hardcoded copies of w=h=1.0. Training stayed self-consistent
only because targets are absolute boxes, so the priors were doing nothing for
regression -- half their value silently discarded. All four now read
`utils.anchor_utils.get_anchors`.

Preprocessing had three resamplers: `cv2.resize` in `BlazeDetector.resize_pad`
and the dataloader, `F.interpolate(mode="bilinear")` in
`BlazeEarInference.preprocess`, and canvas `drawImage` in the browser. The cv2
and torch paths differed by up to 0.0072 per pixel on a [-1, 1] scale (0.0013
mean), moving raw outputs by 0.003. Minor, but it is train/serve skew that
nothing would ever have reported. Both Python paths now share
`utils/preprocess.py` and produce bit-identical tensors. The browser cannot call
cv2, so that residual remains until preprocessing moves into the exported graph.

`test_inference_parity.py` pins all of this: 9 tests covering anchor identity
across four paths, decode agreement between `box_utils` and the pipeline, and
bit-identical preprocessing at four aspect ratios.

## P6 — Claims

- [ ] Update `README.md` and issue a correction to the LinkedIn article once numbers
      are re-measured. The current text claims trainable BatchNorm that is not in the
      model, and reports a mAP that is neither standard mAP nor measured against
      fully human labels.
