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

- [ ] **Relabel the missing second ears.** Build a proposal + human-review pipeline:
      auto-propose with `model_weights/yolov11_ear_detector.pt` and the pose model,
      queue anything that disagrees with existing labels, review through
      `utils/image_annotation_viewer.py`. Automated proposals are *proposals* —
      the accept/reject pass is human work and is the schedule driver here.
- [ ] **Ignore band in anchor assignment.** Even with complete labels, anchors that
      overlap a GT box but lose assignment are currently trained as background.
      Exclude a middle IoU band from both the positive set and hard-negative mining.
- [ ] **Stratify the split** by `annotation_source`, and by subject identity
      wherever the source dataset exposes it — a random per-image split leaks the
      same person across train and val.

## P2 — Augmentation correctness

- [ ] **Order: geometric augs before resize/pad.** `dataloader.py:__getitem__` calls
      `_resize_and_pad` then `_augment_image`, so `augment_scale` letterboxes an
      already-letterboxed image and `augment_rotation` rotates the padding bars into
      diagonal wedges that never occur at inference — all at 128px, on ears of
      roughly 10×15px. Augment at native resolution, resize once.
- [ ] **Stop erasing the ear while keeping the label.** `augment_face_cutout` fills
      1.3–1.9× the ear box with a solid random color and returns the GT box
      untouched; ~24% of boxes per epoch. `augment_targeted_ear_occlusion` adds
      25–60% solid fill, also label-preserving. Either drop the box when occlusion
      passes a visibility threshold, or cap occlusion so the ear stays partly
      visible. Also rename `drop_probability` — `> 0.6` means it fires 60% of the
      time, the opposite of what the name reads as.
- [ ] **Contiguity.** `np.fliplr` returns a negative-stride view that is then written
      through by the occlusion augs and handed to `cv2.resize` / `cv2.warpAffine`.
      (Unverified — OpenCV's behavior on non-contiguous input varies by version.)

## P3 — Loss and numerics

- [ ] **Use `binary_cross_entropy_with_logits`.** Training pre-applies `sigmoid`
      inside `autocast`, then the loss hand-rolls BCE with `clamp(1e-7, 1-1e-7)`.
      Under fp16 the sigmoid saturates long before the clamp, capping per-sample
      loss at ~16.1 and truncating gradients on confidently-wrong anchors — exactly
      the samples hard-negative mining just selected. Same for the focal path.
- [ ] **Box regression scale.** `decode_boxes` returns normalized [0,1] coords, so
      the SmoothL1 argument never leaves the quadratic regime and "Huber" is a no-op.
      Set `beta` to the coordinate scale, or regress in anchor-relative units.

## P4 — Architecture (single retrain)

- [ ] **Put normalization into the model.** `BlazeEar._define_layers` uses
      `BlazeBlock_WT` (BN folded into conv) throughout; `BlazeBlock`, the trainable-BN
      block the `trainable_blazeface` lineage exists to provide, is never
      instantiated, and `convert_blazeear_wt_to_trainable` is only reachable from a
      fallback that never fires. Switch to `BlazeBlock`, load MediaPipe weights via
      `unfold_conv_bn`, and add a test asserting eval-mode output parity with the
      folded model at initialization.
- [ ] **Variable-size anchors.** *Not a one-line flag.* `encode_boxes_to_anchors`
      ignores the anchor tensor entirely: it hardcodes square anchors of 0.0625 /
      0.125 and emits targets per grid *cell* (`[16,16,5]`), which
      `flatten_anchor_targets` repeats ×2 and ×6 — so all anchors in a cell share one
      target by construction. Enabling `fixed_anchor_size=False` in decode alone
      would desync training targets from inference decode. Rewrite assignment to be
      driven by the real `[896,4]` anchor tensor with per-anchor targets, then enable
      the variable-size path.
- [ ] **Retrain**, once P1–P3 are in. Invalidates every checkpoint in
      `runs/checkpoints/` and all five ONNX files in `docs/`.

## P5 — Inference parity

- [ ] **One implementation each of anchors, decode, and NMS.** Anchors are generated
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

## P6 — Claims

- [ ] Update `README.md` and issue a correction to the LinkedIn article once numbers
      are re-measured. The current text claims trainable BatchNorm that is not in the
      model, and reports a mAP that is neither standard mAP nor measured against
      fully human labels.
