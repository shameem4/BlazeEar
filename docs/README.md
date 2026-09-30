# BlazeEar JavaScript Demo

Browser-based ear detection using ONNX Runtime Web.

## Quick Start

1. **Start a local server** (required for module loading):
   ```bash
   cd BlazeEar
   python -m http.server 8000
   ```

2. **Open in browser**: http://localhost:8000/docs/

3. **Use the demo**:
   - Click "Start Webcam" for live detection
   - Or upload an image for single-frame detection

## The two-stage pipeline

The demo runs the pipeline MediaPipe itself uses: a coarse detector on the full
frame, then a fine model on a crop around each hit. This is the only path the
page offers; the single-stage detector it replaced is still in the API and
documented below.

```
frame -> BlazeFace_web.onnx -> square crop at 1.5x each face -> BlazeEar_web.onnx -> map back -> NMS
```

The reason is resolution. An ear is a median **14 px** once a whole frame is
squeezed into the 128x128 input, which is below the smallest anchor the
architecture has. Inside a face crop it is **32 px**. On images that no model
in this repo trained on, that is worth:

| pipeline | mAP@0.5 | mAP@[.5:.95] |
|---|---|---|
| single-stage, full frame | 0.3142 | 0.0938 |
| **two-stage** | **0.5809** | **0.2445** |

**The catch, stated plainly:** an ear whose face BlazeFace misses never reaches
the second stage. At the default `faceThreshold` of 0.2 that is 1.3% of
validation images, down from 4.1% at 0.3. Lower it further to trade crops per
frame for recall.

Falling back to a full-frame pass on those images was measured and is a wash.
The reason is worth knowing: the ears in images where no face is found are a
median **4.4 px**, three times smaller than average, because BlazeFace misses
a face precisely when the subject is tiny. They are not close-ups that a
full-frame detector would catch. Nothing currently recovers them.

The two graphs do **not** share anchors: the face graph carries MediaPipe's
original `w=h=1.0` squares, the ear graph the fitted ear priors. Each has its
own baked in, so nothing in JavaScript needs to know. Swapping them silently
produces boxes at the wrong scale rather than an error.

```javascript
import { BlazeEarTwoStage } from './blazeear_inference.js';

const detector = new BlazeEarTwoStage({
    confidenceThreshold: 0.70,  // ear stage
    faceThreshold: 0.2,         // face stage; this sets the recall ceiling
    expand: 1.5,                // crop side, in multiples of the face box
});
await detector.load('BlazeFace_web.onnx', 'BlazeEar_web.onnx');

const detections = await detector.detect(videoElement);
console.log(detections.faceCount);  // 0 means nothing could be detected
```

These defaults mirror `FACE_CROP_*` in `utils/config.py`. The Python reference
implementation is `evaluate_two_stage.py`; if you change the crop geometry in
one place, change it in the other.

## Why single-stage is not offered

Running one ear detector over the whole frame is what this demo used to do. It
is no longer exported, because there is no regime where it is the better
choice:

| | median ear at 128 px | single-stage | two-stage |
|---|---|---|---|
| a face is found (96%) | 14.0 px | 0.3694 | **0.6593** |
| no face found (4%) | **4.4 px** | **0.0284** | unreachable |

The second row is the one that settles it. It is tempting to assume the images
the face stage misses are ear close-ups, where a full-frame detector would
shine. They are the opposite: the ears there are a median 4.4 px, three times
*smaller* than average, because BlazeFace misses those faces precisely when the
subject is tiny. Single-stage scores 0.0284 on them, which is noise. It does
not rescue the images two-stage cannot reach -- it fails on them harder.

That is also why falling back to a full-frame pass measured as a wash: it was
not recovering ears, it was adding false positives.

`BlazeEarInference` still exists inside the module, because it is the stage the
pipeline runs twice -- once on the frame for faces, once per crop for ears. It
is an implementation detail, not an entry point, and is not exported.

## Model

Both `BlazeFace_web.onnx` and `BlazeEar_web.onnx` are web-optimized graphs
that:
- output all 896 decoded boxes and their scores, already in original-image
  coordinates
- leave thresholding and NMS to JavaScript, because TopK and NMS emit int64
  and ONNX Runtime Web rejects it
- use ONNX opset 17

## Usage in Your Project

```html
<!-- Include ONNX Runtime Web -->
<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/ort.min.js"></script>

<script type="module">
import { BlazeEarTwoStage } from './blazeear_inference.js';

const detector = new BlazeEarTwoStage({
    confidenceThreshold: 0.70,
    iouThreshold: 0.3
});

await detector.load('BlazeFace_web.onnx', 'BlazeEar_web.onnx');

// Detect from video, canvas, or image
const detections = await detector.detect(videoElement);

// Each detection: { ymin, xmin, ymax, xmax, confidence, x, y, width, height }
// detections.faceCount is 0 when no face was found, and therefore nothing
// could be detected at all.
console.log(detections, detections.faceCount);
</script>
```

## API

### BlazeEarTwoStage

```javascript
const detector = new BlazeEarTwoStage(options);
```

**Options:**
- `confidenceThreshold` (default: 0.70) - minimum confidence, ear stage
- `faceThreshold` (default: 0.2) - minimum confidence, face stage. This sets
  the recall ceiling: lower it to reach more ears at the cost of more crops
  per frame. Swept on the validation split, 0.3 leaves 4.1% of images with no
  face and 0.2 leaves 1.3%, for mAP@0.5 0.6336 -> 0.6447.
- `expand` (default: 1.5) - crop side, in multiples of the face box. Larger
  reaches more ears and gives each one fewer pixels.
- `maxFaces` (default: 8) - crops per frame, at most
- `iouThreshold` (default: 0.3) - NMS IoU threshold

**Methods:**
- `load(facePath, earPath)` - load both graphs
- `detect(source)` - run the pipeline on an image/video/canvas. The returned
  array carries a `faceCount` property.
- `drawDetections(ctx, detections, options)` - draw boxes on a canvas

`BlazeEarTwoStage` and `createTwoStageDetector` are the only exports.

## Regenerating the models

```bash
python export_two_stage_web.py
```

This writes both `docs/BlazeFace_web.onnx` and `docs/BlazeEar_web.onnx`, then
runs each graph against the PyTorch model it came from and refuses to ship a
graph that disagrees. Boxes are compared in pixels (tolerance 0.01 px) because
they come out in original-image coordinates, and the probe input is seeded --
a check that passes or fails depending on the draw is not a check.

`python export_e2e_web.py` still exports a single-stage graph from the
full-frame checkpoint.
