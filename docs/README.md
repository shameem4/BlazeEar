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

## The two-stage pipeline (default)

The demo now runs the pipeline MediaPipe itself uses: a coarse detector on the
full frame, then a fine model on a crop around each hit.

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
| **two-stage** | **0.5760** | **0.2444** |

Untick "Two-stage" in the demo to run the ear model over the whole frame, the
way this demo used to.

**The catch, stated plainly:** an ear whose face BlazeFace misses never reaches
the second stage. That is about 4% of images at the default face threshold, and
it is a hard ceiling, not a tuning issue. Lower `faceThreshold` to trade crops
per frame for recall. Falling back to a full-frame pass on those images was
measured and is a wash -- it contributes about as many false positives as it
recovers ears.

The two graphs do **not** share anchors: the face graph carries MediaPipe's
original `w=h=1.0` squares, the ear graph the fitted ear priors. Each has its
own baked in, so nothing in JavaScript needs to know. Swapping them silently
produces boxes at the wrong scale rather than an error.

```javascript
import { BlazeEarTwoStage } from './blazeear_inference.js';

const detector = new BlazeEarTwoStage({
    confidenceThreshold: 0.70,  // ear stage
    faceThreshold: 0.3,         // face stage; this sets the recall ceiling
    expand: 1.5,                // crop side, in multiples of the face box
});
await detector.load('BlazeFace_web.onnx', 'BlazeEar_web.onnx');

const detections = await detector.detect(videoElement);
console.log(detections.faceCount);  // 0 means nothing could be detected
```

These defaults mirror `FACE_CROP_*` in `utils/config.py`. The Python reference
implementation is `evaluate_two_stage.py`; if you change the crop geometry in
one place, change it in the other.

## Model

The single-stage `BlazeEar_web.onnx` is still there, and `BlazeFace_web.onnx`
joins it. Both are web-optimized graphs that:
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
import { BlazeEarInference } from './blazeear_inference.js';

const detector = new BlazeEarInference({
    confidenceThreshold: 0.75,
    iouThreshold: 0.3
});

await detector.load('BlazeEar_web.onnx');

// Detect from video, canvas, or image
const detections = await detector.detect(videoElement);

// Each detection: { ymin, xmin, ymax, xmax, confidence, x, y, width, height }
console.log(detections);
</script>
```

## API

### BlazeEarInference

```javascript
const detector = new BlazeEarInference(options);
```

**Options:**
- `confidenceThreshold` (default: 0.75) - Minimum detection confidence
- `iouThreshold` (default: 0.3) - NMS IoU threshold

**Methods:**
- `load(modelPath)` - Load ONNX model
- `detect(source)` - Run detection on image/video/canvas
- `drawDetections(ctx, detections, options)` - Draw boxes on canvas

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
