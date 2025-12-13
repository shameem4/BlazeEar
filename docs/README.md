# BlazeEar JavaScript Demo

Browser-based ear detection using ONNX Runtime Web.

## Quick Start

1. **Start a local server** (required for module loading):
   ```bash
   cd BlazeEar
   python -m http.server 8000
   ```

2. **Open in browser**: http://localhost:8000/js_demo/

3. **Use the demo**:
   - Click "Start Webcam" for live detection
   - Or upload an image for single-frame detection

## Model

The demo uses `BlazeEar_web.onnx` - a web-optimized model that:
- Outputs decoded boxes (896, 4) and scores (896,)
- NMS is performed in JavaScript for browser compatibility
- Uses ONNX opset 14 for maximum compatibility

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

## Regenerating the Model

```bash
python export_e2e_web.py
```

This creates `js_demo/BlazeEar_web.onnx` from the trained weights.
