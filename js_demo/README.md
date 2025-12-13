# BlazeEar JavaScript Demo

Browser-based ear detection using the BlazeEar ONNX model with ONNX Runtime Web.

## Quick Start

1. **Export the ONNX model** (if not already done):
   ```bash
   python blazeear_inference.py --weights runs/checkpoints/BlazeEar_best.pth --export-onnx-e2e runs/checkpoints/BlazeEar_e2e.onnx
   ```

2. **Serve the files** (required for ES modules and model loading):
   ```bash
   # From the BlazeEar root directory
   python -m http.server 8000
   ```

3. **Open in browser**:
   ```
   http://localhost:8000/js_demo/
   ```

## Files

- `blazeear_inference.js` - Main inference module (ES6 module)
- `index.html` - Demo page with webcam and image detection

## Usage

### ES6 Module

```javascript
import { BlazeEarInference, createDetector } from './blazeear_inference.js';

// Option 1: Manual initialization
const detector = new BlazeEarInference({
    confidenceThreshold: 0.75,
    iouThreshold: 0.3
});
await detector.load('path/to/BlazeEar_e2e.onnx');

// Option 2: Convenience function
const detector = await createDetector('path/to/BlazeEar_e2e.onnx', {
    confidenceThreshold: 0.75
});

// Detect from various sources
const detections = await detector.detect(imageElement);
const detections = await detector.detect(videoElement);
const detections = await detector.detect(canvasElement);

// Each detection has:
// { ymin, xmin, ymax, xmax, confidence, x, y, width, height }

// Draw on canvas
detector.drawDetections(ctx, detections, {
    color: '#00FF00',
    lineWidth: 2,
    showConfidence: true
});
```

### Script Tag (non-module)

```html
<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.16.0/dist/ort.min.js"></script>
<script src="blazeear_inference.js"></script>
<script>
    async function main() {
        const detector = await window.createBlazeEarDetector('model.onnx');
        const detections = await detector.detect(document.querySelector('img'));
        console.log(detections);
    }
    main();
</script>
```

## ONNX Model Types

### End-to-End Model (`BlazeEar_e2e.onnx`) - Recommended

Returns bounding boxes in **original image coordinates**. The model takes preprocessing parameters as inputs.

**Inputs:**
- `image`: (1, 3, 128, 128) float32 in [0, 255]
- `scale`: scalar float32
- `pad_y`: scalar float32
- `pad_x`: scalar float32

**Output:**
- `detections`: (N, 5) [ymin, xmin, ymax, xmax, confidence]

### Simple Model (`BlazeEar_inference.onnx`)

Returns bounding boxes in **normalized [0, 1] coordinates**.

**Input:**
- `image`: (1, 3, 128, 128) float32 in [0, 255]

**Output:**
- `detections`: (N, 5) [ymin, xmin, ymax, xmax, confidence]

The JavaScript module automatically detects which model type is loaded and handles denormalization accordingly.

## Preprocessing

The model expects images preprocessed as follows:

1. Scale the larger dimension to 256 (preserving aspect ratio)
2. Center-pad to 256x256 with black pixels
3. Resize to 128x128
4. Keep pixel values in [0, 255] range (model normalizes internally)

The `BlazeEarInference` class handles all preprocessing automatically.

## Browser Compatibility

Requires a modern browser with:
- WebGL 2.0 (for GPU acceleration)
- WebAssembly (fallback)
- ES6 modules (for import/export syntax)

Tested on:
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Performance Tips

1. **Use WebGL backend** (default) for best performance
2. **Lower resolution** images process faster
3. **Reuse detector instance** instead of creating new ones
4. For video, use `requestAnimationFrame` for smooth detection loop
