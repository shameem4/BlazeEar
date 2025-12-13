/**
 * BlazeEar Inference Module for JavaScript/Browser
 * 
 * Pure JavaScript implementation of the BlazeEar ear detection pipeline.
 * Uses ONNX Runtime Web for model inference.
 * 
 * Usage:
 *   import { BlazeEarInference } from './blazeear_inference.js';
 *   
 *   const detector = new BlazeEarInference();
 *   await detector.load('path/to/BlazeEar_web.onnx');
 *   
 *   // From video element, canvas, or ImageData
 *   const detections = await detector.detect(imageSource);
 *   // Returns: Array of { ymin, xmin, ymax, xmax, confidence }
 * 
 * ONNX Model Options:
 *   - BlazeEar_web.onnx: Web-optimized (avoids int64, NMS done in JS)
 */

// Check for ONNX Runtime
const ort = (typeof window !== 'undefined' && window.ort) || 
            (typeof globalThis !== 'undefined' && globalThis.ort) ||
            (typeof require !== 'undefined' ? require('onnxruntime-web') : null);

/**
 * BlazeEar Ear Detection Pipeline
 */
class BlazeEarInference {
    /**
     * Create a BlazeEar detector instance
     * @param {Object} options - Configuration options
     * @param {number} options.confidenceThreshold - Minimum confidence for detections (default: 0.75)
     * @param {number} options.iouThreshold - IoU threshold for NMS (default: 0.3)
     */
    constructor(options = {}) {
        this.confidenceThreshold = options.confidenceThreshold ?? 0.75;
        this.iouThreshold = options.iouThreshold ?? 0.3;
        this.inputSize = 128;
        this.session = null;
        this.isLoaded = false;
        this.modelType = null; // 'web' or 'e2e' or 'simple'
    }

    /**
     * Load the ONNX model
     * @param {string} modelPath - Path or URL to the ONNX model file
     * @param {Object} sessionOptions - ONNX Runtime session options
     */
    async load(modelPath, sessionOptions = {}) {
        if (!ort) {
            throw new Error('ONNX Runtime Web not found. Include ort.min.js or install onnxruntime-web');
        }

        const defaultOptions = {
            // Prefer WASM for better int64 support, fallback to WebGL
            executionProviders: ['wasm', 'webgl'],
            graphOptimizationLevel: 'all',
        };

        this.session = await ort.InferenceSession.create(
            modelPath,
            { ...defaultOptions, ...sessionOptions }
        );

        this.isLoaded = true;
        console.log('BlazeEar model loaded successfully');
        console.log('Input names:', this.session.inputNames);
        console.log('Output names:', this.session.outputNames);

        // Detect model type based on outputs
        if (this.session.outputNames.includes('boxes') && this.session.outputNames.includes('scores')) {
            this.modelType = 'web';  // Web-optimized: boxes + scores output
            console.log('Detected web-optimized model (NMS in JavaScript)');
        } else if (this.session.inputNames.includes('scale')) {
            this.modelType = 'e2e';  // End-to-end with denormalization
            console.log('Detected end-to-end model');
        } else {
            this.modelType = 'simple';  // Simple postprocessed model
            console.log('Detected simple model');
        }
    }

    /**
     * Preprocess an image for model input
     * @param {HTMLImageElement|HTMLVideoElement|HTMLCanvasElement|ImageData} source - Image source
     * @returns {Object} Preprocessed data with tensor and preprocessing params
     */
    preprocess(source) {
        // Get image data from various sources
        const { imageData, width, height } = this._getImageData(source);
        
        // Calculate preprocessing parameters
        const maxDim = Math.max(height, width);
        const scale = maxDim / 256.0;
        const newH = Math.round(height / scale);
        const newW = Math.round(width / scale);
        const padH = 256 - newH;
        const padW = 256 - newW;
        const padH1 = Math.floor(padH / 2);
        const padW1 = Math.floor(padW / 2);
        const padY = padH1 * scale;
        const padX = padW1 * scale;

        // Create 256x256 canvas for resize + pad
        const canvas256 = document.createElement('canvas');
        canvas256.width = 256;
        canvas256.height = 256;
        const ctx256 = canvas256.getContext('2d');
        ctx256.fillStyle = 'black';
        ctx256.fillRect(0, 0, 256, 256);

        // Draw resized image centered
        if (source instanceof ImageData) {
            // Create temporary canvas for ImageData
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = width;
            tempCanvas.height = height;
            tempCanvas.getContext('2d').putImageData(source, 0, 0);
            ctx256.drawImage(tempCanvas, padW1, padH1, newW, newH);
        } else {
            ctx256.drawImage(source, padW1, padH1, newW, newH);
        }

        // Create 128x128 canvas for final resize
        const canvas128 = document.createElement('canvas');
        canvas128.width = 128;
        canvas128.height = 128;
        const ctx128 = canvas128.getContext('2d');
        ctx128.drawImage(canvas256, 0, 0, 128, 128);

        // Get pixel data and convert to CHW float32 tensor
        const imgData = ctx128.getContext ? 
            ctx128.getImageData(0, 0, 128, 128) :
            canvas128.getContext('2d').getImageData(0, 0, 128, 128);
        
        const pixels = imgData.data;
        const tensorData = new Float32Array(3 * 128 * 128);

        // Convert RGBA to RGB CHW format (keep in [0, 255] range for the model)
        for (let y = 0; y < 128; y++) {
            for (let x = 0; x < 128; x++) {
                const srcIdx = (y * 128 + x) * 4;
                const r = pixels[srcIdx];
                const g = pixels[srcIdx + 1];
                const b = pixels[srcIdx + 2];

                tensorData[0 * 128 * 128 + y * 128 + x] = r;
                tensorData[1 * 128 * 128 + y * 128 + x] = g;
                tensorData[2 * 128 * 128 + y * 128 + x] = b;
            }
        }

        return {
            tensorData,
            scale,
            padY,
            padX,
            originalWidth: width,
            originalHeight: height
        };
    }

    /**
     * Get ImageData from various image sources
     * @private
     */
    _getImageData(source) {
        let width, height, imageData;

        if (source instanceof ImageData) {
            return { imageData: source, width: source.width, height: source.height };
        }

        if (source instanceof HTMLCanvasElement) {
            width = source.width;
            height = source.height;
            imageData = source.getContext('2d').getImageData(0, 0, width, height);
        } else if (source instanceof HTMLImageElement) {
            width = source.naturalWidth || source.width;
            height = source.naturalHeight || source.height;
            const canvas = document.createElement('canvas');
            canvas.width = width;
            canvas.height = height;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(source, 0, 0);
            imageData = ctx.getImageData(0, 0, width, height);
        } else if (source instanceof HTMLVideoElement) {
            width = source.videoWidth;
            height = source.videoHeight;
            const canvas = document.createElement('canvas');
            canvas.width = width;
            canvas.height = height;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(source, 0, 0);
            imageData = ctx.getImageData(0, 0, width, height);
        } else {
            throw new Error('Unsupported image source type');
        }

        return { imageData, width, height };
    }

    /**
     * Run detection on an image
     * @param {HTMLImageElement|HTMLVideoElement|HTMLCanvasElement|ImageData} source - Image source
     * @returns {Promise<Array>} Array of detection objects with {ymin, xmin, ymax, xmax, confidence}
     */
    async detect(source) {
        if (!this.isLoaded) {
            throw new Error('Model not loaded. Call load() first.');
        }

        // Preprocess
        const { tensorData, scale, padY, padX, originalWidth, originalHeight } = this.preprocess(source);

        // Create input tensor
        const imageTensor = new ort.Tensor('float32', tensorData, [1, 3, 128, 128]);

        // Prepare feeds based on model type
        let feeds;
        if (this.modelType === 'web' || this.modelType === 'e2e') {
            feeds = {
                'image': imageTensor,
                'scale': new ort.Tensor('float32', [scale], []),
                'pad_y': new ort.Tensor('float32', [padY], []),
                'pad_x': new ort.Tensor('float32', [padX], [])
            };
        } else {
            feeds = { 'image': imageTensor };
        }

        // Run inference
        const results = await this.session.run(feeds);

        let detections;
        if (this.modelType === 'web') {
            // Web model outputs boxes (896, 4) and scores (896,)
            // We need to filter and apply NMS in JavaScript
            detections = this._processWebModelOutput(
                results.boxes.data,
                results.scores.data,
                originalWidth,
                originalHeight
            );
        } else {
            // E2E or simple model outputs detections directly
            detections = this._processDetectionsOutput(
                results.detections.data,
                results.detections.dims[0],
                scale,
                padY,
                padX,
                originalWidth,
                originalHeight
            );
        }

        return detections;
    }

    /**
     * Process web model output (boxes + scores) with JS NMS
     * @private
     */
    _processWebModelOutput(boxesData, scoresData, originalWidth, originalHeight) {
        const numAnchors = 896;
        const candidates = [];

        // Filter by confidence threshold
        for (let i = 0; i < numAnchors; i++) {
            const score = scoresData[i];
            if (score >= this.confidenceThreshold) {
                candidates.push({
                    ymin: boxesData[i * 4 + 0],
                    xmin: boxesData[i * 4 + 1],
                    ymax: boxesData[i * 4 + 2],
                    xmax: boxesData[i * 4 + 3],
                    confidence: score,
                    index: i
                });
            }
        }

        // Sort by confidence (descending)
        candidates.sort((a, b) => b.confidence - a.confidence);

        // Apply NMS
        const detections = this._nms(candidates, this.iouThreshold);

        // Clamp to image bounds and add convenience properties
        for (const det of detections) {
            det.ymin = Math.max(0, Math.min(det.ymin, originalHeight));
            det.xmin = Math.max(0, Math.min(det.xmin, originalWidth));
            det.ymax = Math.max(0, Math.min(det.ymax, originalHeight));
            det.xmax = Math.max(0, Math.min(det.xmax, originalWidth));
            det.x = det.xmin;
            det.y = det.ymin;
            det.width = det.xmax - det.xmin;
            det.height = det.ymax - det.ymin;
            delete det.index;  // Remove internal property
        }

        return detections;
    }

    /**
     * Process detections output from e2e or simple model
     * @private
     */
    _processDetectionsOutput(detectionsData, numDetections, scale, padY, padX, originalWidth, originalHeight) {
        const detections = [];
        const isE2E = this.modelType === 'e2e';

        for (let i = 0; i < numDetections; i++) {
            let ymin = detectionsData[i * 5 + 0];
            let xmin = detectionsData[i * 5 + 1];
            let ymax = detectionsData[i * 5 + 2];
            let xmax = detectionsData[i * 5 + 3];
            const confidence = detectionsData[i * 5 + 4];

            // If not end-to-end, denormalize coordinates here
            if (!isE2E) {
                ymin = ymin * scale * 256 - padY;
                xmin = xmin * scale * 256 - padX;
                ymax = ymax * scale * 256 - padY;
                xmax = xmax * scale * 256 - padX;
            }

            // Clamp to image bounds
            ymin = Math.max(0, Math.min(ymin, originalHeight));
            xmin = Math.max(0, Math.min(xmin, originalWidth));
            ymax = Math.max(0, Math.min(ymax, originalHeight));
            xmax = Math.max(0, Math.min(xmax, originalWidth));

            detections.push({
                ymin,
                xmin,
                ymax,
                xmax,
                confidence,
                x: xmin,
                y: ymin,
                width: xmax - xmin,
                height: ymax - ymin
            });
        }

        return detections;
    }

    /**
     * Non-Maximum Suppression
     * @private
     */
    _nms(candidates, iouThreshold) {
        const selected = [];
        const suppressed = new Set();

        for (let i = 0; i < candidates.length; i++) {
            if (suppressed.has(i)) continue;

            const current = candidates[i];
            selected.push(current);

            for (let j = i + 1; j < candidates.length; j++) {
                if (suppressed.has(j)) continue;

                const other = candidates[j];
                const iou = this._computeIoU(current, other);

                if (iou > iouThreshold) {
                    suppressed.add(j);
                }
            }
        }

        return selected;
    }

    /**
     * Compute Intersection over Union
     * @private
     */
    _computeIoU(boxA, boxB) {
        const xA = Math.max(boxA.xmin, boxB.xmin);
        const yA = Math.max(boxA.ymin, boxB.ymin);
        const xB = Math.min(boxA.xmax, boxB.xmax);
        const yB = Math.min(boxA.ymax, boxB.ymax);

        const interWidth = Math.max(0, xB - xA);
        const interHeight = Math.max(0, yB - yA);
        const interArea = interWidth * interHeight;

        const areaA = (boxA.xmax - boxA.xmin) * (boxA.ymax - boxA.ymin);
        const areaB = (boxB.xmax - boxB.xmin) * (boxB.ymax - boxB.ymin);

        const unionArea = areaA + areaB - interArea;

        return unionArea > 0 ? interArea / unionArea : 0;
    }

    /**
     * Draw detections on a canvas
     * @param {CanvasRenderingContext2D} ctx - Canvas context to draw on
     * @param {Array} detections - Array of detection objects
     * @param {Object} options - Drawing options
     */
    drawDetections(ctx, detections, options = {}) {
        const color = options.color || '#00FF00';
        const lineWidth = options.lineWidth || 2;
        const showConfidence = options.showConfidence ?? true;
        const fontSize = options.fontSize || 14;

        ctx.strokeStyle = color;
        ctx.lineWidth = lineWidth;
        ctx.font = `${fontSize}px Arial`;
        ctx.fillStyle = color;

        for (const det of detections) {
            // Draw bounding box
            ctx.strokeRect(det.xmin, det.ymin, det.width, det.height);

            // Draw confidence label
            if (showConfidence) {
                const label = `${(det.confidence * 100).toFixed(1)}%`;
                const labelWidth = ctx.measureText(label).width;
                
                // Background for label
                ctx.fillStyle = color;
                ctx.fillRect(det.xmin, det.ymin - fontSize - 4, labelWidth + 6, fontSize + 4);
                
                // Label text
                ctx.fillStyle = '#000000';
                ctx.fillText(label, det.xmin + 3, det.ymin - 4);
                ctx.fillStyle = color;
            }
        }
    }

    /**
     * Dispose of the model and free resources
     */
    async dispose() {
        if (this.session) {
            // ONNX Runtime Web doesn't have explicit dispose, but we can null the reference
            this.session = null;
            this.isLoaded = false;
        }
    }
}

/**
 * Convenience function to create and load a detector
 * @param {string} modelPath - Path to ONNX model
 * @param {Object} options - Detector options
 * @returns {Promise<BlazeEarInference>} Loaded detector instance
 */
async function createDetector(modelPath, options = {}) {
    const detector = new BlazeEarInference(options);
    await detector.load(modelPath);
    return detector;
}

// Export for different module systems
if (typeof module !== 'undefined' && module.exports) {
    // CommonJS
    module.exports = { BlazeEarInference, createDetector };
} else if (typeof window !== 'undefined') {
    // Browser global
    window.BlazeEarInference = BlazeEarInference;
    window.createBlazeEarDetector = createDetector;
}

export { BlazeEarInference, createDetector };
