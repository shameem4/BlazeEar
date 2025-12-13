"""Export ONNX model for web deployment with int32 compatibility."""

from blazeear_inference import BlazeEarInference

# Load pipeline
pipeline = BlazeEarInference(
    weights_path='runs/checkpoints/BlazeEar_best.pth',
    confidence_threshold=0.75,
    device='cuda'
)

# Export end-to-end ONNX
pipeline.to_onnx('runs/checkpoints/BlazeEar_e2e.onnx', end_to_end=True)

# Also export to js_demo
pipeline.to_onnx('js_demo/BlazeEar_e2e.onnx', end_to_end=True)

print('Done!')
