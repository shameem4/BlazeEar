"""Export ONNX model for web deployment with int32 compatibility."""

from blazeear_inference import BlazeEarInference

# Load pipeline
pipeline = BlazeEarInference(
    weights_path='runs/checkpoints/BlazeEar_best.pth',
    confidence_threshold=0.75,
    device='cuda'
)

# Export web-optimized ONNX (avoids int64 ops)
pipeline.to_onnx('js_demo/BlazeEar_web.onnx', for_web=True)

print('Done!')
print()
print('The web model outputs decoded boxes (896, 4) and scores (896,)')
print('JavaScript must filter by confidence and apply NMS.')
