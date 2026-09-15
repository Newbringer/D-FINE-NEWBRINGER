# RTMO-taught pose adapter export gate

## Outcome

The seed-456 adapter can be inserted into the existing single ModelSurgery graph and exported to a
syntactically valid five-output ONNX model. The export is **not semantically valid for pose** and
must not be promoted.

| Gate | Result |
|---|---|
| ONNX checker | Pass |
| Detection box CPU PyTorch/ONNX mean absolute difference | 0.00000054 |
| Segmentation maximum absolute logit difference | 0.000038 |
| Segmentation pixel agreement | 99.9985% |
| Pose keypoint raw mean absolute difference | 0.1127 normalized units — fail |
| TensorRT FP32 full COCO pose | 1.11 AP — fail versus 51.7 PyTorch |
| TensorRT FP16 full COCO pose | NaN outputs — fail |

The FP16 candidate and identically built baseline both measured 5.619 ms median model execution
with CUDA Graph and transfers disabled. This proves the adapter has no measurable graph execution
overhead under that build, but **the latency is not a deployable result** because pose quality
failed first.

The defect is localized to the DETRPose export path: detection and segmentation preserve numerical
parity while pose does not. Legacy TorchScript and new Dynamo ONNX export both fail, and applying
the existing deploy conversions or replacing the feature `Split` with static slices did not repair
pose parity. Those unsuccessful code changes were removed.

## Next step

Implement an export-specific, numerically tested DETRPose deformable-attention path. Require
PyTorch/ONNX task parity on fixed real frames before building another TensorRT engine, then require
full COCO AP parity before accepting latency.

No TagTwo or production files were modified.

## Resolution

This initial failure was resolved for ONNX by making DETRPose eval TopK deterministic across
runtimes. The corrected ONNX scores 51.83 AP versus 51.70 PyTorch on full COCO and is documented in
`ONNX_DEPLOYMENT.md`. TensorRT remains rejected because its recursive decoder numerics still fail;
ONNX Runtime CUDA is the qualified research path.
