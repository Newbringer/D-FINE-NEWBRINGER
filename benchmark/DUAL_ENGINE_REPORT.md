# D-FINE plus RTMO dual-engine research verdict

## Accuracy

| Pose path | Full COCO OKS AP |
|---|---:|
| Current merged pose | 48.20 |
| RTMO official ONNX | 64.75 |
| RTMO TensorRT FP16 | 64.70 |
| RTMO TensorRT FP32 | 64.75 |

The first apparent FP16 score of 61.21 was an evaluator defect: the official RTMO wrapper applies a
second NMS after the end-to-end graph. Applying the identical postprocess restores FP16 to within
0.05 AP of ONNX. FP16 is accepted.

## Runtime budget on RTX 5070 Ti

| Engine | p50 model time | Execution memory |
|---|---:|---:|
| Existing D-FINE ModelSurgery TensorRT | 13.983 ms | 206.28 MiB |
| RTMO-L FP16 TensorRT | 2.224 ms | 780.49 MiB |
| Sequential dual-engine budget | **16.207 ms** | **986.77 MiB** |

This is model execution, not complete camera-to-result latency. The current D-FINE engine still
contains its old pose branch, so a future det+seg-only export could reduce duplicated work.

## Association design

RTMO is one-stage and emits person detections together with their keypoints. Its pose does not need
heuristic assignment to D-FINE person queries. In a research composition, RTMO owns person pose
entities while D-FINE retains general-object detection and the seven-class dense body mask. The
existing office comparison videos validate both paths on identical frames; production message
composition remains out of scope.

## Decision

The dual-engine design is technically viable and is the recommended next architecture candidate:
retain D-FINE detection/segmentation and add RTMO-L FP16 for pose. It gains 16.50 pose AP over the
merged pose head for a measured 2.224 ms model-time increment and ~780 MiB execution memory.

No production code or model was changed.
