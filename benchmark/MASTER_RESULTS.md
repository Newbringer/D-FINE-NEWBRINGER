# Consolidated model research results

All local metrics use pinned COCO/Pascal manifests. Paper claims are excluded from the measured
tables below.

## Detection

| Model | COCO AP | Change vs merged | Decision |
|---|---:|---:|---|
| Stock D-FINE-X obj2coco | **59.31** | +15.71 | strongest detection teacher |
| RF-DETR-Large 2026 | **56.54** | +12.94 | monitor; below stock D-FINE |
| Encoder-retention alpha 0.75 | 45.70 | +2.10 | reject; pose/domain recall regressed |
| Current merged ModelSurgery | 43.60 | baseline | retain for current combined graph |

RF-DETR raw predictions SHA-256:
`47690dcfddc20370618b04f8641da93a2992afafdde42f3dbb57b06f09568a93`.
Checkpoint SHA-256:
`0f4e20e19a99c0f8a62b5685f57f6c8b5c371c59081feda6752a0561a79ccf38`.

## Pose

| Model | COCO OKS AP | Change vs merged | Decision |
|---|---:|---:|---|
| Official DETRPose-X standalone | **74.41** | +26.21 | best teacher/control |
| RTMO-L Body7 official ONNX | **64.75** | +16.55 | strongest independent deployable candidate |
| Current merged ModelSurgery | 48.20 | baseline | retain until integration candidate passes |
| Hungarian-matched adapter | 43.60 screen | -3.50 screen | reject |
| Direct-query adapter | 43.40 screen | -3.70 screen | reject |
| Adapter + pose decoder | 41.60 screen | -5.50 screen | reject |
| Direct DETRPose decoder transplant | 0.51 | -47.69 | reject |

RTMO raw predictions SHA-256:
`1fbcc2b2cd9d459ac24d1087864e710be4694eb7a201bb0b1858cf6bb53bf0db`.
Official ONNX SHA-256:
`090096ca90f29163cc4f67137dcc0cd4b2ee95ea0af11764fbfda88dd2ae1140`.

RTMO was evaluated through the official ONNX graph using the documented 640 letterbox and NMS.
The available ONNX Runtime build was CPU-only, so this run establishes quality, not comparable GPU
latency. Its published 74.8 Body7 result is not substituted for the locally measured 64.75.

## Seven-class body-part segmentation

| Model | Project-semantic mIoU | Decision |
|---|---:|---|
| Current FPN-ASPP head | **73.61** | retain |
| SCHP Pascal-7 | **49.07** | reject |

SCHP's raw same-ID score was 51.60%, but IDs 3..6 mean upper/lower limbs rather than the project's
arms/hands/legs/feet. The semantically valid remap cannot predict hands or feet and scores 0% IoU
for both. Current weak points are feet (56.06%) and hands (64.18%).

## Overall answer

- **Detection:** no tested replacement beats stock D-FINE-X. The merged model's lower AP comes from
  shared-training history, not lack of a newer detector name.
- **Pose:** RTMO-L is the first independent model to produce a large local gain. It should be tested
  on office videos and GPU/TensorRT before integration is considered.
- **Segmentation:** the current head is verified strong; SCHP is materially worse for the actual
  project label contract.
- **Combined model:** no single-pass replacement has won. Keep the current graph while testing RTMO
  as a separate pose path or future teacher.

## Next decisive work

1. Export/benchmark RTMO-L on GPU/TensorRT and render office-video comparisons.
2. Measure RF-DETR person recall and false positives on the soldier-domain validation set.
3. Evaluate whether separate RTMO latency is acceptable versus another shared-head training cycle.
4. Do not replace segmentation; focus any future work specifically on hands and feet.

## Product-control update

- RTMO-L visibly runs on both office videos and retains its quality lead, but CPU ONNX latency is
  124–177 ms mean; GPU/TensorRT proof is still required.
- RF-DETR lowers soldier-domain false positives from 0.41 to 0.17/image and improves matched IoU,
  but recall falls from 86.3% to 84.2%. It is a precision-biased alternative, not a clear winner.
