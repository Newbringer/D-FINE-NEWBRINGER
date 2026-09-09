# Evaluate RTMO and RF-DETR on product controls

- Status: Done
- Repository scope: research checkout only

## Goal

Measure RTMO-L on office videos and RF-DETR-Large on soldier-domain validation before the final recommendation.

## Acceptance criteria

- [x] RTMO renders David and SogO with measured local latency.
- [x] RF-DETR reports soldier-domain recall, false positives and matched IoU.
- [x] All artifacts and model revisions are hash-pinned.
- [x] Consolidated recommendation is updated from local results.

## Next step

Pursue RTMO GPU/TensorRT export as the only remaining deployment gate. Retain D-FINE detection and
the existing segmentation head.
