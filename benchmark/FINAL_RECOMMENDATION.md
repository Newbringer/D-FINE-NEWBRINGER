# Final research recommendation

## Decision

Keep the current combined model and pursue RTMO-L as a separate pose upgrade candidate.

| Area | Current | Best tested alternative | Recommendation |
|---|---:|---:|---|
| Detection | 43.60 AP merged; 59.31 stock | RF-DETR 56.54 AP | Keep D-FINE |
| Pose | 48.20 OKS AP | RTMO-L 64.75 OKS AP | Advance RTMO-L |
| Segmentation | 73.61% mIoU | SCHP 49.07% mIoU | Keep in-house head |
| RTMO deployment | — | 2.224 ms p50 FP16 TensorRT | Technically viable |

RTMO-L is the only tested new model with both a material local quality gain and viable GPU engine
time. It adds approximately 780 MiB of TensorRT execution memory and has not been integrated into
the production pipeline. The research result therefore supports an integration proposal, not an
automatic model replacement.

RF-DETR reduces soldier-domain false positives but also reduces recall, while stock D-FINE remains
more accurate on COCO. SCHP is incompatible with the required hands/feet semantics and substantially
less accurate. All tested DETRPose adapter and distillation approaches regressed pose.

No files in `tagtwo-monorepo` were modified.
