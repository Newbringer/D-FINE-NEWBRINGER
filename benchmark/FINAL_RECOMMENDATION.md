# Final research recommendation

## Decision

Keep the current combined model in production while promoting the RTMO-taught, single-graph pose
adapter through export and product-control gates. A separate RTMO runtime is no longer the preferred
research direction.

| Area | Current | Best tested alternative | Recommendation |
|---|---:|---:|---|
| Detection | 43.60 AP merged; 59.31 stock | RF-DETR 56.54 AP | Keep D-FINE |
| Pose | 48.20 OKS AP | RTMO-taught adapter 51.43 mean AP | Advance single-graph adapter |
| Segmentation | 73.61% mIoU | SCHP 49.07% mIoU | Keep in-house head |
| RTMO deployment | — | 2.224 ms p50 FP16 TensorRT | Technically viable |

RTMO-L remains the strongest standalone control, but the RTMO-taught adapter now reproduces a
single-model gain across three seeds (51.1/51.7/51.5 AP versus 48.2). It changes only the pose feature
path and requires no second RTMO engine at inference. Export parity, latency and office-video review
must pass before it can replace the current checkpoint.

RF-DETR reduces soldier-domain false positives but also reduces recall, while stock D-FINE remains
more accurate on COCO. SCHP is incompatible with the required hands/feet semantics and substantially
less accurate. The earlier DETRPose adapter approaches regressed pose; RTMO pseudo-label training is
the first adapter approach to improve it reproducibly.

No files in `tagtwo-monorepo` were modified.
