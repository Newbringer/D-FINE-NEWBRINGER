# RTMO association diagnosis

| David frame classification | Count | Share |
|---|---:|---:|
| Fully matched | 482 | 85.6% |
| Geometry rejected | 14 | 2.5% |
| RTMO pose missing | 27 | 4.8% |
| D-FINE detection missing | 17 | 3.0% |
| Both missing | 23 | 4.1% |

The observed average match rate of 0.87 is not primarily a matching-threshold problem. Perfectly
recovering all geometry rejections would still leave the long chair-occlusion interval dominated
by absent model outputs. Loosening center/IoU constraints is therefore rejected because it adds
multi-person swap risk for little possible gain.

Recommended next research, if required, is an occlusion-specific temporal test: compare existing
Kalman-predicted tracks with lower-confidence RTMO/D-FINE candidates and measure ID switches against
manual sequence annotations. This is separate from choosing RTMO as the pose model.
