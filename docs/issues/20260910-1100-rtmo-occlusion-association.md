# Diagnose RTMO association under heavy occlusion

- Status: Done
- Repository scope: research checkout only

## Goal

Determine whether David's 0.87 match rate is caused by association geometry or missing model outputs.

## Result

Per-frame instrumentation over all 563 frames found:

- 482 fully matched frames
- 14 geometry-rejection frames
- 27 pose-missing frames
- 17 detection-missing frames
- 23 frames where both were missing

Only 2.5% of frames are addressable by geometry matching. The heavy-occlusion gap is primarily
model availability and must be handled through temporal tracker prediction or deliberately lower
acquisition thresholds, not looser matching that risks cross-person swaps.

## Validation

- Every frame records detection count, pose count, pairs, method, boxes, scores and match rate.
- Existing tracker/smoothing behavior remained unchanged.
- No production files were modified.
