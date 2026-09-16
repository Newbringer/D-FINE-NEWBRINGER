# Fine-tune the ModelSurgery segmentation head for low light

- Status: Done (research candidate)
- Repository scope: research checkout only

## Goal

Improve extreme-low-light segmentation using dark/noisy/colour-cast Pascal Person Parts
augmentation and stronger hand/foot losses without changing backbone, detection or pose.

## Result

Two independent one-epoch runs were blended conservatively with the original segmentation weights.
The resulting single-head candidate improves 10x-dark validation mIoU from 49.59% to 50.63%, while
normal mIoU changes from 73.61% to 73.36%. Dark feet improve from 23.63% to 27.26%. Dark hands are
effectively unchanged across the stable blend; the improvement seen in one seed did not reproduce.

Protected modules remained hash-identical. No production or TagTwo files were modified.

## Decision

Retain as a research candidate, not a production replacement. Validate on real arena captures next.

## Corrected v2 follow-up

The original experiment was superseded by a three-seed protocol with frozen-teacher consistency,
realistic augmentation, fixed multi-condition validation and validation-safe BatchNorm handling.
All seeds improve 10x-dark mIoU from 49.59% to 50.83–50.96% while keeping normal mIoU within the
0.5-point gate. Additional moderate and strong curriculum phases both select epoch 0 and add no
gain. Seed 20260916 epoch 6 is the retained research candidate.

A forced 30-epoch progressive control saved and evaluated every epoch. Epoch 29 won on the screen
but scored 58.25% robust mean on full validation versus 58.30% for the selected balanced model, and
was worse on 10x darkness and noisy low light. More epochs are therefore not the missing factor.
