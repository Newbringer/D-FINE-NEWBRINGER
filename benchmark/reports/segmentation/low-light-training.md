# Low-light segmentation training v2

## Protocol

Only the existing segmentation head was trained. Backbone, encoder, detection and pose stayed
frozen and hash-identical. The corrected protocol uses:

- 50% normal images plus exposure, gamma, sensor noise, LED colour, shadow, blur/compression and
  overexposure augmentation;
- moderate hand/foot class weighting;
- consistency distillation from the frozen original segmentation head;
- fixed normal, 3x-dark, 10x-dark, noisy-low-light and overexposure validation;
- three seeds, cosine learning rate, early stopping and an absolute normal-light regression gate;
- validation in `eval()` mode so BatchNorm state cannot leak from validation into training.

## Three-seed full-validation result

| Run | Best epoch | Normal | 3x dark | 10x dark | Low-light noise |
|---|---:|---:|---:|---:|---:|
| Current baseline | — | **73.61%** | 67.05% | 49.59% | 42.28% |
| Seed 20260916 | 6 | 73.33% | **67.39%** | **50.96%** | 43.55% |
| Seed 20260917 | 5 | 73.34% | 67.38% | 50.83% | 43.52% |
| Seed 20260918 | 10 | 73.25% | 67.42% | 50.87% | **43.56%** |

The improvement reproduces across all seeds. Normal mIoU changes by -0.27 to -0.36 percentage
points, inside the predefined -0.5 gate. Extreme-dark mIoU improves by +1.23 to +1.36 points and
noisy-low-light mIoU by +1.24 to +1.27 points.

For selected seed 20260916, 10x-dark feet improve from 23.63% to 28.35%. Hands improve from 24.64%
to 25.20%, but the hand gain is not consistent across all seeds.

## Curriculum result

The best balanced candidate entered two additional phases:

1. moderate-light curriculum, maximum 12 epochs;
2. strong low-light/noise curriculum with targeted hand/foot crops, maximum 5 epochs.

Both phases selected **epoch 0**. Every trained curriculum epoch reduced the combined robustness
score relative to its input, so early stopping retained the balanced seed unchanged. More epochs or
stronger synthetic augmentation do not improve this dataset/model combination.

Selected checkpoint SHA-256:
`8875f4072a4f476d4beb178b33a3a3ff4448645bbf25a808222de3724bba10bb`.

Complete run summary SHA-256:
`acd5be7575d097b50058f60e1a756d35cf6cf300a67c11411196191307ed177f`.

## Decision

Retain seed 20260916 epoch 6 as the best research candidate. Do not promote it to production yet:
the gain is modest and must be validated on real TagTwo camera frames. Further improvement should
come from real camera noise/exposure data or higher-resolution small-part features, not additional
synthetic epochs.
