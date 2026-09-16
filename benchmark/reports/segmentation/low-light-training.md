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

## Forced 30-epoch control

To test whether early stopping was premature, a separate progressive run completed all 30 epochs
without stopping and saved every checkpoint. Augmentations were visually audited at five severity
levels before training. Epoch 29 won on the fixed screen, then underwent full validation:

| Model | Normal | 3x dark | 10x dark | Low-light noise | Robust mean |
|---|---:|---:|---:|---:|---:|
| Balanced seed 20260916 | 73.33% | 67.39% | **50.96%** | **43.55%** | **58.30%** |
| Progressive epoch 29 | **73.55%** | **67.47%** | 50.73% | 43.34% | 58.25% |

The longer run recovers nearly all normal-light performance and slightly improves moderate darkness,
but is worse on the two primary low-light conditions. It also reduces 10x-dark hand IoU from 25.20%
to 23.81%. The hypothesis that more epochs produce a better low-light model is therefore rejected
on full validation; the screen-only improvement did not generalize.

Progressive checkpoint SHA-256:
`baa420bce9b4e3cb51898ab161882138def417a949a3c2a02ab4e6e3d40731a9`.

Selected checkpoint SHA-256:
`8875f4072a4f476d4beb178b33a3a3ff4448645bbf25a808222de3724bba10bb`.

Complete run summary SHA-256:
`acd5be7575d097b50058f60e1a756d35cf6cf300a67c11411196191307ed177f`.

## Decision

Retain seed 20260916 epoch 6 as the best research candidate. Do not promote it to production yet:
the gain is modest and must be validated on real TagTwo camera frames. Further improvement should
come from real camera noise/exposure data or higher-resolution small-part features, not additional
synthetic epochs.

## Video controls

The selected balanced head was rendered against the current head on identical David and SogO
frames in normal light and with camera-style low-light simulation. The simulation uses 0.1x linear
scene illumination, partial exposure/gain recovery, shot/read noise, warm colour and motion blur;
the scene remains visible. Detection and pose were unchanged.

| Video | Current hits | Candidate hits |
|---|---:|---:|
| David, normal | 259 | 276 |
| David, camera low light | 263 | 279 |
| SogO, normal | 315 | 316 |
| SogO, camera low light | 314 | 318 |

These counts show changed decisions, not improved accuracy; the office videos have no hit/body-part
ground truth. Visual files are under `ZKeepResults/segmentation_lowlight_v2/comparisons/`. The
earlier uniform 0.1x sRGB visualization is deprecated because it produced an unrealistic black
background and is excluded from the visual conclusion.
