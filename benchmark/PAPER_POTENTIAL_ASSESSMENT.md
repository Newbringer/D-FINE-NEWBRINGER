# Paper-potential assessment for ModelSurgery

## Verdict

**Current potential: medium for an applied systems/workshop paper; low for a top-tier methods paper.**

The work is already substantial and reproducible, but the present architecture combines known
components (D-FINE, DETRPose and an FPN/ASPP segmentation head). Combining detection, pose and
segmentation is not by itself novel: HumanQueryNet/HQNet, UniHCP, MDSP and older integrated human
sensing work cover overlapping task sets. A paper must therefore avoid a “first multitask model”
claim.

## Strongest defensible research angle

> A resource-aware single-graph human perception system that shares one backbone across person
> detection, 17-point pose and seven-class anatomical parsing, and turns these outputs into
> occlusion-aware body-part hit decisions.

The distinctive part is not any one head. It is the combination of:

- simultaneous anatomical outputs (`head`, `torso`, `arms`, `hands`, `legs`, `feet`), not only a
  binary person instance mask;
- one shared graph intended to replace separate specialist inference;
- measured negative transfer introduced by pose integration;
- offline specialist teaching that improves the single runtime graph without retaining the teacher;
- a downstream decision problem where per-part correctness and occlusion matter more than generic
  COCO AP alone.

## Evidence already available

| Evidence | Current result | Paper value |
|---|---:|---|
| Detection | 43.6 merged COCO AP | documents current multitask cost |
| Pose baseline | 48.2 OKS AP | reproducible starting point |
| RTMO-taught pose, three seeds | 51.1 / 51.7 / 51.5 | reproducible +3.23 mean AP gain |
| Qualified ONNX pose | 51.83 AP | deployable single-file result |
| Segmentation | 73.61% Pascal Person Parts mIoU | strong seven-class task result |
| ONNX CUDA latency | 24.08 ms current / 24.81 ms taught | +0.73 ms for pose improvement |
| Lighting stress test | 75.2% consistency at 10x darker | identifies product robustness limit |
| Specialist controls | RTMO, DETRPose, RF-DETR, SCHP, DEIMv2 | prevents isolated claims |
| HumanQueryNet | visual control completed | closest architecture baseline identified |

These results are enough for a credible technical report. They are not enough for a paper claim
because most metrics come from separate task datasets and do not yet prove the value of sharing.

## Missing evidence that decides publishability

### Required

1. **Shared benchmark against HumanQueryNet.** Measure detection, pose and person-mask outputs on
   COCO-UniHuman under the same resolution and hardware. Separately state that HQNet has no
   seven-class anatomical output.
2. **Shared-versus-separate ablation.** Compare ModelSurgery with the three specialists at equal
   input resolution: task quality, total latency, peak memory, parameters and model size.
3. **Surgery ablation.** Report detection/pose/segmentation before integration, immediately after
   integration, and after each repair. This must expose negative transfer rather than hide it.
4. **TagTwo ground truth.** Annotate a fixed test set and report hit/no-hit, body-part accuracy,
   false-hit rate, missed-hit rate and occlusion slices.
5. **At least three seeds for claimed learned changes.** Pose already satisfies this; any final
   multitask fine-tuning must do the same.

### Strongly recommended

- FLOPs/MACs and parameter breakdown by shared trunk and each head.
- End-to-end camera-to-decision latency, not only model execution.
- Normal light, low light, motion blur, distance and occlusion slices on real arena captures.
- Failure-case analysis showing when pose fallback helps segmentation and when it causes errors.
- A licensing statement: HQNet code/data is non-commercial without a separate agreement.

## Ablation matrix

| Experiment | Detection | Pose | Parts | Hit metrics | Latency/memory |
|---|---:|---:|---:|---:|---:|
| Three independent specialists | required | required | required | required | required |
| Shared backbone + detection only | required | — | — | — | required |
| + segmentation head | required | — | required | required | required |
| + original pose head | required | required | required | required | required |
| + RTMO-taught pose adapter | required | required | required | required | required |
| HumanQueryNet/HQNet | required | required | binary mask only | not compatible | required |

## Go/no-go rule

Proceed toward an applied paper if the shared model demonstrates either:

- materially lower total latency or memory than three specialists while retaining acceptable task
  quality; or
- materially better TagTwo hit/body-part decisions than simpler detection+mask or detection+pose
  baselines.

Do not proceed as a methods paper unless a genuinely new mechanism is added and isolated—for
example a cross-task anatomical consistency loss, an explicit negative-transfer repair method, or a
joint query design that measurably improves more than one task.

## Recommended next action

Do not draft the paper yet. Complete the shared-versus-separate ablation first, because it is the
cheapest experiment that can validate or invalidate the core “one shared model has value” claim.

## Closest prior work

- HumanQueryNet/HQNet: https://arxiv.org/abs/2312.05525
- UniHCP: https://openaccess.thecvf.com/content/CVPR2023/html/Ci_UniHCP_A_Unified_Model_for_Human-Centric_Perceptions_CVPR_2023_paper.html
- MDSP: https://arxiv.org/abs/2205.01515
- Deep Multitask Architecture for Integrated 2D and 3D Human Sensing:
  https://openaccess.thecvf.com/content_cvpr_2017/html/Popa_Deep_Multitask_Architecture_CVPR_2017_paper.html
