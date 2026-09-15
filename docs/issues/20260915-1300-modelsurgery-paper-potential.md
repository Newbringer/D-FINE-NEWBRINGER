# Assess ModelSurgery paper potential

- Status: Done
- Repository scope: research checkout only

## Goal

Determine whether the accumulated ModelSurgery work supports a publishable contribution without
starting a paper draft.

## Result

The project has medium potential as an applied systems/workshop paper, but currently low potential
as a top-tier methods paper. Detection+pose+segmentation is not itself novel. The strongest possible
claim is a resource-aware single graph with seven-class anatomical parsing and downstream
occlusion-aware hit decisions, supported by a reproducible RTMO-teacher pose gain.

The decisive missing evidence is a shared-versus-separate ablation plus TagTwo ground-truth hit and
body-part evaluation. HumanQueryNet remains the closest architecture baseline but lacks anatomical
part masks.

## Decision

Do not draft a paper yet. Run the shared-versus-separate quality/latency/memory ablation first. No
production or TagTwo files were modified.
