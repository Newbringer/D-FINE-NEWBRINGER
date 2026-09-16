#!/usr/bin/env python3
"""Run the corrected three-seed lighting gate and two curriculum phases."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_training(args, out: Path, seed: int, phase: str, epochs: int, lr: float,
                 initial: Path | None = None) -> None:
    if out.exists():
        print(f"[resume] keeping {out}", flush=True); return
    command = [sys.executable, "benchmark/segmentation/lighting/train_v2.py",
        "--det-config", args.det_config, "--pose-config", args.pose_config,
        "--merged-ckpt", args.merged_ckpt, "--dataset", args.dataset,
        "--phase", phase, "--out", str(out), "--epochs", str(epochs),
        "--min-epochs", "4", "--patience", "3", "--batch-size", "4",
        "--lr", str(lr), "--seed", str(seed), "--screen-size", "384"]
    if initial is not None: command += ["--initial-seg-ckpt", str(initial)]
    subprocess.run(command, check=True)


def load_result(path: Path) -> dict:
    value = torch.load(path, map_location="cpu", weights_only=False)
    full = value["full_validation"]
    robust = float(np.mean([full[name]["miou"] for name in full if name != "normal"]))
    return {"path": str(path), "best_epoch": int(value["best_epoch"]),
            "normal_miou": float(full["normal"]["miou"]), "robust_miou": robust,
            "full_validation": full, "sha256": sha256(path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--det-config", required=True); parser.add_argument("--pose-config", required=True)
    parser.add_argument("--merged-ckpt", required=True); parser.add_argument("--dataset", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(); args.out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [20260916, 20260917, 20260918]
    checkpoints = []
    for seed in seeds:
        path = args.out_dir / f"balanced_seed{seed}.pth"
        run_training(args, path, seed, "balanced", 15, 3e-6); checkpoints.append(path)
    seed_results = [load_result(path) for path in checkpoints]
    eligible = [result for result in seed_results if result["normal_miou"] >= 0.7310789]
    if not eligible: raise RuntimeError("no balanced seed passed the normal-light gate")
    selected = max(eligible, key=lambda result: result["robust_miou"])
    phase1 = args.out_dir / "curriculum_phase1_moderate.pth"
    run_training(args, phase1, 20260919, "moderate", 12, 1e-6, Path(selected["path"]))
    phase2 = args.out_dir / "curriculum_phase2_strong.pth"
    run_training(args, phase2, 20260920, "strong", 5, 1e-6, phase1)
    summary = {"balanced_seeds": seed_results, "selected_balanced": selected,
               "phase1": load_result(phase1), "phase2": load_result(phase2)}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
