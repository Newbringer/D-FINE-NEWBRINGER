#!/usr/bin/env python3
"""Robust low-light fine-tuning for the isolated ModelSurgery segmentation head."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from benchmark.harness.infer_crosshair import build_model_from_merged
from segmentation_sivert.core.datasets import PascalPersonPartsDataset

CLASS_NAMES = ["background", "head", "torso", "arms", "hands", "legs", "feet"]


def digest(module: torch.nn.Module) -> str:
    result = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        result.update(name.encode()); result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def clip(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0, 255).astype(np.uint8)


def augment_light(image: np.ndarray, rng: np.random.Generator, phase: str = "balanced",
                  severity: float = 1.0) -> tuple[np.ndarray, str]:
    value = image.astype(np.float32)
    choice = float(rng.random())
    if phase == "progressive":
        severity = float(np.clip(severity, 0.0, 1.0))
        if choice < 0.50: return image, "normal"
        if choice < 0.68:
            low = 0.55 - 0.45 * severity; high = 0.90 - 0.40 * severity
            return clip(value * rng.uniform(low, high)), "progressive_exposure"
        if choice < 0.80:
            gamma = float(rng.uniform(1.05, 1.25 + 1.75 * severity))
            scale = float(rng.uniform(0.85 - 0.40 * severity, 1.0))
            return clip(255.0 * np.power(value / 255.0, gamma) * scale), "progressive_gamma"
        if choice < 0.90:
            factor = float(rng.uniform(0.65 - 0.52 * severity, 0.90 - 0.45 * severity))
            sigma = float(rng.uniform(2.0, 3.0 + 15.0 * severity))
            return clip(value * factor + rng.normal(0.0, sigma, value.shape)), "progressive_noise"
        if choice < 0.95:
            spread = 0.10 + 0.30 * severity
            cast = np.asarray([rng.uniform(1-spread, 1+spread), rng.uniform(0.9, 1.1),
                               rng.uniform(1-spread, 1+spread)], dtype=np.float32)
            return clip(value * cast), "progressive_color"
        if choice < 0.98:
            return clip(value * rng.uniform(1.05, 1.20 + 0.50 * severity)
                        + rng.uniform(0, 35 * severity)), "progressive_overexposed"
        kernel_size = int(rng.choice([3, 5] if severity < 0.6 else [3, 5, 7]))
        kernel = np.zeros((kernel_size, kernel_size)); kernel[kernel_size // 2] = 1.0 / kernel_size
        return cv2.filter2D(image, -1, kernel), "progressive_motion"
    if phase == "moderate":
        if choice < 0.55: return image, "normal"
        if choice < 0.75: return clip(value * rng.uniform(0.30, 0.70)), "moderate_dark"
        if choice < 0.85:
            gamma = float(rng.uniform(1.2, 2.0))
            return clip(255.0 * np.power(value / 255.0, gamma) * rng.uniform(0.65, 0.95)), "moderate_gamma"
        if choice < 0.92: return clip(value * rng.uniform(1.15, 1.45) + rng.uniform(5, 25)), "moderate_bright"
        if choice < 0.97:
            cast = np.asarray([rng.uniform(0.8, 1.2), 1.0, rng.uniform(0.8, 1.2)], dtype=np.float32)
            return clip(value * cast), "moderate_color"
        return clip(value * 0.55 + rng.normal(0.0, 6.0, value.shape)), "moderate_noise"
    if phase == "strong":
        if choice < 0.30: return image, "normal"
        if choice < 0.52: return clip(value * rng.uniform(0.08, 0.35)), "strong_dark"
        if choice < 0.72:
            return clip(value * rng.uniform(0.10, 0.35) + rng.normal(0.0, rng.uniform(8, 20), value.shape)), "strong_noise"
        if choice < 0.82:
            gamma = float(rng.uniform(2.0, 3.5))
            return clip(255.0 * np.power(value / 255.0, gamma) * rng.uniform(0.35, 0.70)), "strong_gamma"
        if choice < 0.90:
            cast = np.asarray([rng.uniform(0.55, 1.45), rng.uniform(0.8, 1.1), rng.uniform(0.55, 1.45)])
            return clip(value * rng.uniform(0.2, 0.6) * cast), "strong_color"
        if choice < 0.96: return clip(value * rng.uniform(1.4, 1.9) + rng.uniform(25, 65)), "strong_overexposed"
        height, width = image.shape[:2]; yy, xx = np.mgrid[:height, :width]
        gradient = 0.12 + 0.88 * (xx + yy) / max(1, width + height - 2)
        return clip(value * gradient[..., None]), "strong_shadow"
    if choice < 0.50:
        return image, "normal"
    if choice < 0.62:
        factor = float(rng.uniform(0.10, 0.50))
        return clip(value * factor), "exposure_dark"
    if choice < 0.72:
        gamma = float(rng.uniform(1.5, 3.0)); scale = float(rng.uniform(0.45, 0.85))
        return clip(255.0 * np.power(value / 255.0, gamma) * scale), "gamma_dark"
    if choice < 0.82:
        factor = float(rng.uniform(0.12, 0.45)); sigma = float(rng.uniform(4.0, 18.0))
        noise = rng.normal(0.0, sigma, value.shape)
        return clip(value * factor + noise), "low_light_noise"
    if choice < 0.89:
        factor = float(rng.uniform(0.25, 0.80))
        cast = np.asarray([rng.uniform(0.65, 1.35), rng.uniform(0.85, 1.10),
                           rng.uniform(0.65, 1.35)], dtype=np.float32)
        return clip(value * factor * cast), "led_color_cast"
    if choice < 0.94:
        return clip(value * rng.uniform(1.25, 1.75) + rng.uniform(15, 55)), "overexposed"
    if choice < 0.98:
        height, width = image.shape[:2]
        yy, xx = np.mgrid[:height, :width]
        gradient = 0.20 + 0.80 * (xx + yy) / max(1, width + height - 2)
        return clip(value * gradient[..., None]), "strong_shadow"
    kernel_size = int(rng.choice([3, 5, 7])); kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2] = 1.0 / kernel_size
    blurred = cv2.filter2D(image, -1, kernel)
    quality = int(rng.integers(45, 80)); ok, encoded = cv2.imencode(".jpg", blurred,
        [cv2.IMWRITE_JPEG_QUALITY, quality])
    return (cv2.imdecode(encoded, cv2.IMREAD_COLOR) if ok else blurred), "motion_compression"


class PairedLightingDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, seed: int, phase: str = "balanced", total_epochs: int = 1):
        self.base = PascalPersonPartsDataset(root_dir=root, split="train", image_size=640,
                                             num_classes=7, tier="standard")
        self.seed = int(seed); self.epoch = 0; self.phase = phase
        self.total_epochs = max(1, int(total_epochs))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image, mask = self.base._load_image_and_mask(index)
        rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + index * 97)
        if self.phase == "strong" and float(rng.random()) < 0.35:
            target_pixels = np.argwhere((mask == 4) | (mask == 6))
            if target_pixels.size:
                center_y, center_x = target_pixels[int(rng.integers(len(target_pixels)))]
                side = int(min(image.shape[:2]) * rng.uniform(0.40, 0.75))
                x1 = int(np.clip(center_x - side // 2, 0, max(0, image.shape[1] - side)))
                y1 = int(np.clip(center_y - side // 2, 0, max(0, image.shape[0] - side)))
                image = image[y1:y1 + side, x1:x1 + side]
                mask = mask[y1:y1 + side, x1:x1 + side]
        if float(rng.random()) < 0.5:
            image, mask = image[:, ::-1].copy(), mask[:, ::-1].copy()
        severity = (self.epoch - 1) / max(1, self.total_epochs - 1)
        clean = image.copy(); augmented, condition = augment_light(image, rng, self.phase, severity)
        return (self.base.preprocess_image(clean), self.base.preprocess_image(augmented),
                self.base.preprocess_mask(mask), condition)


def weighted_dice(logits: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    probabilities = logits.softmax(1)
    one_hot = F.one_hot(target.clamp(0, 6), 7).permute(0, 3, 1, 2).float()
    intersection = (probabilities * one_hot).sum((0, 2, 3))
    denominator = probabilities.sum((0, 2, 3)) + one_hot.sum((0, 2, 3))
    losses = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
    return (losses * weights).sum() / weights.sum()


def validation_transform(image: np.ndarray, condition: str, index: int) -> np.ndarray:
    value = image.astype(np.float32)
    if condition == "normal": return image
    if condition == "dark_3x": return clip(value / 3.0)
    if condition == "dark_10x": return clip(value / 10.0)
    if condition == "low_light_noise":
        rng = np.random.default_rng(9_000_000 + index)
        return clip(value * 0.20 + rng.normal(0.0, 12.0, value.shape))
    if condition == "overexposed": return clip(value * 1.5 + 35.0)
    raise ValueError(condition)


def evaluate(model, dataset, device, condition: str, indices: list[int]) -> dict:
    confusion = np.zeros((7, 7), dtype=np.int64)
    with torch.inference_mode():
        for start in range(0, len(indices), 8):
            images, masks = [], []
            for index in indices[start:start + 8]:
                image, mask = dataset._load_image_and_mask(index)
                image = validation_transform(image, condition, index)
                images.append(dataset.preprocess_image(image)); masks.append(dataset.preprocess_mask(mask))
            tensor = torch.stack(images).to(device)
            target = torch.stack(masks).numpy().reshape(-1)
            logits = model.seg_head(model.backbone(tensor))
            logits = F.interpolate(logits, (640, 640), mode="bilinear", align_corners=False)
            prediction = logits.argmax(1).cpu().numpy().reshape(-1)
            confusion += np.bincount(7 * target + prediction, minlength=49).reshape(7, 7)
    intersection = np.diag(confusion); union = confusion.sum(0) + confusion.sum(1) - intersection
    iou = intersection / np.maximum(union, 1)
    return {"miou": float(iou.mean()),
            "per_class_iou": {name: float(score) for name, score in zip(CLASS_NAMES, iou)}}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--det-config", required=True); parser.add_argument("--pose-config", required=True)
    parser.add_argument("--merged-ckpt", required=True); parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True, type=Path); parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--min-epochs", type=int, default=4); parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4); parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument("--seed", type=int, default=20260916); parser.add_argument("--screen-size", type=int, default=384)
    parser.add_argument("--phase", choices=["balanced", "moderate", "strong", "progressive"], default="balanced")
    parser.add_argument("--initial-seg-ckpt", type=Path,
                        help="Optional prior v2 checkpoint whose seg_head starts this curriculum phase.")
    parser.add_argument("--save-every-epoch", action="store_true")
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda")
    model, _ = build_model_from_merged(args.det_config, args.pose_config, args.merged_ckpt,
                                       7, 384, 0.1, 640)
    protected_names = ("backbone", "encoder", "det_decoder", "pose_decoder")
    protected = {name: digest(getattr(model, name)) for name in protected_names}
    for name in protected_names: getattr(model, name).requires_grad_(False).eval()
    teacher = copy.deepcopy(model.seg_head).requires_grad_(False).eval()
    if args.initial_seg_ckpt:
        initial = torch.load(args.initial_seg_ckpt, map_location="cpu", weights_only=False)
        model.seg_head.load_state_dict(initial["seg_head"], strict=True)
    model.seg_head.requires_grad_(True); model.to(device); teacher.to(device)
    model.seg_head.eval()
    train = PairedLightingDataset(args.dataset, args.seed, args.phase, args.epochs)
    val = PascalPersonPartsDataset(root_dir=args.dataset, split="val", image_size=640,
                                   num_classes=7, tier="standard")
    screen = list(range(min(args.screen_size, len(val)))); full = list(range(len(val)))
    baseline = {condition: evaluate(model, val, device, condition, screen)
                for condition in ("normal", "dark_3x", "dark_10x", "low_light_noise", "overexposed")}
    normal_floor = baseline["normal"]["miou"] - 0.005
    baseline_robust = float(np.mean([baseline[name]["miou"] for name in baseline if name != "normal"]))
    class_weights = ([0.35, 1.0, 1.0, 1.15, 1.75, 1.0, 2.0] if args.phase != "strong"
                     else [0.30, 1.0, 1.0, 1.20, 2.0, 1.0, 2.25])
    weights = torch.tensor(class_weights, device=device)
    optimizer = torch.optim.AdamW(model.seg_head.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)
    baseline_row = {"epoch": 0, "loss": None, "lr": None, "eligible": True,
                    "robust_miou": baseline_robust, "metrics": baseline,
                    "conditions": {"initial_checkpoint": len(train)}}
    history = [baseline_row]
    best = (baseline_robust, 0, copy.deepcopy(model.seg_head.state_dict()), baseline_row)
    stale = 0
    for epoch in range(1, args.epochs + 1):
        train.set_epoch(epoch); generator = torch.Generator().manual_seed(args.seed + epoch)
        loader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True,
            generator=generator, num_workers=4, pin_memory=True, drop_last=True)
        model.seg_head.train(); losses=[]; counts={}
        for clean, augmented, target, conditions in loader:
            clean, augmented, target = clean.to(device), augmented.to(device), target.to(device)
            with torch.no_grad():
                clean_features = model.backbone(clean); teacher_logits = teacher(clean_features)
                augmented_features = model.backbone(augmented)
            logits = model.seg_head(augmented_features)
            logits = F.interpolate(logits, target.shape[-2:], mode="bilinear", align_corners=False)
            teacher_logits = F.interpolate(teacher_logits, target.shape[-2:], mode="bilinear", align_corners=False)
            supervised = F.cross_entropy(logits, target, weight=weights) + 0.5 * weighted_dice(logits, target, weights)
            temperature = 2.0
            distill = F.kl_div(F.log_softmax(logits / temperature, 1),
                               F.softmax(teacher_logits / temperature, 1), reduction="none").sum(1).mean()
            loss = supervised + 0.20 * distill * temperature * temperature
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.seg_head.parameters(), 1.0); optimizer.step()
            losses.append(float(loss.detach()))
            for condition in conditions: counts[condition] = counts.get(condition, 0) + 1
        scheduler.step(); model.seg_head.eval()
        metrics = {condition: evaluate(model, val, device, condition, screen)
                   for condition in baseline}
        robust = float(np.mean([metrics[name]["miou"] for name in metrics if name != "normal"]))
        eligible = metrics["normal"]["miou"] >= normal_floor
        score = robust if eligible else -1.0
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "lr": optimizer.param_groups[0]["lr"],
               "eligible": eligible, "robust_miou": robust, "metrics": metrics, "conditions": counts}
        history.append(row); print(json.dumps(row), flush=True)
        if args.save_every_epoch:
            epoch_dir = args.out.parent / f"{args.out.stem}_epochs"; epoch_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"seg_head": {k: v.detach().cpu() for k, v in model.seg_head.state_dict().items()},
                        "epoch": epoch, "metrics": row, "args": vars(args)},
                       epoch_dir / f"epoch_{epoch:02d}.pth")
        if best is None or score > best[0]:
            best = (score, epoch, copy.deepcopy(model.seg_head.state_dict()), row); stale = 0
        else: stale += 1
        if epoch >= args.min_epochs and stale >= args.patience: break
    model.seg_head.load_state_dict(best[2]); model.seg_head.eval()
    final = {condition: evaluate(model, val, device, condition, full) for condition in baseline}
    after = {name: digest(getattr(model, name)) for name in protected_names}
    assert protected == after, "protected module changed"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"seg_head": {k:v.cpu() for k,v in best[2].items()}, "best_epoch": best[1],
                "screen_baseline": baseline, "screen_best": best[3], "full_validation": final,
                "history": history, "protected_hashes": protected, "protected_unchanged": True,
                "args": vars(args)}, args.out)
    print(json.dumps({"out": str(args.out), "best_epoch": best[1], "full_validation": final,
                      "protected_unchanged": True}, indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
