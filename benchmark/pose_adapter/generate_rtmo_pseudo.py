#!/usr/bin/env python3
"""Generate RTMO pseudo keypoints for an isolated training subset."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools.coco import COCO
from rtmlib import RTMO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--coco-root", required=True, type=Path)
    parser.add_argument("--split", default="train2017")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--score-thr", type=float, default=0.20)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    ann = args.coco_root / "annotations" / f"person_keypoints_{args.split}.json"
    coco = COCO(str(ann))
    ids = sorted(coco.getImgIds(catIds=[1]))[: args.limit]
    model = RTMO(args.model, model_input_size=(640, 640), score_thr=0.01,
                 nms_thr=0.65, device="cpu")
    results = []
    for number, image_id in enumerate(ids, 1):
        info = coco.loadImgs(image_id)[0]
        image = cv2.imread(str(args.coco_root / args.split / info["file_name"]))
        keypoints, scores = model(image)
        keypoints = np.asarray(keypoints)
        scores = np.asarray(scores)
        if keypoints.ndim == 2:
            keypoints = keypoints[None]
            scores = scores[None]
        for pose, joint_scores in zip(keypoints, scores):
            confidence = float(np.mean(joint_scores))
            if confidence < args.score_thr:
                continue
            flat = []
            for (x, y), score in zip(pose, joint_scores):
                flat.extend([float(x), float(y), float(score)])
            results.append({"image_id": int(image_id), "keypoints": flat,
                            "score": confidence})
        if number % 50 == 0:
            print(f"[rtmo-pseudo] {number}/{len(ids)}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results))
    print(json.dumps({"images": len(ids), "poses": len(results), "out": str(args.out)}))


if __name__ == "__main__":
    main()
