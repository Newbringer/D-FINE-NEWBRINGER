#!/usr/bin/env python3
"""Render fixed augmentation types across progressive severity levels."""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from benchmark.segmentation.lighting.train_v2 import clip
from segmentation_sivert.core.datasets import PascalPersonPartsDataset


def variants(image: np.ndarray, severity: float, seed: int) -> dict[str, np.ndarray]:
    value=image.astype(np.float32);rng=np.random.default_rng(seed);spread=.10+.30*severity
    gamma=1.05+1.95*severity;factor=.90-.75*severity;sigma=2+16*severity
    cast=np.asarray([1+spread,1.0,1-spread],np.float32)
    kernel_size=3 if severity<.5 else 5 if severity<.85 else 7
    kernel=np.zeros((kernel_size,kernel_size));kernel[kernel_size//2]=1/kernel_size
    return {
        "exposure":clip(value*(.90-.78*severity)),
        "gamma":clip(255*np.power(value/255,gamma)*(.95-.50*severity)),
        "noise":clip(value*factor+rng.normal(0,sigma,value.shape)),
        "color":clip(value*cast),
        "overexposed":clip(value*(1.05+.65*severity)+35*severity),
        "motion":cv2.filter2D(image,-1,kernel),
    }


def label(tile: np.ndarray, text: str) -> None:
    cv2.rectangle(tile,(0,0),(tile.shape[1],22),(0,0,0),-1)
    cv2.putText(tile,text,(5,16),cv2.FONT_HERSHEY_SIMPLEX,.40,(255,255,255),1,cv2.LINE_AA)


def main() -> int:
    parser=argparse.ArgumentParser();parser.add_argument("--dataset",required=True)
    parser.add_argument("--out",required=True,type=Path);parser.add_argument("--samples",type=int,default=8)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    dataset=PascalPersonPartsDataset(root_dir=args.dataset,split="train",image_size=640,
                                     num_classes=7,tier="standard")
    indices=np.linspace(0,len(dataset)-1,args.samples).round().astype(int);severities=[0,.25,.5,.75,1.0];rows=[]
    for index in indices:
        image,_=dataset._load_image_and_mask(int(index));sheet=[]
        for severity in severities:
            original=cv2.resize(image[:,:,::-1],(192,192));label(original,f"original s={severity:.2f}")
            row=[original]
            for name,augmented in variants(image,severity,20260916+int(index)).items():
                tile=cv2.resize(augmented[:,:,::-1],(192,192));label(tile,name);row.append(tile)
                rows.append({"image_index":int(index),"severity":severity,"condition":name,
                             "mean":float(augmented.mean()),"std":float(augmented.std()),
                             "min":int(augmented.min()),"max":int(augmented.max())})
            sheet.append(np.hstack(row))
        cv2.imwrite(str(args.out/f"sample_{int(index):04d}.jpg"),np.vstack(sheet),
                    [cv2.IMWRITE_JPEG_QUALITY,94])
    with (args.out/"statistics.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print({"sheets":len(indices),"rows":len(rows),"out":str(args.out)});return 0


if __name__=="__main__":raise SystemExit(main())
