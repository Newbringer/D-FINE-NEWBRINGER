#!/usr/bin/env python3
"""Simulate visible low-light camera output rather than multiplying sRGB pixels."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    parser=argparse.ArgumentParser();parser.add_argument("--input",required=True)
    parser.add_argument("--out",required=True);parser.add_argument("--stats",required=True,type=Path)
    parser.add_argument("--seed",type=int,default=20260916);args=parser.parse_args()
    cap=cv2.VideoCapture(args.input);fps=float(cap.get(cv2.CAP_PROP_FPS) or 30)
    width,height=int(cap.get(3)),int(cap.get(4));Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    writer=cv2.VideoWriter(args.out,cv2.VideoWriter_fourcc(*"mp4v"),fps,(width,height))
    rng=np.random.default_rng(args.seed);before=[];after=[];frames=0
    # Ten times less scene illumination, partially recovered by exposure/gain. Remaining darkness,
    # read/shot noise, warm arena colour and mild motion blur approximate a camera output.
    for _ in iter(int,1):
        ok,frame=cap.read()
        if not ok:break
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        linear=np.power(rgb,2.2);sensor=linear*0.10*2.5
        noise=rng.normal(0.0,0.008,sensor.shape)+rng.normal(0.0,1.0,sensor.shape)*np.sqrt(np.maximum(sensor,0))*0.018
        sensor=np.clip((sensor+noise)*np.asarray([1.06,1.0,0.90],np.float32),0,1)
        low=np.power(sensor,1/2.2);low=np.clip(low*255,0,255).astype(np.uint8)
        low=cv2.filter2D(low,-1,np.asarray([[.15,.25,.20,.25,.15]],np.float32))
        output=cv2.cvtColor(low,cv2.COLOR_RGB2BGR);writer.write(output)
        before.append(float(frame.mean()));after.append(float(output.mean()));frames+=1
    cap.release();writer.release();result={"frames":frames,"fps":fps,"mean_input":float(np.mean(before)),
        "mean_output":float(np.mean(after)),"brightness_ratio":float(np.mean(after)/np.mean(before)),
        "simulation":"0.1x linear illumination, 2.5x exposure/gain recovery, shot/read noise, warm cast, motion blur"}
    args.stats.parent.mkdir(parents=True,exist_ok=True);args.stats.write_text(json.dumps(result,indent=2));print(result);return 0


if __name__=="__main__":raise SystemExit(main())
