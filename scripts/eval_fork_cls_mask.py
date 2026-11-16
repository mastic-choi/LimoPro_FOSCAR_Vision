#!/usr/bin/env python3
"""분류기가 화살표를 보는지, 배경(노트북 화면 테두리·흰 막대)을 보는지: 실차 B 크롭을 원본 / 좌우 반전 / 원 바깥 가림 / 둘 다 로 시험.
뒤집은 B 는 화살표가 ↰ 라 A 가 정답. 원 바깥을 가려도 맞으면 원 안(화살표)을 보는 것.

사용: python scripts/eval_fork_cls_mask.py <head.pt> <bag 크롭 폴더> [<bag 크롭 폴더> ...]
"""
import glob, os, sys

import cv2
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_fork_cls as t

c = torch.load(sys.argv[1], map_location="cpu")
head = nn.Linear(t.FEAT_DIM[c.get("feat", "gap")], 3); head.load_state_dict(c["head"])
m = t.ForkCls(t.folded_backbone(c.get("feat", "gap")), head.eval()).eval()


def rate(ims, want):
    X = torch.from_numpy(np.stack([t.to_input(i) for i in ims])[:, None])
    with torch.no_grad():
        P = m(X).numpy()
    sign = (np.maximum(P[:, 0], P[:, 1]) >= 0.6) & (np.abs(P[:, 0] - P[:, 1]) >= 0.2)
    pred = np.where(sign, np.where(P[:, 0] >= P[:, 1], 0, 1), 2)
    return f"맞음 {np.mean(pred == want) * 100:4.0f}%  반대 {np.mean(pred == 1 - want) * 100:3.0f}%  없음 {np.mean(pred == 2) * 100:3.0f}%"


for d in sys.argv[2:]:
    ims = [cv2.imread(f) for f in sorted(glob.glob(os.path.join(d, "B", "*.png")))]
    fl = [cv2.flip(i, 1) for i in ims]
    print(f"== {os.path.basename(d.rstrip('/'))} B {len(ims)}장 (모델 {os.path.basename(os.path.dirname(sys.argv[1]))})")
    print("  원본 B             → B:", rate(ims, 1))
    print("  뒤집은 B (가짜 A)  → A:", rate(fl, 0))
    print("  원 바깥 가린 B      → B:", rate([t.circle_mask(i) for i in ims], 1))
    print("  가린 뒤집은 B       → A:", rate([t.circle_mask(i) for i in fl], 0))
