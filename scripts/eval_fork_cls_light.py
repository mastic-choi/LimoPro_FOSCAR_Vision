#!/usr/bin/env python3
"""대회장 색 조명(빨간빛·파란빛)에서 분류기가 버티는지: 크롭에 light_tint 를 세기별로 씌워 A/B/없음 비율을 본다.
실차 bag: B 원본(→B), 좌우 반전(가짜 A → A). 합성 폴더(<split>/A, B): 그대로.

사용: python scripts/eval_fork_cls_light.py <head.pt> [--strip out.png] <폴더> [<폴더> ...]
  폴더에 B/ 만 있으면 실차 bag 로, A/·B/ 가 다 있으면 합성으로 본다.
"""
import argparse, glob, os, sys

import cv2
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_fork_cls as t

ap = argparse.ArgumentParser()
ap.add_argument("head"); ap.add_argument("dirs", nargs="+"); ap.add_argument("--strip")
a = ap.parse_args()
c = torch.load(a.head, map_location="cpu")
head = nn.Linear(t.FEAT_DIM[c.get("feat", "gap")], 3); head.load_state_dict(c["head"])
m = t.ForkCls(t.folded_backbone(c.get("feat", "gap")), head.eval()).eval()
LIGHTS = [("흰 조명", None, 0.0), ("빨강 0.6", "red", 0.6), ("빨강 1.0", "red", 1.0), ("파랑 0.6", "blue", 0.6), ("파랑 1.0", "blue", 1.0)]


def lit(ims, kind, s):
    return ims if kind is None else [t.light_tint(i, np.random.default_rng(0), kind, s) for i in ims]


def pred(ims):
    X = torch.from_numpy(np.stack([t.to_input(i) for i in ims])[:, None])
    with torch.no_grad():
        P = m(X).numpy()
    sign = (np.maximum(P[:, 0], P[:, 1]) >= 0.6) & (np.abs(P[:, 0] - P[:, 1]) >= 0.2)
    return np.where(sign, np.where(P[:, 0] >= P[:, 1], 0, 1), 2)


def row(name, ims, want):
    cells = []
    for _, kind, s in LIGHTS:
        p = pred(lit(ims, kind, s)) if ims else np.zeros(0)
        cells.append(f"{np.mean(p == want) * 100:3.0f}/{np.mean(p == 1 - want) * 100:2.0f}")
    print(f"  {name:<18}" + "".join(f"{x:>11}" for x in cells))


print(f"모델 {os.path.basename(os.path.dirname(a.head))}  (칸 = 맞음% / 반대쪽 오답%)")
print(" " * 20 + "".join(f"{n:>10}" for n, _, _ in LIGHTS))
strip = []
for d in a.dirs:
    B = [cv2.imread(f) for f in sorted(glob.glob(os.path.join(d, "B", "*.png")))]
    A = [cv2.imread(f) for f in sorted(glob.glob(os.path.join(d, "A", "*.png")))]
    print(f"== {d.rstrip('/').split('fork_crops_v1/')[-1]}")
    if A:
        row(f"A {len(A)}장 → A", A, 0); row(f"B {len(B)}장 → B", B, 1)
    else:
        row(f"실차 B {len(B)}장 → B", B, 1); row("뒤집은 B → A", [cv2.flip(i, 1) for i in B], 0)
    if a.strip and B:
        ex = cv2.resize(B[len(B) // 2], (96, 96))
        strip.append(np.hstack([lit([ex], k, s)[0] for _, k, s in LIGHTS] + [cv2.cvtColor(t.rule.preprocess(lit([ex], k, s)[0], rgb=False, size=96), cv2.COLOR_GRAY2BGR) for _, k, s in LIGHTS]))
if a.strip and strip:
    cv2.imwrite(a.strip, np.vstack(strip)); print("그림:", a.strip, "(왼쪽 5칸 = 색 크롭, 오른쪽 5칸 = 분류기 입력 흑백)")
