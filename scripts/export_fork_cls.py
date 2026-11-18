#!/usr/bin/env python3
"""갈림길 분류기(head.pt) → ONNX (로봇 ar_fork_sign 이 cv::dnn 으로 읽는다) + 일치 확인용 기준 입력·출력.

- 모델: train_fork_cls.ForkCls (접힌 ResNet18 + 선형 + softmax). 입력 `input` Nx1x96x96 (0~1), 출력 `prob` Nx3 [A, B, none].
- 로봇 OpenCV 4.5.4 dnn 에 맞게 opset 11, 예전 exporter (dynamo=False).
- head.pt 가 --post-mask 로 학습됐으면 원 바깥 가리기(x*m + 0.5*(1-m))를 ONNX 맨 앞에 넣는다: 로봇 입력·전처리는 그대로 (11-18).
- <out>.ref.npz: 실제 크롭 16장의 입력과 torch 출력. check_fork_cls_dnn.py 가 cv2.dnn 결과와 비교한다.

사용: python scripts/export_fork_cls.py <head.pt> <out.onnx> <크롭 폴더> [<크롭 폴더> ...]
"""
import glob, os, sys

import cv2
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_fork_cls as t

ckpt, out, dirs = sys.argv[1], sys.argv[2], sys.argv[3:]
c = torch.load(ckpt, map_location="cpu")
feat = c.get("feat", "gap")
head = nn.Linear(t.FEAT_DIM[feat], 3); head.load_state_dict(c["head"])
model = t.ForkCls(t.folded_backbone(feat), head.eval()).eval()
pm = float(c.get("args", {}).get("post_mask", 0.0) or 0.0)


class Masked(nn.Module):
    def __init__(self, inner, m):
        super().__init__(); self.inner = inner
        self.register_buffer("m", torch.from_numpy(m)[None, None])

    def forward(self, x):   # = x*m + 0.5*(1-m). 상수 텐서 덧셈은 OpenCV 4.5.4 dnn 이 묶음(N>1)에서 못 늘려서 스칼라 덧셈·곱셈만 씀
        return self.inner((x - 0.5) * self.m + 0.5)


if pm > 0:
    model = Masked(model, t.post_mask_array(t.SIZE, pm)).eval()
t.POST_MASK = 0.0   # 기준 입력은 로봇이 넣는 것과 같게(가리기 전), 가리기는 모델 안에서
torch.onnx.export(model, torch.zeros(1, 1, t.SIZE, t.SIZE), out, opset_version=11, dynamo=False,
                  input_names=["input"], output_names=["prob"], dynamic_axes={"input": {0: "n"}, "prob": {0: "n"}},
                  do_constant_folding=True)
files = []
for d in dirs:
    for lab in t.LABELS:
        files += sorted(glob.glob(os.path.join(d, lab, "*.png")))[:6]
rng = np.random.default_rng(0)
files = [files[i] for i in rng.choice(len(files), size=min(16, len(files)), replace=False)]
x = np.stack([t.to_input(cv2.imread(f)) for f in files])[:, None].astype(np.float32)
with torch.no_grad():
    prob = model(torch.from_numpy(x)).numpy()
np.savez(out + ".ref.npz", x=x, prob=prob)
print(f"{out} ({os.path.getsize(out) / 1e6:.1f} MB), 기준 {len(files)}장, 원 바깥 가리기 {pm or '없음'}")
