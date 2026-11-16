#!/usr/bin/env python3
"""ONNX 를 cv2.dnn 으로 돌려 torch 기준 출력(export_fork_cls.py 의 .ref.npz)과 같은지. 로봇과 같은 OpenCV(4.5.4)에서 돌린다.

사용: python3 scripts/check_fork_cls_dnn.py <model.onnx> [<model.onnx.ref.npz>]
"""
import sys

import cv2
import numpy as np

onnx = sys.argv[1]; ref = np.load(sys.argv[2] if len(sys.argv) > 2 else onnx + ".ref.npz")
net = cv2.dnn.readNetFromONNX(onnx)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV); net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
net.setInput(ref["x"]); batch = net.forward().reshape(len(ref["x"]), 3)
single = []
for xi in ref["x"]:
    net.setInput(xi[None]); single.append(net.forward().reshape(3))
d_batch, d_single = np.abs(batch - ref["prob"]).max(), np.abs(np.array(single) - ref["prob"]).max()
print(f"OpenCV {cv2.__version__}: 최대 차이 묶음 {d_batch:.2e}, 한 장씩 {d_single:.2e}, 확률 합 {batch.sum(1).min():.4f}~{batch.sum(1).max():.4f}")
ok = d_batch < 1e-4 and d_single < 1e-4
print("DNN_MATCH", "OK" if ok else "FAIL"); sys.exit(0 if ok else 1)
