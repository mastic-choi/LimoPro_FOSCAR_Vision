#!/usr/bin/env python3
"""검출 점수 보정(calibration) 도구: D-ECE, MCE, NLL, Brier, temperature scaling.

eval_mt_calib.py(측정)와 train_multitask.py(매 epoch val 기록, 학습 후 temperature 저장)가 같이 쓴다.
CUDA를 건드리지 않아 DataLoader 워커에서 import해도 된다.

예측 박스마다 y = 1(같은 클래스 정답과 IoU >= 0.5로 맞음) / 0(틀림, 중복 포함), p = 점수.
  ECE = Σ_b (n_b/N)|mean(y)_b - mean(p)_b|  (등간격 구간 b)
  MCE = max_b |mean(y)_b - mean(p)_b|       (예측이 mce_min개 이상인 구간만)
  NLL = -mean[y log p + (1-y) log(1-p)],  Brier = mean[(p-y)^2]
temperature: p' = sigmoid(logit(p) / T). T < 1이면 점수를 0.5에서 멀리 밀어냄(과소 확신 보정),
T > 1이면 0.5 쪽으로 당김(과신 보정). 같은 클래스 안 점수 순서는 안 바뀌어 AP는 그대로다.
"""
import numpy as np


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-np.asarray(z, float)))


def apply_T(p, T):
    return sigmoid(logit(p) / T)


def metrics(p, y, bins=15, mce_min=10):
    p, y = np.asarray(p, float), np.asarray(y, float)
    if len(p) == 0:
        return dict(n=0)
    e = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, e[1:-1]), 0, bins - 1)
    ece, mce, rel = 0.0, 0.0, []
    for b in range(bins):
        s = idx == b
        if not s.any():
            continue
        gap = abs(y[s].mean() - p[s].mean())
        ece += s.mean() * gap
        if s.sum() >= mce_min:
            mce = max(mce, gap)
        rel.append((float(p[s].mean()), float(y[s].mean()), int(s.sum())))
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    nll = float(-(y * np.log(pc) + (1 - y) * np.log(1 - pc)).mean())
    return dict(n=int(len(p)), acc=round(float(y.mean()), 4), conf=round(float(p.mean()), 4), ece=round(float(ece), 4),
                mce=round(float(mce), 4), nll=round(nll, 4), brier=round(float(((p - y) ** 2).mean()), 4), rel=rel)


def fit_T(p, y, min_n=20):
    """NLL이 가장 작은 temperature (로그 격자 + 국소 탐색). 예측이 min_n개 미만이면 1."""
    z, y = logit(p), np.asarray(y, float)
    if len(z) < min_n:
        return 1.0

    def nll(T):
        q = np.clip(sigmoid(z / T), 1e-6, 1 - 1e-6)
        return -(y * np.log(q) + (1 - y) * np.log(1 - q)).mean()
    grid = np.exp(np.linspace(np.log(0.2), np.log(5), 200))
    T = grid[np.argmin([nll(t) for t in grid])]
    for _ in range(30):   # 국소 탐색도 격자 범위(0.2~5) 안에서만 (예측이 적은 클래스에서 T가 0.05처럼 튀지 않게)
        T = min((max(0.2, T / 1.05), T, min(5.0, T * 1.05)), key=nll)
    return float(T)


def fit_T_per_class(p, y, cls, names, min_n=30):
    """(전체 T, {클래스: T}). 예측이 min_n개 미만인 클래스는 전체 T."""
    p, y, cls = np.asarray(p), np.asarray(y), np.asarray(cls)
    T_all = fit_T(p, y)
    T_cls = {}
    for c, name in enumerate(names):
        s = cls == c
        T_cls[name] = round(fit_T(p[s], y[s]), 3) if s.sum() >= min_n else round(T_all, 3)
    return round(T_all, 3), T_cls


def apply_T_cls(p, cls, T_cls, names):
    return np.array([apply_T(s, T_cls.get(names[c], 1.0)) for s, c in zip(p, cls)])


def match(dets, gts, iou_thr=0.5):
    """dets: [(score, cls, box xyxy)], gts: (N,5) [x1,y1,x2,y2,cls] -> 예측마다 1/0 (점수 높은 순, 정답 하나에 예측 하나)."""
    out = np.zeros(len(dets))
    gts = np.asarray(gts).reshape(-1, 5)
    used = np.zeros(len(gts), bool)
    for j in np.argsort([-d[0] for d in dets]):
        s, c, b = dets[j]
        g = np.where((gts[:, 4] == c) & ~used)[0]
        if not len(g):
            continue
        gg = gts[g, :4]
        x1, y1 = np.maximum(gg[:, 0], b[0]), np.maximum(gg[:, 1], b[1])
        x2, y2 = np.minimum(gg[:, 2], b[2]), np.minimum(gg[:, 3], b[3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        iou = inter / ((gg[:, 2] - gg[:, 0]) * (gg[:, 3] - gg[:, 1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)
        k = int(np.argmax(iou))
        if iou[k] >= iou_thr:
            used[g[k]] = True
            out[j] = 1
    return out
