"""바닥 반사 증강 v3 (대회장 무대 조명이 광택 매트에 비친 것).
v2 대비 (사용자 11-18): 흰빛을 더 강하게, 띠를 차선(ll) 위에 놓아 선을 바래게 하고, 차로(da) 쪽으로 번지게.
반사는 어디서나 곧은 띠 (사용자 11-18). 기울기는 고른 차선의 전체 방향(1차 직선)을 따름.
위치 (사용자 11-18 수정): 선을 지우면 그 자리 ll 을 배울 단서가 없으므로 띠는 ll 과 겹치지 않게, 선 바로 옆 차로(da) 쪽에 둠.
  띠 중심 = 선 + 차로 방향 x 띠 폭 1.3~2.0배 (줄마다). 띠 폭은 v3 의 절반 (화면 폭 6~12.5%).
모양: 지평선 근처에서 화면 아래 끝까지 이어지는 넓은 세로 띠, 아래로 갈수록 넓어짐, 가운데는 하얗게 날아가고 가장자리는 조명색.
라벨(da, ll)은 그대로 = '선이 바래도, 반사가 있어도 차로는 차로'."""
import random, cv2, numpy as np


def _lane_curve(ll, h, w, rng):
    """ll 덩어리 하나를 골라 줄별 가운데 x 에 2차 곡선을 맞춤 -> (곡선 함수, 그 덩어리 아래쪽 한 점) 또는 None."""
    if ll is None: return None
    n, lab, st, _ = cv2.connectedComponentsWithStats(ll[int(0.4 * h):].astype(np.uint8), 8)
    comps = [k for k in range(1, n) if st[k, cv2.CC_STAT_HEIGHT] >= 0.12 * h]
    if not comps: return None
    k = rng.choice(comps); ys, xs = np.nonzero(lab == k); ys = ys + int(0.4 * h)
    uy = np.unique(ys)
    cx = np.array([xs[ys == y].mean() for y in uy])
    if len(uy) < 8: return None
    p = np.polyfit(uy, cx, 1)                                     # 곧은 띠: 차선 전체 방향만 씀
    yb = int(uy.max()); return np.poly1d(p), (int(np.clip(round(np.polyval(p, yb)), 0, w - 1)), yb)   # 화면 밖으로 나간 곡선 끝은 가장자리로


def _anchor(ll, da, h, w, rng):
    """띠 아래쪽 중심: 차선 위 한 점을 고르고, 차로(da)가 있는 쪽으로 밀어 빛이 차로까지 번지게."""
    if ll is not None and rng.random() < 0.8:
        ys, xs = np.nonzero(ll[int(0.55 * h):])
        if len(xs):
            k = rng.randrange(len(xs)); x, y = int(xs[k]), int(ys[k]) + int(0.55 * h)
            side = 0
            if da is not None:
                l, r = da[y, max(0, x - 60):x].mean() if x > 0 else 0, da[y, x:min(w, x + 60)].mean() if x < w else 0
                side = 1 if r > l else -1                     # 차로가 있는 쪽
            return x, side
    if da is not None:
        ys, xs = np.nonzero(da[int(0.55 * h):])
        if len(xs):
            return int(xs[rng.randrange(len(xs))]), 0
    return int(rng.uniform(0.1, 0.9) * w), 0


def floor_glare(img, ll=None, da=None, rng=random):
    h, w = img.shape[:2]
    x = img.astype(np.float32) / 255.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    total = np.zeros((h, w), np.float32)
    for _ in range(1 if rng.random() < 0.65 else 2):
        top = rng.uniform(0.36, 0.5) * h
        bot = rng.uniform(1.0, 1.3) * h
        wb = rng.uniform(0.06, 0.125) * w                        # 아래쪽 폭 (v3 의 절반)
        wt = wb * rng.uniform(0.2, 0.4)
        f = np.clip((yy - top) / (bot - top), 0, 1)
        sw = wt + (wb - wt) * f
        curve = _lane_curve(ll, h, w, rng) if rng.random() < 0.9 else None
        if curve is not None:                                    # 차선 모양을 따라 휘는 띠 (커브면 휘고, 직선이면 곧게)
            poly, (ax, ay) = curve
            side = 0
            if da is not None:
                y_ = min(h - 1, ay); l = da[y_, max(0, ax - 60):max(1, ax)].mean() if ax > 0 else 0
                r = da[y_, min(w - 1, ax):min(w, ax + 60)].mean() if ax < w - 1 else 0
                side = 1 if r > l else -1
            if side == 0: side = rng.choice([-1, 1])
            push = side * rng.uniform(1.3, 2.0)                   # 선과 안 겹치게, 하지만 바로 옆 (띠 폭 기준)
            cx = np.clip(poly(yy[:, :1]), -0.5 * w, 1.5 * w) + push * sw
            top = max(top, float(ll.nonzero()[0].min()) - 0.05 * h) if ll is not None and ll.any() else top
        else:
            ax, side = _anchor(ll, da, h, w, rng)
            cx_bot = ax + (side if side else rng.choice([-1, 1])) * wb * rng.uniform(1.3, 2.0)
            cx_top = cx_bot + rng.uniform(-0.12, 0.12) * w
            cx = cx_top + (cx_bot - cx_top) * f
        g = np.exp(-((xx - cx) / sw) ** 2 * 0.8)
        v = np.clip((yy - top) / (0.1 * h), 0, 1) * (0.8 + 0.2 * np.sin(np.pi * f))
        g = g * v; g[yy < top] = 0
        total = np.maximum(total, g * rng.uniform(0.9, 1.0))
    total = cv2.GaussianBlur(total, (0, 0), 4)
    core = np.clip((total - 0.4) / 0.25, 0, 1)                   # 하얗게 날아가는 영역을 넓게 (v2: 0.6 / 0.35)
    halo = np.clip(total / 0.45, 0, 1)
    local = cv2.GaussianBlur(x, (0, 0), 25)
    tint = 0.5 * local / np.maximum(local.max(2, keepdims=True), 1e-3) + 0.5
    gray = x.mean(2, keepdims=True)
    x = x * (1 - 0.85 * halo[..., None]) + gray * (0.85 * halo[..., None])          # 색이 거의 다 빠짐 → 노란 선도 바램
    x = x + halo[..., None] * rng.uniform(0.55, 0.85) * tint                         # 더 밝게 (v2: 0.35~0.6)
    x = x * (1 - core[..., None]) + core[..., None] * rng.uniform(0.96, 1.0)
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)
