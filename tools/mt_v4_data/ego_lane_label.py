#!/usr/bin/env python3
"""직진 구간 자동 DA 라벨: v4 차선(ll)에 좌우 직선을 맞추고 그 사이를 DA 로 채운다 (사용자 11-18).

배경: multitask-v5 는 대회장 신호등 진입 장면을 DA 0 으로 배워 주행 중 DA 가 꺼졌다 (사람 검수의 'da·ll 비움').
      v4 는 그 장면을 잘 보지만 바닥 반사 자리는 비운다. 직진에서는 차로가 좌우 두 직선 사이라서,
      ll 직선 두 개만 믿을 수 있으면 반사와 상관없이 DA 를 만들 수 있다.

프레임마다:
  1. v4 로 da·ll 확률 (640x360)
  2. 화면 아래쪽(행 HY0~359) 각 행에서 차 기준 왼쪽 ll 의 안쪽 끝, 오른쪽 ll 의 안쪽 끝을 점으로
  3. 좌·우 따로 RANSAC 직선 x = a·y + b (오차 4 px 안을 인라이어)
  4. 판정 ok: 좌우 모두 인라이어 25 행 이상·비율 0.7 이상, 차로 폭(행 YW) 화면의 20~110 %, 두 직선이 화면 안에서 안 엇갈림
  5. ok 이면 ego DA = 두 직선 사이 다각형 ∪ 그 안의 ll 픽셀 (라벨 규칙: da = 도로 + 차선)
     v4 DA 와의 IoU 도 기록 (반사로 v4 가 비운 자리를 ego 가 채운 만큼 IoU 가 떨어짐)

사용 (모듈): from ego_lane_label import EgoLane; e = EgoLane(bundle, v4_ckpt); out = e.label(list_of_bgr_images)
"""
import os, sys

import cv2
import numpy as np
import torch

HY0, STEP, TOL, YW = 150, 2, 4.0, 260   # YW: 차로 폭을 재는 행 (맨 아래는 카메라가 낮아 차로가 화면보다 넓은 게 정상)


def ransac_line(ys, xs, rng, iters=200):
    if len(ys) < 10: return None
    best = None
    for _ in range(iters):
        i, j = rng.choice(len(ys), 2, replace=False)
        if ys[i] == ys[j]: continue
        a = (xs[j] - xs[i]) / (ys[j] - ys[i]); b = xs[i] - a * ys[i]
        inl = np.abs(a * ys + b - xs) < TOL
        if best is None or inl.sum() > best.sum(): best = inl
    if best is None or best.sum() < 10: return None
    a, b = np.polyfit(ys[best], xs[best], 1)
    inl = np.abs(a * ys + b - xs) < TOL
    return a, b, int(inl.sum()), float(inl.mean())


def fit_ego(ll, da, rng):
    """ll, da: (360,640) bool. 반환 dict(ok, xl(y), xr(y), y_top, ...)."""
    H, W = ll.shape
    pts = {"L": ([], []), "R": ([], [])}
    for y in range(HY0, H, STEP):
        row = np.where(ll[y])[0]
        if len(row) == 0: continue
        d = np.where(da[y])[0]
        cx = (d.min() + d.max()) / 2 if len(d) > 20 else W / 2     # 차로 가운데: v4 DA 가운데, 없으면 화면 가운데
        lf, rt = row[row < cx], row[row > cx]
        if len(lf): pts["L"][0].append(y); pts["L"][1].append(lf.max())
        if len(rt): pts["R"][0].append(y); pts["R"][1].append(rt.min())
    fits = {}
    for s in "LR":
        ys, xs = np.array(pts[s][0], float), np.array(pts[s][1], float)
        fits[s] = ransac_line(ys, xs, rng) if len(ys) else None
    r = {"ok": False, "fits": fits}
    if not fits["L"] or not fits["R"]: r["why"] = "선 하나 없음"; return r
    (aL, bL, nL, pL), (aR, bR, nR, pR) = fits["L"], fits["R"]
    if min(nL, nR) < 25 or min(pL, pR) < 0.7: r["why"] = f"직선 아님 L{nL}/{pL:.2f} R{nR}/{pR:.2f}"; return r
    w_bot = (aR * YW + bR) - (aL * YW + bL)
    if not 0.20 * W <= w_bot <= 1.10 * W: r["why"] = f"차로 폭 {w_bot:.0f}px"; return r
    y_top = max(min(pts["L"][0]), min(pts["R"][0]))
    if (aR * y_top + bR) - (aL * y_top + bL) < 4: r["why"] = "선이 엇갈림"; return r
    r.update(ok=True, aL=aL, bL=bL, aR=aR, bR=bR, y_top=int(y_top), w_bot=float(w_bot))
    return r


def ego_mask(r, ll, shape=(360, 640)):
    H, W = shape; m = np.zeros(shape, np.uint8)
    yt = r["y_top"]; xl = lambda y: r["aL"] * y + r["bL"]; xr = lambda y: r["aR"] * y + r["bR"]
    poly = np.array([[xl(yt), yt], [xr(yt), yt], [xr(H - 1), H - 1], [xl(H - 1), H - 1]], np.float32)
    cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 1)
    band = np.zeros(shape, np.uint8)   # 선 자체도 da (좌우 12 px 여유 안의 ll)
    poly2 = poly.copy(); poly2[[0, 3], 0] -= 12; poly2[[1, 2], 0] += 12
    cv2.fillPoly(band, [np.round(poly2).astype(np.int32)], 1)
    m = m.astype(bool) | (ll & band.astype(bool)); m[:yt] = False
    return m


class EgoLane:
    def __init__(self, bundle, ckpt, device="cuda"):
        sys.path.insert(0, os.path.join(os.path.abspath(bundle), "scripts"))
        import train_multitask as tm, make_model_compare_gif as m
        self.tm = tm; cwd = os.getcwd(); os.chdir(m.ROOT)
        self.model = tm.build_model(m, os.path.abspath(ckpt)).to(device).eval(); os.chdir(cwd)
        self.dev = device; self.rng = np.random.default_rng(0)
        self.MEAN, self.STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)

    @torch.no_grad()
    def predict(self, ims):
        tm = self.tm
        x = torch.from_numpy(np.stack([((cv2.cvtColor(cv2.copyMakeBorder(cv2.resize(im, (640, 360), interpolation=cv2.INTER_AREA), tm.PAD, tm.PAD, 0, 0,
                                         cv2.BORDER_CONSTANT, value=(114,) * 3), cv2.COLOR_BGR2RGB).astype(np.float32) / 255 - self.MEAN) / self.STD).transpose(2, 0, 1) for im in ims])).to(self.dev)
        *_, da, ll, sl = tm.heads(self.model, x)
        f = lambda p: (p.float()[:, tm.PAD:-tm.PAD] >= 0.5).cpu().numpy()
        return f(da), f(ll), f(sl)

    def label(self, ims):
        da, ll, sl = self.predict(ims); out = []
        for k in range(len(ims)):
            r = fit_ego(ll[k], da[k], self.rng)
            if r["ok"]:
                e = ego_mask(r, ll[k]); u = (e | da[k]).sum()
                r["iou_v4"] = float((e & da[k]).sum() / u) if u else 0.0; r["ego"] = e
            out.append(dict(r, da=da[k], ll=ll[k], sl=sl[k]))
        return out
