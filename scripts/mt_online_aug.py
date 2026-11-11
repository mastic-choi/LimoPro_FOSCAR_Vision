#!/usr/bin/env python3
"""학습할 때 바로 그려 넣는 음성 샘플 증강 (이미 만들어 둔 합성 데이터 v2 이미지에 적용, 새 이미지를 만들지 않음).

1) 신호등 뒤 단색 방해 모니터 (규정집 미션 1)
   v2 라벨의 신호등 몸체 박스(traffic_light)를 기준으로, 신호등보다 3.5~6배 넓은 16:9 단색 화면(검은 테두리, 받침대)을
   신호등 뒤에 그린다. 그다음 원래 이미지의 신호등 몸체(어두운 픽셀 + 켜진 불)와 기둥을 다시 위에 덮어
   신호등이 모니터 앞에 있게 한다. 모니터는 라벨이 없는 음성 샘플이고 박스 라벨은 바뀌지 않는다.
   지금 달리는 차로(da) 위로 5% 넘게 걸치거나, 다른 라벨 물체(표지판·콘 등)를 가리거나, 화면 위로 잘리면 그리지 않는다.
   이미지 안의 모든 신호등(불빛 포함)은 모니터 앞에 남는다.
   모니터가 덮은 곳은 da/ll/흰 가로선 정답에서 지운다.
   dots_p 확률로 화면을 땡땡이로 그린다: 어두운/흰 바탕에 신호등 불빛 크기의 원을 깔되, 그 신호등에 켜진 불과 다른 색
   (초록불이면 빨강·노랑 원). 원은 라벨 없는 음성 샘플이라 '신호등 몸체 안에서 빛나는 것만 불빛'을 배우게 한다.
2) 트랙 위 흰 전원 케이블
   실차 bag에서 흰 가로선으로 오탐된 것. 그린 곳은 '흰 가로선 아님'(sl=0, 믿을 픽셀)으로 학습한다.
모두 640x360 원본 좌표에서, 좌우 반전·회전 같은 기하 증강 전에 적용한다(train_multitask.Items).
"""
import random

import cv2
import numpy as np

COLORS = [(40, 40, 230), (30, 200, 240), (60, 210, 60), (220, 120, 30), (235, 235, 235),
          (200, 60, 200), (40, 140, 240), (120, 230, 200), (20, 20, 20)]   # BGR 빨강 노랑 초록 파랑 흰 보라 주황 연두 검정
TL_CLS = 6   # train_multitask.CLASSES.index("traffic_light")
LIGHT_CLS = {6, 7, 8, 9}   # traffic_light, light_red/yellow/green (불빛 박스는 신호등 몸체 안에 있어 같이 앞에 남음)
LAMP_COLOR = {7: (40, 40, 235), 8: (40, 200, 240), 9: (110, 225, 60)}   # light_red, light_yellow, light_green (BGR)


def _light(img, box):
    x0, y0, x1, y1 = [int(v) for v in box]
    H, W = img.shape[:2]
    w, h = max(1, x1 - x0), max(1, y1 - y0)
    reg = img[max(0, y0 - 2 * h):min(H, y1 + 2 * h), max(0, x0 - 2 * w):min(W, x1 + 2 * w)]
    if reg.size < 30:
        reg = img
    p = np.percentile(reg.reshape(-1, 3), 92, axis=0)
    return np.clip(p / 225.0, 0.12, 1.25)


def tl_foreground(img, box):
    """신호등 박스 안에서 몸체(어두운 픽셀)와 켜진 불(밝고 색이 진한 픽셀)만 골라 낸 마스크(박스 크기)."""
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    v, s = hsv[..., 2].astype(np.float32), hsv[..., 1].astype(np.float32)
    bg = np.percentile(v, 90)
    fg = (v < min(90, bg * 0.6)) | ((s > 90) & (v > 120))
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(fg, 8)   # 가장 큰 덩어리 + 구멍 메우기
    if n <= 1:
        return None
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    m = (lab == k).astype(np.uint8)
    # 구멍 메우기: 바깥(박스 밖 테두리 1px)에서 닿지 않는 곳은 전부 안쪽. 사진 끝에 잘린 신호등은 그쪽 테두리를
    # 몸체로 쳐서, 잘린 쪽으로 열린 구멍(하얗게 날아간 불빛 가운데 등)도 메움
    H, W = img.shape[:2]
    p = np.pad(m, 1)
    if y0 <= 0: p[0, :] = 1
    if y1 >= H: p[-1, :] = 1
    if x0 <= 0: p[:, 0] = 1
    if x1 >= W: p[:, -1] = 1
    p[0, 0] = p[0, -1] = p[-1, 0] = p[-1, -1] = 0
    outside = np.zeros((p.shape[0] + 2, p.shape[1] + 2), np.uint8)
    flood = p.copy()
    for sy, sx in ((0, 0), (0, p.shape[1] - 1), (p.shape[0] - 1, 0), (p.shape[0] - 1, p.shape[1] - 1)):
        if flood[sy, sx] == 0:
            cv2.floodFill(flood, outside, (sx, sy), 2)
    return (flood != 2)[1:-1, 1:-1]


REAL_ALL = ("match", "persp", "screen", "jitter")


def _noise_sigma(gray):
    """Immerkær(1996) 빠른 노이즈 표준편차 추정 (회색조, 0~255)."""
    if gray.shape[0] < 4 or gray.shape[1] < 4:
        return 2.0
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    c = cv2.filter2D(gray.astype(np.float32), -1, k)[1:-1, 1:-1]
    return float(np.sqrt(np.pi / 2) * np.abs(c).sum() / (6 * c.size))


def _reblur_ratio(gray, mask=None):
    """선명도: 센 경계(기울기 상위 10%)에서 σ=1로 한 번 더 흐렸을 때 기울기가 남는 비율. 흐린 사진일수록 1에 가까움."""
    g = gray.astype(np.float32)
    gm = lambda x: cv2.magnitude(cv2.Sobel(x, cv2.CV_32F, 1, 0), cv2.Sobel(x, cv2.CV_32F, 0, 1))
    g0, g1 = gm(g), gm(cv2.GaussianBlur(g, (0, 0), 1.0))
    v = g0[mask] if mask is not None else g0.ravel()
    if v.size < 20:
        return None
    t = np.percentile(v, 90)
    sel = (g0 >= t) & (mask if mask is not None else True)
    return float(g1[sel].sum() / max(g0[sel].sum(), 1e-6)) if sel.sum() >= 10 and t > 4 else None


def add_monitor(img, da, ll, sl, slv, boxes, rng, dots_p=0.0, real=(), dots=None):
    """boxes: (N,5) 원본 좌표 [x1,y1,x2,y2,cls]. 성공하면 True (이미지·마스크를 제자리에서 바꿈).
    dots_p: 화면을 단색 대신 땡땡이(켜진 불과 다른 색의 불빛 크기 원)로 그릴 확률. dots=True/False면 확률 대신 고정.
    real: 실제 카메라처럼 보이게 하는 효과 (REAL_ALL 중 고름). 효과를 켜고 꺼도 모니터 위치·크기·색은 같다.
      match  신호등 주변 사진의 선명도·노이즈를 재서 모니터도 같은 정도로 흐리게 하고 노이즈를 넣음
      persp  좌우 ±25° 비스듬히 본 사다리꼴(+ 약간 기울기)
      screen 비네팅, 좌우 밝기 차이, 반사 줄, 밝은 색이 하얗게 날아감(자체 발광 화면을 카메라로 찍은 것처럼)
      jitter 땡땡이 원의 위치·크기·밝기를 조금씩 흔들고, 원 간격(촘촘함) 범위를 넓힘"""
    tls = [b for b in boxes if int(b[4]) == TL_CLS and (b[2] - b[0]) >= 6]
    if not tls:
        return False
    b = tls[int(rng.integers(len(tls)))]
    H, W = img.shape[:2]
    R = {k: np.random.default_rng(int(s)) for k, s in zip(REAL_ALL, rng.integers(1 << 31, size=len(REAL_ALL)))}
    tw, th = b[2] - b[0], b[3] - b[1]
    mw = tw * rng.uniform(3.5, 6.0); mh = mw * 9 / 16; bz = max(1.0, mw * 0.035)
    cx = (b[0] + b[2]) / 2 + rng.uniform(-0.3, 0.3) * mw
    bot = b[3] - th * rng.uniform(0.0, 0.25)           # 화면 아래쪽이 신호등 아래 근처
    x0, x1, y0, y1 = cx - mw / 2 - bz, cx + mw / 2 + bz, bot - mh - 2 * bz, bot
    ow_, oh_ = x1 - x0, y1 - y0                         # 테두리 포함 화면 바깥 크기
    cyr = (y0 + y1) / 2
    theta = R["persp"].uniform(-25, 25) if "persp" in real else 0.0
    roll = R["persp"].uniform(-2, 2) if "persp" in real else 0.0
    hw = ow_ / 2 * np.cos(np.radians(theta)); s_ = 0.3 * np.sin(np.radians(theta)); hh = oh_ / 2
    dst = np.array([[cx - hw, cyr - hh * (1 + s_)], [cx + hw, cyr - hh * (1 - s_)],
                    [cx + hw, cyr + hh * (1 - s_)], [cx - hw, cyr + hh * (1 + s_)]], np.float32)
    if roll:
        c_, s2 = np.cos(np.radians(roll)), np.sin(np.radians(roll))
        dst = (dst - [cx, cyr]) @ np.array([[c_, s2], [-s2, c_]], np.float32) + [cx, cyr]
    dst = dst.astype(np.float32)
    if dst[:, 0].max() < 0 or dst[:, 0].min() > W or dst[:, 1].min() < 0:   # 화면 위로 잘리면(아주 가까운 신호등) 안 그림
        return False
    mm = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(mm, np.round(dst * 16).astype(np.int32), 1, cv2.LINE_8, 4)
    mm = mm > 0
    if mm.sum() < 16:
        return False
    y1 = int(round(float(dst[2:, 1].mean())))          # 받침대가 붙는 화면 아래 가운데
    # 받침대: 신호등과 같은 방향 아래로(화면 아래 ~ 바닥). 바닥 위치는 신호등 기둥 끝 근처로 근사
    sw = max(2, int(mw * 0.05)); sx = int(cx)
    ground = int(min(H - 1, b[3] + th * rng.uniform(0.6, 1.4)))
    stand = np.zeros((H, W), bool); stand[max(0, y1 - 1):ground, max(0, sx - sw // 2):min(W, sx + sw // 2 + 1)] = True
    if ((mm | stand) & (da > 0)).sum() > 0.05 * max((mm | stand).sum(), 1):
        return False
    # 다른 라벨 물체(표지판·콘)를 가리면 라벨이 틀어지므로 그리지 않음
    for o in boxes:
        if int(o[4]) not in LIGHT_CLS and (mm | stand)[int(max(0, o[1])):int(min(H, o[3])), int(max(0, o[0])):int(min(W, o[2]))].any():
            return False
    # 모든 신호등 몸체·기둥은 앞에 남김 (고른 신호등 말고 옆 신호등도 가려지지 않게)
    keep = np.zeros((H, W), bool)
    for o in boxes:
        if int(o[4]) != TL_CLS:
            continue
        ox0, oy0, ox1, oy1 = [int(round(np.clip(v, 0, lim))) for v, lim in zip(o[:4], (W, H, W, H))]
        if ox1 <= ox0 or oy1 <= oy0:
            continue
        fg = tl_foreground(img, (ox0, oy0, ox1, oy1))
        if fg is not None:
            keep[oy0:oy0 + fg.shape[0], ox0:ox0 + fg.shape[1]] = fg
        else:
            keep[oy0:oy1, ox0:ox1] = True
        ow, oh = ox1 - ox0, oy1 - oy0
        og = int(min(H - 1, oy1 + (ground - b[3] if np.array_equal(o, b) else oh)))
        pw = max(2, int(ow * 0.18)); pcx = (ox0 + ox1) // 2
        keep[oy1:og, max(0, pcx - pw // 2):min(W, pcx + pw // 2 + 1)] = True   # 신호등 기둥
    for o in boxes:   # 불빛 라벨 박스는 안에 그린 타원만큼 무조건 앞에 (하얗게 날아간 불빛 가운데가 뚫리지 않게)
        if int(o[4]) in LAMP_COLOR:
            lc = (int(round((o[0] + o[2]) / 2 * 16)), int(round((o[1] + o[3]) / 2 * 16)))
            ax = (max(1, int((o[2] - o[0]) / 2 * 16)), max(1, int((o[3] - o[1]) / 2 * 16)))
            e = np.zeros((H, W), np.uint8)
            cv2.ellipse(e, lc, ax, 0, 0, 360, 1, -1, cv2.LINE_8, 4)
            keep |= e > 0
    orig = img.copy()
    light = _light(img, b[:4])
    col = np.array(COLORS[int(rng.integers(len(COLORS)))], np.float32) * rng.uniform(0.7, 1.05)
    # 화면을 2배 크기 캔버스(정면, 테두리 포함)에 그린 뒤 dst 네 꼭짓점으로 원근 변환해 붙임
    S = 2.0
    cw, ch = max(4, int(np.ceil(ow_ * S))), max(4, int(np.ceil(oh_ * S)))
    cv_ = np.empty((ch, cw, 3), np.float32); cv_[:] = np.array([22, 22, 24], np.float32) * light * 2   # 테두리
    bzs = max(1, int(round(bz * S)))
    iy0, iy1, ix0, ix1 = bzs, ch - bzs, bzs, cw - bzs
    is_dots = (rng.random() < dots_p) if dots is None else bool(dots)
    if is_dots:   # 땡땡이: 바탕은 어둡거나 흰 화면, 원은 지금 켜진 불과 다른 색
        col = np.array([(20, 20, 20), (30, 30, 34), (235, 235, 235)][int(rng.integers(3))], np.float32)
    grad = 0.15 * rng.random()
    if ix1 > ix0 and iy1 > iy0:
        yy = np.linspace(0, 1, iy1 - iy0)[:, None, None]
        cv_[iy0:iy1, ix0:ix1] = col * (1 - grad * yy) + light * 8   # 단색 화면(자체 발광)
        if is_dots:
            lamps = [o for o in boxes if int(o[4]) in LAMP_COLOR and b[0] - 2 <= (o[0] + o[2]) / 2 <= b[2] + 2
                     and b[1] - 2 <= (o[1] + o[3]) / 2 <= b[3] + 2]
            lit = int(lamps[0][4]) if lamps else None
            others = [c for c in LAMP_COLOR if c != lit]
            dc = np.array(LAMP_COLOR[others[int(rng.integers(len(others)))]], np.float32) * rng.uniform(0.8, 1.05)
            lr = (lamps[0][2] - lamps[0][0]) / 2 if lamps else tw * 0.3   # 원 크기는 신호등 불빛 크기 근처
            r = max(1.5, lr * rng.uniform(0.7, 1.4)) * S
            step = r * rng.uniform(2.3, 3.6)
            ox, oy = rng.uniform(0, step, 2)
            stagger = rng.random() < 0.5   # 엇갈린 줄 또는 바둑판
            halo_k = rng.uniform(0.2, 0.5)
            jt = R["jitter"] if "jitter" in real else None
            if jt is not None:
                step = r * jt.uniform(2.6, 5.0)
            dm = np.zeros((ch, cw), np.float32)
            for k, yc in enumerate(np.arange(iy0 + oy, iy1 + step, step)):
                xs = ox + (step / 2 if stagger and k % 2 else 0)
                for xc in np.arange(ix0 + xs - step, ix1 + step, step):
                    rr, px, py, br = r, xc, yc, 1.0
                    if jt is not None:   # 이웃 원과 겹치지 않는 범위에서 흔듦
                        px += np.clip(jt.normal(0, 0.05 * step), -0.08 * step, 0.08 * step)
                        py += np.clip(jt.normal(0, 0.05 * step), -0.08 * step, 0.08 * step)
                        rr = r * jt.uniform(0.9, 1.1); br = jt.uniform(0.8, 1.05)
                    cv2.circle(dm, (int(px * 16), int(py * 16)), max(1, int(rr * 16)), float(br), -1, cv2.LINE_AA, 4)
            inner = np.zeros_like(dm); inner[iy0:iy1, ix0:ix1] = 1
            dm *= inner
            halo = cv2.GaussianBlur(dm, (0, 0), max(1.0, r * 0.6)) * halo_k * inner   # 불빛처럼 번짐
            dm = np.clip(dm + halo, 0, 1)[..., None]
            cv_ = cv_ * (1 - dm) + (dc + light * 8) * dm
        if "screen" in real:   # 카메라로 찍은 자체 발광 화면처럼
            q = R["screen"]
            u = np.linspace(-1, 1, ix1 - ix0)[None, :]; w_ = np.linspace(-1, 1, iy1 - iy0)[:, None]
            f = 1 - q.uniform(0.1, 0.35) * (u ** 2 + w_ ** 2) / 2              # 비네팅
            f = f * (1 + q.uniform(-0.15, 0.15) * u)                            # 보는 각도에 따른 좌우 밝기 차이
            scr = cv_[iy0:iy1, ix0:ix1] * f[..., None]
            if q.random() < 0.6:                                               # 비스듬한 반사 줄
                ang = np.radians(q.uniform(30, 70) * q.choice([-1, 1]))
                d = u * np.cos(ang) + w_ * np.sin(ang) - q.uniform(-0.8, 0.8)
                band = np.exp(-(d / q.uniform(0.08, 0.25)) ** 2)
                scr = scr + (255 * light.mean() - scr) * (q.uniform(0.06, 0.2) * band)[..., None]
            scr = scr * q.uniform(1.0, 1.15 if is_dots else 1.5)   # 노출: 밝은 색은 하얗게 날아감 (땡땡이는 원이 불빛처럼 보이게 약하게)
            ex = np.clip(scr.max(2, keepdims=True) - 255, 0, None)
            cv_[iy0:iy1, ix0:ix1] = scr + ex * 0.7
    src = np.array([[0, 0], [cw, 0], [cw, ch], [0, ch]], np.float32)
    Hm = cv2.getPerspectiveTransform(src, dst)
    warp = lambda x: cv2.warpPerspective(x, Hm, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    scr_full = warp(np.ascontiguousarray(cv_))
    a_scr = warp(np.ones((ch, cw), np.float32))
    layer = img.astype(np.float32)
    layer = layer * (1 - a_scr[..., None]) + scr_full
    layer[stand & (a_scr < 0.5)] = np.array([35, 35, 35], np.float32) * light * 2
    a0 = np.maximum(a_scr, stand.astype(np.float32))
    sig = rng.uniform(0.5, 1.1)
    glow_k = rng.uniform(0.03, 0.12)
    nsig = 2.0
    if "match" in real:   # 신호등 주변 사진의 선명도·노이즈에 맞춤
        bx0, by0 = int(max(0, dst[:, 0].min() - 0.3 * ow_)), int(max(0, dst[:, 1].min() - 0.3 * oh_))
        bx1, by1 = int(min(W, dst[:, 0].max() + 0.3 * ow_)), int(min(H, ground + 2))
        g0 = cv2.cvtColor(orig[by0:by1, bx0:bx1], cv2.COLOR_BGR2GRAY)
        tgt = _reblur_ratio(g0)
        if tgt is not None:
            band = (a0[by0:by1, bx0:bx1] > 0.02)
            best_s, best_d = sig, 9.0
            for s in (0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 2.6, 3.2):
                aa = cv2.GaussianBlur(a0[by0:by1, bx0:bx1], (0, 0), s)[..., None]
                cmp_ = orig[by0:by1, bx0:bx1].astype(np.float32) * (1 - aa) + cv2.GaussianBlur(layer[by0:by1, bx0:bx1], (0, 0), s) * aa
                rr = _reblur_ratio(cv2.cvtColor(np.clip(cmp_, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY), band)
                if rr is not None and abs(rr - tgt) < best_d:
                    best_s, best_d = s, abs(rr - tgt)
            sig = best_s
        ring = g0.copy()
        nsig = float(np.clip(_noise_sigma(ring), 0.5, 12.0))
    a = cv2.GaussianBlur(a0, (0, 0), sig)[..., None]
    out = img.astype(np.float32) * (1 - a) + cv2.GaussianBlur(layer, (0, 0), sig) * a
    mmf = (a_scr > 0.5).astype(np.float32)
    glow = cv2.GaussianBlur(mmf, (0, 0), max(2.0, mw * 0.08))[..., None]
    out += glow * col * glow_k
    if "match" in real:   # 밝기·색 노이즈를 주변과 같은 세기로, 센서처럼 약간 뭉친 알갱이
        q = R["match"]
        nz = q.normal(0, 1, (H, W, 1)) + 0.35 * q.normal(0, 1, (H, W, 3))
        nz = cv2.GaussianBlur(nz.astype(np.float32), (0, 0), 0.6)
        nz = nz / max(float(nz.std()), 1e-6) * nsig
        out += nz * a
    else:
        out += rng.normal(0, nsig, out.shape) * (a > 0.01)
    out = np.clip(out, 0, 255).astype(np.uint8)
    out[keep] = orig[keep]                                                            # 신호등을 다시 앞으로
    img[:] = out
    mm = a_scr > 0.5
    cover = (mm | stand) & ~keep
    da[cover] = 0; ll[cover] = 0; sl[cover] = 0
    return True


def add_cable(img, sl, slv, rng, vh=None):
    """흰 케이블 1개. 그린 곳 주변은 sl=0, slv=1 (흰 가로선 아님을 믿을 수 있음)."""
    H, W = img.shape[:2]
    vh = vh if vh is not None else rng.uniform(0.36, 0.47) * H
    y_top = int(np.clip(vh + 0.12 * (H - vh), 0, H - 1))
    p = np.array([[rng.uniform(-0.2, 0.3) * W, rng.uniform(y_top, H)], [rng.uniform(0, 1) * W, rng.uniform(y_top, H)],
                  [rng.uniform(0, 1) * W, rng.uniform(y_top, H)], [rng.uniform(0.7, 1.2) * W, rng.uniform(y_top, H)]])
    if rng.random() < 0.5:
        p = p[::-1]
    t = np.linspace(0, 1, 200)[:, None]
    pts = (1 - t) ** 3 * p[0] + 3 * (1 - t) ** 2 * t * p[1] + 3 * (1 - t) * t ** 2 * p[2] + t ** 3 * p[3]
    m = np.zeros((H, W), np.float32)
    base = rng.uniform(0.006, 0.012)
    for a_, b_ in zip(pts[:-1], pts[1:]):
        th = max(1, int(base * H * (b_[1] - vh) / max(H - vh, 1) * 2))
        cv2.line(m, tuple(int(v) for v in a_), tuple(int(v) for v in b_), 1.0, th, cv2.LINE_AA)
    m = cv2.GaussianBlur(m, (0, 0), 0.7)
    col = np.array(random.choice([(235, 235, 235), (215, 220, 225), (200, 200, 200), (230, 230, 210)]), np.float32)
    light = _light(img, (0, y_top, W, H))
    shadow = cv2.GaussianBlur(np.roll(m, 2, 0), (0, 0), 2) * 0.35
    f = img.astype(np.float32) * (1 - shadow[..., None])
    f = f * (1 - m[..., None]) + (col * light)[None, None] * m[..., None]
    img[:] = np.clip(f, 0, 255).astype(np.uint8)
    neg = cv2.dilate((m > 0.2).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    sl[neg] = 0
    slv[neg] = 1
    return True
