#!/usr/bin/env python3
"""주행 프레임에 표지판·신호등을 원근에 맞춰 합성하고 YOLO 박스 라벨을 만든다.

실물 표지판·신호등이 아직 없어서 규정집(Autorace2026_규정집.pdf 11~13쪽) 그림을 붙인다.
원근은 LIMO 카메라 보정값(fx=915.6 @1280, 높이 0.19m)으로 계산하고, 장면 조명 색을 곱해서
붉은 조명이면 표지판도 붉게 물든다. 켜진 신호등 불은 자체 발광이라 조명 영향을 적게 받는다.

균형
  - 배경 풀 2개: venue(11/06 대회장 + 11/07 bag, 붉은·파란·어두운 조명) / general(10/27 트랙)
  - 클래스마다 두 풀에서 정확히 반씩 나오도록 "가장 모자란 (풀, 클래스)"부터 채운다.
다양화
  - 거리 0.4~3.0m 균등, 가로 위치는 화면 전체(가장자리 밖 일부 포함) 또는 차선 옆
  - 화면 가장자리 잘림, 일부 가림(같은 프레임 다른 부분으로 덮음), 회전(yaw 0~40°, roll ±5°)
  - 흐림·모션블러·노이즈·JPEG 품질 랜덤, 노출 ±15%

클래스(classes.txt 순서)
  0 pedestrian 1 sign_A 2 sign_B 3 bump 4 slow 5 parking
  6 traffic_light(몸체) 7 light_red 8 light_yellow 9 light_green(켜진 불) 10 limo(실사진용 예약) 11 cone(공개 데이터, prepare_cones.py)

출력 <out>/images/*.jpg, <out>/labels/*.txt (YOLO: cls cx cy w h, 0~1), meta.csv, classes.txt
사용 예:
  .venv/Scripts/python scripts/synth_signs.py --assets <assets 폴더> --out dataset/synth_signs_v1 --per-class 1500
"""
import argparse, csv, glob, json, os, random
from multiprocessing import Pool

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSES = ["pedestrian", "sign_A", "sign_B", "bump", "slow", "parking",
           "traffic_light", "light_red", "light_yellow", "light_green", "limo", "cone"]
SIGNS = ["pedestrian", "sign_A", "sign_B", "bump", "slow", "parking"]
LIGHTS = ["light_red", "light_yellow", "light_green"]
TARGETS = SIGNS + LIGHTS                      # 균형을 맞추는 단위 (신호등은 켜진 불 기준)
SIGN_W = {"pedestrian": 0.10, "bump": 0.10}   # 삼각형 판면 폭(m), 나머지 원형 Ø0.09
FX1280, CX1280, CAM_H = 915.6, 640.7, 0.19   # tools/camera_calib/result/limo.yaml, 실측 높이
TL_W = 0.12                                   # 신호등 폭(차양 포함) 12cm

POOLS = {
    "venue": ["dataset/raw_20261106/session_180809/images/*.jpg",
              "dataset/bags_20261107_analysis/*/frames/*.jpg"],
    "general": ["dataset/LimoPro_frames_20261027/session_183234/*.jpg"],
}


# ---------------------------------------------------------------- 자산
def load_assets(adir):
    A = {}
    for n in SIGNS:
        A[n] = cv2.imread(os.path.join(adir, n + ".png"), cv2.IMREAD_UNCHANGED)
    for st in ["red", "yellow", "green"]:
        A["tl_" + st] = cv2.imread(os.path.join(adir, f"tl_{st}.png"), cv2.IMREAD_UNCHANGED)
    A["tl_meta"] = json.load(open(os.path.join(adir, "tl_meta.json")))
    return A


# ---------------------------------------------------------------- 기하
def warp_asset(rgba, tw, th, yaw, roll, rng):
    """자산을 tw x th로 줄이고 yaw(한쪽 변이 짧아짐)·roll 회전. (이미지, 3x3 H) 반환.
    H는 자산 원본 좌표 → 결과 이미지 좌표."""
    ah, aw = rgba.shape[:2]
    c = np.cos(yaw)
    shrink = 1 - 0.18 * abs(np.sin(yaw))      # 먼 쪽 변 높이 비율
    w2 = tw * c
    src = np.float32([[0, 0], [aw, 0], [aw, ah], [0, ah]])
    if yaw >= 0:   # 오른쪽 변이 멀다
        dst = np.float32([[0, 0], [w2, th * (1 - shrink) / 2], [w2, th * (1 + shrink) / 2], [0, th]])
    else:
        dst = np.float32([[0, th * (1 - shrink) / 2], [w2, 0], [w2, th], [0, th * (1 + shrink) / 2]])
    # roll: 중심 기준 회전
    ctr = dst.mean(0)
    R = cv2.getRotationMatrix2D(tuple(ctr.tolist()), np.degrees(roll), 1.0)
    dst = (np.hstack([dst, np.ones((4, 1), np.float32)]) @ R.T).astype(np.float32)
    pad = 2
    dst -= dst.min(0) - pad
    W = int(np.ceil(dst[:, 0].max())) + pad
    Hh = int(np.ceil(dst[:, 1].max())) + pad
    H = cv2.getPerspectiveTransform(src, dst)
    # 큰 축소는 2단계(먼저 INTER_AREA로 2배 크기까지) 해서 계단 현상 방지
    scale = max(tw / aw, th / ah)
    if scale < 0.5:
        k = min(1.0, scale * 2)
        pre = cv2.resize(rgba, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        S = np.diag([1 / k, 1 / k, 1]).astype(np.float64)
        out = cv2.warpPerspective(pre, H @ S, (W, Hh), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
    else:
        out = cv2.warpPerspective(rgba, H, (W, Hh), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
    return out, H


def apply_H(H, pts):
    p = np.hstack([np.float32(pts), np.ones((len(pts), 1), np.float32)]) @ H.T
    return p[:, :2] / p[:, 2:3]


# ---------------------------------------------------------------- 합성
def scene_light(frame, box, rng):
    """물체 주변(4배)과 화면 전체의 밝은 쪽(92%) 색 = 그 자리 조명. 붉은 조명이면 R이 커진다."""
    x0, y0, x1, y1 = [int(v) for v in box]
    w, h = max(1, x1 - x0), max(1, y1 - y0)
    Hh, W = frame.shape[:2]
    reg = frame[max(0, y0 - 2 * h):min(Hh, y1 + 2 * h), max(0, x0 - 2 * w):min(W, x1 + 2 * w)]
    if reg.size < 30:
        reg = frame
    loc = np.percentile(reg.reshape(-1, 3), 92, axis=0)
    glob_ = np.percentile(frame[::4, ::4].reshape(-1, 3), 92, axis=0)
    light = (0.7 * loc + 0.3 * glob_) / 225.0
    return np.clip(light * rng.uniform(0.85, 1.15), 0.12, 1.25)


def draw_object(frame, obj, rng, omask=None):
    """frame(uint8)에 obj를 그린다. 라벨 박스 목록(cls, x0,y0,x1,y1, 전체크기 대비 보이는 비율) 반환."""
    Hh, W = frame.shape[:2]
    f = W / 1280.0
    fx, cx = FX1280 * f, CX1280 * f
    vh, d, u = obj["vh"] * f, obj["d"], obj["u"] * f
    ppm = fx / d
    x_m = (u - cx) / ppm
    if obj["kind"] == "sign":
        rgba = obj["asset"]
        tw = SIGN_W.get(obj["name"], 0.09) * ppm
        th = tw * rgba.shape[0] / rgba.shape[1]
        h_center = obj["h"]
        cy = vh + fx * (CAM_H - h_center) / d
        pole_top = cy
        pole_w = 0.008 * ppm
    else:
        rgba = obj["asset"]
        tw = TL_W * ppm
        th = tw * rgba.shape[0] / rgba.shape[1]
        h_bottom = obj["h"]
        cy = vh + fx * (CAM_H - h_bottom) / d - th / 2
        pole_top = cy + th / 2 - 2
        pole_w = 0.016 * ppm
    v_ground = vh + fx * CAM_H / d

    warped, Hm = warp_asset(rgba, tw, th, obj["yaw"], obj["roll"], rng)
    wh, ww = warped.shape[:2]
    ox, oy = int(round(u - ww / 2)), int(round(cy - wh / 2))

    # 그릴 영역(ROI): 물체 + 지주 + 그림자 + 번짐 여유
    pad = int(0.06 * ppm) + 8
    X0, Y0 = max(0, ox - pad), max(0, oy - pad)
    X1, Y1 = min(W, ox + ww + pad), min(Hh, int(v_ground) + pad)
    if X1 <= X0 + 2 or Y1 <= Y0 + 2:
        return []
    roi = frame[Y0:Y1, X0:X1].astype(np.float32)
    rh, rw = roi.shape[:2]
    layer = np.zeros((rh, rw, 3), np.float32)
    alpha = np.zeros((rh, rw), np.float32)
    lit = np.zeros((rh, rw), np.float32)

    # 바닥 그림자
    sh = np.zeros((rh, rw), np.float32)
    cv2.ellipse(sh, (int(u + 0.02 * ppm - X0), int(v_ground - Y0)),
                (max(1, int(0.05 * ppm)), max(1, int(0.012 * ppm))), 0, 0, 360, 1, -1)
    sh = cv2.GaussianBlur(sh, (0, 0), max(0.5, 0.01 * ppm)) * rng.uniform(0.15, 0.4)
    roi *= (1 - sh[..., None])

    # 지주
    pc = obj["pole_col"]
    p0 = (int(u - pole_w / 2 - X0), int(pole_top - Y0))
    p1 = (int(np.ceil(u + pole_w / 2)) - X0, int(v_ground - Y0))
    if p1[1] > p0[1]:
        cv2.rectangle(layer, p0, p1, pc, -1)
        cv2.rectangle(alpha, p0, p1, 1.0, -1)

    # 물체
    a = warped[..., 3].astype(np.float32) / 255
    sx0, sy0 = ox - X0, oy - Y0
    tx0, ty0 = max(0, sx0), max(0, sy0)
    tx1, ty1 = min(rw, sx0 + ww), min(rh, sy0 + wh)
    if tx1 <= tx0 or ty1 <= ty0:
        return []
    ws = (slice(ty0 - sy0, ty1 - sy0), slice(tx0 - sx0, tx1 - sx0))
    rs = (slice(ty0, ty1), slice(tx0, tx1))
    layer[rs] = warped[ws][..., :3] * a[ws][..., None] + layer[rs] * (1 - a[ws][..., None])
    alpha[rs] = np.maximum(alpha[rs], a[ws])

    labels = []
    def add_box(cls, pts):
        p = apply_H(Hm, pts) + [ox, oy]
        bx0, by0 = p[:, 0].min(), p[:, 1].min()
        bx1, by1 = p[:, 0].max(), p[:, 1].max()
        full = (bx1 - bx0) * (by1 - by0)
        cx0, cy0, cx1, cy1 = max(0, bx0), max(0, by0), min(W, bx1), min(Hh, by1)
        if cx1 <= cx0 or cy1 <= cy0:
            return
        labels.append([cls, cx0, cy0, cx1, cy1, (cx1 - cx0) * (cy1 - cy0) / max(full, 1e-6)])

    ah_, aw_ = rgba.shape[:2]
    if obj["kind"] == "sign":
        ys, xs = np.where(rgba[..., 3] > 127)
        add_box(obj["name"], [[xs.min(), ys.min()], [xs.max(), ys.min()], [xs.max(), ys.max()], [xs.min(), ys.max()]])
    else:
        add_box("traffic_light", [[0, 0], [aw_, 0], [aw_, ah_], [0, ah_]])
        m = obj["tl_meta"]
        lc, r = m["lamps"][obj["state"]], m["lamp_r"]
        add_box("light_" + obj["state"], [[lc[0] - r, lc[1] - r], [lc[0] + r, lc[1] - r],
                                          [lc[0] + r, lc[1] + r], [lc[0] - r, lc[1] + r]])
        # 켜진 불 마스크 (자체 발광)
        lm = np.zeros((ah_, aw_), np.uint8)
        cv2.circle(lm, (int(lc[0]), int(lc[1])), int(r), 255, -1)
        lmw = cv2.warpPerspective(lm, Hm, (ww, wh)).astype(np.float32) / 255
        lit[rs] = lmw[ws]

    # 조명: 반사체는 조명 색을 곱하고, 켜진 불은 거의 그대로
    light = scene_light(frame, (ox, oy, ox + ww, oy + wh), rng)
    emis = rng.uniform(0.9, 1.1)
    layer = layer * light * (1 - lit[..., None]) + np.clip(layer * emis, 0, 255) * lit[..., None]

    # 카메라 흐림(+모션) : 물체와 알파를 같이
    sig = rng.uniform(0.5, 1.4) * max(0.6, f)
    k = int(rng.integers(1, max(2, int(6 * f))))
    pre = layer * alpha[..., None]
    pre = cv2.GaussianBlur(pre, (0, 0), sig)
    al = cv2.GaussianBlur(alpha, (0, 0), sig)
    if k > 1:
        kern = np.zeros((1, k), np.float32) + 1.0 / k
        pre = cv2.filter2D(pre, -1, kern)
        al = cv2.filter2D(al, -1, kern)
    col = np.where(al[..., None] > 1e-3, pre / np.maximum(al[..., None], 1e-3), 0)
    out = roi * (1 - al[..., None]) + col * al[..., None]

    # 불 번짐
    if lit.any():
        g = cv2.GaussianBlur(lit, (0, 0), max(1.0, 0.25 * np.sqrt(lit.sum() / np.pi)))
        lcol = (layer * lit[..., None]).sum((0, 1)) / max(lit.sum(), 1)
        out += g[..., None] * lcol * rng.uniform(0.3, 0.7)

    # 노이즈
    out += rng.normal(0, rng.uniform(1.5, 4.5), out.shape) * (al[..., None] > 0.01)
    frame[Y0:Y1, X0:X1] = np.clip(out, 0, 255).astype(np.uint8)
    if omask is not None:   # 물체가 덮은 픽셀(분할 마스크에서 지울 곳)
        omask[Y0:Y1, X0:X1] |= al > 0.5
    return labels


def add_occluder(frame, o, box, rng, A, omask=None):
    """o(이미 그린 물체)의 박스 box 일부를 카메라 쪽 더 가까운 실제 물체로 가린다.
    절반은 다른 표지판/신호등(라벨도 붙음), 절반은 가까운 기둥(라벨 없음, 초점 밖이라 더 흐림).
    가린 물체의 박스는 전체 크기를 유지한다. (추가 라벨 목록, 가림 종류) 반환."""
    Hh, W = frame.shape[:2]
    f = W / 1280.0
    fx = FX1280 * f
    vh = o["vh"] * f
    bx0, by0, bx1, by1 = box
    bw, bh = bx1 - bx0, by1 - by0
    frac = o["occlude"]
    left = rng.random() < 0.5
    d2 = o["d"] * rng.uniform(0.45, 0.8)
    edge = bx0 + frac * bw if left else bx1 - frac * bw      # 가림이 끝나는 x
    if rng.random() < 0.5:
        t2 = random.choice(TARGETS)
        o2 = sample_object(t2, rng, A)
        o2.update(d=d2, occlude=0.0, vh=o["vh"])
        wphys = SIGN_W.get(t2, 0.09) if t2 in SIGNS else TL_W
        w2 = wphys * fx / d2 * np.cos(o2["yaw"])
        o2["u"] = float((edge - w2 / 2 if left else edge + w2 / 2) / f)
        cy_t = (by0 + by1) / 2 + rng.uniform(-0.3, 0.3) * bh
        h = CAM_H - (cy_t - vh) * d2 / fx                     # 그 높이에 판면 중심이 오도록
        if t2 not in SIGNS:
            h -= TL_W * A["tl_" + t2.split("_")[1]].shape[0] / A["tl_" + t2.split("_")[1]].shape[1] / 2
        o2["h"] = float(np.clip(h, 0.02, 0.5))
        return draw_object(frame, o2, rng, omask), "object:" + t2
    # 가까운 기둥
    ppm = fx / d2
    pw = max(2.0, rng.uniform(0.015, 0.04) * ppm)
    cxp = edge - pw / 2 if left else edge + pw / 2
    top = vh + fx * (CAM_H - rng.uniform(0.35, 0.8)) / d2
    bot = vh + fx * CAM_H / d2
    X0, X1 = int(max(0, cxp - pw / 2 - 12)), int(min(W, cxp + pw / 2 + 12))
    Y0, Y1 = int(max(0, top - 12)), int(min(Hh, bot + 12))
    if X1 <= X0 + 2 or Y1 <= Y0 + 2:
        return [], "pole"
    roi = frame[Y0:Y1, X0:X1].astype(np.float32)
    rh, rw = roi.shape[:2]
    al = np.zeros((rh, rw), np.float32)
    cv2.rectangle(al, (int(cxp - pw / 2 - X0), int(top - Y0)), (int(cxp + pw / 2 - X0), int(bot - Y0)), 1.0, -1)
    col = np.array(random.choice([(40, 40, 40), (90, 90, 90), (200, 200, 200), (230, 230, 225), (60, 70, 120)]), np.float32)
    xs = (np.arange(rw) - (cxp - X0)) / max(pw / 2, 1)          # 원기둥 음영
    shade = np.clip(1.15 - 0.45 * xs ** 2, 0.5, 1.2)[None, :, None]
    light = scene_light(frame, (cxp - pw, top, cxp + pw, bot), rng)
    layer = np.broadcast_to(col * light, (rh, rw, 3)) * shade
    sig = rng.uniform(1.2, 3.0) * max(0.6, f)                   # 가까워서 초점 밖
    pre = cv2.GaussianBlur(layer * al[..., None], (0, 0), sig)
    a2 = cv2.GaussianBlur(al, (0, 0), sig)[..., None]
    c2 = np.where(a2 > 1e-3, pre / np.maximum(a2, 1e-3), 0)
    out = roi * (1 - a2) + c2 * a2 + rng.normal(0, 2.5, roi.shape) * (a2 > 0.01)
    frame[Y0:Y1, X0:X1] = np.clip(out, 0, 255).astype(np.uint8)
    if omask is not None:
        omask[Y0:Y1, X0:X1] |= a2[..., 0] > 0.5
    return [], "pole"


# ---------------------------------------------------------------- 계획
def sample_object(target, rng, A):
    """target 클래스 물체 하나의 배치 파라미터."""
    o = {"d": float(rng.uniform(0.4, 3.0)),
         "vh": float(rng.uniform(262, 335)),                 # 1280x720 기준 지평선 (프레임에서 눈대중 범위)
         "yaw": float(np.radians(rng.uniform(-40, 40))),
         "roll": float(np.radians(rng.uniform(-5, 5))),
         "occlude": float(rng.uniform(0.15, 0.45)) if rng.random() < 0.15 else 0.0}
    if rng.random() < 0.7:   # 화면 위치 균등 (가장자리 밖 10% 포함 → 잘림)
        o["u"] = float(rng.uniform(-0.1, 1.1) * 1280)
    else:                    # 차선 옆 실제 위치
        x = rng.uniform(0.3, 0.6) * (1 if rng.random() < 0.5 else -1)
        o["u"] = float(CX1280 + FX1280 * x / o["d"])
    if target in SIGNS:
        o.update(kind="sign", name=target, asset=A[target], h=float(rng.uniform(0.12, 0.26)))
        o["pole_col"] = tuple(float(v) for v in random.choice([(205, 205, 200), (235, 235, 235), (40, 40, 40), (150, 150, 150)]))
    else:
        st = target.split("_")[1]
        o.update(kind="tl", state=st, asset=A["tl_" + st], tl_meta=A["tl_meta"], h=float(rng.uniform(0.05, 0.20)))
        o["pole_col"] = tuple(float(v) for v in random.choice([(30, 30, 32), (60, 60, 60), (200, 200, 200)]))
    return o


def overlap(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy / max(1e-6, min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])))


def make_one(job):
    idx, pool, bg, targets, seed, adir, out = job
    rng = np.random.default_rng(seed)
    random.seed(seed)
    A = load_assets(adir) if "A" not in globals() else globals()["A"]
    frame = cv2.imread(bg)
    labs, metas, pending = [], [], []
    for t in targets:
        for _ in range(8):  # 겹치거나 보이는 부분이 너무 작으면 다시 뽑기
            o = sample_object(t, rng, A)
            trial = frame.copy()
            got = draw_object(trial, o, rng)
            main = [g for g in got if g[0] in (t, "traffic_light")]
            key = [g for g in got if g[0] == t]
            if not key or key[0][5] < 0.35:
                continue
            if any(overlap(g[1:5], l[1:5]) > 0.1 for g in main for l in labs):
                continue
            frame = trial
            labs += got
            if o["occlude"] > 0:   # 가림은 모든 대상을 그린 뒤 마지막에(가까운 물체가 항상 위에 오도록)
                box = [g for g in got if g[0] in ("traffic_light",) or g[0] in SIGNS][0][1:5]
                pending.append((len(metas), o, box))
            metas.append(dict(cls=t, d=round(o["d"], 2), u=round(o["u"]), yaw=round(np.degrees(o["yaw"])),
                              occ=round(o["occlude"], 2), occ_type="", vis=round(key[0][5], 2)))
            break
    for mi, o, box in pending:
        extra, metas[mi]["occ_type"] = add_occluder(frame, o, box, rng, A)
        labs += extra
    q = int(rng.integers(70, 96))
    name = f"{idx:06d}"
    cv2.imwrite(os.path.join(out, "images", name + ".jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, q])
    Hh, W = frame.shape[:2]
    with open(os.path.join(out, "labels", name + ".txt"), "w") as fh:
        for c, x0, y0, x1, y1, vis in labs:
            fh.write(f"{CLASSES.index(c)} {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / Hh:.6f} "
                     f"{(x1 - x0) / W:.6f} {(y1 - y0) / Hh:.6f}\n")
    return name, pool, os.path.relpath(bg, ROOT), W, Hh, q, metas


def init_worker(adir):
    globals()["A"] = load_assets(adir)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assets", required=True, help="make_assets 결과 폴더 (표지판 6종, tl_*.png, tl_meta.json)")
    ap.add_argument("--out", default="dataset/synth_signs_v1")
    ap.add_argument("--per-class", type=int, default=1500, help="클래스별 목표 박스 수 (풀마다 절반)")
    ap.add_argument("--empty", type=int, default=1000, help="물체 없는 배경 이미지 수 (풀마다 절반)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--regen-occluded", action="store_true",
                    help="--out의 기존 데이터에서 가림(occ>0) 이미지만 같은 번호로 다시 만든다")
    args = ap.parse_args()

    os.chdir(ROOT)
    rng = np.random.default_rng(args.seed)
    bgs = {p: sorted(sum([glob.glob(g) for g in gl], [])) for p, gl in POOLS.items()}
    for p, l in bgs.items():
        print(f"pool {p}: {len(l)} frames")
    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "labels"), exist_ok=True)

    # (풀, 클래스)별 남은 개수를 가장 많이 남은 것부터 1~2개씩 묶어 이미지 계획
    need = {(p, t): args.per_class // 2 for p in POOLS for t in TARGETS}
    jobs, idx = [], 0
    while any(v > 0 for v in need.values()):
        p = max(POOLS, key=lambda q: sum(need[(q, t)] for t in TARGETS))
        cand = sorted([t for t in TARGETS if need[(p, t)] > 0], key=lambda t: -need[(p, t)] + rng.random())
        k = 2 if (len(cand) >= 2 and rng.random() < 0.5) else 1
        ts = cand[:k]
        if k == 2 and all(t in LIGHTS for t in ts):   # 신호등 2개는 드묾 → 하나는 표지판으로
            alt = [t for t in cand if t in SIGNS]
            ts = [ts[0]] + alt[:1] if alt else ts[:1]
        for t in ts:
            need[(p, t)] -= 1
        jobs.append((idx, p, str(rng.choice(bgs[p])), ts, int(rng.integers(1 << 31)), args.assets, args.out))
        idx += 1
    for i in range(args.empty):
        p = list(POOLS)[i % 2]
        jobs.append((idx, p, str(rng.choice(bgs[p])), [], int(rng.integers(1 << 31)), args.assets, args.out))
        idx += 1
    keep_rows = []
    if args.regen_occluded:
        # 기존 데이터에서 가림이 들어간 이미지만 같은 번호·배경·클래스로 다시 만든다
        old = list(csv.DictReader(open(os.path.join(args.out, "meta.csv"), encoding="utf-8")))
        jobs = []
        for r in old:
            objs = json.loads(r["objects"])
            if any(ob.get("occ", 0) > 0 for ob in objs):
                jobs.append((int(r["image"]), r["pool"], os.path.join(ROOT, r["background"]),
                             [ob["cls"] for ob in objs], int(rng.integers(1 << 31)), args.assets, args.out))
            else:
                keep_rows.append((r["image"], r["pool"], r["background"], int(r["width"]), int(r["height"]),
                                  int(r["jpeg_q"]), objs))
        print(f"regenerating {len(jobs)} occluded images, keeping {len(keep_rows)}")
    else:
        print(f"planned {len(jobs)} images")

    rows = list(keep_rows)
    with Pool(args.workers, initializer=init_worker, initargs=(args.assets,)) as pool:
        for i, r in enumerate(pool.imap_unordered(make_one, jobs, chunksize=16)):
            rows.append(r)
            if (i + 1) % 1000 == 0:
                print(f"{i + 1}/{len(jobs)}", flush=True)
    if args.regen_occluded:
        print("regenerated:", ",".join(sorted(r[0] for r in rows if r not in keep_rows)[:5]), "...")

    rows.sort()
    with open(os.path.join(args.out, "meta.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "pool", "background", "width", "height", "jpeg_q", "objects"])
        for name, p, bg, W, Hh, q, metas in rows:
            w.writerow([name, p, bg, W, Hh, q, json.dumps(metas, ensure_ascii=False)])
    with open(os.path.join(args.out, "classes.txt"), "w") as fh:
        fh.write("\n".join(CLASSES) + "\n")

    # 실제 들어간 박스 수 (풀 x 클래스)
    cnt = {(p, c): 0 for p in POOLS for c in CLASSES}
    for name, p, *_ in rows:
        for line in open(os.path.join(args.out, "labels", name + ".txt")):
            cnt[(p, CLASSES[int(line.split()[0])])] += 1
    print(f"{'class':14s} " + " ".join(f"{p:>8s}" for p in POOLS))
    for c in CLASSES:
        print(f"{c:14s} " + " ".join(f"{cnt[(p, c)]:8d}" for p in POOLS))
    json.dump({f"{p}/{c}": v for (p, c), v in cnt.items()}, open(os.path.join(args.out, "counts.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
