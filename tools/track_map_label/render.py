"""도면 + 차 위치 -> 사진 라벨(da: 내 차로, ll: 내 차로 경계선). 기존 라벨 규칙: 횡단보도·글자·마름모·정지선은 ll 아님, da는 그 위도 칠함."""
import os, sys, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import config; SP = config.WORK
import cam
RES = 5.0
PAINT = np.load(SP + "/map/paint.npy"); Hm, Wm = PAINT.shape
# 경계 후보: 작은 조각(횡단보도 막대, 글자, 마름모)은 빼되, 점선(가늘고 긴 작은 조각)은 남김
n, lab, st, _ = cv2.connectedComponentsWithStats(PAINT, 8)
BAR = np.zeros_like(PAINT)
for k in range(1, n):
    w, h, a = st[k, cv2.CC_STAT_WIDTH] * RES, st[k, cv2.CC_STAT_HEIGHT] * RES, st[k, cv2.CC_STAT_AREA]
    big = max(w, h) > 900
    dash = 100 < max(w, h) < 400 and min(w, h) < 45
    if big or dash: BAR[lab == k] = 1
# 차로 안쪽 표시(횡단보도, 마름모+글자, 출발 체커, A/B 글자) 영역: 도면 단위(1:30, 원점 128.5,122) 상자. 차로 경계선은 상자 밖이라 남음
EXCL = [(439.3, 182.5, 455.2, 206.0), (236.0, 275.6, 257.0, 291.5), (214.0, 300.5, 241.0, 316.0), (206.5, 276.0, 234.0, 291.0),
        (442.5, 222.0, 452.5, 263.0), (159.5, 299.5, 165.0, 316.5), (246.0, 160.5, 257.0, 169.0), (246.0, 130.0, 257.0, 141.0)]
for x0, y0, x1, y1 in EXCL:
    X0, X1 = (x0 - 128.5) * 30 / RES, (x1 - 128.5) * 30 / RES; Y1, Y0 = (6000 - (y0 - 122) * 30) / RES, (6000 - (y1 - 122) * 30) / RES
    BAR[int(Y0):int(Y1) + 1, int(X0):int(X1) + 1] = 0
def m2px(p): return np.array([p[0] / RES, (6000 - p[1]) / RES])
def is_bar(p):
    x, y = m2px(p).astype(int)
    return 0 <= x < Wm and 0 <= y < Hm and BAR[y, x] > 0
def runs_along(c, nrm, hi):
    """c에서 nrm 방향 0~hi mm의 경계 픽셀 구간 [(시작, 끝), ...] (mm)."""
    ds = np.arange(0, hi, 2.5); pts = c[None] + ds[:, None] * nrm[None]
    px = np.round(np.c_[pts[:, 0] / RES, (6000 - pts[:, 1]) / RES]).astype(int)
    ok = (px[:, 0] >= 0) & (px[:, 0] < Wm) & (px[:, 1] >= 0) & (px[:, 1] < Hm)
    v = np.zeros(len(ds), bool); v[ok] = BAR[px[ok, 1], px[ok, 0]] > 0
    out, k = [], 0
    while k < len(v):
        if v[k]:
            e = k
            while e + 1 < len(v) and v[e + 1]: e += 1
            out.append((ds[k], ds[e])); k = e + 1
        else: k += 1
    return out
def ray_hit(c, nrm, lo, hi):
    """lo~hi 안에서 시작하는(또는 lo를 덮지만 60mm 뒤에서 시작한) 첫 경계선. 중심이 선 위(0부터 칠해짐)면 그 구간은 건너뜀."""
    for s0, e0 in runs_along(c, nrm, hi):
        if s0 < 7.5: continue          # 중심이 선(정지선 등) 위에 있음
        if e0 < lo: continue           # 너무 가까운 표시
        if s0 <= hi and s0 >= 60: return s0, e0
    return None
def trace(X, Y, yaw_deg, length=4500, step=30):
    h = np.radians(yaw_deg); c = np.array([X, Y], float)
    L, R = [], []
    def side(c, h, sign, expect):
        nrm = np.array([np.cos(h + sign * np.pi / 2), np.sin(h + sign * np.pi / 2)])
        r = ray_hit(c, nrm, max(60, expect - 140), expect + 140)
        return (r, nrm)
    # 시작: 좌우 0~700mm를 다 훑어 가장 가까운 경계. 차 중심이 선 위면 선 양쪽 중 폭이 500mm에 가까운 차로를 고름
    nl = np.array([np.cos(h + np.pi / 2), np.sin(h + np.pi / 2)]); nr = -nl
    fwd = np.array([np.cos(h), np.sin(h)])
    for _ in range(12):  # 중심이 가로선(정지선·출발선) 위면 좌우로 길게 칠해져 있음 -> 선을 벗어날 때까지 앞으로
        if not is_bar(c): break
        lat = sum(10 for sg in (1, -1) for d in range(10, 160, 10) if is_bar(c + sg * d * nl))
        if lat < 200: break  # 세로 차선을 밟은 것
        c = c + 15 * fwd
    rl_, rr_ = runs_along(c, nl, 700), runs_along(c, nr, 700)
    if rl_ and rr_ and rl_[0][0] < 7.5 and rr_[0][0] < 7.5:  # 선 위
        lw = next((x for x, _ in rl_[1:]), None); rw = next((x for x, _ in rr_[1:]), None)
        cand = []
        if lw is not None: cand.append((abs(lw - rl_[0][1] - 500), +1, rl_[0][1], lw))
        if rw is not None: cand.append((abs(rw - rr_[0][1] - 500), -1, rr_[0][1], rw))
        if cand:
            _, sg, e0, far = min(cand); c = c + (nl if sg > 0 else nr) * (e0 + far) / 2
            rl_, rr_ = runs_along(c, nl, 700), runs_along(c, nr, 700)
    # 점선 틈을 피하려고 앞뒤 5군데(-120~+120mm)에서 재서 가장 가까운 경계를 씀. 450mm 넘게 먼 것은 이 차로 경계가 아님
    def near(nrm):
        best = None
        for off in (-120, -60, 0, 60, 120):
            r = next((x for x, _ in runs_along(c + off * fwd, nrm, 700) if x >= 7.5), None)
            if r is not None and r <= 450 and (best is None or r < best): best = r
        return best
    dl, dr = near(nl), near(nr)
    if dl is None and dr is None: dl = dr = 250.0
    elif dl is None: dl = 500 - dr
    elif dr is None: dr = 500 - dl
    if not 380 < dl + dr < 700:  # 폭이 이상하면 가까운 쪽 하나만 믿고 500mm 차로
        if dl <= dr: dr = 500 - dl
        else: dl = 500 - dr
    w = float(np.clip(dl + dr, 420, 620)); c = c + nl * (dl - dr) / 2; dl = dr = w / 2
    prev = c.copy(); t = 0
    while t < length:
        c = c + step * np.array([np.cos(h), np.sin(h)]); t += step
        if is_bar(c):  # 앞을 가로막는 선: 80mm 더 앞에서 좌우 경계가 이어지면 정지선(통과), 아니면 차로 끝(섬, T자 끝)
            c2 = c + 80 * np.array([np.cos(h), np.sin(h)])
            l2, _ = side(c2, h, 1, dl); r2, _ = side(c2, h, -1, dr)
            if is_bar(c2) or (l2 is None and r2 is None): break
        rl, nl = side(c, h, 1, dl); rr, nr = side(c, h, -1, dr)
        gl = rl[0] if rl else None; gr = rr[0] if rr else None
        if gl is not None and gr is not None and 380 < gl + gr < 680:
            w = 0.85 * w + 0.15 * (gl + gr); shift = (gl - gr) / 2
        elif gl is not None and abs(gl - w / 2) < 110: shift = gl - w / 2
        elif gr is not None and abs(gr - w / 2) < 110: shift = w / 2 - gr
        else: shift = 0.0
        c = c + nl * shift * 0.7
        d = c - prev
        if np.hypot(*d) > 1e-3:
            hn = np.arctan2(d[1], d[0]); dh = (hn - h + np.pi) % (2 * np.pi) - np.pi
            h = h + np.clip(dh, -0.12, 0.12) * 0.6
        prev = c.copy()
        L.append(c + nl * w / 2); R.append(c + nr * w / 2)
        px = m2px(c)
        if not (0 <= px[0] < Wm and 0 <= px[1] < Hm): break
    return np.array(L), np.array(R)
def map_labels(X, Y, yaw):
    L, R = trace(X, Y, yaw)
    da = np.zeros_like(PAINT); llmask = np.zeros_like(PAINT)
    if len(L) < 3: return da, llmask
    poly = np.round(np.r_[[m2px(p) for p in L], [m2px(p) for p in R[::-1]]] * 4).astype(np.int32)
    cv2.fillPoly(da, [poly], 1, cv2.LINE_8, shift=2)
    band = np.zeros_like(PAINT)
    for side in (L, R):
        cv2.polylines(band, [np.round(np.array([m2px(p) for p in side]) * 4).astype(np.int32)], False, 1, int(90 / RES), cv2.LINE_8, shift=2)
    llmask = band & BAR
    da = da | (llmask & cv2.dilate(da, np.ones((15, 15), np.uint8)))  # 경계선 안쪽 절반까지 da와 맞닿게
    return da, llmask
_grid = {}
def image_labels(pose, pitch, shape=(360, 640)):
    X, Y, yaw = pose
    da_m, ll_m = map_labels(X, Y, yaw)
    if pitch not in _grid:
        vv, uu = np.mgrid[0:shape[0], 0:shape[1]]
        g = cam.ipm(np.c_[uu.ravel(), vv.ravel()], pitch).reshape(shape + (2,))
        _grid[pitch] = g
    g = _grid[pitch]; P = np.radians(yaw)
    gx_, gy_ = g[..., 0] * 1000, g[..., 1] * 1000
    mx = X + np.cos(P) * gx_ - np.sin(P) * gy_; my = Y + np.sin(P) * gx_ + np.cos(P) * gy_
    px, py = mx / RES, (6000 - my) / RES
    ok = np.isfinite(px) & (g[..., 0] < 6.0) & (px >= 0) & (px < Wm - 1) & (py >= 0) & (py < Hm - 1)
    pxi, pyi = np.where(ok, px, 0).astype(np.float32), np.where(ok, py, 0).astype(np.float32)
    da = cv2.remap(da_m.astype(np.float32), pxi, pyi, cv2.INTER_LINEAR) > 0.5
    ll = cv2.remap(ll_m.astype(np.float32), pxi, pyi, cv2.INTER_LINEAR) > 0.3
    return da & ok, ll & ok
