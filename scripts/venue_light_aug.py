"""대회장 조명 증강 시안. 아직 train_seg.py에 연결하지 않았다.

대회장 raw 사진(라벨 불필요)을 색 견본으로 써서, 기존 학습 사진에 대회장 조명을 입힌다.
  v1: 사진 전체에 견본의 LAB 평균·표준편차를 옮김 (Reinhard)
  v2: 바닥(da 위쪽 끝 아래)과 배경을 따로 옮김 + 조명색 반사 띠, 조명 번짐, 좌우·위아래 색 그라데이션
  v3: v2 + 차선 탈색(ll 라벨 사용, 라벨은 그대로), 매트 광택, 과노출과 하이라이트 채도 빠짐  <- 권장

실제 대회장 사진과의 비교(바닥 영역, 2차 HybridNets 특징 공간): 원본 train FD 224 -> v3 86
(대회장 사진끼리 37), 비슷한 사진이 있는 대회장 사진 비율 8% -> 34%.
v3를 입혀도 test 542장에서 2차 모델 점수는 크게 안 떨어진다(da 0.950 -> 0.937).
즉 모델을 무너뜨리는 원인을 아직 다 재현하지 못한다. 대회장 라벨 데이터와 같이 쓰는 것을 권한다.

  VENUE_REF_DIR: 색 견본으로 쓸 대회장 raw 사진 폴더 (*.jpg)
"""
import os, glob, random, cv2, numpy as np

# 색 견본으로 쓸 대회장 raw 사진 폴더 (라벨 불필요)
NEW = sorted(glob.glob(os.path.join(os.environ.get('VENUE_REF_DIR', '/media/foscar/80BD-730A/dataset_raw/20261106/session_180809/images'), '*.jpg')))
_cache = {}

def load_ref(p):
    if p not in _cache:
        _cache[p] = cv2.resize(cv2.imread(p), (640, 360), interpolation=cv2.INTER_AREA)
    return _cache[p]

def _lab(img): return cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
def _stats(lab, m): x = lab[m]; return x.mean(0), x.std(0) + 1e-3

# ---------------- v1 ----------------
def v1(img, ref, da=None):
    a = _lab(img); (ms, ss), (mr, sr) = _stats(a, np.ones(a.shape[:2], bool)), _stats(_lab(ref), np.ones(ref.shape[:2], bool))
    k = random.uniform(0.6, 1.0)
    out = np.clip(a + k * ((a - ms) / ss * sr + mr - a), 0, 255).astype(np.uint8)
    out = cv2.cvtColor(out, cv2.COLOR_LAB2BGR)
    if random.random() < 0.4:
        h, w = out.shape[:2]; c1, c2 = np.random.uniform(0.7, 1.3, 3), np.random.uniform(0.7, 1.3, 3)
        t = np.linspace(0, 1, w)[None, :, None] if random.random() < 0.5 else np.linspace(0, 1, h)[:, None, None]
        out = np.clip(out * (c1 * (1 - t) + c2 * t), 0, 255).astype(np.uint8)
    return out

# ---------------- v2 ----------------
def _floor_weight(h, w, y_h):
    yy = np.arange(h, dtype=np.float32)[:, None]
    return np.repeat(1 / (1 + np.exp(-(yy - y_h) / 12.0)), w, 1)

def _horizon(da, h):
    if da is not None and da.any():
        top = np.flatnonzero(da.any(1))[0]
        return float(np.clip(top - 8, 0.30 * h, 0.58 * h))
    return 0.45 * h

def _transfer(lab, src_m, ref_lab, ref_m):
    (ms, ss), (mr, sr) = _stats(lab, src_m), _stats(ref_lab, ref_m)
    ratio = np.clip(sr / ss, 0.5, 2.5)  # 무채색 매트의 작은 색 노이즈가 과하게 커지지 않게
    return (lab - ms) * ratio + mr

def _tinted_streak(img, color, wfloor):
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.float32)
    x0 = random.uniform(0.1, 0.9) * w
    cv2.line(mask, (int(x0 + random.uniform(-.05, .05) * w), int(random.uniform(.4, .6) * h)),
             (int(x0 + random.uniform(-.35, .35) * w), h), 1.0, random.randint(25, 90))
    mask = cv2.GaussianBlur(mask, (0, 0), random.uniform(10, 25)) * wfloor
    mask *= np.linspace(0.3, 1.0, h, dtype=np.float32)[:, None]
    return np.clip(img + mask[..., None] * random.uniform(90, 220) * color, 0, 255)

def _tinted_bloom(img, color):
    h, w = img.shape[:2]
    cx, cy, r = random.uniform(0, w), random.uniform(-0.2 * h, 0.45 * h), random.uniform(0.25, 0.8) * w
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    g = np.clip(1 - np.hypot(xx - cx, yy - cy) / r, 0, 1) ** 1.6
    return np.clip(img + g[..., None] * random.uniform(120, 280) * color, 0, 255)

def _highlight_color(ref, m):
    px = ref[m].astype(np.float32); L = px.mean(1)
    c = px[L >= np.percentile(L, 95)].mean(0)
    return c / (c.max() + 1e-6)

def v2(img, ref, da=None):
    h, w = img.shape[:2]
    lab, rlab = _lab(img), _lab(ref)
    wf = _floor_weight(h, w, _horizon(da, h))
    rf = np.zeros((h, w), bool); rf[int(.55 * h):] = True     # 견본 사진: 아래쪽 = 바닥
    rb = np.zeros((h, w), bool); rb[:int(.40 * h)] = True     #            위쪽 = 벽/배경
    tf = _transfer(lab, wf > 0.5, rlab, rf)
    tb = _transfer(lab, wf <= 0.5, rlab, rb)
    t = tf * wf[..., None] + tb * (1 - wf[..., None])
    k = random.uniform(0.75, 1.0)
    out = cv2.cvtColor(np.clip(lab + k * (t - lab), 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
    if random.random() < 0.45:   # 매끈한 매트에 조명이 반사된 색 띠
        out = _tinted_streak(out, _highlight_color(ref, rf), wf)
    if random.random() < 0.30:   # 색 조명 자체가 번져 보임
        out = _tinted_bloom(out, _highlight_color(ref, rb))
    if random.random() < 0.30:   # 조명 색이 화면 한쪽에서 다른 쪽으로 바뀜
        c1, c2 = np.random.uniform(0.75, 1.25, 3), np.random.uniform(0.75, 1.25, 3)
        tt = np.linspace(0, 1, w)[None, :, None] if random.random() < 0.6 else np.linspace(0, 1, h)[:, None, None]
        out = out * (c1 * (1 - tt) + c2 * tt)
    return np.clip(out, 0, 255).astype(np.uint8)

# ---------------- v3 = v2 + 노출 과다로 밝은 곳이 조명색 섞인 흰색으로 날아감 + 차선 탈색 + 바닥 광택 ----------------
def _sheen(h, w, wf):
    """매트 광택: 바닥에 크고 부드러운 밝기 얼룩 (조명이 매트에 비친 넓은 반사)."""
    n = np.random.rand(max(2, h // 60), max(2, w // 60)).astype(np.float32)
    n = cv2.GaussianBlur(cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC), (0, 0), 25)
    n = (n - n.min()) / (np.ptp(n) + 1e-6)
    return n ** 2 * wf

def v3(img, ref, da=None, ll=None):
    h, w = img.shape[:2]
    out = v2(img, ref, da).astype(np.float32)
    wf = _floor_weight(h, w, _horizon(da, h))
    rf = np.zeros((h, w), bool); rf[int(.55 * h):] = True
    if ll is not None and ll.any() and random.random() < 0.5:
        # 차선 탈색: 색 조명 + 노출 과다에서 노란 차선이 연한/흰 차선으로 보임 (라벨은 그대로)
        a = cv2.GaussianBlur(cv2.dilate(ll.astype(np.uint8), np.ones((3, 3))).astype(np.float32), (0, 0), 1.2)[..., None]
        gray = out.mean(2, keepdims=True)
        s = random.uniform(0.4, 0.9)
        lane = gray + s * (out - gray) + random.uniform(0, 25)
        out = out * (1 - a) + lane * a
    if random.random() < 0.5:
        out = out + _sheen(h, w, wf)[..., None] * random.uniform(25, 70) * _highlight_color(ref, rf)
    # 노출: 1.0~1.5배. 밝은 채널부터 255에서 잘려서 하이라이트가 조명색 섞인 흰색이 됨
    out = out * random.uniform(1.0, 1.3) if random.random() < 0.4 else out
    out = np.clip(out, 0, 255)
    # 잘린 근처 하이라이트는 채도가 빠짐 (센서 포화)
    L = out.max(2, keepdims=True); hi = np.clip((L - 230) / 25, 0, 1)
    out = out * (1 - 0.4 * hi) + out.mean(2, keepdims=True) * 0.4 * hi
    return np.clip(out, 0, 255).astype(np.uint8)
