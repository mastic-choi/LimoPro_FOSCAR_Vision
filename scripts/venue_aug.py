"""대회장 조명 증강 (train_seg.py --venue-ref-dir 로 켬). 동아리 PC 시안(v1~v3) + 실제 조명 분포를 옮기는 v4.

견본은 대회장 사진 폴더(라벨 불필요). 평가에 쓰는 대회장 test 사진은 견본에 넣지 말 것.
라벨은 바꾸지 않는다(색과 밝기만 바뀜). 원본: dataset/track_label_20261106/aug/venue_aug.py
"""
import glob, os, random, cv2, numpy as np

REFS = []
_cache = {}


def set_refs(folder):
    REFS[:] = sorted(glob.glob(os.path.join(folder, "*.jpg")) + glob.glob(os.path.join(folder, "*.png")))
    return len(REFS)


def load_ref(p):
    if p not in _cache:
        _cache[p] = cv2.resize(cv2.imread(p), (640, 360), interpolation=cv2.INTER_AREA)
    return _cache[p]


def random_ref():
    return load_ref(random.choice(REFS))


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

# ---------------- v4 = 실제 대회장 사진의 조명 분포를 통째로 옮김 (2026-11-06) ----------------
# 대회장은 한 장 안에서 여러 색 조명이 섞이고(가까운 쪽 빨강, 먼 쪽 파랑, 매트 위 자홍 웅덩이), 조명이 매트에 몰린다.
# v1~v3는 사진 한 장의 색 통계를 옮기거나 직선 그라데이션만 더해서 이 공간 분포를 못 만든다.
# 방법: 사진 = 조명(아주 흐린 성분) x 반사(세부 모양)라고 보고, 학습 사진의 조명을 대회장 사진의 조명으로 바꾼다.
#       차선·매트 무늬 같은 세부는 학습 사진 그대로라 라벨이 그대로 맞는다.
def _illum(img, sigma):
    return cv2.GaussianBlur(img.astype(np.float32), (0, 0), sigma) + 4.0

def v4(img, ref, da=None, ll=None):
    h, w = img.shape[:2]
    if random.random() < 0.5:
        ref = ref[:, ::-1]                                       # 조명 배치 좌우 뒤집기
    dx, dy = random.randint(-40, 40), random.randint(-15, 15)    # 조명 위치 조금 옮기기
    ref = cv2.warpAffine(np.ascontiguousarray(ref), np.float32([[1, 0, dx], [0, 1, dy]]), (w, h), borderMode=cv2.BORDER_REFLECT)
    sigma = random.uniform(18, 40)
    x = img.astype(np.float32)
    gain = _illum(ref, sigma) / _illum(img, sigma)               # 픽셀별 3채널 조명 비율
    gain = np.clip(gain, 0.25, 6.0)
    k = random.uniform(0.7, 1.0)
    out = x * (1 - k + k * gain)
    if ll is not None and ll.any() and random.random() < 0.35:   # 노란 차선이 조명에 물들어 하얗게 날아감 (v3와 같음)
        a = cv2.GaussianBlur(cv2.dilate(ll.astype(np.uint8), np.ones((3, 3))).astype(np.float32), (0, 0), 1.2)[..., None]
        gray = out.mean(2, keepdims=True)
        out = out * (1 - a) + (gray + random.uniform(0.4, 0.9) * (out - gray) + random.uniform(0, 25)) * a
    if random.random() < 0.3:                                    # 자동 화이트밸런스·노출이 프레임마다 흔들림
        out = out * np.random.uniform(0.85, 1.15, 3) * random.uniform(0.85, 1.25)
    out = np.clip(out, 0, 255)
    L = out.max(2, keepdims=True); hi = np.clip((L - 230) / 25, 0, 1)   # 포화된 하이라이트는 채도가 빠짐
    out = out * (1 - 0.4 * hi) + out.mean(2, keepdims=True) * 0.4 * hi
    if random.random() < 0.4:                                    # 어두운 조명에서 늘어나는 센서 노이즈
        out = out + np.random.normal(0, random.uniform(2, 6), out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------- v5 = 실제 어둠·색 조명 주행의 색을 그대로 옮김 (2026-11-08, 4차) ----------------
# 11/07 실차 bag 중 어둠·색 조명 주행 5개(darkout3, dark4, redout5, blue6, purple7)에서 1초마다
# 벽(0~110행) / 먼 바닥(200~250행) / 가까운 바닥(300~360행)의 중앙값 색을 모아 둔 표(v5_bag_colors.json, 336개)를 쓴다.
# 학습 사진의 같은 세 구역 중앙값을 그 실제 값으로 맞추는 채널별 배율을 행마다 이어 붙여 곱한다.
#   - 실제 빨강 조명: 매트가 탁한 진빨강, 차선은 분홍빛, 조명 받은 벽은 분홍~흰색으로 날아감 -> 배율이 커서 R이 포화되며 저절로 생김
#   - 실제 어둠: 바닥 중앙값 10~30, 카메라가 노이즈를 지워 매끈함 -> 노이즈는 아주 약하게만
# 첫 시안(단색 배율·웅덩이를 손으로 정함)은 벽이 하얗게 날아가고 바닥 색이 실제보다 선명했고, 어둠은 덜 어두웠다.
# 라벨은 그대로.
_BAG = []


def _bag_colors():
    if not _BAG:
        import json
        _BAG.extend(json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "v5_bag_colors.json"))))
    return _BAG


def _rows_gain(h, g_wall, g_far, g_near):
    """행마다 채널별 배율: 0~130행 벽, 130~225 벽->먼 바닥, 225~330 먼->가까운 바닥."""
    y = np.arange(h, dtype=np.float32)
    def ramp(a, b): return np.clip((y - a) / (b - a), 0, 1)[:, None]
    g = g_wall + (g_far - g_wall) * ramp(130, 225)
    return g + (g_near - g) * ramp(225, 330)


def v5(img, ref=None, da=None, ll=None):
    """ref는 쓰지 않음 (train_seg의 다른 증강과 같은 호출 모양)."""
    h, w = img.shape[:2]
    c = random.choice(_bag_colors())
    x = img.astype(np.float32)
    med = lambda a, b: np.median(x[a:b].reshape(-1, 3), 0) + 3.0
    j = lambda: np.random.uniform(0.9, 1.1, 3)                        # 같은 실제 장면 안에서 조금씩 흔들기
    gains = [np.clip((np.float32(c[k]) + 3.0) * j() / med(a, b), 0.03, 8.0)
             for k, (a, b) in (("wall", (0, 110)), ("far", (200, 250)), ("near", (300, 360)))]
    g = _rows_gain(h, *gains)[:, None, :]                             # (h, 1, 3)
    if random.random() < 0.5:                                         # 조명이 한쪽으로 치우침 (좌우 기울기)
        g = g * np.linspace(random.uniform(0.7, 1.0), random.uniform(1.0, 1.3), w, dtype=np.float32)[None, :, None][:, ::random.choice([1, -1])]
    out = x * g
    over = np.clip(out.max(2, keepdims=True) - 255, 0, None)          # 포화된 빛이 다른 채널로 번짐 -> 분홍/흰색
    out = out + over * random.uniform(0.15, 0.4)
    if random.random() < 0.5:
        out = out + np.random.normal(0, random.uniform(0.3, 1.2), out.shape[:2])[..., None]
    if random.random() < 0.3:                                         # 저조도 카메라의 노이즈 제거로 살짝 뭉개짐
        out = cv2.GaussianBlur(out, (0, 0), random.uniform(0.5, 1.0))
    return np.clip(out, 0, 255).astype(np.uint8)
