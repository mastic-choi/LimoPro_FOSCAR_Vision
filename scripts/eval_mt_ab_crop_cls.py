#!/usr/bin/env python3
"""확인 실험: 파란 사각형 표지판(sign_A / sign_B / parking)을 잘라서 작은 CNN 분류기로 구분할 수 있는가.

검출 모델(multitask-v1)은 A/B 2택 정답률이 train에서도 59%였다. 같은 사진에서 잘라 낸 표지판으로
작은 분류기를 처음부터 학습해서 높게 나오면, 정보는 사진에 있고 검출 머리가 못 배운 것이다.

- 자르기: 정답 박스 중심, 박스 긴 변의 2배 정사각형을 96x96으로 저장 (640x360 사진 그대로에서 자름)
- 학습 증강: 박스 위치·크기 흔들기(검출 박스 오차 흉내), 해상도 낮췄다 올리기(작은 표지판 흉내), 색·밝기, 흐림.
  좌우 반전은 하지 않음(A/B 화살표 방향이 바뀜)
- 입력 64x64, 작은 CNN(약 1.2M 파라미터), 30 epoch
- 평가: val / test / test_venue / test_bag. 정답 박스 그대로 + 박스를 ±10% 흔든 것 둘 다.

사용: python scripts/eval_mt_ab_crop_cls.py . <출력 폴더>
"""
import os, sys, json, random, time, collections
import cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

root, out = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
os.makedirs(out, exist_ok=True)
D = os.path.join(root, "dataset/synth_signs_v2")
CLS = {1: 0, 2: 1, 5: 2}   # sign_A, sign_B, parking -> 0, 1, 2
NAMES = ["A", "B", "P"]
CTX, IN = 96, 64
random.seed(0); np.random.seed(0); torch.manual_seed(0)
dev = torch.device("cuda")


def load_split(split):
    X, Y, S, N = [], [], [], []
    d = os.path.join(D, split)
    for n in sorted(os.listdir(os.path.join(d, "images"))):
        lab = os.path.join(d, "labels", n[:-4] + ".txt")
        if not os.path.exists(lab):
            continue
        rows = [l.split() for l in open(lab) if l.strip()]
        rows = [r for r in rows if int(r[0]) in CLS]
        if not rows:
            continue
        img = cv2.imread(os.path.join(d, "images", n))
        H, W = img.shape[:2]
        pad = cv2.copyMakeBorder(img, 200, 200, 200, 200, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        for r in rows:
            cx, cy, w, h = float(r[1]) * W, float(r[2]) * H, float(r[3]) * W, float(r[4]) * H
            s = max(w, h) * 2
            x1, y1 = int(round(cx - s / 2)) + 200, int(round(cy - s / 2)) + 200
            c = pad[y1:y1 + int(round(s)), x1:x1 + int(round(s))]
            X.append(cv2.resize(c, (CTX, CTX), interpolation=cv2.INTER_LINEAR))
            Y.append(CLS[int(r[0])]); S.append(np.sqrt(w * h)); N.append(f"{split}/{n}")
    return np.stack(X), np.array(Y), np.array(S), N


t0 = time.time()
data = {sp: load_split(sp) for sp in ("train", "val", "test", "test_venue", "test_bag")}
for sp, (X, Y, S, _) in data.items():
    print(f"{sp:11s} crops {len(Y):5d}  A {np.sum(Y == 0):4d} B {np.sum(Y == 1):4d} P {np.sum(Y == 2):4d}  size median {np.median(S):.1f}px", flush=True)
print(f"load {time.time() - t0:.0f}s", flush=True)

MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)


def to_t(X):   # BGR uint8 N,96,96,3 -> RGB float N,3,96,96 on GPU
    return torch.from_numpy(X[..., ::-1].copy()).to(dev).permute(0, 3, 1, 2).float() / 255


def crop_batch(x, sizes, jitter, train):
    """x: N,3,96,96 (박스 = 가운데 48px 정사각형). 박스 주변 1.5배를 잘라 IN x IN 으로."""
    N = x.shape[0]
    scale = torch.full((N,), 0.75, device=dev)   # 96 중 72px = 박스 1.5배
    tx = torch.zeros(N, device=dev); ty = torch.zeros(N, device=dev)
    if jitter > 0:
        scale = scale * (1 + (torch.rand(N, device=dev) * 2 - 1) * jitter)
        tx = (torch.rand(N, device=dev) * 2 - 1) * jitter * 0.5
        ty = (torch.rand(N, device=dev) * 2 - 1) * jitter * 0.5
    th = torch.zeros(N, 2, 3, device=dev)
    th[:, 0, 0] = scale; th[:, 1, 1] = scale; th[:, 0, 2] = tx; th[:, 1, 2] = ty
    g = F.affine_grid(th, (N, 3, IN, IN), align_corners=False)
    y = F.grid_sample(x, g, mode="bilinear", padding_mode="border", align_corners=False)
    if train:
        # 해상도 낮췄다 올리기: 박스가 실제로 가졌던 픽셀 수 근처까지 (작은 표지판 흉내)
        lo = (torch.rand(N, device=dev) < 0.7)
        for k in torch.nonzero(lo).flatten().tolist():
            r = random.randint(14, 40)
            y[k:k + 1] = F.interpolate(F.interpolate(y[k:k + 1], size=(r, r), mode="area"), size=(IN, IN), mode="bilinear", align_corners=False)
        b = 1 + (torch.rand(N, 1, 1, 1, device=dev) * 2 - 1) * 0.4      # 밝기
        c = 1 + (torch.rand(N, 1, 1, 1, device=dev) * 2 - 1) * 0.4      # 대비
        m = y.mean((1, 2, 3), keepdim=True)
        y = ((y - m) * c + m) * b
        tint = 1 + (torch.rand(N, 3, 1, 1, device=dev) * 2 - 1) * 0.25   # 색 조명
        y = y * tint
        gray = (torch.rand(N, 1, 1, 1, device=dev) < 0.1).float()
        y = y * (1 - gray) + y.mean(1, keepdim=True) * gray
        y = y + torch.randn_like(y) * 0.02 * torch.rand(N, 1, 1, 1, device=dev)
        y = y.clamp(0, 1)
    return (y - MEAN) / STD


def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(block(3, 32), nn.MaxPool2d(2), block(32, 64), nn.MaxPool2d(2),
                               block(64, 128), nn.MaxPool2d(2), block(128, 256), nn.AdaptiveAvgPool2d(1))
        self.h = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, 3))

    def forward(self, x):
        return self.h(self.f(x))


net = Net().to(dev)
print(f"params {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M", flush=True)
Xtr, Ytr = to_t(data["train"][0]), torch.from_numpy(data["train"][1]).to(dev)
EPOCHS, BS = 30, 256
opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=5e-4)
steps = EPOCHS * ((len(Ytr) + BS - 1) // BS)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=0.15)


@torch.no_grad()
def predict(sp, jitter=0.0, seed=0):
    net.eval()
    torch.manual_seed(seed)
    X = to_t(data[sp][0])
    P = torch.cat([F.softmax(net(crop_batch(X[i:i + 512], None, jitter, False)), 1) for i in range(0, len(X), 512)])
    return P.cpu().numpy()


def acc(sp, P):
    Y = data[sp][1]
    pred = P.argmax(1)
    ab = Y < 2
    two = (P[ab][:, :2].argmax(1) == Y[ab]).mean() if ab.any() else float("nan")
    return float((pred == Y).mean()), float(two)


log = []
for ep in range(EPOCHS):
    net.train()
    perm = torch.randperm(len(Ytr), device=dev)
    tl, t0 = 0.0, time.time()
    for i in range(0, len(perm), BS):
        idx = perm[i:i + BS]
        x = crop_batch(Xtr[idx], None, 0.25, True)
        loss = F.cross_entropy(net(x), Ytr[idx], label_smoothing=0.05)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        tl += loss.item() * len(idx)
    va3, va2 = acc("val", predict("val"))
    log.append(dict(epoch=ep + 1, loss=round(tl / len(Ytr), 4), val_acc3=round(va3, 4), val_ab2=round(va2, 4)))
    print(f"ep {ep + 1:2d} loss {tl / len(Ytr):.4f} val 3종 {va3:.3f} A/B {va2:.3f} ({time.time() - t0:.1f}s)", flush=True)

torch.save(net.state_dict(), os.path.join(out, "ab_crop_cls.pt"))
res = {}
for sp in ("train", "val", "test", "test_venue", "test_bag"):
    P0 = predict(sp)
    a3, a2 = acc(sp, P0)
    js = [acc(sp, predict(sp, 0.10, seed=s)) for s in range(5)]
    Y, S = data[sp][1], data[sp][2]
    conf = np.zeros((3, 3), int)
    for y, p in zip(Y, P0.argmax(1)):
        conf[y, p] += 1
    bysz = {}
    for k, (lo, hi) in {"<18px": (0, 18), "18-26px": (18, 26), ">=26px": (26, 1e9)}.items():
        mk = (S >= lo) & (S < hi) & (Y < 2)
        if mk.any():
            bysz[k] = f"{int((P0[mk][:, :2].argmax(1) == Y[mk]).sum())}/{int(mk.sum())}"
    res[sp] = dict(n=int(len(Y)), acc3=round(a3, 4), ab2=round(a2, 4),
                   acc3_jitter10=round(float(np.mean([j[0] for j in js])), 4), ab2_jitter10=round(float(np.mean([j[1] for j in js])), 4),
                   confusion_rows_gt_cols_pred_ABP=conf.tolist(), ab2_by_size=bysz)
    print(f"{sp:11s} n {len(Y):5d}  3종 {a3:.3f}  A/B {a2:.3f}  | 박스 ±10% 흔듦: 3종 {res[sp]['acc3_jitter10']:.3f} A/B {res[sp]['ab2_jitter10']:.3f}  크기별 A/B {bysz}  혼동 {conf.tolist()}", flush=True)

json.dump(dict(results=res, log=log), open(os.path.join(out, "ab_crop_cls.json"), "w"), ensure_ascii=False, indent=1)

# 몽타주: 대회장 test 전부 (틀린 것 먼저)
sp = "test_venue"
X, Y, S, N = data[sp]
P = predict(sp)
order = sorted(range(len(Y)), key=lambda i: (P[i].argmax() == Y[i], Y[i]))
tiles = []
for i in order:
    t = cv2.resize(X[i], (144, 144), interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(t, (36, 36), (108, 108), (255, 255, 255), 1)
    canvas = np.full((184, 144, 3), 25, np.uint8); canvas[:144] = t
    ok = P[i].argmax() == Y[i]
    col = (60, 200, 60) if ok else (60, 60, 230)
    cv2.putText(canvas, f"GT {NAMES[Y[i]]} {S[i]:.0f}px", (4, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"pred {NAMES[P[i].argmax()]} {P[i].max():.2f}", (4, 178), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (143, 183), col, 2)
    tiles.append(canvas)
COLS = 12
while len(tiles) % COLS:
    tiles.append(np.full((184, 144, 3), 25, np.uint8))
g = np.vstack([np.hstack(tiles[r:r + COLS]) for r in range(0, len(tiles), COLS)])
head = np.full((36, g.shape[1], 3), 30, np.uint8)
cv2.putText(head, f"crop classifier on test_venue A/B/P ({len(Y)}): acc {res[sp]['acc3']:.3f}, A/B {res[sp]['ab2']:.3f} | wrong first",
            (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
cv2.imwrite(os.path.join(out, "ab_crop_cls_test_venue.jpg"), np.vstack([head, g]), [cv2.IMWRITE_JPEG_QUALITY, 90])
print("saved", out, flush=True)
