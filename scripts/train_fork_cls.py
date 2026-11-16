#!/usr/bin/env python3
"""갈림길 표지판 분류기 학습: 고정 ResNet18 특징 + 선형층 (A / B / none).

교수님 제안 구조: 검출 문턱을 낮춰 후보를 늘리고, 원본 크롭을 2단계 분류기로 거른다. 분류기는
"사전학습 CNN 특징(고정) + 선형" (linear probe). 특징 추출부는 ImageNet ResNet18 을 그대로 쓰고 학습은 선형층만.
  특징은 평균(gap)이 아니라 layer3 공간 맵(--feat l3, 기본, 256x6x6): ImageNet 특징은 좌우 반전에 둔감해 평균을 내면 화살표 방향이 사라진다.

- 입력: fork_crop_rule.preprocess (흑백 → 96x96 → CLAHE) /255, 1채널.
- ResNet18 첫 conv 를 1채널로 접는다: 같은 흑백을 3채널로 복사한 뒤 ImageNet 정규화한 것과 같게
  W' = Σ_c W[:,c]/std_c, b' = −Σ_c Σ W[:,c]·mean_c/std_c. 학습·ONNX·로봇이 모두 이 접힌 모델을 쓴다.
- 증강(전처리 전, 원본 크롭에): 위치·크기 흔들기, 밝기·대비, 색 조명(채널 배율), 흐림, 축소 후 확대, JPEG. 좌우 반전 없음.
  --flip-real-b: 실차 B 크롭을 좌우 반전해 A 로도 쓴다(실물 A 가 없을 때의 대용. 화살표는 맞고 글자는 거울 B).
  --mask-prob: 이 확률로 표지판 원 바깥을 회색으로 가린다. 뒤집은 B 는 배경(노트북 흰 막대 등)도 뒤집혀 '배경으로 A/B 고르기'를 배울 수 있어서.
- 데이터 폴더: <dir>/<A|B|none>/*.png (fork_crops_synth.py, fork_crops_bag.py 결과).

사용:
  python scripts/train_fork_cls.py --out outputs/fork_cls_v1/run \
      --train dataset/fork_crops_v1/synth/train dataset/fork_crops_v1/bag_030233 \
      --real dataset/fork_crops_v1/bag_030233 \
      --eval synth_val=dataset/fork_crops_v1/synth/val venue=dataset/fork_crops_v1/synth/test_venue \
             bag=dataset/fork_crops_v1/synth/test_bag real_030611=dataset/fork_crops_v1/bag_030611 [--flip-real-b]
"""
import argparse, glob, json, os, random, sys, time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fork_crop_rule as rule

LABELS = ["A", "B", "none"]
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
SIZE = 96


class Features(nn.Module):
    """접힌 ResNet18 의 앞부분 → 특징 벡터. gap: 전역 평균 512, l4: layer4 맵 512x3x3, l3: layer3 맵 256x6x6 (96px 입력).
    ImageNet 특징은 좌우 반전에 둔감하게 학습돼 있어, 화살표 방향(A ↰ / B ↱)은 평균(gap)보다 공간 맵(l4·l3)에 남는다."""

    def __init__(self, m, feat):
        super().__init__()
        self.m, self.feat = m, feat

    def forward(self, x):
        m = self.m
        x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
        x = m.layer3(m.layer2(m.layer1(x)))
        if self.feat == "l3":
            return torch.flatten(x, 1)
        x = m.layer4(x)
        return torch.flatten(m.avgpool(x) if self.feat == "gap" else x, 1)


FEAT_DIM = {"gap": 512, "l4": 512 * 3 * 3, "l3": 256 * 6 * 6}


def folded_backbone(feat="gap"):
    """ImageNet ResNet18 → 1채널 입력(0~1 흑백)을 받는 고정 특징 추출부."""
    m = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    w = m.conv1.weight.data                                   # 64x3x7x7
    s = torch.tensor(STD).view(1, 3, 1, 1); mu = torch.tensor(MEAN).view(1, 3, 1, 1)
    conv = nn.Conv2d(1, 64, 7, 2, 3, bias=True)
    conv.weight.data = (w / s).sum(1, keepdim=True)
    conv.bias.data = -(w * mu / s).sum((1, 2, 3))
    m.conv1 = conv
    m.fc = nn.Identity()
    f = Features(m, feat)
    for p in f.parameters():
        p.requires_grad_(False)
    return f.eval()


class ForkCls(nn.Module):
    """접힌 ResNet18 + 선형 + softmax. 입력 Nx1x96x96 (0~1), 출력 Nx3 확률 [A, B, none]."""

    def __init__(self, backbone, head):
        super().__init__()
        self.backbone, self.head = backbone, head

    def forward(self, x):
        return torch.softmax(self.head(self.backbone(x)), dim=1)


def load_dir(d):
    items = []
    for i, lab in enumerate(LABELS):
        for f in sorted(glob.glob(os.path.join(d, lab, "*.png"))):
            items.append((f, i))
    return items


def circle_mask(img, scale=1.08, dx=0.0, dy=0.0, fill=128):
    """표지판 원 바깥을 회색으로. 크롭 한 변 = 박스 x 1.6 이라 원 반지름 ≈ 한 변 / 3.2.
    배경(노트북 화면 테두리·흰 막대 등)으로 A/B 를 고르는 지름길을 막는다 (뒤집은 B 를 A 로 쓰면 배경도 같이 뒤집힌다)."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[:h, :w]
    r = min(h, w) / 3.2 * scale
    out = img.copy()
    out[np.hypot(xx - (w / 2 + dx * w), yy - (h / 2 + dy * h)) > r] = fill
    return out


def light_tint(img, rng, kind=None, strength=None):
    """대회장 색 조명 흉내 (11-16): 빨간빛·파란빛 아래 파란 표지판. 크롭(BGR)에 조명 색 이득을 곱하고 그 색 빛을 조금 더한다.
    strength 0 = 흰 조명, 1 = 아주 진한 색. 흑백으로 바꾸기 전에 해야 화살표·원 대비 변화가 학습된다 (파란빛이면 파란 원이 밝아져 흰 화살표와 대비가 줄어듦)."""
    kind = kind or ("red" if rng.random() < 0.5 else "blue")
    s = rng.uniform(0.3, 1.0) if strength is None else strength
    gains = [1 - 0.9 * s, 1 - 0.8 * s, 1 + 0.2 * s] if kind == "red" else [1 + 0.2 * s, 1 - 0.75 * s, 1 - 0.9 * s]   # B, G, R
    out = img.astype(np.float32) * np.float32(gains)
    out[..., 2 if kind == "red" else 0] += (rng.uniform(0, 40) if strength is None else 20) * s   # 그 색 빛 번짐
    return np.clip(out, 0, 255).astype(np.uint8)


def augment(img, rng, mask_prob=0.0, tint_prob=0.0):
    h, w = img.shape[:2]
    if rng.random() < mask_prob:   # 원 바깥 가리기 (위치 흔들기 전에: 가림이 표지판과 같이 움직이게)
        img = circle_mask(img, rng.uniform(1.0, 1.25), rng.uniform(-0.03, 0.03), rng.uniform(-0.03, 0.03),
                          int(rng.integers(60, 200)))
    if rng.random() < 0.7:   # 위치·크기 흔들기 (검출 박스 오차)
        sc = rng.uniform(0.88, 1.12); tx, ty = rng.uniform(-0.08, 0.08) * w, rng.uniform(-0.08, 0.08) * h
        M = np.float32([[sc, 0, (1 - sc) * w / 2 + tx], [0, sc, (1 - sc) * h / 2 + ty]])
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if rng.random() < tint_prob:   # 진한 빨간빛·파란빛 (대회장 조명)
        img = light_tint(img, rng)
    if rng.random() < 0.5:   # 약한 색 조명 (대회장 조명이 파랑·노랑·빨강으로 바뀐다)
        img = np.clip(img.astype(np.float32) * rng.uniform(0.7, 1.3, 3), 0, 255).astype(np.uint8)
    a, b = rng.uniform(0.6, 1.4), rng.uniform(-35, 35)   # 밝기·대비
    img = np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:   # 작은 표지판 흉내
        f = rng.uniform(0.25, 1.0); s = max(8, int(round(min(h, w) * f)))
        img = cv2.resize(cv2.resize(img, (s, s), interpolation=cv2.INTER_AREA), (w, h), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.3:
        k = int(rng.choice([3, 5])); img = cv2.GaussianBlur(img, (k, k), 0)
    if rng.random() < 0.4:
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(30, 90))])
        img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return img


def to_input(img):
    return rule.preprocess(img, rgb=False, size=SIZE).astype(np.float32) / 255.0   # 크롭은 BGR png


@torch.no_grad()
def features(backbone, arrs, dev, bs=256):
    out = []
    for i in range(0, len(arrs), bs):
        x = torch.from_numpy(np.stack(arrs[i:i + bs])[:, None]).to(dev)
        out.append(backbone(x).cpu())
    return torch.cat(out) if out else torch.zeros(0, backbone.m.fc.in_features)


def build_set(items, copies, rng, flip_b=False, mask_prob=0.0, tint_prob=0.0):
    X, Y = [], []
    for f, y in items:
        img = cv2.imread(f)
        if img is None:
            continue
        variants = [(img, y)]
        if flip_b and y == 1:
            variants.append((cv2.flip(img, 1), 0))   # 거울 B → A 대용
        for im, yy in variants:
            X.append(to_input(im)); Y.append(yy)
            for _ in range(copies):
                X.append(to_input(augment(im, rng, mask_prob, tint_prob))); Y.append(yy)
    return X, np.array(Y, np.int64)


def evaluate(probs, y, min_prob=0.6, min_margin=0.2):
    """로봇 고르기 규칙과 같게: max(pA,pB) >= min_prob 이고 |pA-pB| >= min_margin 이면 A/B, 아니면 none."""
    pa, pb = probs[:, 0], probs[:, 1]
    sign = (np.maximum(pa, pb) >= min_prob) & (np.abs(pa - pb) >= min_margin)
    pred = np.where(sign, np.where(pa >= pb, 0, 1), 2)
    cm = np.zeros((3, 3), int)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    r = {"n": int(len(y)), "confusion[true][pred] A,B,none": cm.tolist(),
         "A<->B": int(cm[0, 1] + cm[1, 0]),
         "sign_recall": float((cm[0, 0] + cm[1, 1]) / max(1, cm[0].sum() + cm[1].sum())),
         "none_as_sign": float((cm[2, 0] + cm[2, 1]) / max(1, cm[2].sum()))}
    return r, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--real", nargs="*", default=[], help="실차 폴더 (증강 많이, --flip-real-b 대상)")
    ap.add_argument("--eval", nargs="*", default=[], help="이름=폴더")
    ap.add_argument("--flip-real-b", action="store_true")
    ap.add_argument("--mask-prob", type=float, default=0.0, help="증강에서 표지판 원 바깥을 가릴 확률 (배경 지름길 막기)")
    ap.add_argument("--tint-prob", type=float, default=0.0, help="증강에서 진한 빨간빛·파란빛 조명을 씌울 확률 (대회장 조명)")
    ap.add_argument("--synth-none-max", type=int, default=4000)
    ap.add_argument("--synth-copies", type=int, default=1)
    ap.add_argument("--real-copies", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--feat", choices=list(FEAT_DIM), default="l3", help="특징: l3(9216, 기본) | l4(4608) | gap(512)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed); random.seed(a.seed); torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    backbone = folded_backbone(a.feat).to(dev)
    t0 = time.time()

    Xtr, Ytr = [], []
    for d in a.train:
        items = load_dir(d)
        real = os.path.abspath(d) in {os.path.abspath(r) for r in a.real}
        if not real:   # 합성 none 은 너무 많아 일부만
            nn_ = [it for it in items if it[1] == 2]; rest = [it for it in items if it[1] != 2]
            random.shuffle(nn_); items = rest + nn_[:a.synth_none_max]
        X, Y = build_set(items, a.real_copies if real else a.synth_copies, rng, flip_b=real and a.flip_real_b, mask_prob=a.mask_prob, tint_prob=a.tint_prob)
        Xtr += X; Ytr.append(Y)
        print(f"train {d}: {len(Y)} (A {int((Y == 0).sum())}, B {int((Y == 1).sum())}, none {int((Y == 2).sum())})")
    Ytr = np.concatenate(Ytr)
    Ftr = features(backbone, Xtr, dev)
    print(f"특징 {tuple(Ftr.shape)} ({time.time() - t0:.0f} s)")

    head = nn.Linear(FEAT_DIM[a.feat], 3)
    cnt = np.bincount(Ytr, minlength=3).astype(np.float32)
    lossf = nn.CrossEntropyLoss(weight=torch.tensor(cnt.sum() / (3 * np.maximum(cnt, 1))))
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    yt = torch.from_numpy(Ytr)
    for ep in range(a.epochs):   # 특징을 미리 뽑아 두었으므로 전체 배치
        opt.zero_grad(); loss = lossf(head(Ftr), yt); loss.backward(); opt.step()
    print(f"선형층 학습 끝 loss {loss.item():.4f}")
    head = head.eval()
    torch.save({"head": head.state_dict(), "labels": LABELS, "size": SIZE, "feat": a.feat, "args": vars(a)}, os.path.join(a.out, "head.pt"))

    model = ForkCls(backbone, head.to(dev)).eval()
    report = {"train_counts": cnt.tolist()}
    for spec in a.eval:
        name, d = spec.split("=", 1)
        items = load_dir(d)
        X = [to_input(cv2.imread(f)) for f, _ in items]; y = np.array([l for _, l in items])
        with torch.no_grad():
            P = torch.cat([model(torch.from_numpy(np.stack(X[i:i + 256])[:, None]).to(dev)).cpu()
                           for i in range(0, len(X), 256)]).numpy() if X else np.zeros((0, 3))
        r, pred = evaluate(P, y)
        if a.flip_real_b or name.startswith("real"):   # 실차 B 를 거울로 뒤집어 A 대용 시험도
            Xf = [to_input(cv2.flip(cv2.imread(f), 1)) for f, l in items if l == 1]
            if Xf:
                with torch.no_grad():
                    Pf = model(torch.from_numpy(np.stack(Xf)[:, None]).to(dev)).cpu().numpy()
                rf, _ = evaluate(Pf, np.zeros(len(Xf), np.int64))
                r["flipped_B_as_A"] = {"n": rf["n"], "recall_A": rf["sign_recall"], "called_B": int(rf["confusion[true][pred] A,B,none"][0][1])}
        report[name] = r
        print(f"{name:14s} n={r['n']:4d} A<->B {r['A<->B']:3d}  sign_recall {r['sign_recall']:.3f}  none_as_sign {r['none_as_sign']:.3f}"
              + (f"  flipB→A recall {r['flipped_B_as_A']['recall_A']:.3f}" if "flipped_B_as_A" in r else ""))
        bad = [(items[i][0], int(y[i]), int(pred[i]), P[i]) for i in range(len(y)) if pred[i] != y[i]]
        if bad:   # 틀린 크롭 몽타주 (최대 48장)
            tiles = []
            for f, t, p, pr in bad[:48]:
                im = cv2.resize(cv2.imread(f), (96, 96))
                cv2.putText(im, f"{LABELS[t]}>{LABELS[p]}", (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                cv2.putText(im, f"{pr[0]:.2f}/{pr[1]:.2f}/{pr[2]:.2f}", (2, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 255, 255), 1)
                tiles.append(im)
            tiles += [np.zeros_like(tiles[0])] * ((-len(tiles)) % 12)
            cv2.imwrite(os.path.join(a.out, f"errors_{name}.png"),
                        np.vstack([np.hstack(tiles[i:i + 12]) for i in range(0, len(tiles), 12)]))
    json.dump(report, open(os.path.join(a.out, "metrics.json"), "w"), indent=1, ensure_ascii=False)
    print(f"끝 ({time.time() - t0:.0f} s) → {a.out}")


if __name__ == "__main__":
    main()
