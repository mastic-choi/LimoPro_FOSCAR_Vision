#!/usr/bin/env python3
"""앙상블 멤버(m.ENSEMBLE)를 ego lane 라벨(build_train_dataset.py 결과)로 파인튜닝한다.

모델마다 출력 형식이 달라서 어댑터로 "da 전경 확률, ll 전경 확률"(B,H,W)로 통일하고 같은 손실/평가를 쓴다.
확률 정의는 make_model_compare_gif.py의 probs()와 같다(앙상블 추론과 일치).

  | 모델       | 입력                 | 출력 -> 확률                                   |
  |------------|----------------------|------------------------------------------------|
  | tln_large  | RGB /255             | da/ll 2ch logit -> softmax[1]                   |
  | yolop      | BGR ImageNet 정규화  | da/ll 채널별 sigmoid 2ch -> c1/(c0+c1)          |
  | hybridnets | RGB ImageNet 정규화  | 3ch logit(bg/road/lane) -> da = road+lane, ll = lane |
  | yolopv2    | RGB /255             | da 2ch -> c1/(c0+c1), ll 1ch 확률 그대로 (TorchScript) |

입력: 라벨 640x360 그대로, 이미지는 위아래 12px 패딩해서 640x384 (공식 BDD100K 학습/our pad_to_stride와 같음).
손실: BCE + soft Dice, da + ll_weight * ll.  평가: val IoU(da/ll, 0.5 threshold) 매 epoch.
저장: <out>/best_da.pt, best_ll.pt, last.pt(재개용), log.csv, test.json(best_da 기준 test IoU, 커브/직진 별도).
재개: <out>/last.pt가 있으면 자동으로 이어서 학습(매 epoch 저장, 임시 파일에 쓴 뒤 교체해서 전원이 꺼져도 안 깨짐).
GPU 사용률은 GPU_DUTY 환경변수(기본 0.9 = 90%)로 제한. 4개를 순서대로 돌리려면 scripts/train_all.py.

사용 예:
  .venv/Scripts/python scripts/train_seg.py --model tln_large --data dataset/ego_v1 --seed 0
"""
import argparse
import csv
import json
import os
import random
import sys
import time
from argparse import Namespace

import cv2
import numpy as np
import torch
import torch.nn.functional as F

PAD = 12  # 360 -> 384


# ---------------------------------------------------------------------------
# 사진 증강 (밝기/흐림만 바꿔서 라벨은 그대로)
# ---------------------------------------------------------------------------
def motion_blur(img):
    """빠르게 달릴 때/차체 흔들림: 임의 방향 직선 블러."""
    k = random.randint(9, 25)
    ker = np.zeros((k, k), np.float32)
    ker[k // 2] = 1
    ker = cv2.warpAffine(ker, cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), random.uniform(0, 180), 1), (k, k))
    return cv2.filter2D(img, -1, ker / max(ker.sum(), 1e-6))


def zoom_blur(img):
    """전진 주행 블러: 소실점(화면 위쪽 가운데 근처)에서 바깥으로 번짐 = 조금씩 확대한 사본 평균."""
    h, w = img.shape[:2]
    cx, cy = w * random.uniform(0.4, 0.6), h * random.uniform(0.2, 0.45)
    acc = img.astype(np.float32)
    n, amount = 8, random.uniform(0.04, 0.10)
    for i in range(1, n):
        M = cv2.getRotationMatrix2D((cx, cy), 0, 1 + amount * i / n)
        acc += cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT).astype(np.float32)
    return (acc / n).astype(np.uint8)


def overexpose(img):
    """노출 과다: 밝은 곳(벽/하늘/조명)이 하얗게 날아감."""
    gain, gamma = random.uniform(1.6, 2.6), random.uniform(0.5, 0.8)
    x = np.clip((img.astype(np.float32) / 255.0) * gain, 0, 1) ** gamma
    return (x * 255).astype(np.uint8)


def light_bloom(img):
    """위쪽 조명/창문에서 강한 빛이 들어와 번짐: 화면 위쪽에 큰 밝은 원형 그라디언트를 더함."""
    h, w = img.shape[:2]
    cx, cy = random.uniform(0, w), random.uniform(-0.3 * h, 0.4 * h)
    r = random.uniform(0.4, 1.0) * w
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    g = np.clip(1 - np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / r, 0, 1) ** 1.5
    tint = np.array([random.uniform(0.85, 1.0), random.uniform(0.9, 1.0), 1.0], np.float32)  # 살짝 따뜻한 흰빛
    x = img.astype(np.float32) + g[..., None] * random.uniform(180, 320) * tint
    return np.clip(x, 0, 255).astype(np.uint8)


def floor_streak(img):
    """바닥(매트)에 빛이 반사돼 생기는 길쭉한 밝은 띠 (KMU 증강의 바닥 반사 스트릭)."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.float32)
    x0 = random.uniform(0.1, 0.9) * w
    top = (int(x0 + random.uniform(-0.05, 0.05) * w), int(random.uniform(0.35, 0.55) * h))
    bot = (int(x0 + random.uniform(-0.3, 0.3) * w), h)
    cv2.line(mask, top, bot, 1.0, random.randint(20, 60))
    mask = cv2.GaussianBlur(mask, (0, 0), random.uniform(8, 20))
    mask *= np.linspace(0.3, 1.0, h, dtype=np.float32)[:, None]  # 가까울수록 밝게
    x = img.astype(np.float32) + mask[..., None] * random.uniform(100, 200)
    return np.clip(x, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 데이터 (DataLoader worker가 import하므로 여기선 make_model_compare_gif를 import하지 않음: CUDA 초기화 방지)
# ---------------------------------------------------------------------------
class EgoDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, train, rgb, imagenet_norm):
        self.dir = os.path.join(root, split)
        self.names = sorted(os.listdir(os.path.join(self.dir, "images")))
        self.train, self.rgb, self.norm = train, rgb, imagenet_norm

    def __len__(self):
        return len(self.names)

    def augment(self, img, da, ll):
        h, w = da.shape
        if random.random() < 0.5:  # 좌우 반전: ego lane 개념은 좌우 대칭
            img, da, ll = img[:, ::-1], da[:, ::-1], ll[:, ::-1]
        if random.random() < 0.5:  # 작은 회전/확대/이동 (차가 차선과 평행하지 않은 경우 흉내)
            M = cv2.getRotationMatrix2D((w / 2, h / 2), random.uniform(-5, 5), random.uniform(0.9, 1.1))
            M[:, 2] += (random.uniform(-0.05, 0.05) * w, random.uniform(-0.05, 0.05) * h)
            img = cv2.warpAffine(np.ascontiguousarray(img), M, (w, h), borderValue=(114, 114, 114))
            da = cv2.warpAffine(np.ascontiguousarray(da), M, (w, h), flags=cv2.INTER_NEAREST)
            ll = cv2.warpAffine(np.ascontiguousarray(ll), M, (w, h), flags=cv2.INTER_NEAREST)
        if random.random() < 0.5:  # HSV (BDD100K.augment_hsv 와 같은 범위)
            r = np.random.uniform(-1, 1, 3) * (0.015, 0.7, 0.4) + 1
            hsv = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[..., 0] = (hsv[..., 0] * r[0]) % 180
            hsv[..., 1:] = np.clip(hsv[..., 1:] * r[1:], 0, 255)
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        img = np.ascontiguousarray(img)
        if random.random() < 0.15:
            img = overexpose(img)
        if random.random() < 0.25:
            img = light_bloom(img)
        if random.random() < 0.2:
            img = floor_streak(img)
        r = random.random()
        if r < 0.15:
            img = motion_blur(img)
        elif r < 0.3:
            img = zoom_blur(img)
        elif r < 0.4:
            img = cv2.GaussianBlur(img, (0, 0), random.uniform(0.5, 1.5))
        return img, np.ascontiguousarray(da), np.ascontiguousarray(ll)

    def __getitem__(self, i):
        n = self.names[i]
        img = cv2.imread(os.path.join(self.dir, "images", n))
        da = (cv2.imread(os.path.join(self.dir, "da_masks", n), 0) > 0).astype(np.uint8)
        ll = (cv2.imread(os.path.join(self.dir, "ll_masks", n), 0) > 0).astype(np.uint8)
        if self.train:
            img, da, ll = self.augment(img, da, ll)
        img = cv2.copyMakeBorder(img, PAD, PAD, 0, 0, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        if self.rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = img.astype(np.float32) / 255.0
        if self.norm:
            x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        return torch.from_numpy(x.transpose(2, 0, 1).copy()), torch.from_numpy(da).float(), torch.from_numpy(ll).float(), n


# ---------------------------------------------------------------------------
# 모델 어댑터: build(m) -> (model, rgb, imagenet_norm, forward(model, x) -> (da_prob, ll_prob), save(model, path))
# ---------------------------------------------------------------------------
def ratio_fg(t):
    return t[:, 1] / (t[:, 0] + t[:, 1] + 1e-6)


def atomic(save_fn, path):
    """임시 파일에 저장 후 교체: 저장 중 전원이 꺼져도 기존 파일은 멀쩡함."""
    tmp = path + ".tmp"
    save_fn(tmp)
    os.replace(tmp, path)


def save_state(model, path):
    atomic(lambda p: torch.save(model.state_dict(), p), path)


def build_tln_large(m):
    repo = os.path.join(m.MODELS_DIR, "TwinLiteNetPlus")
    TwinLiteNetPlus = m.import_from_repo(repo, "model.model", "TwinLiteNetPlus")
    model = TwinLiteNetPlus(Namespace(config="large"))
    model.load_state_dict(torch.load(os.path.join(repo, "pretrained", "large.pth"), map_location="cpu"))

    def fwd(model, x):
        da, ll = model(x)
        return torch.softmax(da, 1)[:, 1], torch.softmax(ll, 1)[:, 1]
    return model, True, False, fwd, save_state


def build_yolop(m):
    repo = os.path.join(m.MODELS_DIR, "YOLOP")
    get_net = m.import_from_repo(repo, "lib.models", "get_net")
    model = get_net(False)
    model.load_state_dict(torch.load(os.path.join(repo, "weights", "End-to-end.pth"), map_location="cpu")["state_dict"])

    def fwd(model, x):
        _, da, ll = model(x)
        return ratio_fg(da), ratio_fg(ll)
    return model, False, True, fwd, save_state


def build_hybridnets(m):
    repo = os.path.join(m.MODELS_DIR, "HybridNets")
    HybridNetsBackbone = m.import_from_repo(repo, "backbone", "HybridNetsBackbone")
    model = HybridNetsBackbone(compound_coef=3, num_classes=1, ratios=[(0.62, 1.58), (1.0, 1.0), (1.58, 0.62)],
                               scales=[2**0, 2**0.70, 2**1.32], seg_classes=2, backbone_name=None, seg_mode="multiclass")
    model.load_state_dict(torch.load(os.path.join(repo, "weights", "hybridnets.pth"), map_location="cpu"))

    def fwd(model, x):
        p = torch.softmax(model(x)[-1], 1)
        return p[:, 1] + p[:, 2], p[:, 2]
    return model, True, True, fwd, save_state


def load_yolopv2_trainable(path, device):
    """TorchScript YOLOPv2를 학습 가능하게 로드.
    - 배포본이라 파라미터가 requires_grad=False로 저장돼 있음 -> 켬
    - batch_norm에 cudnn_enabled=True가 상수로 박혀 있어 ZLUDA에서 CUDNN_STATUS_EXECUTION_FAILED
      -> 추론용 로더처럼 freeze하면 가중치가 상수가 돼 학습 불가. 대신 그래프를 inline한 뒤 8번째 입력만 False로 교체
    - 그래프의 batch_norm은 추적 당시 eval 모드로 고정 -> 통계는 안 바뀌고 가중치만 학습됨
    """
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_texpr_fuser_enabled(False)
    torch._C._jit_override_can_fuse_on_gpu(False)
    model = torch.jit.load(path, map_location=device)
    if not torch.backends.cudnn.enabled:
        graph = model.graph
        torch._C._jit_pass_inline(graph)
        false_const = graph.insertConstant(False)
        false_const.node().moveBefore(next(graph.nodes()))
        for node in graph.findAllNodes("aten::batch_norm"):
            node.replaceInput(8, false_const)
    for p in model.parameters():
        p.requires_grad_(True)
    return model


def build_yolopv2(m):
    # TorchScript (학습 코드 미공개). 저장은 torch.jit.save, 다시 읽을 땐 load_yolopv2_trainable로 패치
    model = load_yolopv2_trainable(os.path.join(m.MODELS_DIR, "YOLOPv2", "data", "weights", "yolopv2.pt"), m.device)

    def fwd(model, x):
        _, seg, ll = model(x)
        return ratio_fg(seg), ll[:, 0]
    return model, True, False, fwd, lambda model, path: atomic(lambda p: torch.jit.save(model, p), path)


ADAPTERS = {"tln_large": build_tln_large, "yolop": build_yolop, "hybridnets": build_hybridnets, "yolopv2": build_yolopv2}
DEFAULT_LR = {"tln_large": 5e-5, "yolop": 1e-4, "hybridnets": 1e-4, "yolopv2": 1e-5}


# ---------------------------------------------------------------------------
def seg_loss(p, t):
    p = p.clamp(1e-6, 1 - 1e-6)
    bce = F.binary_cross_entropy(p, t)
    dice = 1 - (2 * (p * t).sum((1, 2)) + 1) / (p.sum((1, 2)) + t.sum((1, 2)) + 1)
    return bce + dice.mean()


def crop(p):
    return p[:, PAD:-PAD]


class IoU:
    def __init__(self):
        self.tp = self.fp = self.fn = 0

    def add(self, p, t):
        pb, tb = p >= 0.5, t >= 0.5
        self.tp += int((pb & tb).sum())
        self.fp += int((pb & ~tb).sum())
        self.fn += int((~pb & tb).sum())

    def value(self):
        return self.tp / max(self.tp + self.fp + self.fn, 1)


@torch.no_grad()
def evaluate(model, fwd, loader, device, groups=None):
    """groups: {name: set(파일명)} 이면 그룹별 IoU도 같이."""
    model.eval()
    res = {"all": (IoU(), IoU())}
    for g in groups or {}:
        res[g] = (IoU(), IoU())
    for x, da, ll, names in loader:
        pda, pll = fwd(model, x.to(device))
        pda, pll = crop(pda).cpu(), crop(pll).cpu()
        for k in range(len(names)):
            for g, (a, b) in res.items():
                if g == "all" or names[k] in groups[g]:
                    a.add(pda[k], da[k])
                    b.add(pll[k], ll[k])
    return {g: {"da_iou": round(a.value(), 4), "ll_iou": round(b.value(), 4), "n": None} for g, (a, b) in res.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(ADAPTERS), required=True)
    ap.add_argument("--data", required=True, help="build_train_dataset.py 결과 폴더")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=None, help="기본: 모델별 DEFAULT_LR")
    ap.add_argument("--ll-weight", type=float, default=2.0, help="ll 손실 가중치 (KMU에서 2.0이 효과 있었음)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=None, help="기본: outputs/train/<model>_seed<seed>")
    ap.add_argument("--fresh", action="store_true", help="last.pt가 있어도 무시하고 처음부터")
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import make_model_compare_gif as m  # ZLUDA 설정(cuDNN 끔 등) 포함

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = args.out or os.path.join(m.ROOT, "outputs", "train", f"{args.model}_seed{args.seed}")
    os.makedirs(out, exist_ok=True)
    lr0 = args.lr or DEFAULT_LR[args.model]

    model, rgb, norm, fwd, save = ADAPTERS[args.model](m)
    model = model.to(m.device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr0, weight_decay=5e-4)

    mk = lambda split, train: torch.utils.data.DataLoader(
        EgoDataset(args.data, split, train, rgb, norm), batch_size=args.batch, shuffle=train, drop_last=train,
        num_workers=args.workers, persistent_workers=args.workers > 0, pin_memory=True)
    train_loader, val_loader, test_loader = mk("train", True), mk("val", False), mk("test", False)

    start, best = 0, {"da": -1.0, "ll": -1.0}
    if not args.fresh and os.path.exists(os.path.join(out, "last.pt")):
        ck = torch.load(os.path.join(out, "last.pt"), map_location=m.device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start, best = ck["epoch"], ck["best"]
        print(f"재개: epoch {start}, best {best}", flush=True)

    print(f"[{args.model} seed {args.seed}] train {len(train_loader.dataset)} / val {len(val_loader.dataset)} / "
          f"test {len(test_loader.dataset)}, lr {lr0}, batch {args.batch}, epochs {args.epochs} -> {out}", flush=True)
    log_path = os.path.join(out, "log.csv")
    if start == 0:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("epoch,lr,loss,val_da_iou,val_ll_iou,sec\n")
    limiter = m.GpuDutyLimiter()

    for ep in range(start, args.epochs):
        lr = lr0 * (1 - ep / args.epochs) ** 1.5  # TwinLiteNetPlus poly_lr_scheduler 와 같은 모양
        for g in opt.param_groups:
            g["lr"] = lr
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, da, ll, _ in train_loader:
            with limiter:
                x, da, ll = x.to(m.device), da.to(m.device), ll.to(m.device)
                pda, pll = fwd(model, x)
                loss = seg_loss(crop(pda), da) + args.ll_weight * seg_loss(crop(pll), ll)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            tot, n = tot + float(loss), n + 1
        v = evaluate(model, fwd, val_loader, m.device)["all"]
        for k in ("da", "ll"):
            if v[f"{k}_iou"] > best[k]:
                best[k] = v[f"{k}_iou"]
                save(model, os.path.join(out, f"best_{k}.pt"))
        ck = {"model": model.state_dict(), "opt": opt.state_dict(), "epoch": ep + 1, "best": best}
        atomic(lambda p: torch.save(ck, p), os.path.join(out, "last.pt"))
        sec = time.time() - t0
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{ep + 1},{lr:.3g},{tot / max(n, 1):.4f},{v['da_iou']},{v['ll_iou']},{sec:.0f}\n")
        print(f"epoch {ep + 1}/{args.epochs}  loss {tot / max(n, 1):.4f}  val da {v['da_iou']:.4f} ll {v['ll_iou']:.4f}"
              f"  (best da {best['da']:.4f} ll {best['ll']:.4f})  {sec:.0f}s", flush=True)

    # test: best_da 가중치로, 커브/직진 따로
    with open(os.path.join(args.data, "manifest.csv"), encoding="utf-8") as f:
        man = [r for r in csv.DictReader(f) if r["split"] == "test"]
    groups = {"curve": {r["frame"] + ".png" for r in man if r["curve"] == "1"},
              "straight": {r["frame"] + ".png" for r in man if r["curve"] == "0"}}
    if args.model == "yolopv2":
        model = load_yolopv2_trainable(os.path.join(out, "best_da.pt"), m.device)
    else:
        model.load_state_dict(torch.load(os.path.join(out, "best_da.pt"), map_location=m.device))
    res = evaluate(model, fwd, test_loader, m.device, groups)
    for g in res:
        res[g]["n"] = len(test_loader.dataset) if g == "all" else len(groups[g])
    def write_json(p):
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"model": args.model, "seed": args.seed, "best_val": best, "test": res}, f, indent=2)
    atomic(write_json, os.path.join(out, "test.json"))
    print("test (best_da):", json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
