#!/usr/bin/env python3
"""HybridNets 다중 작업 학습: 분할(da/ll/흰 가로선) + 검출(표지판·신호등·불빛·limo·cone 12클래스).

데이터
  --seg-data (synth_signs_v2): ego_v4 프레임 + 합성 표지판/신호등. 한 장에 분할 마스크(da, ll)와 YOLO 박스가 같이 있음.
             11/06 대회장 프레임 일부만 흰 가로선 라벨(sl_masks)과 그 라벨을 믿을 픽셀(sl_valid)이 있음.
  --det-data (cones_v1): 공개 라바콘(FSOCO, TraCon). 박스만 있음. 콘이 있는 사진만, --cone-train-n장 고정 추출.
손실 (이미지마다 가진 라벨만)
  분할: 출력 4채널 softmax(bg/도로/차선/흰 가로선). da = 도로+차선+가로선, ll = 차선, sl = 가로선
        seg_loss(da) + ll_weight * seg_loss(ll)               ← 분할 라벨이 있는 이미지만
        sl_weight * (BCE + Dice), sl_valid 픽셀만               ← 가로선 라벨이 있는 이미지만
  검출: HybridNets FocalLoss(분류 + 회귀) × det_weight        ← 모든 이미지(박스 없는 이미지는 음성 샘플)
시작 가중치: --init(기본 4차 best_da.pt). 분할 머리는 4번째 채널을 도로 채널에서 복사(편향 -4로 처음엔 거의 0),
            검출 분류 머리는 12클래스로 새로 만들고, 예전 'car' 클래스 가중치를 limo 자리에 옮김.
증강: train_seg.py와 같은 사진 증강 + 대회장 조명 증강(분할 이미지만). 좌우 반전은 A/B 표지판이 있으면 하지 않음
      (화살표 방향이 바뀌어 라벨이 틀어짐). 회전·확대는 박스도 같이 변환.
평가(매 epoch, val): da/ll IoU, 클래스별 AP50(합성 val + 콘 val). best는 da+ll IoU + 표지판·불빛 mAP50 합으로 고름.
     끝나면 best로 test / test_venue(가로선 IoU 포함) / test_bag 채점 -> test.json
저장: <out>/best.pt, last.pt(재개), log.csv, test.json. 같은 명령을 다시 실행하면 last.pt부터 이어서.

사용 예:
  python scripts/train_multitask.py --seg-data dataset/synth_signs_v2 --det-data dataset/cones_v1 \\
      --venue-ref-dir dataset/venue_ref_20261106 --init outputs/train/hybridnets_v4_seed0/best_da.pt --batch 16
"""
import argparse, csv, json, os, random, sys, time
from argparse import Namespace

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_seg as ts  # 사진 증강 함수, seg_loss, IoU (CUDA 초기화 없음)

PAD = ts.PAD
H_IN, W_IN = 360, 640
CLASSES = ["pedestrian", "sign_A", "sign_B", "bump", "slow", "parking",
           "traffic_light", "light_red", "light_yellow", "light_green", "limo", "cone"]
NC = len(CLASSES)
AB = {CLASSES.index("sign_A"), CLASSES.index("sign_B")}
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
MAX_BOX = 64


# ---------------------------------------------------------------------------
def read_yolo(path, W, H):
    out = []
    if os.path.exists(path):
        for line in open(path):
            p = line.split()
            if len(p) == 5:
                c, cx, cy, w, h = int(p[0]), *map(float, p[1:])
                out.append([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H, c])
    return np.array(out, np.float32).reshape(-1, 5)


def affine_boxes(b, M, W, H, min_vis=0.4):
    if not len(b):
        return b
    pts = np.stack([b[:, [0, 1]], b[:, [2, 1]], b[:, [2, 3]], b[:, [0, 3]]], 1)            # N,4,2
    pts = pts @ M[:, :2].T + M[:, 2]
    x0, y0 = pts[..., 0].min(1), pts[..., 1].min(1)
    x1, y1 = pts[..., 0].max(1), pts[..., 1].max(1)
    full = (x1 - x0) * (y1 - y0)
    cx0, cy0, cx1, cy1 = np.clip(x0, 0, W), np.clip(y0, 0, H), np.clip(x1, 0, W), np.clip(y1, 0, H)
    vis = (cx1 - cx0) * (cy1 - cy0) / np.maximum(full, 1e-6)
    keep = (vis >= min_vis) & (cx1 - cx0 >= 2) & (cy1 - cy0 >= 2)
    return np.stack([cx0, cy0, cx1, cy1, b[:, 4]], 1)[keep].astype(np.float32)


class Items(torch.utils.data.Dataset):
    """items: dict(img, da?, ll?, lab, sl?, slv?, kind='seg'|'det')"""

    def __init__(self, items, train, venue=None):
        self.items, self.train, self.venue = items, train, venue

    def __len__(self):
        return len(self.items)

    def load(self, it):
        img = cv2.imread(it["img"])
        if it["kind"] == "det":   # 콘 사진: 비율 유지해서 640x360 안에 넣고 회색으로 채움
            h, w = img.shape[:2]
            s = min(W_IN / w, H_IN / h)
            nw, nh = round(w * s), round(h * s)
            canvas = np.full((H_IN, W_IN, 3), 114, np.uint8)
            ox, oy = (W_IN - nw) // 2, (H_IN - nh) // 2
            canvas[oy:oy + nh, ox:ox + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
            b = read_yolo(it["lab"], w, h)
            if len(b):
                b[:, :4] = b[:, :4] * s + [ox, oy, ox, oy]
            z = np.zeros((H_IN, W_IN), np.uint8)
            return canvas, z, z.copy(), z.copy(), z.copy(), b
        h, w = img.shape[:2]
        assert (h, w) == (H_IN, W_IN), it["img"]
        rd = lambda p: (cv2.imread(p, 0) > 0).astype(np.uint8) if p and os.path.exists(p) else np.zeros((H_IN, W_IN), np.uint8)
        return img, rd(it["da"]), rd(it["ll"]), rd(it.get("sl")), rd(it.get("slv")), read_yolo(it["lab"], w, h)

    def augment(self, img, masks, b, seg):
        h, w = img.shape[:2]
        lit = False
        if seg and self.venue and random.random() < self.venue[1]:
            import venue_aug as va
            if not va.REFS:
                va.set_refs(self.venue[0])
            fn = va.v4 if random.random() < self.venue[2] else va.v3
            if random.random() < self.venue[3]:
                fn = va.v5
            img, lit = fn(img, va.random_ref(), masks[0] > 0, masks[1] > 0), True
        if random.random() < 0.5 and not (len(b) and np.isin(b[:, 4], list(AB)).any()):
            img = img[:, ::-1]
            masks = [m[:, ::-1] for m in masks]
            if len(b):
                b = b.copy(); b[:, [0, 2]] = w - b[:, [2, 0]]
        if random.random() < 0.5:
            M = cv2.getRotationMatrix2D((w / 2, h / 2), random.uniform(-5, 5), random.uniform(0.9, 1.1))
            M[:, 2] += (random.uniform(-0.05, 0.05) * w, random.uniform(-0.05, 0.05) * h)
            img = cv2.warpAffine(np.ascontiguousarray(img), M, (w, h), borderValue=(114, 114, 114))
            masks = [cv2.warpAffine(np.ascontiguousarray(m), M, (w, h), flags=cv2.INTER_NEAREST) for m in masks]
            b = affine_boxes(b, M, w, h)
        if random.random() < 0.5:
            r = np.random.uniform(-1, 1, 3) * (0.015, 0.7, 0.4) + 1
            hsv = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[..., 0] = (hsv[..., 0] * r[0]) % 180
            hsv[..., 1:] = np.clip(hsv[..., 1:] * r[1:], 0, 255)
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        img = np.ascontiguousarray(img)
        masks = [np.ascontiguousarray(m) for m in masks]
        if seg and random.random() < 0.3:
            img = ts.lane_gap(img, masks[1])
        weak = 0.3 if lit else 1.0
        if random.random() < 0.15 * weak:
            img = ts.overexpose(img)
        if random.random() < 0.25 * weak:
            img = ts.light_bloom(img)
        if seg and random.random() < 0.2:
            img = ts.floor_streak(img)
        r = random.random()
        if r < 0.15:
            img = ts.motion_blur(img)
        elif r < 0.25:
            img = ts.zoom_blur(img)
        elif r < 0.35:
            img = cv2.GaussianBlur(img, (0, 0), random.uniform(0.5, 1.5))
        return img, masks, b

    def __getitem__(self, i):
        it = self.items[i]
        img, da, ll, sl, slv, b = self.load(it)
        seg = it["kind"] == "seg"
        if self.train:
            img, (da, ll, sl, slv), b = self.augment(img, [da, ll, sl, slv], b, seg)
        img = cv2.copyMakeBorder(img, PAD, PAD, 0, 0, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        x = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 - MEAN) / STD
        ann = np.full((MAX_BOX, 5), -1, np.float32)
        if len(b):
            b = b[:MAX_BOX].copy(); b[:, [1, 3]] += PAD
            ann[:len(b)] = b
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float()
        return (torch.from_numpy(x.transpose(2, 0, 1).copy()), t(da), t(ll), t(sl), t(slv),
                torch.tensor(float(seg)), torch.tensor(float(it.get("sl") is not None)), torch.from_numpy(ann), it["name"])


def seg_items(root, split):
    d = os.path.join(root, split)
    if not os.path.isdir(os.path.join(d, "images")):
        return []
    out = []
    for n in sorted(os.listdir(os.path.join(d, "images"))):
        stem = n[:-4]
        sl = os.path.join(d, "sl_masks", n)
        has_sl = os.path.exists(sl)
        out.append(dict(kind="seg", name=f"{split}/{stem}", img=os.path.join(d, "images", n),
                        da=os.path.join(d, "da_masks", n), ll=os.path.join(d, "ll_masks", n),
                        lab=os.path.join(d, "labels", stem + ".txt"),
                        sl=sl if has_sl else None, slv=os.path.join(d, "sl_valid", n) if has_sl else None))
    return out


def det_items(root, split, n_max, seed, positives_only=True):
    d = os.path.join(root, "images", split)
    if not os.path.isdir(d):
        return []
    out = []
    for n in sorted(os.listdir(d)):
        stem = os.path.splitext(n)[0]
        lab = os.path.join(root, "labels", split, stem + ".txt")
        if positives_only and (not os.path.exists(lab) or os.path.getsize(lab) == 0):
            continue
        out.append(dict(kind="det", name=f"cone/{split}/{stem}", img=os.path.join(d, n), lab=lab))
    rnd = random.Random(seed)
    rnd.shuffle(out)
    return out[:n_max] if n_max else out


# ---------------------------------------------------------------------------
def add_det_skip(model):
    """실험 B: 백본 p3(stride 8) 특징을 1x1 conv로 BiFPN P3에 더해 검출 머리에만 넣는 지름길.
    linear probe에서 백본 p3는 A/B를 약 95% 구분하는데 BiFPN P3는 약 85%라, 고운 특징을 검출 머리에 직접 준다.
    가중치·편향을 0으로 시작해서 처음에는 원래 모델과 출력이 같다. 분할(bifpndecoder)에는 원래 BiFPN 특징을 쓴다."""
    c_in = model.encoder.out_channels[-3]
    c_out = model.fpn_num_filters[model.compound_coef]
    model.det_skip = torch.nn.Conv2d(c_in, c_out, 1)
    torch.nn.init.zeros_(model.det_skip.weight); torch.nn.init.zeros_(model.det_skip.bias)


def build_model(m, init, enc_skip=False):
    repo = os.path.join(m.MODELS_DIR, "HybridNets")
    HB = m.import_from_repo(repo, "backbone", "HybridNetsBackbone")
    model = HB(compound_coef=3, num_classes=NC, ratios=[(0.62, 1.58), (1.0, 1.0), (1.58, 0.62)],
               scales=[2 ** 0, 2 ** 0.70, 2 ** 1.32], seg_classes=3, backbone_name=None, seg_mode="multiclass")
    sd = torch.load(init, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd and not any(k.startswith("encoder.") for k in sd):
        sd = sd["model"]
    if enc_skip or any(k.startswith("det_skip.") for k in sd):
        add_det_skip(model)
    own = model.state_dict()
    new = {}
    for k, v in sd.items():
        if k not in own:
            continue
        if own[k].shape == v.shape:
            new[k] = v
        elif k.startswith("segmentation_head") and v.shape[0] == 3 and own[k].shape[0] == 4:   # 4번째 채널 = 흰 가로선
            w = torch.cat([v, v[1:2]], 0)
            if v.dim() == 1:
                w[3] = v[1] - 4.0
            new[k] = w
        elif k.startswith("classifier.header.pointwise_conv.conv") and v.shape[0] * NC == own[k].shape[0]:   # car -> limo
            # 새 클래스는 처음 점수가 약 1%가 되게(EfficientDet 관례: 편향 -log(99)), 가중치는 작게
            w = own[k].clone()
            if w.dim() == 1:
                w.fill_(-float(np.log(99.0)))
            else:
                torch.nn.init.normal_(w, std=0.01)
            A = v.shape[0]
            for a in range(A):
                w[a * NC + CLASSES.index("limo")] = v[a]
            new[k] = w
    missing = [k for k in own if k not in new]
    model.load_state_dict(new, strict=False)
    print(f"init {init}: loaded {len(new)}/{len(own)} tensors, new {len(missing)} (예: {missing[:3]})", flush=True)
    return model


def forward_mt(model, x):
    """HybridNetsBackbone.forward 와 같고, det_skip 이 있으면 검출 머리에 들어가는 P3에만 더한다."""
    p2, p3, p4, p5 = model.encoder(x)[-4:]
    feats = model.bifpn((p3, p4, p5))
    seg = model.segmentation_head(model.bifpndecoder((p2, *feats)))
    if hasattr(model, "det_skip"):
        feats = (feats[0] + model.det_skip(p3), *feats[1:])
    return feats, model.regressor(feats), model.classifier(feats), model.anchors(x, x.dtype), seg


def heads(model, x):
    _, reg, cls, anchors, seg = forward_mt(model, x)
    p = torch.softmax(seg.float(), 1)
    return reg, cls, anchors, p[:, 1] + p[:, 2] + p[:, 3], p[:, 2], p[:, 3]


def masked_seg_loss(p, t, v):
    p = p.clamp(1e-6, 1 - 1e-6)
    bce = (F.binary_cross_entropy(p, t, reduction="none") * v).sum() / v.sum().clamp(min=1)
    inter = (p * t * v).sum((1, 2))
    dice = 1 - (2 * inter + 1) / ((p * v).sum((1, 2)) + (t * v).sum((1, 2)) + 1)
    return bce + dice.mean()


# ---------------------------------------------------------------------------
def ap50(dets, gts, ncls):
    """dets: [(img_id, cls, score, box)], gts: {img_id: array(N,5)} -> {cls: AP or None}"""
    res = {}
    for c in range(ncls):
        g = {k: v[v[:, 4] == c, :4] for k, v in gts.items()}
        npos = sum(len(v) for v in g.values())
        if npos == 0:
            res[c] = None
            continue
        used = {k: np.zeros(len(v), bool) for k, v in g.items()}
        dc = sorted([d for d in dets if d[1] == c], key=lambda d: -d[2])
        tp = np.zeros(len(dc)); fp = np.zeros(len(dc))
        for j, (k, _, s, b) in enumerate(dc):
            gg = g.get(k, np.zeros((0, 4)))
            if len(gg):
                ix0, iy0 = np.maximum(gg[:, 0], b[0]), np.maximum(gg[:, 1], b[1])
                ix1, iy1 = np.minimum(gg[:, 2], b[2]), np.minimum(gg[:, 3], b[3])
                inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
                iou = inter / ((gg[:, 2] - gg[:, 0]) * (gg[:, 3] - gg[:, 1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)
                bi = int(np.argmax(iou))
                if iou[bi] >= 0.5 and not used[k][bi]:
                    used[k][bi] = True; tp[j] = 1; continue
            fp[j] = 1
        tp, fp = np.cumsum(tp), np.cumsum(fp)
        rec = tp / npos
        prec = tp / np.maximum(tp + fp, 1e-9)
        mrec = np.r_[0, rec, 1]; mpre = np.r_[0, prec, 0]
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        res[c] = float(((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).sum())
    return res


@torch.no_grad()
def evaluate(model, loader, device, util, amp, groups=None):
    model.eval()
    reg_t, clip_t = util.BBoxTransform(), util.ClipBoxes()
    iou = {"all": [ts.IoU(), ts.IoU()]}
    for g in groups or {}:
        iou[g] = [ts.IoU(), ts.IoU()]
    sl_iou = ts.IoU()
    dets, gts = [], {}
    for x, da, ll, sl, slv, segv, slf, ann, names in loader:
        x = x.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            reg, cls, anchors, pda, pll, psl = heads(model, x)
        pda, pll, psl = ts.crop(pda).float().cpu(), ts.crop(pll).float().cpu(), ts.crop(psl).float().cpu()
        out = util.postprocess(x, anchors.float(), reg.float(), cls.float(), reg_t, clip_t, 0.05, 0.5)
        for k in range(len(names)):
            if segv[k] > 0:
                for g, (a, b) in iou.items():
                    if g == "all" or names[k] in groups[g]:
                        a.add(pda[k], da[k]); b.add(pll[k], ll[k])
                if slf[k] > 0:
                    v = slv[k] > 0
                    sl_iou.add(psl[k][v], sl[k][v])
            a = ann[k].numpy(); a = a[a[:, 4] >= 0]; a[:, [1, 3]] -= PAD
            gts[names[k]] = a
            o = out[k]
            top = np.argsort(-np.asarray(o["scores"]))[:100] if len(o["scores"]) else []   # 이미지당 상위 100개만
            for j in top:
                b, c, s = o["rois"][j], o["class_ids"][j], o["scores"][j]
                bb = np.array(b, np.float32); bb[[1, 3]] -= PAD
                dets.append((names[k], int(c), float(s), bb))
    ap = ap50(dets, gts, NC)
    res = {g: {"da_iou": round(a.value(), 4), "ll_iou": round(b.value(), 4)} for g, (a, b) in iou.items()}
    res["all"]["sl_iou"] = round(sl_iou.value(), 4) if (sl_iou.tp + sl_iou.fn) else None
    res["all"]["ap50"] = {CLASSES[c]: (round(v, 4) if v is not None else None) for c, v in ap.items()}
    sign_ap = [v for c, v in ap.items() if v is not None and CLASSES[c] not in ("traffic_light",)]
    res["all"]["map50"] = round(float(np.mean(sign_ap)), 4) if sign_ap else None
    return res


# ---------------------------------------------------------------------------
def collate(b):
    return [torch.stack(z) if torch.is_tensor(z[0]) else list(z) for z in zip(*b)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seg-data", default="dataset/synth_signs_v2")
    ap.add_argument("--det-data", default="dataset/cones_v1")
    ap.add_argument("--cone-train-n", type=int, default=2500, help="학습에 쓸 콘 사진 수(콘 있는 사진만)")
    ap.add_argument("--cone-val-n", type=int, default=300)
    ap.add_argument("--init", default="outputs/train/hybridnets_v4_seed0/best_da.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ll-weight", type=float, default=2.0)
    ap.add_argument("--sl-weight", type=float, default=2.0)
    ap.add_argument("--det-weight", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--venue-ref-dir", default=None)
    ap.add_argument("--venue-p", type=float, default=0.5)
    ap.add_argument("--venue-v4", type=float, default=0.8)
    ap.add_argument("--venue-v5", type=float, default=0.25)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--amp", action="store_true", help="bf16 autocast (H100 권장)")
    ap.add_argument("--max-steps", type=int, default=0, help="시험용: epoch당 최대 step")
    ap.add_argument("--val-max", type=int, default=0, help="시험용: val 이미지 수 제한")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/train/hybridnets_mt_v1_seed0")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--head-lr-mult", type=float, default=1.0,
                    help="검출 머리(regressor, classifier)·BiFPN·det_skip 의 lr 배수 (실험 A/B: 10)")
    ap.add_argument("--enc-skip", action="store_true", help="실험 B: 백본 p3 -> 검출 머리 지름길(det_skip) 추가")
    ap.add_argument("--warmup-steps", type=int, default=0, help="처음 이 step 동안 lr을 0에서 선형으로 올림")
    a = ap.parse_args()

    import make_model_compare_gif as m   # 장치·ZLUDA 설정
    os.chdir(m.ROOT)
    sys.path.insert(0, os.path.join(m.MODELS_DIR, "HybridNets"))
    from hybridnets.loss import FocalLoss
    import utils.utils as util
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    dev = m.device
    amp = a.amp and dev.type == "cuda"

    venue = (a.venue_ref_dir, a.venue_p, a.venue_v4, a.venue_v5) if a.venue_ref_dir and a.venue_ref_dir != "none" else None
    tr = seg_items(a.seg_data, "train") + det_items(a.det_data, "train", a.cone_train_n, a.seed)
    va_ = seg_items(a.seg_data, "val") + det_items(a.det_data, "val", a.cone_val_n, a.seed)
    if a.val_max:
        va_ = va_[:a.val_max // 2] + va_[-(a.val_max // 2):]
    print(f"train {len(tr)} (seg {sum(i['kind'] == 'seg' for i in tr)}, sl {sum(i.get('sl') is not None for i in tr)}, "
          f"cone {sum(i['kind'] == 'det' for i in tr)}) / val {len(va_)}", flush=True)
    mk = lambda items, train: torch.utils.data.DataLoader(
        Items(items, train, venue if train else None), batch_size=a.batch, shuffle=train, drop_last=train,
        num_workers=a.workers, persistent_workers=a.workers > 0, pin_memory=True, collate_fn=collate)
    tl, vl = mk(tr, True), mk(va_, False)

    model = build_model(m, a.init, a.enc_skip).to(dev)
    head_names = ("regressor.", "classifier.", "bifpn.", "det_skip.")
    g_head = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith(head_names)]
    g_rest = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith(head_names)]
    opt = torch.optim.AdamW([{"params": g_rest, "mult": 1.0}, {"params": g_head, "mult": a.head_lr_mult}],
                            lr=a.lr, weight_decay=5e-4)
    print(f"param groups: rest {sum(p.numel() for p in g_rest) / 1e6:.1f}M x1, "
          f"head+bifpn {sum(p.numel() for p in g_head) / 1e6:.1f}M x{a.head_lr_mult}", flush=True)
    focal = FocalLoss()
    start, best = 0, {"score": -1.0, "epoch": 0, "stopped": False}
    if not a.fresh and os.path.exists(os.path.join(a.out, "last.pt")):
        ck = torch.load(os.path.join(a.out, "last.pt"), map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start, best = ck["epoch"], ck["best"]
        print(f"재개: epoch {start}, best {best}", flush=True)
    log = os.path.join(a.out, "log.csv")
    if start == 0:
        with open(log, "w", encoding="utf-8") as f:
            f.write("epoch,lr,loss,seg,sl,cls,reg,val_da,val_ll,val_map50,sec\n")
    limiter = m.GpuDutyLimiter()

    for ep in range(start, a.epochs):
        if best["stopped"]:
            break
        lr = a.lr * (1 - ep / a.epochs) ** 1.5
        for g in opt.param_groups:
            g["lr"] = lr * g.get("mult", 1.0)
        model.train()
        t0, n, acc = time.time(), 0, np.zeros(5)
        for step, (x, da, ll, sl, slv, segv, slf, ann, _) in enumerate(tl):
            if a.max_steps and step >= a.max_steps:
                break
            gstep = ep * len(tl) + step
            if a.warmup_steps and gstep <= a.warmup_steps:
                for g in opt.param_groups:
                    g["lr"] = lr * g.get("mult", 1.0) * min(1.0, (gstep + 1) / a.warmup_steps)
            with limiter:
                x, da, ll, sl, slv = x.to(dev), da.to(dev), ll.to(dev), sl.to(dev), slv.to(dev)
                segv, slf, ann = segv.to(dev), slf.to(dev), ann.to(dev)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    reg, cls, anchors, pda, pll, psl = heads(model, x)
                pda, pll, psl = ts.crop(pda).float(), ts.crop(pll).float(), ts.crop(psl).float()
                zero = x.sum() * 0
                sv = segv > 0
                l_seg = (ts.seg_loss(pda[sv], da[sv]) + a.ll_weight * ts.seg_loss(pll[sv], ll[sv])) if sv.any() else zero
                lv = (slf > 0)
                l_sl = masked_seg_loss(psl[lv], sl[lv], slv[lv]) if lv.any() else zero
                c_l, r_l = focal(cls.float(), reg.float(), anchors.float(), ann)
                l_det = c_l.mean() + r_l.mean()
                loss = l_seg + a.sl_weight * l_sl + a.det_weight * l_det
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                opt.step()
            acc += [float(loss), float(l_seg), float(l_sl), float(c_l.mean()), float(r_l.mean())]
            n += 1
            if step % 100 == 0:
                print(f"  ep {ep + 1} step {step}/{len(tl)} loss {float(loss):.3f} seg {float(l_seg):.3f} "
                      f"sl {float(l_sl):.3f} cls {float(c_l.mean()):.3f} reg {float(r_l.mean()):.3f}", flush=True)
        acc /= max(n, 1)
        v = evaluate(model, vl, dev, util, amp)["all"]
        score = v["da_iou"] + v["ll_iou"] + (v["map50"] or 0)
        if score > best["score"] + 1e-3:
            best.update(score=score, epoch=ep + 1, val=v)
            ts.save_state(model, os.path.join(a.out, "best.pt"))
        best["stopped"] = a.patience > 0 and ep + 1 - best["epoch"] >= a.patience
        ck = {"model": model.state_dict(), "opt": opt.state_dict(), "epoch": ep + 1, "best": best}
        ts.atomic(lambda p: torch.save(ck, p), os.path.join(a.out, "last.pt"))
        sec = time.time() - t0
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"{ep + 1},{lr:.3g},{acc[0]:.4f},{acc[1]:.4f},{acc[2]:.4f},{acc[3]:.4f},{acc[4]:.4f},"
                    f"{v['da_iou']},{v['ll_iou']},{v['map50']},{sec:.0f}\n")
        print(f"epoch {ep + 1}/{a.epochs} loss {acc[0]:.4f} | val da {v['da_iou']:.4f} ll {v['ll_iou']:.4f} "
              f"mAP50 {v['map50']} | AP {json.dumps(v['ap50'])} | {sec:.0f}s", flush=True)

    # test: best 가중치로 test / test_venue / test_bag
    model.load_state_dict(torch.load(os.path.join(a.out, "best.pt"), map_location=dev))
    res = {"best": best, "args": vars(a)}
    for split in ("test", "test_venue", "test_bag"):
        items = seg_items(a.seg_data, split)
        if split == "test":
            items += det_items(a.det_data, "test", 0, a.seed) + det_items(a.det_data, "val", 0, a.seed)[a.cone_val_n:]
        if items:
            res[split] = evaluate(model, mk(items, False), dev, util, amp)
            print(split, json.dumps(res[split]["all"]), flush=True)
    ts.atomic(lambda p: json.dump(res, open(p, "w", encoding="utf-8"), indent=2, default=str), os.path.join(a.out, "test.json"))


if __name__ == "__main__":
    main()
