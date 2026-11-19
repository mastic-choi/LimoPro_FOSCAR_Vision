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
저장: <out>/best.pt, last.pt(재개), log.csv, test.json, calib.json. 같은 명령을 다시 실행하면 last.pt부터 이어서.

v3 옵션 (기본값이면 v2와 같은 학습)
  --ignore-small PX   √(박스 넓이) < PX인 박스는 '무시'(검출 손실에서 빠짐, 맞혀도 틀려도 벌 없음). 먼 물체는 차에서 안 쓰므로.
                      불빛(light_red/yellow/green)은 원래 작아서 제외
  --oversample-ab W   A/B/주차가 들어간 이미지를 W배 자주 뽑음 (헷갈리는 세 클래스에 집중)
  --calib-loss {none,mdca,sece,both} --calib-weight λ
                      학습 중 보정 손실 (mt_det_loss.DetLoss, 검출 분류 점수에만). sece = 미분 가능한 ECE(soft-binned)
  --patience N --min-delta d --es-smooth k   정체 정지: 최근 k epoch 평균 점수가 N epoch 동안 d 이상 안 오르면 멈춤 (0 = 끔)
  --overfit-drop g --overfit-epochs m   과적합 정지: train loss는 best 때보다 낮은데 val 점수가 best보다 g 넘게 낮은
                      epoch이 m번 연속이면 멈춤
  --es-val-loss N --es-min-delta d --best-by val_loss   val loss 조기 종료: 매 epoch val에서 학습과 같은 손실(증강 없음,
                      같은 작은 박스 무시)을 재고, 최저 epoch 뒤 N epoch 동안 d 넘게 안 낮아지면 멈춤. best.pt = val loss 최저.
                      점수(da+ll IoU+mAP50) 최고 가중치는 항상 best_score.pt
  흰 케이블 음성 샘플: <split>/neg_sl/<이름>.png가 있으면 그 픽셀은 '가로선 아님'(sl=0, 믿을 픽셀)으로 학습
  --aug-monitor-p p --aug-cable-p q   이미 만든 합성 이미지(v2)에 학습 중 바로 그려 넣는 음성 샘플(mt_online_aug.py):
                      신호등 뒤 단색 방해 모니터(확률 p, 신호등이 있는 이미지만), 트랙 위 흰 케이블(확률 q)
  --aug-monitor-dots r  그린 모니터 중 비율 r은 땡땡이 화면(켜진 불과 다른 색의 불빛 크기 원)
  --aug-monitor-real    모니터를 실제 카메라처럼(match,persp,screen,jitter 또는 all). 비교 그림: preview_monitor_real.py
  매 epoch val 검출 점수 보정 지표(D-ECE, MCE, NLL)를 log.csv에 기록, 학습 후 val로 temperature를 맞춰 calib.json에 저장

실물 신호등 옵션 (v3real, 기본값이면 쓰지 않음)
  --real-data DIR --real-repeat N   로봇 카메라로 찍은 실물 신호등 사진(박스만, scripts/make_real_tl_labels.py). train은 N배로,
                      val은 val에 더하고, test는 test_real로 따로 채점. 분할 라벨이 없어 차선 손실에는 들어가지 않음.
                      방해 모니터 증강(--aug-monitor-p)은 실물 사진에도 그림
  --det-data none     콘 데이터를 쓰지 않음 (출력 12클래스는 그대로, cone 출력은 학습 안 됨)
  --freeze-bn         BatchNorm 통계를 시작 가중치 값으로 고정. 큰 batch로 학습한 가중치를 작은 batch로 이어서 학습할 때

v4 옵션 (2026-11-16, 원본 HybridNets 에서 새로 학습, H100)
  --merge-ab          sign_A·sign_B 를 한 클래스(1번 칸 fork_sign)로 합침. A/B 는 갈림길 분류기(CNN)가 정함. 2번 칸은 비움.
                      출력 12칸은 그대로라 TensorRT·C++ 노드 구조는 같음. 합치면 표지판 이미지도 좌우 반전 증강을 함
  --seg-extra DIR     분할+박스 데이터 추가(여러 번): <DIR>/<split>/{images,da_masks,ll_masks,labels,sl_masks,sl_valid}.
                      train 은 학습에, 그 밖의 split 은 시험(test_<split>)에. 예: bag_mt_v1 (train, test_bag1116)
  --det-extra DIR:N[:all][:tl]  박스만 데이터 추가(여러 번): <DIR>/images/<split>, labels/<split>. train 은 N배, test 는 test_<이름>.
                      all = 빈 라벨(물체 없음) 프레임도 음성 샘플로 씀. tl = 실물 신호등(방해 모니터 증강 대상)
H100 에서 느려지지 않게 (H100 은 작은 커널이 많은 모델에서 CPU 쪽 실행·동기화·데이터 준비가 병목이 되기 쉬움)
  - 매 step 동기화 없음: 손실은 GPU 에 모아 --log-every step 마다만 꺼냄. GPU_DUTY < 1 일 때만 점유율 제한(그때는 매 step 동기화)
  - --workers 0 = 자동(코어 수 - 2, 최대 64), --prefetch 4, persistent_workers, pin_memory + non_blocking 복사
  - TF32·cudnn.benchmark 켬, fused AdamW, --channels-last, --compile (모델 forward 를 torch.compile)
  - log.csv 에 epoch 마다 초당 이미지 수(ips)와 데이터를 기다린 비율(data_wait) → data_wait 가 크면 CPU(증강)가 병목
  - 대회장 조명 증강(venue_aug)의 큰 블러를 저해상도에서 계산 (VENUE_FAST=0 이면 예전 방식)

사용 예:
  python scripts/train_multitask.py --seg-data dataset/synth_signs_v2 --det-data dataset/cones_v1 \\
      --venue-ref-dir dataset/venue_ref_20261106 --init outputs/train/hybridnets_v4_seed0/best_da.pt --batch 16
"""
import argparse, contextlib, csv, json, os, random, sys, time
from argparse import Namespace

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_seg as ts  # 사진 증강 함수, seg_loss, IoU (CUDA 초기화 없음)
import mt_calib as mc   # 검출 점수 보정 지표·temperature
from mt_det_loss import DetLoss

PAD = ts.PAD
H_IN, W_IN = 360, 640
CLASSES = ["pedestrian", "sign_A", "sign_B", "bump", "slow", "parking",
           "traffic_light", "light_red", "light_yellow", "light_green", "limo", "cone"]
NC = len(CLASSES)
LAMP_CLS = [CLASSES.index(c) for c in ("light_red", "light_yellow", "light_green")]   # --ignore-small 제외
AB = {CLASSES.index("sign_A"), CLASSES.index("sign_B")}
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
MAX_BOX = 64
GLARE_P = 0.0     # --aug-glare-p: 바닥 반사 증강 확률 (floor_glare.py, 대회장 무대 조명 반사, 사용자 11-18)
MERGE_AB = False   # --merge-ab: main 에서 켬 (DataLoader 워커는 fork 라 그대로 물려받음)


# ---------------------------------------------------------------------------
def read_yolo(path, W, H):
    out = []
    if os.path.exists(path):
        for line in open(path):
            p = line.split()
            if len(p) == 5:
                c, cx, cy, w, h = int(p[0]), *map(float, p[1:])
                if MERGE_AB and c == 2:   # sign_B -> fork_sign(1)
                    c = 1
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

    def __init__(self, items, train, venue=None, ignore_small=0.0, aug_monitor_p=0.0, aug_cable_p=0.0, aug_dots_p=0.0,
                 aug_real=()):
        self.items, self.train, self.venue, self.ignore_small = items, train, venue, ignore_small
        self.aug_monitor_p, self.aug_cable_p, self.aug_dots_p = aug_monitor_p, aug_cable_p, aug_dots_p
        self.aug_real = tuple(aug_real)

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
        sl, slv = rd(it.get("sl")), rd(it.get("slv"))
        if it.get("neg"):   # 흰 케이블 등: 이 픽셀은 가로선이 아님(믿을 수 있음)
            ng = rd(it["neg"]) > 0
            sl[ng] = 0
            slv[ng] = 1
        return img, rd(it["da"]), rd(it["ll"]), sl, slv, read_yolo(it["lab"], w, h)

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
        if seg and GLARE_P > 0 and random.random() < GLARE_P:   # 바닥 반사: 선 바로 옆 차로에 곧은 흰 띠, 라벨은 그대로
            from floor_glare import floor_glare
            img = floor_glare(img, masks[1] > 0, masks[0] > 0)
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
        neg_aug = False
        # 실물 신호등 사진(--real-data, kind=det, real_tl=True)에도 뒤 모니터를 그린다. 케이블은 분할 이미지만
        if self.train and (seg or it.get("real_tl")) and (self.aug_monitor_p > 0 or self.aug_cable_p > 0):
            import mt_online_aug as oa
            r = np.random.default_rng()
            if len(b) and r.random() < self.aug_monitor_p:
                oa.add_monitor(img, da, ll, sl, slv, b, r, self.aug_dots_p, real=self.aug_real)
            if seg and r.random() < self.aug_cable_p:
                neg_aug = oa.add_cable(img, sl, slv, r)
        if self.train:
            img, (da, ll, sl, slv), b = self.augment(img, [da, ll, sl, slv], b, seg)
        img = cv2.copyMakeBorder(img, PAD, PAD, 0, 0, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        x = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 - MEAN) / STD
        ann = np.full((MAX_BOX, 5), -1, np.float32)
        if len(b):
            b = b[:MAX_BOX].copy(); b[:, [1, 3]] += PAD
            if self.ignore_small > 0:   # 먼(작은) 물체는 무시 박스(-2)로. 불빛(light_*)은 신호등 몸체의 일부라 원래 작아서(중앙값 13px) 제외
                small = np.sqrt(np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)) < self.ignore_small
                small &= ~np.isin(b[:, 4], LAMP_CLS)
                b[small, 4] = -2
            ann[:len(b)] = b
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float()
        return (torch.from_numpy(x.transpose(2, 0, 1).copy()), t(da), t(ll), t(sl), t(slv),
                torch.tensor(float(seg)), torch.tensor(float(it.get("sl") is not None or bool(it.get("neg")) or neg_aug)),
                torch.from_numpy(ann), it["name"])


def seg_items(root, split):
    d = os.path.join(root, split)
    if not os.path.isdir(os.path.join(d, "images")):
        return []
    out = []
    for n in sorted(os.listdir(os.path.join(d, "images"))):
        stem = n[:-4]
        sl = os.path.join(d, "sl_masks", n)
        has_sl = os.path.exists(sl)
        neg = os.path.join(d, "neg_sl", n)
        out.append(dict(kind="seg", name=f"{split}/{stem}", img=os.path.join(d, "images", n),
                        da=os.path.join(d, "da_masks", n), ll=os.path.join(d, "ll_masks", n),
                        lab=os.path.join(d, "labels", stem + ".txt"),
                        sl=sl if has_sl else None, slv=os.path.join(d, "sl_valid", n) if has_sl else None,
                        neg=neg if os.path.exists(neg) else None))
    return out


def det_items(root, split, n_max, seed, positives_only=True, prefix="cone", **extra):
    d = os.path.join(root, "images", split)
    if not os.path.isdir(d):
        return []
    out = []
    for n in sorted(os.listdir(d)):
        stem = os.path.splitext(n)[0]
        lab = os.path.join(root, "labels", split, stem + ".txt")
        if positives_only and (not os.path.exists(lab) or os.path.getsize(lab) == 0):
            continue
        out.append(dict(kind="det", name=f"{prefix}/{split}/{stem}", img=os.path.join(d, n), lab=lab, **extra))
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


def ignore_small_ann(ann, px):
    """ann (B,M,5) torch: √넓이 < px인 박스는 무시(-2). 불빛(light_*)과 빈 칸(-1)은 그대로. 학습 Items와 같은 규칙."""
    if px <= 0:
        return ann
    ann = ann.clone()
    w, h = (ann[..., 2] - ann[..., 0]).clamp(min=0), (ann[..., 3] - ann[..., 1]).clamp(min=0)
    c = ann[..., 4]
    small = ((w * h).sqrt() < px) & (c >= 0)
    for k in LAMP_CLS:
        small &= c != k
    ann[..., 4] = torch.where(small, torch.full_like(c, -2), c)
    return ann


def compute_loss(a, focal, reg, cls, anchors, pda, pll, psl, da, ll, sl, slv, segv, slf, ann):
    """학습과 val에서 같이 쓰는 손실. p* 는 crop 한 확률 (B,H,W). 반환: (loss, [seg, sl, cls, reg, cal])"""
    zero = pda.sum() * 0
    sv = segv > 0
    l_seg = (ts.seg_loss(pda[sv], da[sv]) + a.ll_weight * ts.seg_loss(pll[sv], ll[sv])) if sv.any() else zero
    lv = (slf > 0)
    l_sl = masked_seg_loss(psl[lv], sl[lv], slv[lv]) if lv.any() else zero
    c_l, r_l, k_l = focal(cls.float(), reg.float(), anchors.float(), ann)
    l_det = c_l.mean() + r_l.mean()
    loss = l_seg + a.sl_weight * l_sl + a.det_weight * l_det + a.calib_weight * k_l.mean()
    return loss, [l_seg, l_sl, c_l.mean(), r_l.mean(), k_l.mean()]


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
def evaluate(model, loader, device, util, amp, groups=None, T_cls=None, loss_fn=None):
    """T_cls가 있으면 그 temperature로 보정한 점수의 보정 지표도 같이 낸다(res['all']['calib_T']).
    loss_fn(reg, cls, anchors, pda, pll, psl, da, ll, sl, slv, segv, slf, ann) -> 학습과 같은 손실. 주면 res['all']['loss']."""
    model.eval()
    reg_t, clip_t = util.BBoxTransform(), util.ClipBoxes()
    iou = {"all": [ts.IoU(), ts.IoU()]}
    for g in groups or {}:
        iou[g] = [ts.IoU(), ts.IoU()]
    sl_iou = ts.IoU()
    dets, gts = [], {}
    loss_sum, loss_parts, loss_n = 0.0, np.zeros(5), 0
    for x, da, ll, sl, slv, segv, slf, ann, names in loader:
        x = x.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            reg, cls, anchors, pda, pll, psl = heads(model, x)
        if loss_fn is not None:
            td = lambda t: t.to(device)
            lo, parts = loss_fn(reg, cls, anchors, ts.crop(pda).float(), ts.crop(pll).float(), ts.crop(psl).float(),
                                td(da), td(ll), td(sl), td(slv), td(segv), td(slf), td(ann))
            loss_sum += float(lo) * len(names); loss_parts += np.array([float(p) for p in parts]) * len(names)
            loss_n += len(names)
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
    by_img = {}   # 검출 점수 보정 지표: 예측마다 맞음/틀림
    for k, c, s, b in dets:
        by_img.setdefault(k, []).append((s, c, b))
    cs, cc, cy = [], [], []
    for k, d in by_img.items():
        y = mc.match(d, gts[k])
        cs += [x[0] for x in d]; cc += [x[1] for x in d]; cy += list(y)
    res = {g: {"da_iou": round(a.value(), 4), "ll_iou": round(b.value(), 4)} for g, (a, b) in iou.items()}
    res["all"]["calib"] = {k: v for k, v in mc.metrics(cs, cy).items() if k != "rel"}
    if T_cls:
        res["all"]["calib_T"] = {k: v for k, v in mc.metrics(mc.apply_T_cls(cs, cc, T_cls, CLASSES), cy).items() if k != "rel"}
    res["_raw"] = (np.array(cs), np.array(cc, int), np.array(cy))
    res["all"]["sl_iou"] = round(sl_iou.value(), 4) if (sl_iou.tp + sl_iou.fn) else None
    res["all"]["ap50"] = {CLASSES[c]: (round(v, 4) if v is not None else None) for c, v in ap.items()}
    sign_ap = [v for c, v in ap.items() if v is not None and CLASSES[c] not in ("traffic_light",)]
    res["all"]["map50"] = round(float(np.mean(sign_ap)), 4) if sign_ap else None
    if loss_n:
        res["all"]["loss"] = round(loss_sum / loss_n, 5)
        res["all"]["loss_parts"] = dict(zip(("seg", "sl", "cls", "reg", "cal"), np.round(loss_parts / loss_n, 5).tolist()))
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
    ap.add_argument("--min-delta", type=float, default=1e-3, help="조기 종료: 이만큼 이상 올라야 '오름'")
    ap.add_argument("--es-smooth", type=int, default=1, help="조기 종료 판단에 쓰는 최근 epoch 평균 개수 (1 = 그 epoch만)")
    ap.add_argument("--ignore-small", type=float, default=0.0, help="√넓이가 이 px 미만인 박스는 무시 (0 = 끔)")
    ap.add_argument("--oversample-ab", type=float, default=1.0, help="A/B/주차가 든 이미지를 이 배수로 자주 뽑음")
    ap.add_argument("--calib-loss", choices=["none", "mdca", "sece", "both"], default="none",
                    help="학습 중 보정 손실 (검출 분류 점수에만). sece = 미분 가능한 ECE")
    ap.add_argument("--calib-weight", type=float, default=0.0, help="학습 중 보정 손실 비중")
    ap.add_argument("--aug-monitor-p", type=float, default=0.0, help="신호등이 있는 이미지에 뒤 단색 모니터를 그릴 확률")
    ap.add_argument("--aug-monitor-dots", type=float, default=0.0,
                    help="그린 모니터 중 화면을 땡땡이(켜진 불과 다른 색 원)로 할 비율")
    ap.add_argument("--aug-monitor-real", default="",
                    help="모니터를 실제 카메라처럼: match(선명도·노이즈),persp(원근),screen(비네팅·반사·날아감),jitter(땡땡이 흔들기) "
                         "중 쉼표로. 'all' = 전부")
    ap.add_argument("--overfit-drop", type=float, default=0.0,
                    help="과적합 정지: val 점수가 best보다 이만큼 넘게 낮고 train loss는 best 때보다 낮은 epoch이 "
                         "--overfit-epochs번 연속이면 멈춤 (0 = 끔)")
    ap.add_argument("--overfit-epochs", type=int, default=3)
    ap.add_argument("--es-val-loss", type=int, default=0,
                    help="val loss 조기 종료: val loss 최저 epoch 뒤로 이만큼 epoch 동안 더 낮아지지 않으면(= 올라가기 시작) 멈춤 (0 = 끔)")
    ap.add_argument("--es-min-delta", type=float, default=0.0, help="val loss가 최저보다 이만큼 넘게 낮아야 '새 최저'")
    ap.add_argument("--best-by", choices=["score", "val_loss"], default="score",
                    help="best.pt 기준: score = da+ll IoU+mAP50 최고, val_loss = val loss 최저. 점수 최고는 항상 best_score.pt에도 저장")
    ap.add_argument("--lane-guard", type=float, default=0.0,
                    help="차선 보호: val da+ll IoU가 시작 가중치보다 이만큼 넘게 낮은 epoch은 best.pt로 고르지 않음 (0 = 끔). "
                         "학습 전에 시작 가중치를 val로 한 번 잼")
    ap.add_argument("--aug-cable-p", type=float, default=0.0, help="흰 케이블을 그릴 확률 (그 자리는 흰 가로선 아님)")
    ap.add_argument("--real-data", default="",
                    help="실물 신호등 사진(로봇 카메라, 박스만): images/<split>, labels/<split> (scripts/make_real_tl_labels.py). "
                         "train은 --real-repeat배로 넣고, val은 val에 더하고, test는 test_real로 따로 채점. 빈 값 = 안 씀")
    ap.add_argument("--real-repeat", type=int, default=1, help="실물 train 사진을 한 epoch에 몇 번 넣을지")
    ap.add_argument("--aug-glare-p", type=float, default=0.0, help="바닥 반사 증강 확률 (분할 이미지, 대회장 조명 증강 다음). 0 = 끔")
    ap.add_argument("--merge-ab", action="store_true", help="sign_A·sign_B -> fork_sign (1번 칸), A/B 는 CNN 이 정함")
    ap.add_argument("--seg-extra", action="append", default=[], help="분할+박스 데이터 추가 (여러 번)")
    ap.add_argument("--det-extra", action="append", default=[], help="박스만 데이터 DIR:N[:all][:tl] (여러 번)")
    ap.add_argument("--prefetch", type=int, default=4, help="DataLoader prefetch_factor (워커당 미리 만들어 둘 batch 수)")
    ap.add_argument("--channels-last", action="store_true", help="모델·입력을 channels_last 로 (합성곱 비전 모델 권장)")
    ap.add_argument("--compile", action="store_true", help="모델 forward 를 torch.compile (첫 epoch 에 컴파일 시간)")
    ap.add_argument("--log-every", type=int, default=100, help="이 step 마다만 손실을 꺼내 출력 (꺼낼 때만 GPU 동기화)")
    ap.add_argument("--freeze-bn", action="store_true",
                    help="BatchNorm 통계(running mean/var)를 시작 가중치 값으로 고정 (학습 중에도 eval 모드). "
                         "batch 32로 학습한 가중치를 batch 4~8로 이어서 학습할 때 차선이 흔들리지 않게")
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
    global MERGE_AB, AB
    global GLARE_P
    GLARE_P = a.aug_glare_p
    if a.merge_ab:
        MERGE_AB = True
        CLASSES[1], CLASSES[2] = "fork_sign", "unused_2"
        AB = set()   # 합치면 화살표 방향이 라벨에 상관없어 표지판 이미지도 좌우 반전
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True   # 입력 크기가 640x384 로 고정
        pr = torch.cuda.get_device_properties(dev)
        print(f"GPU {pr.name} {pr.total_memory / 2**30:.0f} GiB, SM {pr.multi_processor_count}, CPU {os.cpu_count()}개"
              + ("  ⚠ MIG 조각 (SM 이 적음) — nvidia-smi -L 로 확인" if "MIG" in pr.name else ""), flush=True)
    if a.workers <= 0:
        a.workers = max(1, min(64, (os.cpu_count() or 4) - 2))
    print(f"DataLoader workers {a.workers}, prefetch {a.prefetch}, batch {a.batch}, GPU_DUTY {m.GPU_DUTY}", flush=True)

    venue = (a.venue_ref_dir, a.venue_p, a.venue_v4, a.venue_v5) if a.venue_ref_dir and a.venue_ref_dir != "none" else None
    import mt_online_aug as oa
    real = oa.REAL_ALL if a.aug_monitor_real.strip() == "all" else tuple(x for x in a.aug_monitor_real.split(",") if x.strip())
    bad = [x for x in real if x not in oa.REAL_ALL]
    if bad:
        raise SystemExit(f"--aug-monitor-real에 모르는 값 {bad} (가능: {','.join(oa.REAL_ALL)}, all)")
    tr = seg_items(a.seg_data, "train") + det_items(a.det_data, "train", a.cone_train_n, a.seed)
    va_ = seg_items(a.seg_data, "val") + det_items(a.det_data, "val", a.cone_val_n, a.seed)
    if a.real_data:
        rt = det_items(a.real_data, "train", 0, a.seed, prefix="real", real_tl=True)
        rv = det_items(a.real_data, "val", 0, a.seed, prefix="real", real_tl=True)
        if not rt:
            raise SystemExit(f"--real-data {a.real_data}: images/train 이 비어 있음")
        tr += rt * a.real_repeat
        va_ += rv
        print(f"real: train {len(rt)} x{a.real_repeat} = {len(rt) * a.real_repeat}, val {len(rv)}", flush=True)
    extra_tests = {}   # 이름 -> items
    for d in a.seg_extra:
        tr += seg_items(d, "train")
        va_ += seg_items(d, "val")
        for sp in sorted(os.listdir(d)):
            if sp not in ("train", "val", "excluded") and os.path.isdir(os.path.join(d, sp, "images")):
                extra_tests[sp] = seg_items(d, sp)
        print(f"seg-extra {d}: train {sum(1 for _ in seg_items(d, 'train'))}, test {list(extra_tests)}", flush=True)
    for spec in a.det_extra:
        parts = spec.split(":"); d, rep = parts[0], int(parts[1]) if len(parts) > 1 else 1
        allf, tl_ = "all" in parts[2:], "tl" in parts[2:]
        nm = os.path.basename(os.path.normpath(d))
        dt = det_items(d, "train", 0, a.seed, positives_only=not allf, prefix=nm, real_tl=tl_)
        tr += dt * rep
        va_ += det_items(d, "val", 0, a.seed, positives_only=not allf, prefix=nm, real_tl=tl_)
        te = det_items(d, "test", 0, a.seed, positives_only=not allf, prefix=nm)
        if te:
            extra_tests[f"test_{nm}"] = te
        print(f"det-extra {nm}: train {len(dt)} x{rep}{' (빈 프레임 포함)' if allf else ''}, test {len(te)}", flush=True)
    if a.val_max:
        va_ = va_[:a.val_max // 2] + va_[-(a.val_max // 2):]
    print(f"train {len(tr)} (seg {sum(i['kind'] == 'seg' for i in tr)}, sl {sum(i.get('sl') is not None for i in tr)}, "
          f"박스만 {sum(i['kind'] == 'det' and not i.get('real_tl') for i in tr)}, 실물 신호등 {sum(bool(i.get('real_tl')) for i in tr)}) "
          f"/ val {len(va_)}", flush=True)
    mk = lambda items, train: torch.utils.data.DataLoader(
        Items(items, train, venue if train else None, a.ignore_small if train else 0.0,
              a.aug_monitor_p if train else 0.0, a.aug_cable_p if train else 0.0, a.aug_monitor_dots, real), batch_size=a.batch,
        shuffle=train, drop_last=train, num_workers=a.workers, persistent_workers=a.workers > 0, pin_memory=True,
        prefetch_factor=a.prefetch if a.workers > 0 else None, collate_fn=collate)
    tl, vl = mk(tr, True), mk(va_, False)
    if a.oversample_ab != 1.0:   # A/B/주차가 든 이미지를 더 자주
        ABP = {1, 2, 5}   # sign_A(또는 fork_sign), sign_B, parking
        has = [bool(it.get("lab")) and os.path.exists(it["lab"])
               and any(int(l.split()[0]) in ABP for l in open(it["lab"]) if l.strip()) for it in tr]
        w = torch.tensor([a.oversample_ab if h else 1.0 for h in has], dtype=torch.double)
        sampler = torch.utils.data.WeightedRandomSampler(w, num_samples=len(tr), replacement=True,
                                                         generator=torch.Generator().manual_seed(a.seed))
        tl = torch.utils.data.DataLoader(Items(tr, True, venue, a.ignore_small, a.aug_monitor_p, a.aug_cable_p, a.aug_monitor_dots, real),
                                         batch_size=a.batch, sampler=sampler,
                                         drop_last=True, num_workers=a.workers, persistent_workers=a.workers > 0,
                                         pin_memory=True, prefetch_factor=a.prefetch if a.workers > 0 else None, collate_fn=collate)
        print(f"oversample A/B/P x{a.oversample_ab}: {sum(has)} images", flush=True)

    model = build_model(m, a.init, a.enc_skip).to(dev)
    if a.channels_last:
        model = model.to(memory_format=torch.channels_last)
    head_names = ("regressor.", "classifier.", "bifpn.", "det_skip.")
    g_head = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith(head_names)]
    g_rest = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith(head_names)]
    opt = torch.optim.AdamW([{"params": g_rest, "mult": 1.0}, {"params": g_head, "mult": a.head_lr_mult}],
                            lr=a.lr, weight_decay=5e-4, **({"fused": True} if dev.type == "cuda" else {}))
    print(f"param groups: rest {sum(p.numel() for p in g_rest) / 1e6:.1f}M x1, "
          f"head+bifpn {sum(p.numel() for p in g_head) / 1e6:.1f}M x{a.head_lr_mult}", flush=True)
    focal = DetLoss(calib=a.calib_loss if a.calib_weight > 0 else "none")
    # val loss: 학습과 같은 손실(같은 가중치, 같은 작은 박스 무시). val은 증강 없음
    val_loss_fn = lambda reg, cls, anchors, pda, pll, psl, da, ll, sl, slv, segv, slf, ann: compute_loss(
        a, focal, reg, cls, anchors, pda, pll, psl, da, ll, sl, slv, segv, slf, ignore_small_ann(ann, a.ignore_small))
    start, best = 0, {"score": -1.0, "epoch": 0, "stopped": False}
    if not a.fresh and os.path.exists(os.path.join(a.out, "last.pt")):
        ck = torch.load(os.path.join(a.out, "last.pt"), map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start, best = ck["epoch"], ck["best"]
        print(f"재개: epoch {start}, best {best}", flush=True)
    log = os.path.join(a.out, "log.csv")
    if start == 0:
        with open(log, "w", encoding="utf-8") as f:
            f.write("epoch,lr,loss,seg,sl,cls,reg,cal,val_loss,val_seg,val_sl,val_det,val_da,val_ll,val_map50,val_ece,val_mce,val_nll,sec,ips,data_wait\n")
    # GPU_DUTY < 1 (공용 RTX PC 보호) 일 때만 점유율 제한. 제한기는 매 step 동기화하므로 H100(GPU_DUTY=1)에서는 안 씀
    limiter_on = m.GPU_DUTY < 1.0 and dev.type == "cuda"
    fwd = (lambda x_: heads(model, x_))
    if a.compile:
        fwd = torch.compile(fwd)
        print("torch.compile: 첫 step 들에서 컴파일 (수 분)", flush=True)
    if a.lane_guard > 0 and "start_val" not in best:   # 시작 가중치의 val 차선 IoU (차선 보호 기준)
        s0 = evaluate(model, vl, dev, util, amp, loss_fn=val_loss_fn)["all"]
        best["start_val"] = {k: s0[k] for k in ("da_iou", "ll_iou", "map50", "loss")}
        print(f"시작 가중치 val: da {s0['da_iou']} ll {s0['ll_iou']} mAP50 {s0['map50']} val_loss {s0['loss']}", flush=True)
        if a.best_by == "val_loss":   # 학습해도 차선이 계속 나빠지면 시작 가중치가 best.pt로 남음
            best.update(best_pt_epoch=0)
            ts.save_state(model, os.path.join(a.out, "best.pt"))

    for ep in range(start, a.epochs):
        if best["stopped"]:
            break
        lr = a.lr * (1 - ep / a.epochs) ** 1.5
        for g in opt.param_groups:
            g["lr"] = lr * g.get("mult", 1.0)
        model.train()
        if a.freeze_bn:   # 작은 batch(RTX 4~8)로 이어서 학습할 때: BatchNorm은 시작 가중치의 통계를 그대로 쓰고 갱신하지 않음
            for mod in model.modules():
                if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
                    mod.eval()
        t0, n = time.time(), 0
        acc_t = torch.zeros(6, device=dev)   # 손실 합은 GPU 에 (매 step float() 하면 그때마다 동기화)
        wait, n_img, it = 0.0, 0, iter(tl)
        for step in range(len(tl)):
            tw = time.perf_counter()
            try:
                x, da, ll, sl, slv, segv, slf, ann, _ = next(it)
            except StopIteration:
                break
            wait += time.perf_counter() - tw; n_img += x.shape[0]
            if a.max_steps and step >= a.max_steps:
                break
            gstep = ep * len(tl) + step
            if a.warmup_steps and gstep <= a.warmup_steps:
                for g in opt.param_groups:
                    g["lr"] = lr * g.get("mult", 1.0) * min(1.0, (gstep + 1) / a.warmup_steps)
            with (m.GpuDutyLimiter() if limiter_on else contextlib.nullcontext()):
                nb = dict(non_blocking=True)
                x = x.to(dev, **nb, memory_format=torch.channels_last) if a.channels_last else x.to(dev, **nb)
                da, ll, sl, slv = da.to(dev, **nb), ll.to(dev, **nb), sl.to(dev, **nb), slv.to(dev, **nb)
                segv, slf, ann = segv.to(dev, **nb), slf.to(dev, **nb), ann.to(dev, **nb)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    reg, cls, anchors, pda, pll, psl = fwd(x)
                pda, pll, psl = ts.crop(pda).float(), ts.crop(pll).float(), ts.crop(psl).float()
                loss, (l_seg, l_sl, c_l, r_l, k_l) = compute_loss(a, focal, reg, cls, anchors, pda, pll, psl,
                                                                  da, ll, sl, slv, segv, slf, ann)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                opt.step()
            acc_t += torch.stack([t_.detach().float() for t_ in (loss, l_seg, l_sl, c_l, r_l, k_l)])
            n += 1
            if step % a.log_every == 0:
                lv = acc_t.tolist(); el = time.time() - t0
                print(f"  ep {ep + 1} step {step}/{len(tl)} loss(평균) {lv[0] / n:.3f} seg {lv[1] / n:.3f} sl {lv[2] / n:.3f} "
                      f"cls {lv[3] / n:.3f} reg {lv[4] / n:.3f} | {n_img / max(el, 1e-6):.0f} img/s, 데이터 대기 {wait / max(el, 1e-6):.0%}", flush=True)
        acc = acc_t.cpu().numpy() / max(n, 1)
        t_train = time.time() - t0
        ips, wait_frac = n_img / max(t_train, 1e-6), wait / max(t_train, 1e-6)
        v = evaluate(model, vl, dev, util, amp, loss_fn=val_loss_fn)["all"]
        score = v["da_iou"] + v["ll_iou"] + (v["map50"] or 0)
        vloss = v["loss"]
        if score > best.get("raw_best", -1.0):   # 점수가 가장 높은 가중치 (참고용 best_score.pt)
            best.update(raw_best=score, best_score_epoch=ep + 1, val_at_best_score=v)
            ts.save_state(model, os.path.join(a.out, "best_score.pt"))
        if vloss < best.get("best_val_loss", float("inf")) - a.es_min_delta:   # val loss 최저 = best.pt
            best.update(best_val_loss=vloss, best_loss_epoch=ep + 1, val=v, best_train_loss=float(acc[0]))
            ts.save_state(model, os.path.join(a.out, "best_val_loss.pt"))   # 차선 보호와 상관없이 val loss 최저는 항상 남김
            lane_ok = True
            if a.lane_guard > 0 and "start_val" in best:   # 차선(da+ll IoU)이 시작 가중치보다 lane_guard 넘게 나빠진 epoch은 best.pt로 안 씀
                s0 = best["start_val"]
                lane_ok = v["da_iou"] + v["ll_iou"] >= s0["da_iou"] + s0["ll_iou"] - a.lane_guard
                if not lane_ok:
                    print(f"  차선 보호: val loss는 최저지만 da+ll {v['da_iou'] + v['ll_iou']:.4f} < 시작 "
                          f"{s0['da_iou'] + s0['ll_iou']:.4f} - {a.lane_guard} 라 best.pt로 고르지 않음 "
                          f"(best_val_loss.pt에는 저장)", flush=True)
                    best.setdefault("lane_guard_skipped", []).append(ep + 1)
            if a.best_by == "val_loss" and lane_ok:
                best["best_pt_epoch"] = ep + 1
                ts.save_state(model, os.path.join(a.out, "best.pt"))
        if a.best_by == "score" and best.get("best_score_epoch") == ep + 1:
            best["best_pt_epoch"] = ep + 1
            ts.save_state(model, os.path.join(a.out, "best.pt"))
        best.setdefault("loss_log", []).append(dict(epoch=ep + 1, train_loss=round(float(acc[0]), 5), val_loss=vloss, val_parts=v["loss_parts"],
                                                    score=round(score, 4)))
        stop_valloss = a.es_val_loss > 0 and ep + 1 - best["best_loss_epoch"] >= a.es_val_loss
        hist = best.setdefault("hist", []); hist.append(score)
        sm = float(np.mean(hist[-a.es_smooth:]))   # 정체 정지(--patience)는 최근 es_smooth epoch 평균으로
        if sm > best["score"] + a.min_delta:
            best.update(score=sm, epoch=ep + 1)
        # 과적합 의심: train loss는 best 때보다 더 내려갔는데 val 점수는 best보다 overfit_drop 넘게 낮음
        sus = a.overfit_drop > 0 and score < best["raw_best"] - a.overfit_drop and acc[0] < best.get("best_train_loss", float("inf"))
        best["overfit_run"] = best.get("overfit_run", 0) + 1 if sus else 0
        best.setdefault("overfit_log", []).append(dict(epoch=ep + 1, score=round(score, 4), train_loss=round(float(acc[0]), 4),
                                                       gap=round(best["raw_best"] - score, 4), suspect=bool(sus)))
        stop_plateau = a.patience > 0 and ep + 1 - best["epoch"] >= a.patience
        stop_overfit = a.overfit_drop > 0 and best["overfit_run"] >= a.overfit_epochs
        best["stopped"] = stop_plateau or stop_overfit or stop_valloss
        if best["stopped"]:
            best["stop_reason"] = "val_loss" if stop_valloss else ("overfit" if stop_overfit else "plateau")
            print(f"조기 종료 ({best['stop_reason']}): best.pt epoch {best.get('best_pt_epoch')}, "
                  f"val loss 최저 {best.get('best_val_loss')} (epoch {best.get('best_loss_epoch')}) -> 지금 {vloss}, "
                  f"점수 최고 {best['raw_best']:.4f} (epoch {best.get('best_score_epoch')}) -> 지금 {score:.4f}", flush=True)
        ck = {"model": model.state_dict(), "opt": opt.state_dict(), "epoch": ep + 1, "best": best}
        ts.atomic(lambda p: torch.save(ck, p), os.path.join(a.out, "last.pt"))
        sec = time.time() - t0
        with open(log, "a", encoding="utf-8") as f:
            cb = v["calib"]; lp = v["loss_parts"]
            f.write(f"{ep + 1},{lr:.3g},{acc[0]:.4f},{acc[1]:.4f},{acc[2]:.4f},{acc[3]:.4f},{acc[4]:.4f},{acc[5]:.4f},"
                    f"{vloss},{lp['seg']},{lp['sl']},{round(lp['cls'] + lp['reg'], 5)},{v['da_iou']},{v['ll_iou']},{v['map50']},{cb.get('ece')},{cb.get('mce')},{cb.get('nll')},{sec:.0f},{ips:.1f},{wait_frac:.3f}\n")
        print(f"epoch {ep + 1}/{a.epochs} loss {acc[0]:.4f} val_loss {vloss} | val da {v['da_iou']:.4f} ll {v['ll_iou']:.4f} "
              f"mAP50 {v['map50']} | ECE {v['calib'].get('ece')} NLL {v['calib'].get('nll')} | "
              f"AP {json.dumps(v['ap50'])} | {sec:.0f}s, 학습 {ips:.0f} img/s, 데이터 대기 {wait_frac:.0%}", flush=True)

    # test: best 가중치로 test / test_venue / test_bag
    model.load_state_dict(torch.load(os.path.join(a.out, "best.pt"), map_location=dev))
    # 학습 후 보정: val 예측으로 temperature(전체, 클래스별)를 맞춰 calib.json에 저장. 차에서는 p' = sigmoid(logit(p) / T)
    vres = evaluate(model, vl, dev, util, amp)
    s_, c_, y_ = vres["_raw"]
    T_all, T_cls = mc.fit_T_per_class(s_, y_, c_, CLASSES)
    calib = {"T_all": T_all, "T_cls": T_cls, "how": "p_calibrated = sigmoid(logit(p) / T_cls[class])",
             "val_raw": {k: v for k, v in mc.metrics(s_, y_).items() if k != "rel"},
             "val_T": {k: v for k, v in mc.metrics(mc.apply_T_cls(s_, c_, T_cls, CLASSES), y_).items() if k != "rel"}}
    ts.atomic(lambda p: json.dump(calib, open(p, "w", encoding="utf-8"), indent=1), os.path.join(a.out, "calib.json"))
    print("calib:", json.dumps(calib, ensure_ascii=False), flush=True)
    res = {"best": best, "args": vars(a), "calib": calib}
    test_sets = {}
    for split in ("test", "test_venue", "test_bag", "test_real"):
        if split == "test_real":   # 실물 신호등 (로봇 카메라). --real-data 없으면 건너뜀
            items = det_items(a.real_data, "test", 0, a.seed, prefix="real") if a.real_data else []
        else:
            items = seg_items(a.seg_data, split)
        if split == "test":
            items += det_items(a.det_data, "test", 0, a.seed) + det_items(a.det_data, "val", 0, a.seed)[a.cone_val_n:]
        if items:
            test_sets[split] = items
    for split, items in extra_tests.items():   # bag_mt_v1 test_bag1116 (분할), real_fork_v1 test (박스만) 등
        if items:
            test_sets[split] = items
    for split, items in test_sets.items():
        if split not in res:
            res[split] = evaluate(model, mk(items, False), dev, util, amp, T_cls=T_cls)
            res[split].pop("_raw", None)
            print(split, json.dumps(res[split]["all"]), flush=True)
    # 다른 후보 가중치도 같은 test로 채점 (best.pt와 epoch이 다를 때만): 차선 보호로 빠진 val loss 최저, 점수 최고
    cands = {"best_val_loss.pt": best.get("best_loss_epoch"), "best_score.pt": best.get("best_score_epoch")}
    res["candidates"] = {"best.pt": {"epoch": best.get("best_pt_epoch")}}
    for fn, epn in cands.items():
        p = os.path.join(a.out, fn)
        if not os.path.exists(p) or epn == best.get("best_pt_epoch"):
            continue
        model.load_state_dict(torch.load(p, map_location=dev))
        r = {"epoch": epn}
        for split, items in test_sets.items():
            e = evaluate(model, mk(items, False), dev, util, amp)["all"]
            r[split] = {k: e.get(k) for k in ("da_iou", "ll_iou", "sl_iou", "map50", "ap50")}
        res["candidates"][fn] = r
        print("후보", fn, json.dumps({s: {k: r[s][k] for k in ("da_iou", "ll_iou", "map50")} for s in test_sets}), flush=True)
    for split in test_sets:
        res["candidates"]["best.pt"][split] = {k: res[split]["all"].get(k) for k in ("da_iou", "ll_iou", "sl_iou", "map50", "ap50")}
    ts.atomic(lambda p: json.dump(res, open(p, "w", encoding="utf-8"), indent=2, default=str), os.path.join(a.out, "test.json"))


if __name__ == "__main__":
    main()
