#!/usr/bin/env python3
"""대회장 bag 자동 라벨 (사람 검수 최소화, 사용자 11-18): DA·차선 = v4 예측 그대로, ego lane 은 확인용.

배경: v5 는 사람 검수의 'da·ll 비움'(= 빈 정답으로 학습) 때문에 대회장 신호등 장면을 '도로 없음'으로 배웠다.
      v4 는 11-18 대회장 bag 전부에서 DA 가 꺼진 적이 없고, 먼 곳까지 끝까지 칠한다 → v4 예측을 라벨로 쓰고,
      믿을 만한 프레임만 자동으로 고른다. 빈 DA 정답은 절대 만들지 않는다.

프레임 고르기: frames.csv 의 mission_id != -1 (출발 뒤), 0.25 s 간격. 화면이 거의 안 변하면(차가 섬) 2 s 에 1 장만.
판정 (프레임마다):
  - 꺼짐/약함: 화면 아래 절반 v4 DA < 10 %                       → 버림 (drop_weak)
  - 흔들림: 0.1 s 앞·뒤 프레임과 v4 DA IoU 중 작은 값 < 0.80       → 버림 (drop_unstable)
  - 직진(ego lane 직선 2 개가 맞음): ego 와 v4 DA IoU >= 0.85      → 자동 통과 (straight_ok)
                                     < 0.85                       → 검수 대기 (review_straight)
  - 직진 아님(커브·회전): 위 두 검사를 통과했으면                   → 자동 통과 (curve_ok)
박스: v4 검출. 점수 >= 0.5 정답, 0.2~0.5 무시 박스(-2), cone 은 0.3 이상 무시 박스. 정지선(sl)은 라벨 없음(학습 안 함).
출력: <out>/train/{images,da_masks,ll_masks,labels}/<bag>_<t>.png|txt  (seg-extra 형식, 640x360)
      <out>/excluded/review/...   검수 대기 (seg-extra 가 읽지 않음)
      <out>/auto_meta.jsonl       프레임마다 판정·수치
사용: python auto_label_v4.py --bundle <LimoPro_mt_v4> --ckpt <v4 best.pt> --out <dataset> <bag 프레임 폴더>...
"""
import argparse, csv, json, os, sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ego_lane_label as el  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--bundle", required=True); ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--dt", type=float, default=0.25); ap.add_argument("--static-dt", type=float, default=2.0)
ap.add_argument("runs", nargs="+")
a = ap.parse_args()
OUT = os.path.abspath(a.out); runs = [os.path.abspath(r) for r in a.runs]
E = el.EgoLane(a.bundle, a.ckpt); tm = E.tm
sys.path.insert(0, os.path.join(os.path.dirname(tm.__file__)))
import make_model_compare_gif as m  # noqa: E402
cwd = os.getcwd(); os.chdir(m.ROOT); sys.path.insert(0, os.path.join(m.MODELS_DIR, "HybridNets")); import utils.utils as util  # noqa: E402
os.chdir(cwd)
reg_t, clip_t = util.BBoxTransform(), util.ClipBoxes()
CONE = tm.CLASSES.index("cone") if "cone" in tm.CLASSES else 11


@torch.no_grad()
def dets(ims):
    x = torch.from_numpy(np.stack([((cv2.cvtColor(cv2.copyMakeBorder(cv2.resize(im, (640, 360), interpolation=cv2.INTER_AREA), tm.PAD, tm.PAD, 0, 0, cv2.BORDER_CONSTANT,
                                    value=(114,) * 3), cv2.COLOR_BGR2RGB).astype(np.float32) / 255 - E.MEAN) / E.STD).transpose(2, 0, 1) for im in ims])).to(E.dev)
    reg, cls, anchors, *_ = tm.heads(E.model, x)
    outs = util.postprocess(x, anchors.float(), reg.float(), cls.float(), reg_t, clip_t, 0.2, 0.5)
    res = []
    for o in outs:
        L = []
        for b, c, s in zip(o["rois"], o["class_ids"], o["scores"]):
            x1, y1, x2, y2 = float(b[0]), float(b[1]) - tm.PAD, float(b[2]), float(b[3]) - tm.PAD
            c = int(c); s = float(s)
            cid = -2 if (c == CONE and s >= 0.3) or s < 0.5 else c
            if c == CONE and s < 0.3: continue
            L.append(f"{cid} {(x1 + x2) / 1280:.6f} {(y1 + y2) / 720:.6f} {(x2 - x1) / 640:.6f} {(y2 - y1) / 360:.6f}")
        res.append(L)
    return res


def load(r, row):
    return cv2.imread(os.path.join(r, "frames", row["name"]))


iou = lambda p, q: float((p & q).sum() / max(1, (p | q).sum()))
for sub in ("train", "excluded/review"):
    for k in ("images", "da_masks", "ll_masks", "labels"): os.makedirs(os.path.join(OUT, sub, k), exist_ok=True)
meta = open(os.path.join(OUT, "auto_meta.jsonl"), "a"); stat = {}
for r in runs:
    bag = os.path.basename(r); rows = list(csv.DictReader(open(os.path.join(r, "frames.csv"))))
    T = np.array([float(x["t"]) for x in rows]); fps = (len(T) - 1) / max(1e-6, T[-1] - T[0]); nb = max(1, int(round(0.1 * fps)))
    pick, last_t, last_small, last_static = [], -1e9, None, -1e9
    for i, x in enumerate(rows):
        if x["mission_id"] == "-1" or T[i] - last_t < a.dt or i < nb or i >= len(rows) - nb: continue
        small = cv2.resize(cv2.cvtColor(load(r, x), cv2.COLOR_BGR2GRAY), (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)
        static = last_small is not None and np.abs(small - last_small).mean() < 2.0
        if static and T[i] - last_static < a.static_dt: continue
        if static: last_static = T[i]
        pick.append(i); last_t = T[i]; last_small = small
    for k0 in range(0, len(pick), 16):
        idx = pick[k0:k0 + 16]
        ims = [load(r, rows[i]) for i in idx]
        nbr = [load(r, rows[j]) for i in idx for j in (i - nb, i + nb)]
        lab = E.label(ims); nda, _, _ = E.predict(nbr); D = dets(ims)
        for q, i in enumerate(idx):
            L = lab[q]; da, ll = L["da"], L["ll"]
            frac = float(da[180:].mean()); st = min(iou(da, nda[2 * q]), iou(da, nda[2 * q + 1]))
            if frac < 0.10: dec = "drop_weak"
            elif st < 0.80: dec = "drop_unstable"
            elif L["ok"]: dec = "straight_ok" if L["iou_v4"] >= 0.85 else "review_straight"
            else: dec = "curve_ok"
            stat[dec] = stat.get(dec, 0) + 1
            name = f"{bag}_{T[i]:07.2f}"
            rec = dict(name=name, bag=bag, t=round(float(T[i]), 3), mission=rows[i]["mission_id"], decision=dec, da_frac=round(frac, 3), stable_iou=round(st, 3),
                       ego_ok=bool(L["ok"]), ego_iou=round(L.get("iou_v4", 0.0), 3), ego_why=L.get("why", ""), boxes=len(D[q]))
            meta.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if dec.startswith("drop"): continue
            sub = "train" if dec.endswith("_ok") else "excluded/review"
            p = lambda k, ext: os.path.join(OUT, sub, k, name + ext)
            cv2.imwrite(p("images", ".png"), cv2.resize(ims[q], (640, 360), interpolation=cv2.INTER_AREA))
            cv2.imwrite(p("da_masks", ".png"), da.astype(np.uint8) * 255); cv2.imwrite(p("ll_masks", ".png"), ll.astype(np.uint8) * 255)
            open(p("labels", ".txt"), "w").write("".join(l + "\n" for l in D[q]))
    print(f"{bag}: 고른 프레임 {len(pick)} | 누적 {stat}", flush=True)
meta.close(); print("끝:", stat)
