#!/usr/bin/env python3
"""bag 프레임(원본 1280x720 JPEG) → multitask 학습 형식 pseudo-label (da / ll / 박스).

  da, ll : multitask-v3real 분할 확률 > 0.5 (640x360, 0/255 png)
  박스   : multitask-v3real 검출 (점수 >= --det-thr, 클래스별 NMS 0.5) + 아래 규칙
  흰 가로선(sl)은 만들지 않는다 (sl_masks 없음 → 학습에서 그 이미지는 sl 손실 제외).

갈림길 A/B 는 HybridNets 가 흔들려서(같은 B 를 sign_B / slow / sign_A) 다음 순서로 정한다.
  1) 위치 = HybridNets 박스.  2) 표지판인가 = 갈림길 분류기(ONNX, 원본 1280x720 크롭, 로봇과 같은 크롭 규칙).
  3) 글자 = bag 단위 표지판(--bag-sign, 한 주행 안에서 안 바뀜). 분류기 글자가 bag 과 다르면 flag 만 남기고 bag 글자를 쓴다.
  - sign_A/B 박스(점수 >= 0.3): 분류기가 표지판(A/B) → bag 글자.  분류기가 없음 → 무시 박스(-2).
  - slow/parking/pedestrian/bump 박스(점수 >= 0.3): M7(갈림길) 구간에서 분류기가 표지판이면 bag 글자로 바꿈.
    M7 밖에서 분류기가 표지판이라 하면 무시 박스(-2) (11-16 bag: 케이블 릴을 slow 0.76 으로 봄). 아니면 그대로(점수 >= det-thr).
  - cone: v3real 에서 콘 출력은 학습 안 됨 → 점수 >= 0.3 이면 무시 박스(-2).
  - 그 밖: 점수 >= det-thr(0.5) 정답, ign-thr(0.2) ~ det-thr 무시 박스(-2), 그 아래 버림.
    (0.3~0.5 에는 먼 신호등·작은 표지판 같은 진짜와 콘을 신호등·slow 로 본 가짜가 섞여 있음, 11-16 bag 확인)
  - 바꾼 뒤 같은 표지판에 박스가 겹치면(IoU >= 0.5) 하나만 남긴다.
무시 박스(-2)는 학습 코드(train_multitask.py read_yolo / mt_det_loss)에서 그 자리 앵커를 손실에서 뺀다.

사용 (v3real 묶음 venv):
  python pseudo_label_mt.py <bundle> <frames 폴더> <출력 dataset 폴더> <fork onnx> --ckpt <pt> --bag-sign B
       [--holdout arall_1116_183018] [--det-thr 0.5] [--motion <frames>/frames_motion.csv]
출력: <출력>/<split>/{images,da_masks,ll_masks,labels}/<name>.png|txt, <출력>/pseudo_meta.jsonl
  split = train / test_bag1116(--holdout bag) / excluded(멈춤·기울어짐·da 불확실, 이유는 meta 의 excluded)
  멈춤(속도 < --min-speed)만 걸렸어도 신호등이 정답으로 잡히면 남긴다. 마스크는 작은 조각(da < 2000 px, ll < 60 px)을 지우고
  da 의 작은 구멍(가장자리에 안 닿는 2000 px 미만)을 메운다.
"""
import argparse, csv, json, os, sys

import cv2
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("bundle"); ap.add_argument("frames"); ap.add_argument("out"); ap.add_argument("onnx")
ap.add_argument("--ckpt", required=True); ap.add_argument("--bag-sign", default="B", choices=["A", "B"])
ap.add_argument("--holdout", default=""); ap.add_argument("--det-thr", type=float, default=0.5)
ap.add_argument("--cand-thr", type=float, default=0.3); ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--ign-thr", type=float, default=0.2, help="이 점수 ~ det-thr 사이 박스는 무시 박스(-2)")
ap.add_argument("--motion", default="", help="annotate_motion.py 의 frames_motion.csv. 주면 아래 기준으로 excluded/ 로 뺀다")
ap.add_argument("--min-speed", type=float, default=0.03); ap.add_argument("--max-tilt", type=float, default=10.0)
ap.add_argument("--max-da-unc", type=float, default=1.0)
ap.add_argument("--da-min-px", type=int, default=2000); ap.add_argument("--ll-min-px", type=int, default=60)
ap.add_argument("--vision-scripts", default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "scripts"),
                help="fork_crop_rule.py 가 있는 폴더 (fork-sign-crop-cls 브랜치의 scripts)")
a = ap.parse_args()
out, frames = os.path.abspath(a.out), os.path.abspath(a.frames)

sys.path.insert(0, os.path.join(a.bundle, "scripts")); sys.path.append(a.vision_scripts)   # Vision 에도 같은 이름 스크립트가 있어 뒤에 둠
import fork_crop_rule as rule   # noqa: E402
import make_model_compare_gif as m   # noqa: E402
import train_multitask as tm   # noqa: E402
os.chdir(m.ROOT)
sys.path.insert(0, os.path.join(m.MODELS_DIR, "HybridNets"))
import utils.utils as util   # noqa: E402
import train_seg as ts   # noqa: E402

C = tm.CLASSES
SIGN_LIKE = {"slow", "parking", "pedestrian", "bump"}
AB = {"sign_A", "sign_B"}
dev = m.device
model = tm.build_model(m, os.path.abspath(a.ckpt)).to(dev).eval()
net = cv2.dnn.readNetFromONNX(a.onnx)
reg_t, clip_t = util.BBoxTransform(), util.ClipBoxes()
holdout = {x for x in a.holdout.split(",") if x}


def fork_cls(img, boxes):
    if not boxes:
        return np.zeros((0, 3), np.float32)
    X = []
    for b in boxes:
        x, y, s, _ = rule.crop_rect(b, 640, 360, img.shape[1], img.shape[0])
        X.append(rule.preprocess(img[y:y + s, x:x + s], rgb=False, size=96).astype(np.float32) / 255.0)
    net.setInput(np.stack(X)[:, None])
    return net.forward().reshape(-1, 3)


def iou(p, q):
    ix = max(0.0, min(p[2], q[2]) - max(p[0], q[0])); iy = max(0.0, min(p[3], q[3]) - max(p[1], q[1]))
    u = (p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


def clean(mask, min_px, fill_px=0):
    """작은 조각 지우기, (fill_px > 0 이면) 작은 구멍 메우기. mask bool → bool"""
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    keep = np.zeros(n, bool); keep[1:] = st[1:, cv2.CC_STAT_AREA] >= min_px
    m = keep[lab]
    if fill_px > 0:
        n, lab, st, _ = cv2.connectedComponentsWithStats((~m).astype(np.uint8), 4)
        H, W = m.shape
        for k in range(1, n):
            x, y, w, h, area = st[k]
            if area < fill_px and x > 0 and y > 0 and x + w < W and y + h < H:   # 가장자리에 닿지 않는 작은 구멍만
                m[lab == k] = True
    return m


rows = list(csv.DictReader(open(os.path.join(frames, "frames.csv"))))
motion = {r["name"]: r for r in csv.DictReader(open(a.motion))} if a.motion else {}
for sp in ("train", "test_bag1116", "excluded"):
    for d in ("images", "da_masks", "ll_masks", "labels"):
        os.makedirs(os.path.join(out, sp, d), exist_ok=True)
meta = open(os.path.join(out, "pseudo_meta.jsonl"), "w")
stats = {"frames": 0, "boxes": {}, "ignore": 0, "flags": {}}

with torch.no_grad():
    for i in range(0, len(rows), a.batch):
        chunk = rows[i:i + a.batch]
        origs = [cv2.imread(os.path.join(frames, "orig", r["bag"], r["name"] + ".jpg")) for r in chunk]
        smalls = [cv2.resize(o, (640, 360), interpolation=cv2.INTER_AREA) for o in origs]
        xs = []
        for s in smalls:
            p = cv2.copyMakeBorder(s, tm.PAD, tm.PAD, 0, 0, cv2.BORDER_CONSTANT, value=(114, 114, 114))
            xs.append(((cv2.cvtColor(p, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 - tm.MEAN) / tm.STD).transpose(2, 0, 1))
        x = torch.from_numpy(np.stack(xs)).to(dev)
        reg, cls, anchors, pda, pll, _ = tm.heads(model, x)
        pda, pll = ts.crop(pda).float().cpu().numpy(), ts.crop(pll).float().cpu().numpy()
        res = util.postprocess(x, anchors.float(), reg.float(), cls.float(), reg_t, clip_t, 0.05, 0.5)
        for k, r in enumerate(chunk):
            sp = "test_bag1116" if r["bag"] in holdout else "train"
            name, mission = r["name"], int(r["mission_id"])
            da_raw, ll_raw = pda[k], pll[k]
            unc = lambda p: float(((p > 0.3) & (p < 0.7)).sum() / max(1, (p > 0.5).sum()))
            why = []
            if name in motion:
                mo = motion[name]
                if float(mo["tilt"]) > a.max_tilt: why.append(f"tilt{float(mo['tilt']):.0f}")
                if 0 <= float(mo["speed"]) < a.min_speed: why.append("still")
            if unc(da_raw) > a.max_da_unc: why.append("da_unc")
            o = res[k]
            dets = []
            for b, c, s in zip(o["rois"], o["class_ids"], o["scores"]):
                bb = np.array(b, np.float32); bb[[1, 3]] -= tm.PAD
                bb[[0, 2]] = bb[[0, 2]].clip(0, 640); bb[[1, 3]] = bb[[1, 3]].clip(0, 360)
                dets.append([C[int(c)], float(s), bb.tolist()])
            cand = [d for d in dets if (d[0] in AB or d[0] in SIGN_LIKE) and d[1] >= a.cand_thr]
            P = fork_cls(origs[k], [d[2] for d in cand])
            fork = {}
            for d, p in zip(cand, P):
                is_sign = max(p[0], p[1]) >= 0.6 and abs(p[0] - p[1]) >= 0.2
                fork[id(d)] = (is_sign, "AB"[int(p[1] > p[0])], [round(float(v), 3) for v in p])
            labels, flags = [], []   # (cls_name 또는 'ignore', box, 우선순위)
            for d in dets:
                cn, sc, bb = d
                f = fork.get(id(d))
                if cn == "cone":
                    if sc >= a.cand_thr: labels.append(("ignore", bb, 0))
                elif cn in AB and sc >= a.cand_thr:
                    if f[0]:
                        labels.append((f"sign_{a.bag_sign}", bb, 2))
                        if f[1] != a.bag_sign: flags.append(f"cnn_{f[1]}_vs_bag_{a.bag_sign}")
                        if cn != f"sign_{a.bag_sign}": flags.append(f"hn_{cn}_to_bag")
                    else:
                        labels.append(("ignore", bb, 0))
                        if sc >= a.det_thr: flags.append(f"hn_{cn}_cnn_none")
                elif cn in SIGN_LIKE and sc >= a.cand_thr and f[0] and mission == 7:
                    labels.append((f"sign_{a.bag_sign}", bb, 2)); flags.append(f"m7_{cn}_to_sign")
                    if f[1] != a.bag_sign: flags.append(f"cnn_{f[1]}_vs_bag_{a.bag_sign}")
                elif cn in SIGN_LIKE and sc >= a.cand_thr and f[0]:   # M7 밖인데 분류기가 갈림길 표지판이라 함 → 정체 불명
                    labels.append(("ignore", bb, 0)); flags.append(f"cnn_sign_on_{cn}_outside_m7")
                elif sc >= a.det_thr:
                    labels.append((cn, bb, 1))
                elif sc >= a.ign_thr:   # 진짜·가짜가 섞인 중간 점수: 정답도 배경도 아님
                    labels.append(("ignore", bb, 0))
            # 멈춤만 걸렸고 신호등이 정답으로 잡힌 프레임은 남긴다 (신호 바뀜 = 쓸모 있는 장면)
            if why == ["still"] and any(l[0] == "traffic_light" for l in labels):
                why = []
            if why:
                sp = "excluded"
            # 같은 자리 겹침 정리: 우선순위 높은(갈림길 확정 > 일반 > 무시) 것부터
            labels.sort(key=lambda l: -l[2]); keep = []
            for l in labels:
                if all(iou(l[1], q[1]) < 0.5 for q in keep if (l[0] in AB or l[0] in SIGN_LIKE or l[0] == "ignore")
                       and (q[0] in AB or q[0] in SIGN_LIKE or q[0] == "ignore")):
                    keep.append(l)
            with open(os.path.join(out, sp, "labels", name + ".txt"), "w") as fh:
                for cn, bb, _ in keep:
                    cid = -2 if cn == "ignore" else C.index(cn)
                    cx, cy = (bb[0] + bb[2]) / 1280, (bb[1] + bb[3]) / 720
                    w, h = (bb[2] - bb[0]) / 640, (bb[3] - bb[1]) / 360
                    if w <= 0 or h <= 0: continue
                    fh.write(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                    if sp != "excluded": stats["boxes"][cn] = stats["boxes"].get(cn, 0) + 1
            da = clean(da_raw > 0.5, a.da_min_px, fill_px=a.da_min_px)
            ll = clean(ll_raw > 0.5, a.ll_min_px)
            cv2.imwrite(os.path.join(out, sp, "images", name + ".png"), smalls[k])
            cv2.imwrite(os.path.join(out, sp, "da_masks", name + ".png"), (da * 255).astype(np.uint8))
            cv2.imwrite(os.path.join(out, sp, "ll_masks", name + ".png"), (ll * 255).astype(np.uint8))
            for fl in flags: stats["flags"][fl] = stats["flags"].get(fl, 0) + 1
            stats.setdefault("split", {}); stats["split"][sp] = stats["split"].get(sp, 0) + 1
            for wy in why: stats.setdefault("excluded_why", {}); stats["excluded_why"][wy[:4]] = stats["excluded_why"].get(wy[:4], 0) + 1
            meta.write(json.dumps({"name": name, "split": sp, "bag": r["bag"], "t": float(r["t"]), "mission": mission,
                                   "excluded": why,
                                   "speed": float(motion[name]["speed"]) if name in motion else None,
                                   "tilt": float(motion[name]["tilt"]) if name in motion else None,
                                   "da_frac": round(float(da.mean()), 4), "ll_frac": round(float(ll.mean()), 4),
                                   "da_unc": round(unc(da_raw), 4), "ll_unc": round(unc(ll_raw), 4),
                                   "labels": [[cn, [round(v, 1) for v in bb]] for cn, bb, _ in keep],
                                   "dets": [[cn, round(sc, 3), [round(v, 1) for v in bb]] for cn, sc, bb in dets if sc >= 0.1],
                                   "fork": [[d[0], round(d[1], 3), fork[id(d)][2]] for d in cand], "flags": flags},
                                  ensure_ascii=False) + "\n")
            stats["frames"] += 1
        print(f"  {min(i + a.batch, len(rows))}/{len(rows)}", flush=True) if (i // a.batch) % 25 == 0 else None
meta.close()
json.dump(stats, open(os.path.join(out, "pseudo_stats.json"), "w"), indent=1, ensure_ascii=False)
print(json.dumps(stats, indent=1, ensure_ascii=False))
