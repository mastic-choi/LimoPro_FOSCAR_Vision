#!/usr/bin/env python3
"""실물 Ø80 mm A·B 표지판 M7 주행(원본 프레임) → H100 multitask 의 '박스만' 데이터 (real_tl_v1 과 같은 형식, 분할 라벨 없음).

  프레임: 30 fps 원본에서 3 장에 1 장(10 Hz). 1280x720 → 640x360 (INTER_AREA) png.
  갈림길 표지판: SAM 2.1 추적 박스(sign_track.json) + 주행 글자(사람 확인) → sign_A(1) / sign_B(2).
                A·B 를 fork_sign 하나로 합치는 것은 데이터셋을 합칠 때 한 번에 (기존 합성 데이터와 같이).
                가로/세로 비 0.6~1.67 밖(화면 끝에 잘리거나 가려진 조각)은 무시 박스(-2). 작은 박스도 정답 (--ignore-small 끔, 사용자 11-16).
  다른 물체: multitask-v3real 검출 — 신호등·불빛 점수 >= 0.5 정답, 0.2~0.5 무시, 콘 >= 0.3 무시.
            이 주행들에는 갈림길 A/B 말고 표지판이 하나도 없다(사용자 11-16). 그래서 표지판류 검출(sign_A/B, slow, parking,
            pedestrian, bump)은: SAM 박스와 IoU >= 0.3 이면 같은 갈림길 표지판이라 뺌. 안 겹치면 갈림길 분류기(--fork-onnx)로
            원본 크롭을 보고, A/B 라 하면(SAM 이 놓친 그 표지판일 수 있음) 무시 박스, 아니면 **배경**(라벨 없음 = 오탐을 배경으로 배움,
            예: 파란 원통을 parking 0.5 로 봄).
  split: --test 에 준 주행은 test, 나머지 train.
출력: <out>/images/<split>/<run>_<frame>.png, <out>/labels/<split>/...txt (YOLO 정규화, 640x360 기준), <out>/meta.csv
사용 (v3real 묶음 venv): python real_fork_det_labels.py <bundle> <ckpt> <out> <run>:<A|B> ... --test 220858,215318
"""
import argparse, csv, json, os, sys

import cv2
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("bundle"); ap.add_argument("ckpt"); ap.add_argument("out"); ap.add_argument("runs", nargs="+")
ap.add_argument("--test", default=""); ap.add_argument("--every", type=int, default=3)
ap.add_argument("--fork-onnx", required=True, help="갈림길 분류기 ONNX (fork-sign-v0.2)")
a = ap.parse_args()
out = os.path.abspath(a.out); ckpt = os.path.abspath(a.ckpt)
runs = [(os.path.abspath(r.rsplit(":", 1)[0]), r.rsplit(":", 1)[1]) for r in a.runs]
test = {x for x in a.test.split(",") if x}
sys.path.insert(0, os.path.join(a.bundle, "scripts"))
import make_model_compare_gif as m   # noqa: E402
import train_multitask as tm   # noqa: E402
os.chdir(m.ROOT)
sys.path.insert(0, os.path.join(m.MODELS_DIR, "HybridNets"))
import utils.utils as util   # noqa: E402
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "scripts"))   # fork_crop_rule (fork-sign-crop-cls 브랜치)
import fork_crop_rule as rule   # noqa: E402
fnet = cv2.dnn.readNetFromONNX(os.path.abspath(a.fork_onnx))


def fork_is_sign(orig, b640):
    x, y, s, _ = rule.crop_rect(tuple(b640), 640, 360, orig.shape[1], orig.shape[0])
    fnet.setInput((rule.preprocess(orig[y:y + s, x:x + s], rgb=False, size=96).astype(np.float32) / 255.0)[None, None])
    p = fnet.forward().reshape(3)
    return max(p[0], p[1]) >= 0.6 and abs(p[0] - p[1]) >= 0.2

C = tm.CLASSES
SIGNLIKE = {"slow", "parking", "pedestrian", "bump"}
model = tm.build_model(m, ckpt).to(m.device).eval()
reg_t, clip_t = util.BBoxTransform(), util.ClipBoxes()


def iou(p, q):
    ix = max(0.0, min(p[2], q[2]) - max(p[0], q[0])); iy = max(0.0, min(p[3], q[3]) - max(p[1], q[1]))
    u = (p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


meta = []
for sp in ("train", "test"):
    for d in ("images", "labels"): os.makedirs(os.path.join(out, d, sp), exist_ok=True)
for run, let in runs:
    name = os.path.basename(run); sp = "test" if any(t in name for t in test) else "train"
    F = json.load(open(os.path.join(run, "sign_track.json")))["frames"]
    pick = [r for r in F if r["i"] % a.every == 0]
    stat = {"frames": 0, "sign": 0, "sign_ign": 0, "other": 0, "ign": 0, "fp_bg": 0, "fp_cnn_ign": 0}
    for k in range(0, len(pick), 8):
        ch = pick[k:k + 8]
        origs = [cv2.imread(os.path.join(run, "frames", r["frame"])) for r in ch]
        smalls = [cv2.resize(o_, (640, 360), interpolation=cv2.INTER_AREA) for o_ in origs]
        xs = []
        for s in smalls:
            p = cv2.copyMakeBorder(s, tm.PAD, tm.PAD, 0, 0, cv2.BORDER_CONSTANT, value=(114, 114, 114))
            xs.append(((cv2.cvtColor(p, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 - tm.MEAN) / tm.STD).transpose(2, 0, 1))
        x = torch.from_numpy(np.stack(xs)).to(m.device)
        with torch.no_grad():
            reg, cls, anchors, *_ = tm.heads(model, x)
            res = util.postprocess(x, anchors.float(), reg.float(), cls.float(), reg_t, clip_t, 0.2, 0.5)
        for r, s, o, orig in zip(ch, smalls, res, origs):
            labs = []   # (cls_id, box640)
            sb = [v / 2 for v in r["sam_box"]] if r["sam_box"] else None
            if sb:
                w, h = sb[2] - sb[0], sb[3] - sb[1]
                ok = 0.6 <= w / max(h, 1e-6) <= 1.67
                labs.append((C.index(f"sign_{let}") if ok else -2, sb)); stat["sign" if ok else "sign_ign"] += 1
            for b, c, sc in zip(o["rois"], o["class_ids"], o["scores"]):
                bb = np.array(b, np.float32); bb[[1, 3]] -= tm.PAD
                bb[[0, 2]] = bb[[0, 2]].clip(0, 640); bb[[1, 3]] = bb[[1, 3]].clip(0, 360); bb = bb.tolist()
                cn = C[int(c)]
                if cn in SIGNLIKE or cn in ("sign_A", "sign_B"):
                    if sb and iou(bb, sb) >= 0.3:
                        continue   # 같은 갈림길 표지판
                    if fork_is_sign(orig, bb):
                        labs.append((-2, bb)); stat["fp_cnn_ign"] += 1   # SAM 이 놓친 갈림길 표지판일 수 있음
                    else:
                        stat["fp_bg"] += 1                               # 이 주행엔 다른 표지판이 없음 → 배경
                    continue
                if cn == "cone" or sc < 0.5:
                    if cn == "cone" and sc < 0.3: continue
                    labs.append((-2, bb)); stat["ign"] += 1
                else:
                    labs.append((int(c), bb)); stat["other"] += 1
            fn = f"{name}_{r['i']:05d}"
            cv2.imwrite(os.path.join(out, "images", sp, fn + ".png"), s)
            with open(os.path.join(out, "labels", sp, fn + ".txt"), "w") as fh:
                for cid, b in labs:
                    if b[2] - b[0] <= 0 or b[3] - b[1] <= 0: continue
                    fh.write(f"{cid} {(b[0] + b[2]) / 1280:.6f} {(b[1] + b[3]) / 720:.6f} {(b[2] - b[0]) / 640:.6f} {(b[3] - b[1]) / 360:.6f}\n")
            meta.append([fn, sp, name, let, r["i"], r.get("pass") or ("near" if sb else ""), len(labs)])
            stat["frames"] += 1
    print(f"{name} ({let}, {sp}): {stat}", flush=True)
with open(os.path.join(out, "meta.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["image", "split", "run", "sign", "frame", "sign_pass", "n_boxes"]); w.writerows(meta)
