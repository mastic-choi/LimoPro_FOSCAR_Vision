#!/usr/bin/env python3
"""실물 신호등(로봇 카메라) 라벨로 가중치 여러 개를 나란히 채점한다.

세트: <real>/images/<split>/ (make_real_tl_labels.py 결과). 기본 test(수집 영상에서 뗀 구간)와 test_1m2(약 1.2m, 별도 촬영 12장).
지표 (IoU 0.5 이상 맞힘, 점수 기준 0.3 / 0.5):
  recall      클래스별 정답 중 같은 클래스 박스로 잡힌 비율 (몸체, 빨강, 노랑, 초록)
  wrong_color 켜진 불빛 정답 자리에 다른 색 불빛이 그 점수 이상으로 잡힌 수
  fp          정답과 안 맞는 신호등·불빛 박스 수 (같은 클래스 IoU 0.5 미만)
  ap50        학습 코드와 같은 AP50 (점수 순위 기준이라 점수가 낮아도 순위만 맞으면 높게 나옴 -> recall과 같이 봄)
전처리·후처리는 학습 평가와 같다 (train_multitask.Items, utils.postprocess 0.05 / NMS 0.5).

사용: python scripts/eval_real_tl.py --real dataset/real_tl_v1 --ckpt v3=weights/hybridnets_mt_v3_best.pt \
          --ckpt new=outputs/train/hybridnets_mt_v3real_seed0/best.pt --out outputs/.../eval_real
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import cv2
import numpy as np
import torch

LAMPS = ("light_red", "light_yellow", "light_green")
SHOW = ("traffic_light",) + LAMPS
THRS = (0.3, 0.5)


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


@torch.no_grad()
def predict(model, items, tm, util, dev):
    reg_t, clip_t = util.BBoxTransform(), util.ClipBoxes()
    ds = tm.Items(items, train=False)
    out = []
    for i in range(len(ds)):
        x, *_, ann, name = ds[i]
        reg, cls, anchors, *_ = tm.heads(model, x[None].to(dev))
        o = util.postprocess(x[None].cpu(), anchors.float().cpu(), reg.float().cpu(), cls.float().cpu(), reg_t, clip_t, 0.05, 0.5)[0]
        g = ann.numpy(); g = g[g[:, 4] >= 0].copy(); g[:, [1, 3]] -= tm.PAD
        d = []
        for b, c, s in zip(o["rois"], o["class_ids"], o["scores"]):
            bb = np.array(b, np.float32); bb[[1, 3]] -= tm.PAD
            d.append((tm.CLASSES[int(c)], float(s), bb))
        out.append((name, [(tm.CLASSES[int(r[4])], r[:4]) for r in g], d))
    return out


def score(preds, tm):
    res = {"n_images": len(preds)}
    for t in THRS:
        rec = {c: [0, 0] for c in SHOW}
        wrong, fp = 0, {c: 0 for c in SHOW}
        for _, gts, dets in preds:
            for gc, gb in gts:
                if gc not in rec:
                    continue
                rec[gc][1] += 1
                rec[gc][0] += any(dc == gc and s >= t and iou(gb, db) >= 0.5 for dc, s, db in dets)
                if gc in LAMPS:
                    wrong += any(dc in LAMPS and dc != gc and s >= t and iou(gb, db) >= 0.3 for dc, s, db in dets)
            for dc, s, db in dets:
                if dc in fp and s >= t and not any(gc == dc and iou(gb, db) >= 0.5 for gc, gb in gts):
                    fp[dc] += 1
        res[f"thr{t}"] = {"recall": {c: (round(h / n, 3) if n else None) for c, (h, n) in rec.items()},
                          "hits": {c: f"{h}/{n}" for c, (h, n) in rec.items()}, "wrong_color": wrong, "fp": fp}
    dets = [(nm, tm.CLASSES.index(dc), s, db) for nm, _, ds in preds for dc, s, db in ds]
    gts = {nm: np.array([[*gb, tm.CLASSES.index(gc)] for gc, gb in g], np.float32).reshape(-1, 5) for nm, g, _ in preds}
    ap = tm.ap50(dets, gts, tm.NC)
    res["ap50"] = {tm.CLASSES[c]: (round(v, 4) if v is not None else None) for c, v in ap.items() if tm.CLASSES[c] in SHOW}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default="dataset/real_tl_v1")
    ap.add_argument("--splits", default="test,test_1m2")
    ap.add_argument("--ckpt", action="append", required=True, help="이름=경로 (여러 번)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="", help="cuda | mps | cpu (빈 값이면 자동)")
    a = ap.parse_args()
    real, out = os.path.abspath(a.real), os.path.abspath(a.out)
    ckpts = [(c.split("=", 1)[0], os.path.abspath(c.split("=", 1)[1])) for c in a.ckpt]
    import make_model_compare_gif as m
    import train_multitask as tm
    os.chdir(m.ROOT)
    sys.path.insert(0, os.path.join(m.MODELS_DIR, "HybridNets"))
    import utils.utils as util
    dev = torch.device(a.device) if a.device else m.device
    os.makedirs(out, exist_ok=True)
    sets = {s: tm.det_items(real, s, 0, 0, prefix="real") for s in a.splits.split(",")}
    sets = {s: v for s, v in sets.items() if v}
    res = {}
    for name, path in ckpts:
        if not os.path.exists(path):
            print("없음, 건너뜀:", path, flush=True)
            continue
        model = tm.build_model(m, path).to(dev).eval()
        res[name] = {"ckpt": path}
        for s, items in sets.items():
            res[name][s] = score(predict(model, items, tm, util, dev), tm)
        del model
    json.dump(res, open(os.path.join(out, "eval_real.json"), "w"), indent=1, ensure_ascii=False)
    # 표
    lines = []
    for s in sets:
        for t in THRS:
            lines.append(f"\n[{s}, {len(sets[s])}장, 점수 >= {t}]  recall (몸체 / 빨강 / 노랑 / 초록) | 다른 색 | 오탐(몸체, 불빛)")
            for name in res:
                r = res[name][s][f"thr{t}"]
                rc = " / ".join("-" if r["recall"][c] is None else f"{r['recall'][c]:.0%}" for c in SHOW)
                fp_l = sum(r["fp"][c] for c in LAMPS)
                lines.append(f"  {name:<14} {rc:<26} | {r['wrong_color']:>3} | {r['fp']['traffic_light']:>3}, {fp_l:>3}")
    txt = "\n".join(lines)
    open(os.path.join(out, "eval_real.txt"), "w").write(txt + "\n")
    print(txt, flush=True)


if __name__ == "__main__":
    main()
