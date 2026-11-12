#!/usr/bin/env python3
"""신호등 뒤 방해 모니터(단색 / 땡땡이)가 있을 때 다중 작업 모델의 신호등·불빛 인식이 얼마나 흔들리는지 잰다.

synth_signs_v2 test, test_venue 중 신호등이 있는 사진마다 같은 seed로 세 장을 만든다(모니터 위치·크기 같음):
  clean(원본) / solid(단색 모니터) / dots(땡땡이 모니터: 켜진 불과 다른 색 원)
지표 (점수 기준 --thr, 기본 0.3과 0.5):
  AP50          traffic_light, light_red/yellow/green
  불빛 색       정답 불빛마다 IoU≥0.5인 light_* 검출 중 점수 1등의 색: 맞음 / 다른 색 / 놓침
  가짜 불빛     정답 불빛과 IoU<0.3인 light_* 검출(점수≥thr). 그중 모니터 위에 있는 것을 따로 셈
  가짜 신호등   정답 신호등과 IoU<0.3인 traffic_light 검출 중 모니터 위에 있는 것
출력: <out>/<이름>.json, <out>/<이름>_examples.jpg (가짜 불빛이 나온 예)

사용 예:
  .venv/Scripts/python scripts/eval_mt_monitor.py outputs/train/hybridnets_mt_v2/hybridnets_mt_v2_best.pt --name v2
"""
import argparse, json, os, sys, zlib

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_multitask as tm
import mt_online_aug as oa

LAMPS = [tm.CLASSES.index(c) for c in ("light_red", "light_yellow", "light_green")]
TL = tm.CLASSES.index("traffic_light")
VARIANTS = ("clean", "solid", "dots")


def iou(a, b):
    ix0, iy0 = np.maximum(a[0], b[:, 0]), np.maximum(a[1], b[:, 1])
    ix1, iy1 = np.minimum(a[2], b[:, 2]), np.minimum(a[3], b[:, 3])
    inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter + 1e-9)


def make_samples(data, splits, seed0=0, real=()):
    """신호등이 있는 사진마다 (이름, 박스, {clean/solid/dots: 사진}, {solid/dots: 모니터 마스크}). 세 가지 모두 그려진 사진만.
    seed가 사진 이름·순서로 정해져서 실행마다(모델이 달라도) 같은 모니터가 그려진다."""
    ds = tm.Items([], False)
    samples = []
    for split in splits.split(","):
        for it in tm.seg_items(data, split):
            img, da, ll, sl, slv, b = ds.load(it)
            if not len(b) or not (b[:, 4] == TL).any():
                continue
            seed = seed0 * 100003 + len(samples) * 7919 + zlib.crc32(it["name"].encode()) % 1000   # 실행마다 같은 모니터
            outs, masks = {"clean": img}, {}
            for v, dots in (("solid", False), ("dots", True)):
                im = img.copy(); z0 = np.zeros(img.shape[:2], np.uint8)
                if not oa.add_monitor(im, da.copy(), ll.copy(), z0.copy(), z0.copy(), b, np.random.default_rng(seed),
                                      real=real, dots=dots):
                    break
                outs[v] = im
                masks[v] = np.abs(im.astype(np.int16) - img).max(2) > 12
            if len(outs) == 3:
                samples.append((it["name"], b, outs, masks))
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("--name", default="model")
    ap.add_argument("--data", default="dataset/synth_signs_v2")
    ap.add_argument("--splits", default="test,test_venue")
    ap.add_argument("--real", default="", help="모니터 효과 (match,persp,screen,jitter). 비우면 기본 모니터")
    ap.add_argument("--out", default="outputs/eval_monitor")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="", help="비우면 cuda가 있으면 cuda, 없으면 cpu (예: mps)")
    a = ap.parse_args()
    real = tuple(x for x in a.real.split(",") if x)

    import make_model_compare_gif as m
    os.chdir(m.ROOT)
    sys.path.insert(0, os.path.join(m.MODELS_DIR, "HybridNets"))
    import utils.utils as util
    dev = torch.device(a.device) if a.device else m.device
    model = tm.build_model(m, a.weights).to(dev).eval()
    reg_t, clip_t = util.BBoxTransform(), util.ClipBoxes()

    samples = make_samples(a.data, a.splits, a.seed, real)
    print(f"신호등 있는 사진 중 모니터를 그린 것 {len(samples)}장", flush=True)

    dets = {v: [] for v in VARIANTS}
    gts = {s[0]: s[1] for s in samples}
    with torch.no_grad():
        for v in VARIANTS:
            for i in range(0, len(samples), a.batch):
                chunk = samples[i:i + a.batch]
                xs = []
                for _, _, outs, _ in chunk:
                    im = cv2.copyMakeBorder(outs[v], tm.PAD, tm.PAD, 0, 0, cv2.BORDER_CONSTANT, value=(114, 114, 114))
                    xs.append(((cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 - tm.MEAN) / tm.STD).transpose(2, 0, 1))
                x = torch.from_numpy(np.stack(xs)).to(dev)
                reg, cls, anchors, *_ = tm.heads(model, x)
                out = util.postprocess(x, anchors.float(), reg.float(), cls.float(), reg_t, clip_t, 0.05, 0.5)
                for (name, *_), o in zip(chunk, out):
                    for bb, c, s in zip(o["rois"], o["class_ids"], o["scores"]):
                        bb = np.array(bb, np.float32); bb[[1, 3]] -= tm.PAD
                        dets[v].append((name, int(c), float(s), bb))
            print(f"  {v} 추론 끝 ({len(dets[v])} 검출)", flush=True)

    res = {"weights": a.weights, "n_images": len(samples), "real": list(real),
           "n_lamps": int(sum(np.isin(b[:, 4], LAMPS).sum() for b in gts.values()))}
    examples = []
    for v in VARIANTS:
        apd = tm.ap50(dets[v], gts, tm.NC)
        r = {"ap50": {tm.CLASSES[c]: (round(apd[c], 4) if apd[c] is not None else None) for c in [TL] + LAMPS}}
        by = {}
        for d in dets[v]:
            by.setdefault(d[0], []).append(d)
        for thr in (0.3, 0.5):
            ok = wrong = miss = 0; fake = fake_mon = fake_tl_mon = 0; img_fake_mon = 0
            conf = np.zeros((3, 4), int)   # 정답 색 x (red, yellow, green, 놓침)
            for name, b, outs, masks in samples:
                d = [x for x in by.get(name, []) if x[2] >= thr]
                lam = [x for x in d if x[1] in LAMPS]
                g_l = b[np.isin(b[:, 4], LAMPS)]
                for g in g_l:
                    cand = [x for x in lam if iou(g[:4], x[3][None])[0] >= 0.5]
                    gi = LAMPS.index(int(g[4]))
                    if not cand:
                        miss += 1; conf[gi, 3] += 1; continue
                    top = max(cand, key=lambda x: x[2])
                    conf[gi, LAMPS.index(top[1])] += 1
                    ok += top[1] == int(g[4]); wrong += top[1] != int(g[4])
                mm = masks.get(v)
                hit_img = False
                for x in lam:
                    if len(g_l) and iou(x[3], g_l[:, :4]).max() >= 0.3:
                        continue
                    fake += 1
                    cx, cy = int(np.clip((x[3][0] + x[3][2]) / 2, 0, 639)), int(np.clip((x[3][1] + x[3][3]) / 2, 0, 359))
                    if mm is not None and mm[cy, cx]:
                        fake_mon += 1; hit_img = True
                        if thr == 0.3 and len(examples) < 12:
                            examples.append((v, name, outs[v], x, b))
                g_t = b[b[:, 4] == TL]
                for x in d:
                    if x[1] != TL or (len(g_t) and iou(x[3], g_t[:, :4]).max() >= 0.3):
                        continue
                    cx, cy = int(np.clip((x[3][0] + x[3][2]) / 2, 0, 639)), int(np.clip((x[3][1] + x[3][3]) / 2, 0, 359))
                    fake_tl_mon += bool(mm is not None and mm[cy, cx])
                img_fake_mon += hit_img
            n = max(ok + wrong + miss, 1)
            r[f"thr{thr}"] = dict(lamp_correct=round(ok / n, 4), lamp_wrong_color=round(wrong / n, 4), lamp_missed=round(miss / n, 4),
                                  fake_lamps=fake, fake_lamps_on_monitor=fake_mon, images_with_fake_lamp_on_monitor=img_fake_mon,
                                  fake_tl_on_monitor=fake_tl_mon,
                                  confusion_rows_gt_red_yellow_green__cols_pred_red_yellow_green_missed=conf.tolist())
        res[v] = r
        print(v, json.dumps(r, ensure_ascii=False), flush=True)
    os.makedirs(a.out, exist_ok=True)
    tag = a.name + ("_" + "-".join(real) if real else "")
    json.dump(res, open(os.path.join(a.out, tag + ".json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    tiles = []
    for v, name, im, x, b in examples:
        im = im.copy()
        for g in b:
            if int(g[4]) in LAMPS + [TL]:
                cv2.rectangle(im, (int(g[0]), int(g[1])), (int(g[2]), int(g[3])), (255, 255, 255), 1)
        x0, y0, x1, y1 = [int(t) for t in x[3]]
        cv2.rectangle(im, (x0, y0), (x1, y1), (255, 0, 255), 2)
        cv2.putText(im, f"{v} {tm.CLASSES[x[1]]} {x[2]:.2f}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        tiles.append(im)
    if tiles:
        while len(tiles) % 2:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(os.path.join(a.out, tag + "_examples.jpg"),
                    np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("->", os.path.join(a.out, tag + ".json"))


if __name__ == "__main__":
    main()
