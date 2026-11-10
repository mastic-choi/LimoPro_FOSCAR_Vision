#!/usr/bin/env python3
"""검출이 왜 약한지: 클래스별 크기, 앵커 매칭 가능 여부, 클래스 혼동, 클래스 무시 AP.
사용: python scripts/eval_mt_det_diag.py . <best.pt>  (결과는 표준 출력)
"""
import os, sys, collections
root, ckpt = [os.path.abspath(p) for p in sys.argv[1:3]]
sys.path.insert(0, os.path.join(root, "scripts"))
import numpy as np, torch
import train_multitask as tm
import make_model_compare_gif as m
os.chdir(m.ROOT)
sys.path.insert(0, os.path.join(m.MODELS_DIR, "HybridNets"))
import utils.utils as util

model = tm.build_model(m, ckpt).to(m.device).eval()
reg_t, clip_t = util.BBoxTransform(), util.ClipBoxes()
C = tm.CLASSES


def iou(a, b):  # a (N,4) b (M,4)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ar = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (ar(a)[:, None] + ar(b)[None] - inter + 1e-9)


@torch.no_grad()
def run(split):
    items = tm.seg_items(os.path.join(root, "dataset/synth_signs_v2"), split)
    dl = torch.utils.data.DataLoader(tm.Items(items, False), batch_size=32, num_workers=8, collate_fn=tm.collate)
    dets, gts = [], {}
    size = collections.defaultdict(list); match = collections.defaultdict(lambda: [0, 0])
    conf = collections.defaultdict(collections.Counter)
    anc_np = None
    for x, *_, ann, names in dl:
        x = x.to(m.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            reg, cls, anchors, *_ = tm.heads(model, x)
        if anc_np is None:
            a = anchors[0].float().cpu().numpy(); anc_np = a[:, [1, 0, 3, 2]]  # y1x1y2x2 -> x1y1x2y2
        out = util.postprocess(x, anchors.float(), reg.float(), cls.float(), reg_t, clip_t, 0.05, 0.5)
        for k, nm in enumerate(names):
            g = ann[k].numpy(); g = g[g[:, 4] >= 0]   # 패딩 포함 좌표 그대로(앵커와 같은 좌표계)
            gts[nm] = g
            o = out[k]
            pb = np.array(o["rois"], np.float32).reshape(-1, 4); pc = np.array(o["class_ids"]).astype(int); ps = np.array(o["scores"])
            for b, c, s in zip(pb, pc, ps):
                dets.append((nm, c, float(s), b))
            if len(g):
                ia = iou(g[:, :4], anc_np).max(1)
                for b, c, mi in zip(g, g[:, 4].astype(int), ia):
                    w, h = b[2] - b[0], b[3] - b[1]
                    size[c].append(np.sqrt(w * h))
                    need = 0.5 if w * h > 100 else 0.15
                    match[c][0] += 1; match[c][1] += mi >= need
                # 혼동: 정답마다 IoU>=0.5 이고 점수>=0.3 인 예측 중 점수 최고의 클래스
                keep = ps >= 0.3
                if keep.any():
                    I = iou(g[:, :4], pb[keep])
                    for gi, c in enumerate(g[:, 4].astype(int)):
                        j = np.where(I[gi] >= 0.5)[0]
                        conf[c][C[pc[keep][j[np.argmax(ps[keep][j])]]] if len(j) else "(놓침)"] += 1
                else:
                    for c in g[:, 4].astype(int):
                        conf[c]["(놓침)"] += 1
    ap = tm.ap50(dets, gts, tm.NC)
    agn = tm.ap50([(n, 0, s, b) for n, c, s, b in dets], {n: np.c_[g[:, :4], np.zeros(len(g))] for n, g in gts.items()}, 1)[0]
    print(f"\n===== {split}  (클래스 무시 AP50 = {agn:.3f})")
    print(f"{'class':14s} {'AP50':>5s} {'GT':>4s} {'크기중앙(px)':>11s} {'p10':>4s} {'앵커매칭가능':>10s}   혼동(점수>=0.3, IoU>=0.5 예측)")
    for c in range(tm.NC):
        if not match[c][0]:
            continue
        s = np.array(size[c])
        top = ", ".join(f"{k} {v}" for k, v in conf[c].most_common(4))
        apv = ap.get(c); apv = f"{apv:.2f}" if apv is not None else "  -"
        print(f"{C[c]:14s} {apv:>5s} {match[c][0]:4d} {np.median(s):11.1f} {np.percentile(s, 10):4.0f} {match[c][1] / match[c][0]:10.0%}   {top}")
    a = anc_np; sz = np.sqrt((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))
    return sz


sz = None
for sp in ("test", "test_venue", "test_bag"):
    sz = run(sp)
print(f"\n앵커 크기(√면적) 최소 {sz.min():.1f}px, 종류 {sorted(set(np.round(sz, 1)))[:6]} ...")
