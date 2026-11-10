#!/usr/bin/env python3
"""sign_A / sign_B 분류 확인용 몽타주와 통계.
사용: python scripts/eval_mt_ab.py . <best.pt> <출력 폴더> [split,split,...]  (기본 train,test,test_venue,test_bag)

정답 A/B 표지판마다
  - 최종 예측: 점수>=0.3, IoU>=0.5 인 예측 중 점수 최고 (없으면 놓침)
  - A/B 2택: 정답 박스와 IoU>=0.5 인 앵커 중 (A,B,parking) 점수가 가장 높은 앵커에서 A 점수와 B 점수만 비교
    -> "파란 사각형을 찾은 뒤 A/B 둘 중 하나만 고르라면 맞히는가"
"""
import os, sys, json, collections
root, ckpt, out = [os.path.abspath(p) for p in sys.argv[1:4]]
sys.path.insert(0, os.path.join(root, "scripts"))
import cv2, numpy as np, torch
import train_multitask as tm
import make_model_compare_gif as m
os.chdir(m.ROOT)
sys.path.insert(0, os.path.join(m.MODELS_DIR, "HybridNets"))
import utils.utils as util

C = tm.CLASSES
A, B, P = C.index("sign_A"), C.index("sign_B"), C.index("parking")
SHORT = {A: "A", B: "B", P: "P"}
model = tm.build_model(m, ckpt).to(m.device).eval()
reg_t, clip_t = util.BBoxTransform(), util.ClipBoxes()
os.makedirs(out, exist_ok=True)
GREEN, RED, ORANGE, WHITE = (60, 200, 60), (60, 60, 230), (0, 150, 255), (255, 255, 255)


def iou(a, b):
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ar = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (ar(a)[:, None] + ar(b)[None] - inter + 1e-9)


def name(c):
    return SHORT.get(c, C[c])


@torch.no_grad()
def collect(split):
    items = [it for it in tm.seg_items(os.path.join(root, "dataset/synth_signs_v2"), split)
             if os.path.exists(it["lab"]) and any(int(l.split()[0]) in (A, B) for l in open(it["lab"]) if l.strip())]
    if split == "train":   # train 은 고르게 600장만 (증강 없이 채점)
        items = [items[i] for i in np.linspace(0, len(items) - 1, 600).round().astype(int)]
    ds = tm.Items(items, False)
    dl = torch.utils.data.DataLoader(ds, batch_size=32, num_workers=8, collate_fn=tm.collate)
    recs, frames, i0 = [], [], 0
    for x, *_, ann, names in dl:
        x = x.to(m.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            reg, cls, anchors, *_ = tm.heads(model, x)
        reg, cls, anchors = reg.float(), cls.float(), anchors.float()
        boxes = clip_t(reg_t(anchors, reg), x).cpu().numpy()
        clsn = cls.cpu().numpy()
        outs = util.postprocess(x, anchors, reg, cls, reg_t, clip_t, 0.05, 0.5)
        for k, nm in enumerate(names):
            it = ds.items[i0 + k]
            g = ann[k].numpy(); g = g[g[:, 4] >= 0]
            o = outs[k]
            pb = np.array(o["rois"], np.float32).reshape(-1, 4); pc = np.array(o["class_ids"]).astype(int); ps = np.array(o["scores"])
            frames.append(dict(it=it, gt=g, pb=pb, pc=pc, ps=ps))
            for b in g[g[:, 4].astype(int) == A].tolist() + g[g[:, 4].astype(int) == B].tolist():
                b = np.array(b); gc = int(b[4])
                r = dict(split=split, img=os.path.relpath(it["img"], root), name=nm, gt=gc, box=(b[:4] - [0, tm.PAD, 0, tm.PAD]).tolist(),
                         size=float(np.sqrt((b[2] - b[0]) * (b[3] - b[1]))))
                keep = ps >= 0.3
                j = np.where(keep & (iou(b[None, :4], pb)[0] >= 0.5))[0] if len(pb) else []
                if len(j):
                    jj = j[np.argmax(ps[j])]; r["pred"], r["score"] = int(pc[jj]), float(ps[jj])
                else:
                    r["pred"], r["score"] = -1, 0.0
                ia = iou(b[None, :4], boxes[k])[0]
                cand = np.where(ia >= 0.5)[0]
                if len(cand):
                    best = cand[np.argmax(clsn[k, cand][:, [A, B, P]].max(1))]
                    sa, sb, sp = clsn[k, best, A], clsn[k, best, B], clsn[k, best, P]
                    r.update(sA=float(sa), sB=float(sb), sP=float(sp), two=A if sa >= sb else B)
                else:
                    r.update(sA=0.0, sB=0.0, sP=0.0, two=-1)
                recs.append(r)
        i0 += len(names)
    return recs, frames


def tile(r, S=200):
    img = cv2.imread(r["img"])
    x1, y1, x2, y2 = r["box"]
    cx, cy, s = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1) * 3 + 8
    X1, Y1 = int(max(0, cx - s / 2)), int(max(0, cy - s / 2))
    X2, Y2 = int(min(img.shape[1], cx + s / 2)), int(min(img.shape[0], cy + s / 2))
    crop = img[Y1:Y2, X1:X2].copy()
    sc = S / max(crop.shape[:2])
    crop = cv2.resize(crop, (max(1, round(crop.shape[1] * sc)), max(1, round(crop.shape[0] * sc))), interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(crop, (round((x1 - X1) * sc), round((y1 - Y1) * sc)), (round((x2 - X1) * sc), round((y2 - Y1) * sc)), WHITE, 1)
    canvas = np.full((S + 58, S, 3), 25, np.uint8)
    canvas[:crop.shape[0], :crop.shape[1]] = crop
    ok = r["pred"] == r["gt"]
    col = GREEN if ok else (ORANGE if r["pred"] < 0 else RED)
    put = lambda t, y, c: cv2.putText(canvas, t, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
    put(f"GT {name(r['gt'])}  {r['size']:.0f}px", S + 15, WHITE)
    put("pred: miss" if r["pred"] < 0 else f"pred: {name(r['pred'])} {r['score']:.2f}", S + 33, col)
    tcol = GREEN if r["two"] == r["gt"] else RED
    put(f"A/B: {name(r['two']) if r['two'] >= 0 else '-'} ({r['sA']:.2f}/{r['sB']:.2f})", S + 51, tcol)
    cv2.rectangle(canvas, (0, 0), (S - 1, S + 57), col, 3)
    return canvas


def grid(tiles, cols, title, path):
    h, w = tiles[0].shape[:2]
    while len(tiles) % cols:
        tiles.append(np.full((h, w, 3), 25, np.uint8))
    g = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    head = np.full((40, g.shape[1], 3), 30, np.uint8)
    cv2.putText(head, title, (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)
    cv2.imwrite(path, np.vstack([head, g]), [cv2.IMWRITE_JPEG_QUALITY, 90])


def frame_tile(f):
    img = tm.Items([f["it"]], False).load(f["it"])[0].copy()
    for b in f["gt"]:
        c = int(b[4]); y1, y2 = b[1] - tm.PAD, b[3] - tm.PAD
        cv2.rectangle(img, (int(b[0]), int(y1)), (int(b[2]), int(y2)), WHITE, 1)
        if c in (A, B, P):
            cv2.putText(img, f"GT {name(c)}", (int(b[0]), int(y2) + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)
    for b, c, s in zip(f["pb"], f["pc"], f["ps"]):
        if s < 0.3:
            continue
        x1, y1, x2, y2 = int(b[0]), int(b[1] - tm.PAD), int(b[2]), int(b[3] - tm.PAD)
        col = {A: (255, 160, 0), B: (255, 0, 200), P: (255, 255, 0)}.get(int(c), (180, 180, 180))
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        t = f"{name(int(c))} {s:.2f}"
        (tw, th), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw + 4, y1), col, -1)
        cv2.putText(img, t, (x1 + 2, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return img


stats = {}
allrecs = {}
SPLITS = sys.argv[4].split(",") if len(sys.argv) > 4 else ["train", "test", "test_venue", "test_bag"]
for sp in SPLITS:
    recs, frames = collect(sp)
    allrecs[sp] = recs
    st = {}
    for gc in (A, B):
        rr = [r for r in recs if r["gt"] == gc]
        cnt = collections.Counter("miss" if r["pred"] < 0 else name(r["pred"]) for r in rr)
        st[name(gc)] = dict(n=len(rr), pred=dict(cnt), correct=cnt.get(name(gc), 0),
                            two_way_correct=sum(r["two"] == gc for r in rr),
                            size_median=round(float(np.median([r["size"] for r in rr])), 1))
    n = sum(v["n"] for v in st.values())
    per = [st["A"], st["B"]]
    st["final_acc"] = round(sum(v["correct"] for v in per) / n, 3)
    st["two_way_acc"] = round(sum(v["two_way_correct"] for v in per) / n, 3)
    # 크기별 2택 정확도
    sz = collections.defaultdict(lambda: [0, 0])
    for r in recs:
        k = "<18px" if r["size"] < 18 else ("18-26px" if r["size"] < 26 else ">=26px")
        sz[k][0] += 1; sz[k][1] += r["two"] == r["gt"]
    st["two_way_by_size"] = {k: f"{v[1]}/{v[0]} ({v[1] / v[0]:.0%})" for k, v in sorted(sz.items())}
    stats[sp] = st
    print(sp, json.dumps(st, ensure_ascii=False), flush=True)

    # 잘라 낸 표지판 몽타주: test는 A/B 각 24개 고르게, 나머지는 전부
    if sp in ("test", "train"):
        sel = []
        for gc in (A, B):
            rr = [r for r in recs if r["gt"] == gc]
            sel += [rr[i] for i in np.linspace(0, len(rr) - 1, 24).round().astype(int)]
    else:
        sel = sorted(recs, key=lambda r: r["gt"])
    grid([tile(r) for r in sel], 12 if len(sel) > 24 else 8,
         f"{sp}: sign_A / sign_B crops ({len(sel)})  border green=correct red=wrong orange=miss | A/B = A vs B score only",
         os.path.join(out, f"ab_crops_{sp}.jpg"))
    # 전체 프레임 몽타주(대회장만)
    if sp == "test_venue":
        fs = [frames[i] for i in np.linspace(0, len(frames) - 1, min(16, len(frames))).round().astype(int)]
        tiles = [frame_tile(f) for f in fs]
        grid(tiles, 4, "test_venue frames with sign_A/B: pred A(blue) B(magenta) P(cyan) others(gray), score>=0.3 / GT white",
             os.path.join(out, "ab_frames_test_venue.jpg"))

json.dump(dict(stats=stats, records=allrecs), open(os.path.join(out, "ab_stats.json"), "w"), ensure_ascii=False, indent=1)
