#!/usr/bin/env python3
"""다중 작업 모델(best.pt)로 test 사진에 검출 박스를 그려 몽타주를 만든다.
예측은 클래스별 색(점수 0.4 이상), 정답은 흰 얇은 선. test / test_venue / test_bag 24장씩, 콘 test 16장.
사용: python scripts/eval_mt_montage.py . <best.pt> <출력 폴더>
"""
import os, sys
root, ckpt, out = [os.path.abspath(p) for p in sys.argv[1:4]]
sys.path.insert(0, os.path.join(root, "scripts"))
import cv2, numpy as np, torch
import train_multitask as tm
import make_model_compare_gif as m

os.chdir(m.ROOT)
sys.path.insert(0, os.path.join(m.MODELS_DIR, "HybridNets"))
import utils.utils as util

THR, COLS, TW, TH = 0.4, 4, 640, 360
PAL = [(60, 60, 255), (255, 160, 0), (255, 0, 200), (0, 200, 255), (0, 255, 120), (255, 255, 0),
       (200, 200, 200), (0, 0, 255), (0, 220, 255), (0, 255, 0), (255, 120, 120), (0, 140, 255)]

model = tm.build_model(m, ckpt).to(m.device).eval()
reg_t, clip_t = util.BBoxTransform(), util.ClipBoxes()
os.makedirs(out, exist_ok=True)


def pick(items, n):
    pos = [it for it in items if os.path.exists(it["lab"]) and os.path.getsize(it["lab"]) > 0]
    idx = np.linspace(0, len(pos) - 1, min(n, len(pos))).round().astype(int)
    return [pos[i] for i in idx]


def label(img, x1, y1, text, col):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y = max(y1, th + 4)
    cv2.rectangle(img, (x1, y - th - 4), (x1 + tw + 4, y), col, -1)
    cv2.putText(img, text, (x1 + 2, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


@torch.no_grad()
def run(name, items, n):
    ds = tm.Items(pick(items, n), train=False)
    tiles, stats = [], [0, 0]
    for i in range(len(ds)):
        x, *_, ann, nm = ds[i]
        img = ds.load(ds.items[i])[0].copy()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            reg, cls, anchors, *_ = tm.heads(model, x[None].to(m.device))
        o = util.postprocess(x[None], anchors.float(), reg.float(), cls.float(), reg_t, clip_t, THR, 0.5)[0]
        gt = ann.numpy(); gt = gt[gt[:, 4] >= 0]; gt[:, [1, 3]] -= tm.PAD
        for b in gt:   # 정답: 흰색 얇은 선
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 255, 255), 1)
        stats[0] += len(gt)
        for b, c, s in zip(o["rois"], o["class_ids"], o["scores"]):
            b = np.array(b, np.float32); b[[1, 3]] -= tm.PAD
            x1, y1, x2, y2 = b.round().astype(int)
            col = PAL[int(c)]
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
            label(img, x1, y1, f"{tm.CLASSES[int(c)]} {s:.2f}", col)
            stats[1] += 1
        cv2.putText(img, nm.split("/")[-1][:40], (6, TH - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(cv2.resize(img, (TW, TH)))
    while len(tiles) % COLS:
        tiles.append(np.zeros((TH, TW, 3), np.uint8))
    grid = np.vstack([np.hstack(tiles[r:r + COLS]) for r in range(0, len(tiles), COLS)])
    head = np.full((44, grid.shape[1], 3), 30, np.uint8)
    cv2.putText(head, f"hybridnets_mt_v1 best.pt | {name} | {len(ds)} imgs | pred(color, score>={THR}) / GT(white thin)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    p = os.path.join(out, f"montage_{name}.jpg")
    cv2.imwrite(p, np.vstack([head, grid]), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"{name}: {len(ds)} imgs, GT {stats[0]}, pred {stats[1]} -> {p}", flush=True)


seg = os.path.join(root, "dataset/synth_signs_v2")
run("test", tm.seg_items(seg, "test"), 24)
run("test_venue", tm.seg_items(seg, "test_venue"), 24)
run("test_bag", tm.seg_items(seg, "test_bag"), 24)
run("cone_test", tm.det_items(os.path.join(root, "dataset/cones_v1"), "test", 0, 0), 16)
