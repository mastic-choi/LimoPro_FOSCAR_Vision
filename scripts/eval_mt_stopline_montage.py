#!/usr/bin/env python3
"""정지선(흰 가로선) 분할 몽타주: 정답(초록 테두리)과 예측(자홍 칠)을 겹쳐 그리고 사진별 IoU를 적는다.

세트마다 정지선이 있는 사진 16장 + 정지선이 없는 사진 8장(오탐 확인)을 고르게 뽑는다.
IoU는 sl_valid(라벨을 믿을 픽셀) 안에서만 잰다. 라벨을 믿지 않는 픽셀은 어둡게 그린다.
세트 전체 IoU도 같이 출력한다(학습 코드의 sl_iou와 같은 정의).

사용: python scripts/eval_mt_stopline_montage.py <가중치.pt> <출력 폴더> [제목]
"""
import os, sys
ckpt, out = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
TITLE = sys.argv[3] if len(sys.argv) > 3 else os.path.basename(ckpt)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import cv2, numpy as np, torch
import train_multitask as tm
import make_model_compare_gif as m
os.chdir(m.ROOT)

D = "dataset/LimoPro_dataset_mt_v2"
SETS = [("test_venue", f"{D}/synth_signs_v2", "test_venue"), ("test_bag1116", f"{D}/bag_mt_v1", "test_bag1116")]
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
os.makedirs(out, exist_ok=True)
model = tm.build_model(m, ckpt).to(m.device).eval()


def rd(p):
    return (cv2.imread(p, 0) > 0) if p and os.path.exists(p) else None


@torch.no_grad()
def predict(paths):
    xs = []
    for p in paths:
        img = cv2.copyMakeBorder(cv2.imread(p), tm.PAD, tm.PAD, 0, 0, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        xs.append(((cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1))
    x = torch.from_numpy(np.stack(xs)).to(m.device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        *_, psl = tm.heads(model, x)
    return (psl.float()[:, tm.PAD:-tm.PAD] > 0.5).cpu().numpy()


for name, root, split in SETS:
    items = [it for it in tm.seg_items(root, split) if it["sl"]]
    preds, tp = {}, np.zeros(3)   # tp, fp, fn (valid 안)
    for i in range(0, len(items), 16):
        b = items[i:i + 16]
        for it, p in zip(b, predict([it["img"] for it in b])):
            preds[it["img"]] = p
    rows = []
    for it in items:
        g, v, p = rd(it["sl"]), rd(it["slv"]), preds[it["img"]]
        v = np.ones_like(g) if v is None else v
        t, f, n = (p & g & v).sum(), (p & ~g & v).sum(), (~p & g & v).sum()
        tp += (t, f, n)
        rows.append((it, g, v, p, t / max(t + f + n, 1), int((g & v).sum()), int((p & v).sum())))
    total = tp[0] / max(tp.sum(), 1)
    pos = [r for r in rows if r[5] > 0]; neg = [r for r in rows if r[5] == 0]
    neg_fp = sum(r[6] > 300 for r in neg)
    pick = lambda rs, k: [rs[j] for j in np.linspace(0, len(rs) - 1, min(k, len(rs))).round().astype(int)] if rs else []
    sel = pick(pos, 16) + pick(neg, 8)
    tiles = []
    for it, g, v, p, iou, gpx, ppx in sel:
        img = cv2.imread(it["img"])
        img = (img * np.where(v, 1.0, 0.45)[..., None]).astype(np.uint8)
        over = img.copy(); over[p] = (255, 0, 255)
        img = cv2.addWeighted(over, 0.55, img, 0.45, 0)
        cs, _ = cv2.findContours(g.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, cs, -1, (0, 255, 0), 2)
        cv2.rectangle(img, (0, 0), (639, 26), (0, 0, 0), -1)
        txt = f"IoU {iou:.2f}  GT {gpx}px  pred {ppx}px" if gpx else f"no stop line  pred {ppx}px"
        col = (255, 255, 255) if gpx else ((80, 80, 255) if ppx > 300 else (255, 255, 255))
        cv2.putText(img, txt, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)
        cv2.putText(img, it["name"].split("/")[-1][:40], (6, 352), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(img)
    while len(tiles) % 4:
        tiles.append(np.zeros((360, 640, 3), np.uint8))
    grid = np.vstack([np.hstack(tiles[r:r + 4]) for r in range(0, len(tiles), 4)])
    head = np.full((44, grid.shape[1], 3), 30, np.uint8)
    cv2.putText(head, f"{TITLE} | {name} stop line: GT green outline / pred magenta | set IoU {total:.3f} | "
                      f"top 4 rows: with stop line, bottom 2: without", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    path = os.path.join(out, f"stopline_{name}.jpg")
    cv2.imwrite(path, np.vstack([head, grid]), [cv2.IMWRITE_JPEG_QUALITY, 88])
    ious = np.array([r[4] for r in pos])
    print(f"{name}: 라벨 사진 {len(rows)} (정지선 있음 {len(pos)}, 없음 {len(neg)}) | 세트 IoU {total:.3f} | "
          f"정지선 있는 사진 IoU 중앙 {np.median(ious):.2f}, 0.5 미만 {int((ious < 0.5).sum())}장, 놓침(pred 0) {sum(r[6] == 0 for r in pos)}장 | "
          f"없는 사진에서 300px 넘게 칠함 {neg_fp}/{len(neg)}장 -> {path}", flush=True)
