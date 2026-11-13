"""실물 신호등 사진에서 v3와 v3real 예측을 나란히 그린다 (신호등 주변을 확대).

사용: python scripts/eval_real_tl_montage.py <번들 폴더> <out 폴더> v3=<pt> v3real=<pt>
출력: real_tl_1m2.jpg (1.2m 12장 전부), real_tl_test.jpg (test에서 v3가 놓친 장면 위주 12장)
"""
import os
import sys

BUNDLE, OUT = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
CKPTS = [(c.split("=", 1)[0], os.path.abspath(c.split("=", 1)[1])) for c in sys.argv[3:]]
sys.path.insert(0, os.path.join(BUNDLE, "scripts"))
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import make_model_compare_gif as m
import train_multitask as tm
import eval_real_tl as er

os.chdir(m.ROOT)
sys.path.insert(0, os.path.join(m.MODELS_DIR, "HybridNets"))
import utils.utils as util

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
COL = {"traffic_light": (255, 255, 0), "light_red": (0, 0, 255), "light_yellow": (0, 220, 255), "light_green": (0, 230, 0)}
SHORT = {"traffic_light": "몸체", "light_red": "빨강", "light_yellow": "노랑", "light_green": "초록"}
TILE_H, TILE_W, HEAD = 330, 260, 34
THR_LO, THR_HI = 0.3, 0.5


def font(s):
    return ImageFont.truetype(FONT, s)


def texts(img, items):
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    for x, y, t, s, c, bg in items:
        if bg is not None:
            l, tp, r, b = d.textbbox((x, y), t, font=font(s))
            d.rectangle((l - 2, tp - 1, r + 2, b + 1), fill=bg)
        d.text((x, y), t, font=font(s), fill=c)
    img[:] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def crop_box(gts):
    body = [b for c, b in gts if c == "traffic_light"] or [b for _, b in gts]
    b = np.array(body[0], np.float32)
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    h = max((b[3] - b[1]) * 1.8, 90.0)
    w = h * TILE_W / TILE_H
    x0, y0 = int(max(0, cx - w / 2)), int(max(0, cy - h / 2))
    x1, y1 = int(min(640, x0 + w)), int(min(360, y0 + h))
    return x0, y0, x1, y1


def tile(img640, gts, dets, name, crop):
    x0, y0, x1, y1 = crop
    sx, sy = TILE_W / (x1 - x0), TILE_H / (y1 - y0)
    t = cv2.resize(img640[y0:y1, x0:x1], (TILE_W, TILE_H), interpolation=cv2.INTER_CUBIC)
    T = lambda b: (int((b[0] - x0) * sx), int((b[1] - y0) * sy), int((b[2] - x0) * sx), int((b[3] - y0) * sy))
    for c, b in gts:   # 정답: 흰 얇은 선
        p = T(b); cv2.rectangle(t, p[:2], p[2:], (255, 255, 255), 1, cv2.LINE_AA)
    lab = []
    for c, s, b in sorted(dets, key=lambda d: d[1]):
        if c not in COL or s < THR_LO:
            continue
        p = T(b)
        cv2.rectangle(t, p[:2], p[2:], COL[c], 3 if s >= THR_HI else 1, cv2.LINE_AA)
        if c != "traffic_light":
            lab.append((max(2, p[0]), max(2, p[3] + 2), f"{SHORT[c]} {s:.2f}", 15, COL[c][::-1], (0, 0, 0)))
    body = [s for c, s, b in dets if c == "traffic_light"]
    lab.append((4, TILE_H - 24, f"몸체 {max(body):.2f}" if body else "몸체 없음", 14, (0, 255, 255), (0, 0, 0)))
    out = np.zeros((TILE_H + HEAD, TILE_W, 3), np.uint8); out[HEAD:] = t
    texts(out, [(6, 4, name, 18, (255, 255, 255), None)] + [(x, y + HEAD, s, z, c, bg) for x, y, s, z, c, bg in lab])
    return out


def lamp_hit(gts, dets, thr):
    return all(any(dc == gc and s >= thr and er.iou(gb, db) >= 0.5 for dc, s, db in dets) for gc, gb in gts if gc in er.LAMPS)


def cell(item, gts, per_model, caption):
    img = cv2.resize(cv2.imread(item["img"]), (640, 360), interpolation=cv2.INTER_AREA)
    crop = crop_box(gts)
    tiles = [tile(img, gts, d, n, crop) for n, d in per_model]
    c = np.hstack([tiles[0], np.full((TILE_H + HEAD, 4, 3), 60, np.uint8), tiles[1]])
    bar = np.zeros((30, c.shape[1], 3), np.uint8)
    texts(bar, [(6, 4, caption, 16, (230, 230, 230), None)])
    return np.vstack([bar, c])


def grid(cells, cols, title, legend):
    h, w = cells[0].shape[:2]
    rows = []
    for i in range(0, len(cells), cols):
        r = cells[i:i + cols] + [np.zeros_like(cells[0])] * (cols - len(cells[i:i + cols]))
        rows.append(np.hstack([np.pad(x, ((6, 6), (6, 6), (0, 0))) for x in r]))
    body = np.vstack(rows)
    top = np.zeros((84, body.shape[1], 3), np.uint8)
    texts(top, [(12, 6, title, 26, (255, 255, 255), None), (12, 46, legend, 16, (200, 200, 200), None)])
    return np.vstack([top, body])


def main():
    os.makedirs(OUT, exist_ok=True)
    dev = m.device
    real = os.path.join(BUNDLE, "dataset/real_tl_v1")
    sets = {s: tm.det_items(real, s, 0, 0, prefix="real") for s in ("test_1m2", "test")}
    for s in sets:
        sets[s].sort(key=lambda it: it["name"])
    preds = {}
    for n, p in CKPTS:
        model = tm.build_model(m, p).to(dev).eval()
        preds[n] = {s: er.predict(model, items, tm, util, dev) for s, items in sets.items()}
        del model
    names = [n for n, _ in CKPTS]
    legend = (f"왼쪽 {names[0]} · 오른쪽 {names[1]} · 굵은 선 = 점수 {THR_HI} 이상, 얇은 선 = {THR_LO}~{THR_HI} · "
              "흰 얇은 선 = 정답 · 하늘색 = 신호등 몸체 · 빨강/노랑/초록 = 켜진 불빛")
    gt_col = lambda gts: next((SHORT[c] for c, _ in gts if c in er.LAMPS), "불빛 없음")
    # 1.2m 12장 전부
    cells = []
    for i, it in enumerate(sets["test_1m2"]):
        gts = preds[names[0]]["test_1m2"][i][1]
        cells.append(cell(it, gts, [(n, preds[n]["test_1m2"][i][2]) for n in names],
                          f"정답 {gt_col(gts)} · {os.path.basename(it['img'])}"))
    cv2.imwrite(os.path.join(OUT, "real_tl_1m2.jpg"), grid(cells, 3, "실물 신호등 1.2m (따로 찍은 12장 전부)", legend),
                [cv2.IMWRITE_JPEG_QUALITY, 88])
    # test: v3가 점수 0.5에서 놓친 장면을 색마다 고르게, 그리고 v3real이 놓친 장면은 전부
    P0, P1 = preds[names[0]]["test"], preds[names[1]]["test"]
    new_miss = [i for i in range(len(P1)) if not lamp_hit(P1[i][1], P1[i][2], THR_HI)]
    by_col = {}
    for i in range(len(P0)):
        if i not in new_miss and not lamp_hit(P0[i][1], P0[i][2], THR_HI):
            by_col.setdefault(gt_col(P0[i][1]), []).append(i)
    pick = list(new_miss)
    k = 0
    while len(pick) < 12 and any(by_col.values()):
        for c in ("초록", "노랑", "빨강"):
            lst = by_col.get(c, [])
            if lst and len(pick) < 12:
                pick.append(lst.pop(len(lst) // 2 if k % 2 else 0))
        k += 1
    cells = []
    for i in pick:
        gts = P0[i][1]
        tag = f"{names[1]}도 놓침" if i in new_miss else f"{names[0]}가 놓친 장면"
        cells.append(cell(sets["test"][i], gts, [(n, preds[n]["test"][i][2]) for n in names],
                          f"정답 {gt_col(gts)} · {tag} · {os.path.basename(sets['test'][i]['img'])}"))
    cv2.imwrite(os.path.join(OUT, "real_tl_test.jpg"),
                grid(cells, 3, f"실물 신호등 test (171장 중 {len(pick)}장, 점수 {THR_HI} 기준으로 놓친 장면)", legend),
                [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("v3real이 점수 0.5에서 놓친 test 장면:", [os.path.basename(sets["test"][i]["img"]) for i in new_miss])
    print("저장:", OUT)


if __name__ == "__main__":
    main()
