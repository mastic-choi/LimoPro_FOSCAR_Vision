#!/usr/bin/env python3
"""합성 표지판 데이터(dataset/synth_signs_v2) → 갈림길 분류기 크롭 (A / B / none).

- A = sign_A, B = sign_B, none = 다른 표지판류(pedestrian·bump·slow·parking)·신호등 몸체 박스 + 무작위 배경.
  로봇에서 후보로 들어오는 표지판류가 바로 헷갈리는 대상이라 none 에 넣는다.
- 크롭 규칙은 fork_crop_rule.crop_rect (로봇과 같음). 사진이 640x360 이므로 배율 1, 최소 한 변은 64 x 640/1280 = 32.
- 저장: <out>/<split>/<A|B|none>/<사진>_<k>.png (원본 BGR 크롭, 전처리 전) + <out>/<split>/crops.csv

사용: python scripts/fork_crops_synth.py dataset/synth_signs_v2 dataset/fork_crops_v1/synth
"""
import csv, os, random, sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fork_crop_rule as rule

root, out = sys.argv[1], sys.argv[2]
SPLITS = ["train", "val", "test", "test_venue", "test_bag"]
LABEL = {1: "A", 2: "B", 0: "none", 3: "none", 4: "none", 5: "none", 6: "none"}   # classes.txt 번호
BG_PROB = 0.3   # 사진마다 이 확률로 배경 크롭 1장 (none)
random.seed(0)


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


for split in SPLITS:
    d = os.path.join(root, split)
    if not os.path.isdir(d):
        continue
    for lab in ("A", "B", "none"):
        os.makedirs(os.path.join(out, split, lab), exist_ok=True)
    rows, counts = [], {"A": 0, "B": 0, "none": 0}
    for name in sorted(os.listdir(os.path.join(d, "images"))):
        stem = os.path.splitext(name)[0]
        lp = os.path.join(d, "labels", stem + ".txt")
        img = cv2.imread(os.path.join(d, "images", name))
        if img is None:
            continue
        H, W = img.shape[:2]
        min_px = round(64 * W / 1280)
        boxes = []
        for ln in (open(lp) if os.path.exists(lp) else []):
            p = ln.split()
            if len(p) != 5:
                continue
            c, cx, cy, w, h = int(p[0]), *map(float, p[1:])
            boxes.append((c, ((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H)))
        k = 0
        for c, b in boxes:
            if c not in LABEL or b[2] - b[0] < 2 or b[3] - b[1] < 2:
                continue
            x, y, s, _ = rule.crop_rect(b, W, H, W, H, min_px=min_px)
            f = f"{stem}_{k}.png"; k += 1
            cv2.imwrite(os.path.join(out, split, LABEL[c], f), img[y:y + s, x:x + s])
            rows.append([f, LABEL[c], c, *[round(v, 1) for v in b]]); counts[LABEL[c]] += 1
        if random.random() < BG_PROB:   # 표지판과 겹치지 않는 배경
            for _ in range(10):
                s = random.randint(min_px, 120)
                x, y = random.randint(0, W - s), random.randint(0, H - s)
                bb = (x, y, x + s, y + s)
                if all(iou(bb, b) < 0.05 for _, b in boxes):
                    f = f"{stem}_bg.png"
                    cv2.imwrite(os.path.join(out, split, "none", f), img[y:y + s, x:x + s])
                    rows.append([f, "none", -1, *bb]); counts["none"] += 1
                    break
    with open(os.path.join(out, split, "crops.csv"), "w", newline="") as fh:
        wr = csv.writer(fh); wr.writerow(["file", "label", "src_class", "x1", "y1", "x2", "y2"]); wr.writerows(rows)
    print(split, counts)
