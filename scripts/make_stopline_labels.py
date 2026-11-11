#!/usr/bin/env python3
"""synth_signs_v2(ego_v4 기반)의 11/06 대회장 프레임(v1106_*)에 흰 가로선(정지선·미션 시작선) 분할 라벨을 붙인다.

방법 (dataset/track_label_20261106/code/render_stopline.py)
  1) 도면 흰 가로선 지도(map/stopline.npy, build_stopline_map.py)를 그 프레임의 차 위치(results/poses_route.json)로 사진에 투영
  2) 선(성분)마다 위아래 ±8px, 좌우 ±4px 옮겨 실제 흰 픽셀과 가장 많이 겹치는 위치로 맞춤(모양은 도면 그대로)
  3) 겹친 비율이 --min-fit 미만인 선은 라벨에서 빼고 그 주변을 '모름'(valid=0)으로 둔다
  4) 합성 표지판·기둥이 덮은 픽셀(ego_v4 원본과 달라진 곳)은 선에서 지운다
위치 점수가 --max-score보다 크거나 위치가 없는 프레임은 라벨을 만들지 않는다(학습에서 정지선 손실 제외).

출력 <data>/<split>/sl_masks/<stem>.png  (0/255, 흰 가로선)
     <data>/<split>/sl_valid/<stem>.png  (0/255, 이 픽셀의 정지선 라벨을 믿을 수 있음)
     <data>/stopline_labels.csv
사용 예:
  .venv/Scripts/python scripts/make_stopline_labels.py --data dataset/synth_signs_v2 --ego dataset/ego_v4
"""
import argparse, csv, json, os, sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TL = os.path.join(ROOT, "dataset", "track_label_20261106")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="dataset/synth_signs_v2")
    ap.add_argument("--ego", default="dataset/ego_v4")
    ap.add_argument("--max-score", type=float, default=30.0)
    ap.add_argument("--min-fit", type=float, default=0.5)
    a = ap.parse_args()
    os.chdir(ROOT)
    os.environ["TRACK_LABEL_WORK"] = TL
    sys.path.insert(0, os.path.join(TL, "code"))
    import render_stopline as RS
    P = json.load(open(os.path.join(TL, "results", "poses_route.json")))
    rows, K = [], np.ones((3, 3), np.uint8)
    splits = [s for s in sorted(os.listdir(a.data)) if os.path.isdir(os.path.join(a.data, s, "images"))]
    for split in splits:
        names = [n for n in sorted(os.listdir(os.path.join(a.data, split, "images"))) if n.startswith("v1106_")]
        if not names:
            continue
        for sub in ("sl_masks", "sl_valid"):
            os.makedirs(os.path.join(a.data, split, sub), exist_ok=True)
        for n in names:
            stem = n[:-4]
            i = int(stem.split("_")[1])
            ego_name = "_".join(stem.split("_")[:2]) + ".png"          # v1106_000123_s0 -> v1106_000123.png
            p = P[i]
            if p.get("score") is None or p["score"] > a.max_score:
                rows.append(dict(split=split, image=stem, status="no_pose", lines=0, px=0))
                continue
            ego = cv2.imread(os.path.join(a.ego, split, "images", ego_name))
            img = cv2.imread(os.path.join(a.data, split, "images", n))
            proj = RS.image_stopline((p["x"], p["y"], p["yaw"]), p["pitch"])
            sl, info = RS.align_to_white(ego, proj)
            valid = np.ones(sl.shape, bool)
            n_ok = 0
            if info:   # 잘 안 맞은 선은 빼고 그 주변은 모름으로
                n_l, lab = cv2.connectedComponents(proj.astype(np.uint8), 8)
                k_info = 0
                for k in range(1, n_l):
                    comp = lab == k
                    if comp.sum() < 10:
                        continue
                    _, fit, dy, dx = info[k_info]
                    k_info += 1
                    if fit < a.min_fit:
                        shifted = np.roll(np.roll(comp, dy, 0), dx, 1)
                        sl &= ~shifted
                        valid &= ~(cv2.dilate((comp | shifted).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0)
                    else:
                        n_ok += 1
            changed = np.abs(img.astype(np.int16) - ego.astype(np.int16)).max(2) > 25   # 합성 물체가 덮은 곳
            changed = cv2.dilate(changed.astype(np.uint8), K) > 0
            sl &= ~changed
            cv2.imwrite(os.path.join(a.data, split, "sl_masks", n), sl.astype(np.uint8) * 255)
            cv2.imwrite(os.path.join(a.data, split, "sl_valid", n), valid.astype(np.uint8) * 255)
            rows.append(dict(split=split, image=stem, status="ok", lines=n_ok, px=int(sl.sum())))
    with open(os.path.join(a.data, "stopline_labels.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for split in splits:
        r = [x for x in rows if x["split"] == split]
        if r:
            ok = [x for x in r if x["status"] == "ok"]
            print(f"{split:11s} v1106 {len(r):5d}  labeled {len(ok):5d}  with lines {sum(x['lines'] > 0 for x in ok):5d}")


if __name__ == "__main__":
    main()
