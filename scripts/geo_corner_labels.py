#!/usr/bin/env python3
"""코너에서 차선이 하나만 보일 때의 기하 라벨: 앙상블 da를 '보이는 차선에서 바닥 거리 LANE_W 이내'로 자른다.

배경: 시계 방향 코너에서 카메라에 안쪽(또는 바깥) 차선 하나만 보이면, 1~3차 모델 모두 그 차선 너머 매트를 화면 끝까지 da로 칠한다.
     라벨 기준(da = 지금 달리는 차선만)으로는 차선 폭(550mm)만큼만 칠해야 한다.
방법:
  1. 3차 4개 앙상블로 da, ll (da 확률 평균 >= 0.5, ll 4개 중 3개 이상)
  2. 이미지 픽셀 -> 바닥 좌표 (카메라 보정, 높이 0.19m, 기울기 PITCH, cam.py)
  3. da와 맞닿은 ll(내 차로의 경계선)만 남기고, 바닥에서 그 선으로부터 LANE_W + 선 반폭 이내인 da만 남김.
     어느 쪽이 내 차로인지는 앙상블 da가 칠한 쪽을 따름. 화면 아래 가운데와 이어진 덩어리만 남김
  4. da = da | ll (라벨 관례)
입력: rosbag 분석 폴더의 event_labels.csv(사람이 '트랙 안 코너'로 고른 순간)
출력: <out>/{images,da_masks,ll_masks}/<주행>_<시각>.png, <out>/manifest.csv, <out>/geo_corner_montage.jpg

사용 예:
  .venv/Scripts/python scripts/geo_corner_labels.py dataset/bags_20261107_analysis dataset/geo_corner_20261107 \
      --cam dataset/track_label_20261106/code
"""
import argparse, csv, glob, json, os, sys, warnings
warnings.filterwarnings("ignore")
import cv2, numpy as np, torch
from scipy.spatial import cKDTree

LANE_W, LINE_HALF = 0.55, 0.03


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("analysis")
    ap.add_argument("out")
    ap.add_argument("--cam", required=True, help="cam.py가 있는 폴더 (track_label 작업 폴더의 code/)")
    ap.add_argument("--pitch", type=float, default=4.0)
    ap.add_argument("--choice", default="corner")
    args = ap.parse_args()
    sys.path.insert(0, args.cam); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import cam, train_seg as T, make_model_compare_gif as m

    keys = ["hybridnets", "tln_large", "yolop", "yolopv2"]
    models = {}
    for k in keys:
        model, rgb, norm, fwd, _ = T.ADAPTERS[k](m)
        w = os.path.join(m.ROOT, "outputs", "train", f"{k}_v3_seed0", "best_da.pt")
        if k == "yolopv2": model = T.load_yolopv2_trainable(w, m.device)
        else: model.load_state_dict(torch.load(w, map_location="cpu"))
        models[k] = (model.to(m.device).eval(), rgb, norm, fwd)

    def ensemble(img):
        das, lls = [], []
        for k in keys:
            model, rgb, norm, fwd = models[k]
            x = cv2.copyMakeBorder(img, 12, 12, 0, 0, cv2.BORDER_CONSTANT, value=(114, 114, 114))
            if rgb: x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
            x = x.astype(np.float32) / 255
            if norm: x = (x - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
            with torch.no_grad(), m.GpuDutyLimiter():
                a, b = fwd(model, torch.from_numpy(x.transpose(2, 0, 1).copy())[None].to(m.device))
            das.append(T.crop(a)[0].float().cpu().numpy()); lls.append(T.crop(b)[0].float().cpu().numpy())
        ll = (np.stack(lls) >= .5).sum(0) >= 3
        return (np.mean(das, 0) >= .5) | ll, ll

    H, W = 360, 640
    vv, uu = np.mgrid[0:H, 0:W]
    G = cam.ipm(np.c_[uu.ravel(), vv.ravel()].astype(np.float32), args.pitch).reshape(H, W, 2)
    valid = np.isfinite(G).all(2) & (G[..., 0] < 4.0)

    def geo(da, ll):
        near = cv2.dilate(da.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        line = ll & near & valid
        if line.sum() < 30:
            return None, None
        tree = cKDTree(G[line])
        keep = np.zeros_like(da)
        m_ = da & valid
        d, _ = tree.query(G[m_], distance_upper_bound=LANE_W + LINE_HALF + 0.01)
        keep[m_] = np.isfinite(d) & (d <= LANE_W + LINE_HALF)
        n, lab = cv2.connectedComponents(keep.astype(np.uint8), connectivity=8)
        seed = [k for k in np.unique(lab[330:, 160:480]) if k > 0]
        if not seed:
            return None, None
        keep = np.isin(lab, seed)
        llm = ll & cv2.dilate(keep.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        return keep | llm, llm

    A = args.analysis
    L = [r for r in csv.DictReader(open(os.path.join(A, "event_labels.csv"), encoding="utf-8")) if r["choice"] == args.choice]
    bags = {os.path.basename(d).split("_")[0]: d for d in glob.glob(os.path.join(A, "*_1107_*"))}
    for sub in ("images", "da_masks", "ll_masks"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)
    rows, demo = [], []
    for r in L:
        d = bags[r["bag"]]; t = float(r["t"])
        cam_t = np.load(os.path.join(d, "cam_t.npy")); fr = sorted(glob.glob(os.path.join(d, "frames", "*.jpg")))
        idx = np.array([int(os.path.basename(f)[:-4]) for f in fr])
        img = cv2.imread(fr[int(np.argmin(np.abs(cam_t[idx] - t)))])
        da, ll = ensemble(img)
        gda, gll = geo(da, ll)
        name = f"{r['bag']}_{t:05.1f}s"
        row = dict(frame=name, bag=r["bag"], t=t, ens_area=round(float(da.mean()), 4), status="ok" if gda is not None else "no_line")
        if gda is not None:
            row.update(geo_area=round(float(gda.mean()), 4), cut=round(float(1 - gda.sum() / max(da.sum(), 1)), 3))
            cv2.imwrite(os.path.join(args.out, "images", name + ".png"), img)
            cv2.imwrite(os.path.join(args.out, "da_masks", name + ".png"), gda.astype(np.uint8) * 255)
            cv2.imwrite(os.path.join(args.out, "ll_masks", name + ".png"), gll.astype(np.uint8) * 255)
        rows.append(row); demo.append((name, img, da, ll, gda, gll))
        print(name, row["status"], row.get("cut"), flush=True)
    with open(os.path.join(args.out, "manifest.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame", "bag", "t", "ens_area", "geo_area", "cut", "status"], restval="")
        w.writeheader(); w.writerows(rows)

    def ov(img, da_, ll_):
        g = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR).astype(np.float32)
        g[da_] = g[da_] * .5 + np.array([0, 200, 0]) * .5; g[ll_] = (255, 0, 255)
        return cv2.resize(g.astype(np.uint8), (320, 180), interpolation=cv2.INTER_AREA)
    pick = [x for x in demo if x[4] is not None]
    pick = [pick[i] for i in np.linspace(0, len(pick) - 1, min(24, len(pick))).astype(int)]
    tiles = []
    for name, img, da, ll, gda, gll in pick:
        row = np.hstack([cv2.resize(img, (320, 180)), ov(img, da, ll), ov(img, gda, gll)])
        cv2.putText(row, name, (5, 18), 0, .5, (0, 255, 255), 2)
        tiles.append(row)
    while len(tiles) % 2: tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(os.path.join(args.out, "geo_corner_montage.jpg"),
                np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    ok = [r for r in rows if r["status"] == "ok"]
    print(f"\n{len(rows)}장 중 기하 라벨 {len(ok)}장 (선 못 찾음 {len(rows) - len(ok)}), da가 줄어든 비율 중앙값 {np.median([r['cut'] for r in ok]) * 100:.0f}%")


if __name__ == "__main__":
    main()
