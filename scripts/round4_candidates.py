#!/usr/bin/env python3
"""4차 데이터셋 후보 뽑기 (사람이 review_round4.py로 분류하기 전 단계).

후보 세 종류:
  wide   : 3차 데이터셋을 만들 때 'da가 화면의 40% 이상'이라 뺀 라벨 (ego_v3/excluded_wide_da.csv).
           대부분 3차선 전체를 칠한 옛 기준이지만, 차로가 화면 아래를 꽉 채운 정상 코너도 섞여 있다.
  edge   : 11/07 실차 bag에서 3차 HybridNets만 화면 좌/우 끝(코너 안쪽)을 깎고 나머지 3개 모델은 칠한 순간
           (edge_cut.json)의 앞뒤 0.5초. 제안 라벨 = TLN·YOLOP·YOLOPv2 3개 합의
  off    : 사람이 '트랙 이탈'로 분류한 순간(event_labels.csv)의 앞뒤 1초. 제안 라벨 = 3개 합의 또는 da 비움

출력: <out>/items.json, <out>/{img,view,da_ens,ll_ens}/<id>.png
  view = 사람이 볼 비교 그림 (원본 | HybridNets 3차 | 제안 라벨)

사용 예:
  .venv/Scripts/python scripts/round4_candidates.py --out dataset/round4_candidates
"""
import argparse, csv, glob, json, os, sys, warnings
warnings.filterwarnings("ignore")
import cv2, numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_seg as T, make_model_compare_gif as m

A = "dataset/bags_20261107_analysis"
KEYS = ["hybridnets", "tln_large", "yolop", "yolopv2"]


def load_models():
    out = {}
    for k in KEYS:
        model, rgb, norm, fwd, _ = T.ADAPTERS[k](m)
        w = f"outputs/train/{k}_v3_seed0/best_da.pt"
        if k == "yolopv2": model = T.load_yolopv2_trainable(w, m.device)
        else: model.load_state_dict(torch.load(w, map_location="cpu"))
        out[k] = (model.to(m.device).eval(), rgb, norm, fwd)
    return out


def predict(models, img):
    """모델별 (da 확률, ll 확률)."""
    res = {}
    for k, (model, rgb, norm, fwd) in models.items():
        x = cv2.copyMakeBorder(img, 12, 12, 0, 0, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        if rgb: x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = x.astype(np.float32) / 255
        if norm: x = (x - m.IMAGENET_MEAN) / m.IMAGENET_STD
        with torch.no_grad(), m.GpuDutyLimiter():
            a, b = fwd(model, torch.from_numpy(x.transpose(2, 0, 1).copy())[None].to(m.device))
        res[k] = (T.crop(a)[0].float().cpu().numpy(), T.crop(b)[0].float().cpu().numpy())
    return res


def consensus(res):
    """HybridNets를 뺀 3개: da 확률 평균 >= 0.5, ll 3개 중 2개 이상, da |= ll, 화면 아래 가운데와 이어진 덩어리만."""
    ks = KEYS[1:]
    ll = (np.stack([res[k][1] for k in ks]) >= .5).sum(0) >= 2
    da = (np.mean([res[k][0] for k in ks], 0) >= .5) | ll
    n, lab = cv2.connectedComponents(da.astype(np.uint8), connectivity=8)
    seed = [k for k in np.unique(lab[330:, 160:480]) if k > 0]
    if seed:
        da = np.isin(lab, seed); ll = ll & cv2.dilate(da.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    return da, ll


def overlay(img, da, ll, title):
    o = img.copy()
    o[da] = (o[da] * .45 + np.array([0, 255, 0]) * .55).astype(np.uint8)
    o[ll] = (255, 0, 255)
    edge = cv2.morphologyEx(da.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    o[edge] = (0, 255, 255)
    cv2.putText(o, title, (8, 28), 0, .8, (0, 0, 0), 4); cv2.putText(o, title, (8, 28), 0, .8, (255, 255, 255), 2)
    return o


def bag_frames(b):
    d = glob.glob(f"{A}/{b}_*")[0]
    ct = np.load(d + "/cam_t.npy"); fr = sorted(glob.glob(d + "/frames/*.jpg"))
    idx = np.array([int(os.path.basename(f)[:-4]) for f in fr])
    return ct[idx], fr


def pick_times(times, half, step):
    """각 순간의 앞뒤 half초를 step초 간격으로 (중복 제거)."""
    out = set()
    for t in times:
        for dt in np.arange(-half, half + 1e-6, step):
            out.add(round(float(t + dt), 1))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset/round4_candidates")
    args = ap.parse_args()
    for sub in ("img", "view", "da_ens", "ll_ens"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)
    items = []

    # wide: 뺀 넓은 da 라벨 (라벨은 원래 사람 라벨)
    for r in csv.DictReader(open("dataset/ego_v3/excluded_wide_da.csv", encoding="utf-8")):
        base = f"dataset/ego_v2_clean/{r['split']}"
        img = cv2.imread(f"{base}/images/{r['frame']}.png")
        da = cv2.imread(f"{base}/da_masks/{r['frame']}.png", 0) > 0; ll = cv2.imread(f"{base}/ll_masks/{r['frame']}.png", 0) > 0
        iid = f"wide_{r['frame']}"
        view = np.hstack([img, overlay(img, da, ll, f"label (da {float(r['da_area']) * 100:.0f}%)")])
        cv2.imwrite(f"{args.out}/view/{iid}.jpg", cv2.resize(view, (1280, 360)), [cv2.IMWRITE_JPEG_QUALITY, 88])
        items.append(dict(id=iid, kind="wide", src=f"{base}/images/{r['frame']}.png", split=r["split"], frame=r["frame"]))
    print("wide", sum(i["kind"] == "wide" for i in items), flush=True)

    models = load_models()
    ec = json.load(open(f"{A}/edge_cut.json", encoding="utf-8"))
    ev = [r for r in csv.DictReader(open(f"{A}/event_labels.csv", encoding="utf-8")) if r["choice"] == "off"]
    jobs = [("edge", b, t) for b, v in ec.items() for t in pick_times(v["times"], 0.5, 0.2)]
    off = {}
    for r in ev: off.setdefault(r["bag"], []).append(float(r["t"]))
    jobs += [("off", b, t) for b, ts in off.items() for t in pick_times(ts, 1.0, 0.3)]
    seen = set()
    cache = {}
    for kind, b, t in jobs:
        if b not in cache: cache[b] = bag_frames(b)
        ts, fr = cache[b]
        i = int(np.argmin(np.abs(ts - t)))
        if (b, i) in seen: continue
        seen.add((b, i))
        img = cv2.imread(fr[i]); res = predict(models, img)
        hda = (res["hybridnets"][0] >= .5) | (res["hybridnets"][1] >= .5); hll = res["hybridnets"][1] >= .5
        da, ll = consensus(res)
        iid = f"{kind}_{b}_{ts[i]:05.1f}"
        cv2.imwrite(f"{args.out}/img/{iid}.png", img)
        cv2.imwrite(f"{args.out}/da_ens/{iid}.png", da.astype(np.uint8) * 255)
        cv2.imwrite(f"{args.out}/ll_ens/{iid}.png", ll.astype(np.uint8) * 255)
        view = np.hstack([img, overlay(img, hda, hll, "HybridNets v3 (car)"), overlay(img, da, ll, "proposal: 3-model")])
        cv2.imwrite(f"{args.out}/view/{iid}.jpg", cv2.resize(view, (1440, 270)), [cv2.IMWRITE_JPEG_QUALITY, 88])
        items.append(dict(id=iid, kind=kind, bag=b, t=round(float(ts[i]), 2), frame_file=fr[i]))
    items.sort(key=lambda x: ({"edge": 0, "off": 1, "wide": 2}[x["kind"]], x.get("bag", ""), x.get("t", 0), x["id"]))
    json.dump(items, open(f"{args.out}/items.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print({k: sum(i["kind"] == k for i in items) for k in ("edge", "off", "wide")})


if __name__ == "__main__":
    main()
