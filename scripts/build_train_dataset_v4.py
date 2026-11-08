#!/usr/bin/env python3
"""4차 학습 데이터셋 ego_v4 = ego_v3 + 사람이 review_round4.py로 고른 후보.

라벨 관례(ego_v3와 같음): ll = 보이는 차선 전부, da = 지금 달리는 차선 | ll.
추가분 (2026-11-08 사람 검토, da만 보고 판정. ll은 기존 것을 씀):
  wide     '사용'  : 3차 때 뺀 넓은 da 라벨 중 정상 라벨 -> 원래 사람 라벨 그대로, 원래 split
  wide_ego '사용'  : 3차선 라벨을 내 차로만 남기게 바꾼 것(wide_to_ego.py) -> da = 바꾼 da | 원래 사람 ll, ll = 원래 사람 ll
  edge/off '사용'  : 11/07 bag 프레임, da = 3개 모델(TLN·YOLOP·YOLOPv2) 합의 ego da | ll
  edge/off 'da 비움': 트랙 밖이라 내 차로 없음 -> da = ll (차선만)
  bag 프레임의 ll = 3개 모델 중 2개 이상 (자르지 않은 전체)
bag 프레임 split: --test-bag(기본 blue6) 주행은 통째로 test_bag (연속 프레임이라 학습과 섞지 않음), 나머지는 train에 --repeat배로 넣음

사용 예:
  .venv/Scripts/python scripts/build_train_dataset_v4.py --out dataset/ego_v4
"""
import argparse, csv, json, os, shutil, sys, warnings
warnings.filterwarnings("ignore")
import cv2, numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
SUB = ("images", "da_masks", "ll_masks")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="dataset/ego_v3")
    ap.add_argument("--cand", default="dataset/round4_candidates")
    ap.add_argument("--out", default="dataset/ego_v4")
    ap.add_argument("--test-bag", default="blue6")
    ap.add_argument("--repeat", type=int, default=2, help="bag 프레임을 train에 몇 번 넣을지 (새 장면이 적어서 가중)")
    args = ap.parse_args()
    import round4_candidates as R

    if os.path.isdir(args.out):
        shutil.rmtree(args.out)
    shutil.copytree(args.base, args.out)
    man = list(csv.DictReader(open(os.path.join(args.base, "manifest.csv"), encoding="utf-8")))
    fields = list(man[0].keys())

    def put(split, name, img, da, ll):
        for sub, x in zip(SUB, (img, da.astype(np.uint8) * 255, ll.astype(np.uint8) * 255)):
            d = os.path.join(args.out, split, sub); os.makedirs(d, exist_ok=True)
            cv2.imwrite(os.path.join(d, name + ".png"), x)

    items = {it["id"]: it for it in json.load(open(os.path.join(args.cand, "items.json"), encoding="utf-8"))}
    dec = list(csv.DictReader(open(os.path.join(args.cand, "decisions.csv"), encoding="utf-8")))
    models = None
    added = {}
    for r in dec:
        it = items.get(r["id"])
        if it is None or r["choice"] == "drop":
            continue
        kind, choice = it["kind"], r["choice"]
        if kind in ("wide", "wide_ego"):
            base = f"dataset/ego_v2_clean/{it['split']}"
            img = cv2.imread(f"{base}/images/{it['frame']}.png")
            hll = cv2.imread(f"{base}/ll_masks/{it['frame']}.png", 0) > 0
            if kind == "wide":
                da = cv2.imread(f"{base}/da_masks/{it['frame']}.png", 0) > 0
            else:
                da = (cv2.imread(os.path.join(args.cand, "da_ego", it["id"] + ".png"), 0) > 0) | hll
            put(it["split"], it["frame"], img, da, hll)
            rows = [dict(frame=it["frame"], split=it["split"], source=f"ego_v2_clean+{kind}", label=kind)]
        else:
            if models is None:
                models = R.load_models()
            img = cv2.imread(os.path.join(args.cand, "img", it["id"] + ".png"))
            res = R.predict(models, img)
            ll = (np.stack([res[k][1] for k in R.KEYS[1:]]) >= .5).sum(0) >= 2
            ego = cv2.imread(os.path.join(args.cand, "da_ens", it["id"] + ".png"), 0) > 0
            da = (ego if choice == "use" else np.zeros_like(ll)) | ll
            name = f"b1107_{it['bag']}_{it['t']:05.1f}".replace(".", "_")
            test = it["bag"] == args.test_bag
            split = "test_bag" if test else "train"
            reps = 1 if test else args.repeat
            rows = []
            for j in range(reps):
                nm = name if j == 0 else f"{name}_r{j + 1}"
                put(split, nm, img, da, ll)
                rows.append(dict(frame=nm, split=split, source="bag_20261107", label=f"{kind}_{choice}", section=it["bag"]))
        for row in rows:
            man.append({f: row.get(f, "") for f in fields} | {"use": 1, "curve": row.get("curve", "1")})
        added[(kind, choice)] = added.get((kind, choice), 0) + len(rows)

    with open(os.path.join(args.out, "manifest.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(man)
    cnt = {s: len(os.listdir(os.path.join(args.out, s, "images"))) for s in sorted(os.listdir(args.out)) if os.path.isdir(os.path.join(args.out, s))}
    print("추가:", added)
    print("split별 장수:", cnt)


if __name__ == "__main__":
    main()
