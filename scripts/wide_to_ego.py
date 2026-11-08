#!/usr/bin/env python3
"""3차선 전체를 칠한 옛 라벨 -> 내 차로(ego lane)만 남긴 라벨.

사람 라벨은 경계가 정확하지만 3차선을 다 칠했고, 모델(3차, ego 기준으로 학습)은 어느 차로가 내 차로인지 안다.
  1. 3차 4개 모델 앙상블의 ego da (da 확률 평균 >= 0.5, ll 4개 중 3개 이상, da |= ll)
  2. 사람 da를 사람 ll로 잘라 조각을 내고, 앙상블 ego da와 겹치는 정도로 조각을 고름 — 경계는 차선, 어느 차로인지는 모델
  3. 화면 아래 가운데와 이어진 덩어리만 남김
  4. 새 ll = 사람 ll 중 새 da에 닿은 것(내 차로 경계선), da |= ll (라벨 관례)
입력: review_round4.py에서 wide 후보를 '버림'으로 고른 것 (decisions.csv)
출력: <cand>/{da_ego,ll_ego,view}/wide_ego_<frame>.png|jpg, items.json에 'wide_ego' 후보 추가 (앱에서 다시 확인)

사용 예:
  .venv/Scripts/python scripts/wide_to_ego.py dataset/round4_candidates
"""
import csv, json, os, sys, warnings
warnings.filterwarnings("ignore")
import cv2, numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import round4_candidates as R




def main():
    D = sys.argv[1] if len(sys.argv) > 1 else "dataset/round4_candidates"
    items = json.load(open(os.path.join(D, "items.json"), encoding="utf-8"))
    dec = {r["id"]: r["choice"] for r in csv.DictReader(open(os.path.join(D, "decisions.csv"), encoding="utf-8"))}
    wide = [it for it in items if it["kind"] == "wide" and dec.get(it["id"]) == "drop"]
    items = [it for it in items if it["kind"] != "wide_ego"]
    for sub in ("da_ego", "ll_ego"):
        os.makedirs(os.path.join(D, sub), exist_ok=True)
    models = R.load_models()

    areas = []
    for it in wide:
        base = f"dataset/ego_v2_clean/{it['split']}"
        img = cv2.imread(f"{base}/images/{it['frame']}.png")
        hda = cv2.imread(f"{base}/da_masks/{it['frame']}.png", 0) > 0
        hll = cv2.imread(f"{base}/ll_masks/{it['frame']}.png", 0) > 0
        res = R.predict(models, img)
        ell = (np.stack([res[m][1] for m in R.KEYS]) >= .5).sum(0) >= 3
        eda = (np.mean([res[m][0] for m in R.KEYS], 0) >= .5) | ell
        # 사람 da를 사람 ll(차선)로 잘라 조각을 낸다. 실선은 차로를 완전히 가른다.
        # 조각이 앙상블 ego da와 대부분(>= 0.8) 겹치면 조각 전체(경계 = 차선), 일부만 겹치면(점선 틈으로 옆 차로와 붙은 조각)
        # 앙상블 영역과 겹친 부분만, 거의 안 겹치면 버린다. (처음엔 앙상블을 GROW px 키워 겹쳤더니 da가 차선 밖으로 튀어나왔다)
        line = cv2.dilate(hll.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        n, lab = cv2.connectedComponents((hda & ~line).astype(np.uint8), connectivity=4)
        da = np.zeros_like(hda)
        near = cv2.dilate(eda.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        for c in range(1, n):
            part = lab == c
            frac = eda[part].mean()
            if frac >= 0.8:
                da |= part
            elif frac >= 0.2:
                da |= part & near & ~line
        n, lab = cv2.connectedComponents(da.astype(np.uint8), connectivity=8)
        seed = [s for s in np.unique(lab[330:, 160:480]) if s > 0]
        if seed:
            da = np.isin(lab, seed)
        ll = hll & (cv2.dilate(da.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0)
        da = da | ll
        iid = f"wide_ego_{it['frame']}"
        cv2.imwrite(os.path.join(D, "da_ego", iid + ".png"), da.astype(np.uint8) * 255)
        cv2.imwrite(os.path.join(D, "ll_ego", iid + ".png"), ll.astype(np.uint8) * 255)
        view = np.hstack([img, R.overlay(img, hda, hll, "old label (3 lanes)"),
                          R.overlay(img, da, ll, f"ego only ({da.mean() * 100:.0f}%)")])
        cv2.imwrite(os.path.join(D, "view", iid + ".jpg"), cv2.resize(view, (1440, 270)), [cv2.IMWRITE_JPEG_QUALITY, 88])
        items.append(dict(id=iid, kind="wide_ego", src=it["src"], split=it["split"], frame=it["frame"]))
        areas.append((hda.mean(), da.mean()))
    json.dump(items, open(os.path.join(D, "items.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    a = np.array(areas)
    print(f"{len(wide)}장 변환, da 면적 중앙값 {np.median(a[:, 0]) * 100:.0f}% -> {np.median(a[:, 1]) * 100:.0f}%")


if __name__ == "__main__":
    main()
