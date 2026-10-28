#!/usr/bin/env python3
"""make_ego_lane_labels.py --save-labels 결과를 status.csv의 OK / FAIL로 나눈다.

  <labels>/images, da_masks, ll_masks, status.csv
    -> <labels>/ok/{images,da_masks,ll_masks}, <labels>/fail/{images,da_masks,ll_masks}
       + ok/status.csv, fail/status.csv (원본 status.csv는 그대로 둠)

FAIL = ego lane이 da를 못 잡은 프레임(좌/우 차선 한쪽 누락, 추적 끊김, 추적 행 부족). 주로 커브.
이 프레임들은 자동 라벨을 쓰지 않고 사람이 SAM 클릭으로 라벨링한다(docs/EGO_LANE_PSEUDO_LABEL.md §6).

사용 예:
  .venv/Scripts/python scripts/split_ego_labels.py outputs/ego_labels_20261027_183234
"""
import argparse
import collections
import csv
import os
import shutil

SUBDIRS = ("images", "da_masks", "ll_masks")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels", help="make_ego_lane_labels.py --save-labels 폴더")
    args = ap.parse_args()

    with open(os.path.join(args.labels, "status.csv"), encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    groups = {"ok": [r for r in rows if r["status"] == "OK"], "fail": [r for r in rows if r["status"] != "OK"]}

    for name, rs in groups.items():
        for sub in SUBDIRS:
            os.makedirs(os.path.join(args.labels, name, sub), exist_ok=True)
        for r in rs:
            stem = os.path.splitext(r["frame"])[0]
            for sub in SUBDIRS:
                src = os.path.join(args.labels, sub, f"{stem}.png")
                if os.path.exists(src):
                    shutil.move(src, os.path.join(args.labels, name, sub, f"{stem}.png"))
        with open(os.path.join(args.labels, name, "status.csv"), "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rs)
    for sub in SUBDIRS:  # 다 옮겼으면 빈 폴더 정리
        d = os.path.join(args.labels, sub)
        if os.path.isdir(d) and not os.listdir(d):
            os.rmdir(d)

    reasons = collections.Counter(r["status"].split(" (")[0] for r in groups["fail"])
    print(f"OK {len(groups['ok'])} / FAIL {len(groups['fail'])} (총 {len(rows)})")
    for k, v in reasons.most_common():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
