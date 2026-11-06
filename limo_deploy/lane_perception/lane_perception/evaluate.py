"""Scores adapters on the labeled test set, for both .pt and the FP16 engine.

  python -m lane_perception.evaluate --data ~/Downloads/LimoPro_ego_v2_clean --model yolop

IoU pools pixels over a group, as in the dataset README, with threshold 0.5.
da is scored two ways: as the node publishes it (da) and merged with ll (da|ll).
Reference values are the PyTorch scores in the dataset README (ego-lane-v2, best_da).
"""
import argparse
import csv
import json
import os

import cv2
import numpy as np

from .adapters import make_adapter

# model: all da, all ll, easy da, easy ll, new da, new ll
REFERENCE = {
    "hybridnets": (0.9476, 0.8242, 0.9589, 0.8021, 0.9361, 0.8499),
    "tln": (0.9430, 0.8101, 0.9564, 0.7831, 0.9294, 0.8416),
    "yolop": (0.9350, 0.7887, 0.9468, 0.7686, 0.9230, 0.8121),
    "yolopv2": (0.9481, 0.8282, 0.9602, 0.8058, 0.9358, 0.8542),
}


class Pool:
    def __init__(self):
        self.tp = self.fp = self.fn = 0

    def add(self, p, t):
        self.tp += int((p & t).sum())
        self.fp += int((p & ~t).sum())
        self.fn += int((~p & t).sum())

    def value(self):
        return self.tp / max(self.tp + self.fp + self.fn, 1)


def iou(a, b):
    return float((a & b).sum() / max((a | b).sum(), 1))


def score(adapter, frames, data):
    keys = ("da", "da_or", "ll")
    pools = {g: {k: Pool() for k in keys} for g in ("all", "easy", "new", "curve", "corrected")}
    masks = {}
    for name, row in frames.items():
        img = cv2.imread(os.path.join(data, "test", "images", name + ".png"))
        gda = cv2.imread(os.path.join(data, "test", "da_masks", name + ".png"), 0) > 0
        gll = cv2.imread(os.path.join(data, "test", "ll_masks", name + ".png"), 0) > 0
        da, ll = adapter.postprocess(adapter.infer(adapter.preprocess(img)))
        da, ll = da > 0, ll > 0
        masks[name] = (np.packbits(da), np.packbits(ll))
        groups = ["all", "easy" if row["split_old"] == "test" else "new"]
        if row["curve"] == "1":
            groups.append("curve")
        if row["decision"] == "corrected":
            groups.append("corrected")
        for g in groups:
            pools[g]["da"].add(da, gda)
            pools[g]["da_or"].add(da | ll, gda)
            pools[g]["ll"].add(ll, gll)
    return {g: {k: round(p.value(), 4) for k, p in v.items()} for g, v in pools.items()}, masks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True, choices=list(REFERENCE))
    ap.add_argument("--out", default=os.path.expanduser("~/lane_perception/eval"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.data, "manifest.csv"), encoding="utf-8-sig") as f:
        frames = {r["frame"]: r for r in csv.DictReader(f) if r["split"] == "test" and r["use"] == "1"}
    frames = dict(sorted(frames.items()))

    pt = make_adapter(a.model, "pt")
    pt_scores, pt_masks = score(pt, frames, a.data)
    del pt
    trt = make_adapter(a.model, "trt")
    trt_scores, trt_masks = score(trt, frames, a.data)

    diffs = {}
    shape = (360, 640)
    for name in frames:
        for i, k in enumerate(("da", "ll")):
            p = np.unpackbits(pt_masks[name][i]).reshape(shape).astype(bool)
            q = np.unpackbits(trt_masks[name][i]).reshape(shape).astype(bool)
            diffs.setdefault(k, {})[name] = 1 - iou(p, q)
    agree = {}
    for k, v in diffs.items():
        arr = np.array(list(v.values()))
        worst = sorted(v, key=v.get, reverse=True)[:5]
        agree[k] = {"mean": round(float(arr.mean()), 5), "p95": round(float(np.percentile(arr, 95)), 5), "max": round(float(arr.max()), 5),
                    "over_0.005": int((arr > 0.005).sum()), "worst": {n: round(v[n], 4) for n in worst}}
    res = {"model": a.model, "frames": len(frames), "reference": REFERENCE[a.model], "pt": pt_scores, "fp16": trt_scores, "pt_vs_fp16": agree}
    with open(os.path.join(a.out, f"{a.model}.json"), "w") as f:
        json.dump(res, f, indent=1)
    ref = REFERENCE[a.model]
    print(a.model, "ref all da/ll", ref[0], ref[1])
    print(" pt   ", pt_scores["all"])
    print(" fp16 ", trt_scores["all"])
    print(" agree", {k: {x: y for x, y in v.items() if x != "worst"} for k, v in agree.items()}, flush=True)


if __name__ == "__main__":
    main()
