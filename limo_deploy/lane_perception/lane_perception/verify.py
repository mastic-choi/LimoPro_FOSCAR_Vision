"""Label-free check on the same images (test data unavailable, so this does not reproduce test IoU).

  python -m lane_perception.verify --model yolopv2 --frames DIR --refcode LimoPro_train/scripts [--trt]

1) adapter vs training code (eval_ensemble.prep + train_seg forward and crop): masks must be identical.
2) with --trt: .pt adapter vs FP16 engine, per-frame IoU difference from 1.0.
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch

from .adapters import PAD, SPECS, THIRD_PARTY, default_weight, make_adapter

REF = {"tln": "tln_large", "yolop": "yolop", "hybridnets": "hybridnets", "yolopv2": "yolopv2"}


def reference(model, refcode, device):
    """Training code itself: ADAPTERS for the model and forward, eval_ensemble.prep, train_seg.crop."""
    sys.path[:0] = [refcode, os.path.join(THIRD_PARTY, "pydeps")]
    import eval_ensemble as E
    import make_model_compare_gif as m
    T = E.T
    m.MODELS_DIR = THIRD_PARTY
    w = default_weight(model, "pt")
    real_load, real_y2 = torch.load, T.load_yolopv2_trainable
    # ADAPTERS load public pretrained files that are not kept here; the trained weights replace them below.
    pretrained = ("large.pth", "End-to-end.pth", "hybridnets.pth")

    def fake_load(path, *a, **k):
        if str(path).endswith(pretrained):
            sd = real_load(w, map_location="cpu")
            return {"state_dict": sd} if str(path).endswith("End-to-end.pth") else sd
        return real_load(path, *a, **k)

    torch.load, T.load_yolopv2_trainable = fake_load, lambda path, dev: None
    try:
        net, rgb, norm, fwd, _ = T.ADAPTERS[REF[model]](m)
    finally:
        torch.load, T.load_yolopv2_trainable = real_load, real_y2
    if model == "yolopv2":
        net = real_y2(w, device)
    else:
        net.load_state_dict(real_load(w, map_location="cpu"))
    net = net.to(device).eval()

    @torch.no_grad()
    def run(img):
        a, b = fwd(net, E.prep(img, rgb, norm)[None].to(device))
        return (T.crop(a)[0].cpu().numpy() >= .5), (T.crop(b)[0].cpu().numpy() >= .5)
    return run


def iou(a, b):
    return float((a & b).sum() / max((a | b).sum(), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(SPECS))
    ap.add_argument("--frames", required=True)
    ap.add_argument("--refcode", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--trt", action="store_true")
    a = ap.parse_args()
    frames = [cv2.imread(f) for f in sorted(glob.glob(os.path.join(a.frames, "*.png")))]
    if not frames:
        raise SystemExit("no frames")
    dev = torch.device(a.device)
    pt = make_adapter(a.model, "pt", device=a.device)
    ref = reference(a.model, a.refcode, dev)
    diff = 0
    for img in frames:
        da, ll = pt.postprocess(pt.infer(pt.preprocess(img)))
        rda, rll = ref(img)
        diff += int(((da > 0) != rda).sum() + ((ll > 0) != rll).sum())
    print(f"[adapter vs training code] frames={len(frames)} differing_pixels={diff}", flush=True)
    area = [float((pt.postprocess(pt.infer(pt.preprocess(i)))[0] > 0).mean()) for i in frames[:5]]
    print("da area sample:", [round(x, 3) for x in area], "pad:", PAD)
    if a.trt:
        trt = make_adapter(a.model, "trt")
        d, l = [], []
        for img in frames:
            p = pt.postprocess(pt.infer(pt.preprocess(img)))
            q = trt.postprocess(trt.infer(trt.preprocess(img)))
            d.append(iou(p[0] > 0, q[0] > 0))
            l.append(iou(p[1] > 0, q[1] > 0))
        print(f"[pt vs fp16] da IoU mean={np.mean(d):.5f} min={np.min(d):.5f} | ll IoU mean={np.mean(l):.5f} min={np.min(l):.5f}")
        print("max 1-IoU:", round(1 - min(min(d), min(l)), 5), "(criterion 0.005)")


if __name__ == "__main__":
    main()
