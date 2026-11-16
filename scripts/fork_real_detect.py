#!/usr/bin/env python3
"""M7 원본 프레임(1280x720 PNG) 폴더마다 multitask-v3real 검출을 두 가지로 돌리고, 표지판류 후보를 갈림길 분류기로 확인한다.

  full : 640x360 으로 줄인 전체 화면 (로봇과 같음)
  zoom : 원본에서 y 100~460 띠를 640x360 타일 3개(x 0 / 320 / 640)로 잘라 원본 해상도 그대로 = 2배 확대. 먼 표지판용
후보 = 표지판류(sign_A/B, slow, parking, pedestrian, bump) 점수 >= 0.15. 박스는 모두 원본 1280x720 좌표.
분류기(fork-sign ONNX)는 원본에서 로봇 크롭 규칙으로 잘라 [pA, pB, pNone].

사용 (v3real 묶음 venv): python scripts/fork_real_detect.py <bundle> <ckpt> <onnx> <run 폴더>...   → <run>/det.json
"""
import json, os, sys

import cv2
import numpy as np
import torch

bundle, ckpt, onnx = sys.argv[1], os.path.abspath(sys.argv[2]), os.path.abspath(sys.argv[3])
runs = [os.path.abspath(r) for r in sys.argv[4:]]
sys.path.insert(0, os.path.join(bundle, "scripts"))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))   # fork_crop_rule (이 레포 scripts/)
import fork_crop_rule as rule   # noqa: E402
import make_model_compare_gif as m   # noqa: E402
import train_multitask as tm   # noqa: E402
os.chdir(m.ROOT)
sys.path.insert(0, os.path.join(m.MODELS_DIR, "HybridNets"))
import utils.utils as util   # noqa: E402

C = tm.CLASSES
SIGNLIKE = {"sign_A", "sign_B", "slow", "parking", "pedestrian", "bump"}
model = tm.build_model(m, ckpt).to(m.device).eval()
net = cv2.dnn.readNetFromONNX(onnx)
reg_t, clip_t = util.BBoxTransform(), util.ClipBoxes()
Y0, TILES = 100, (0, 320, 640)


def prep(im640):
    p = cv2.copyMakeBorder(im640, tm.PAD, tm.PAD, 0, 0, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return ((cv2.cvtColor(p, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 - tm.MEAN) / tm.STD).transpose(2, 0, 1)


@torch.no_grad()
def detect(batch):
    x = torch.from_numpy(np.stack(batch)).to(m.device)
    reg, cls, anchors, *_ = tm.heads(model, x)
    res = util.postprocess(x, anchors.float(), reg.float(), cls.float(), reg_t, clip_t, 0.15, 0.5)
    out = []
    for o in res:
        d = []
        for b, c, s in zip(o["rois"], o["class_ids"], o["scores"]):
            if C[int(c)] in SIGNLIKE:
                bb = np.array(b, np.float32); bb[[1, 3]] -= tm.PAD
                d.append((C[int(c)], float(s), bb))
        out.append(d)
    return out


for run in runs:
    fs = sorted(os.listdir(os.path.join(run, "frames")))
    res = []
    for i in range(0, len(fs), 4):
        ch = fs[i:i + 4]
        ims = [cv2.imread(os.path.join(run, "frames", f)) for f in ch]
        batch = []
        for im in ims:
            batch.append(prep(cv2.resize(im, (640, 360), interpolation=cv2.INTER_AREA)))
            for x0 in TILES:
                batch.append(prep(im[Y0:Y0 + 360, x0:x0 + 640]))
        dets = detect(batch)
        for k, (f, im) in enumerate(zip(ch, ims)):
            cand = []
            for src, d in zip(("full",) + tuple(f"zoom{x0}" for x0 in TILES), dets[k * 4:(k + 1) * 4]):
                for cn, sc, bb in d:
                    if src == "full":
                        b = bb * 2.0
                    else:
                        x0 = int(src[4:])
                        b = bb + np.array([x0, Y0, x0, Y0], np.float32)
                        if b[0] <= x0 + 2 and x0 > 0 or b[2] >= x0 + 638 and x0 < 640:   # 타일 가장자리에 걸린 박스는 버림
                            continue
                    cand.append({"src": src, "cls": cn, "score": round(sc, 3), "box": [round(float(v), 1) for v in b]})
            if cand:
                X = []
                for c in cand:
                    b = c["box"]
                    x, y, s, _ = rule.crop_rect((b[0] / 2, b[1] / 2, b[2] / 2, b[3] / 2), 640, 360, 1280, 720)
                    X.append(rule.preprocess(im[y:y + s, x:x + s], rgb=False, size=96).astype(np.float32) / 255.0)
                net.setInput(np.stack(X)[:, None]); P = net.forward().reshape(-1, 3)
                for c, p in zip(cand, P): c["cnn"] = [round(float(v), 3) for v in p]
            res.append({"frame": f, "cand": cand})
    json.dump(res, open(os.path.join(run, "det.json"), "w"))
    def has(src_pred):
        return sum(any(src_pred(c) for c in r["cand"]) for r in res)
    sign = lambda c: max(c["cnn"][0], c["cnn"][1]) >= 0.6 and abs(c["cnn"][0] - c["cnn"][1]) >= 0.2
    print(f"{os.path.basename(run)}: {len(fs)}장 | 분류기가 표지판이라 한 후보가 있는 프레임: 일반 {has(lambda c: c['src'] == 'full' and sign(c))}, "
          f"확대 {has(lambda c: c['src'] != 'full' and sign(c))}, 둘 중 하나 {has(sign)}", flush=True)
