#!/usr/bin/env python3
"""HybridNets 체크포인트의 검출 헤드 상태를 확인한다 (표지판·신호등 학습 전 점검용).

확인 항목
  1. 헤드 구조: 검출 클래스 수(classifier 출력 채널 / 앵커 9개), 분할 클래스 수
  2. 공식 가중치 대비 변화량: 모듈 그룹(encoder/bifpn/.../classifier)별로
     학습 파라미터와 BN 통계(running_mean/var)를 나눠서 상대 변화량(||a-b|| / ||b||)을 잰다.
     train_seg.py는 분할 손실만 쓰므로 regressor/classifier의 conv 가중치는 0이어야 정상이고,
     backbone/BiFPN이 바뀐 만큼 검출 헤드 입력 특징이 달라졌다고 본다.
  3. 실제 검출: 같은 이미지에 체크포인트별로 검출을 돌려 개수/최고 점수를 비교하고 몽타주를 저장한다.
     기본 이미지 = HybridNets 공식 demo 도로 사진(차량이 있음) + LimoPro 주행 프레임 일부.
     공식 가중치는 demo 사진에서 차를 잡는데 파인튜닝본이 못 잡으면 검출 성능이 무너진 것.

사용 예:
  .venv/Scripts/python scripts/check_hybridnets_det.py
  .venv/Scripts/python scripts/check_hybridnets_det.py --ckpt outputs/train/hybridnets_v2_seed0/best_da.pt --n-frames 12
출력: outputs/check_hybridnets_det/{summary.json, montage.jpg}
"""
import argparse
import glob
import json
import os
import random
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_model_compare_gif as m  # noqa: E402  (ZLUDA 설정, device, 전처리 유틸)

NUM_ANCHORS = 9  # 3 scales x 3 ratios
GROUPS = ["encoder", "bifpn", "bifpndecoder", "segmentation_head", "regressor", "classifier"]
BN_BUFFERS = ("running_mean", "running_var", "num_batches_tracked")
HYB_REPO = os.path.join(m.MODELS_DIR, "HybridNets")
OFFICIAL = os.path.join(HYB_REPO, "weights", "hybridnets.pth")
OFFICIAL_OBJ_LIST = ["car"]  # projects/bdd100k.yml: obj_list ['car'], obj_combine car/bus/truck/train
STRIDE = 128  # BiFPN이 P7(1/128)까지 내려감

# 그룹 출력 순서와 맞춘 BGR 박스 색 (체크포인트별)
COLORS = [(0, 200, 255), (255, 120, 0), (0, 220, 0), (255, 0, 255), (0, 0, 255)]


def load_state_dict(path):
    """state_dict(.pth/best_*.pt) 또는 train_seg.py의 last.pt({'model': ...})를 받는다."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ck, dict) and not any(k.startswith("encoder.") for k in ck):
        for v in ck.values():
            if isinstance(v, dict) and any(k.startswith("encoder.") for k in v):
                return v
        raise ValueError(f"{path}: HybridNets state_dict를 찾지 못함 (키: {list(ck)[:5]})")
    return ck


def head_info(sd):
    cls_w = sd["classifier.header.pointwise_conv.conv.weight"]
    reg_w = sd["regressor.header.pointwise_conv.conv.weight"]
    seg_w = sd["segmentation_head.0.weight"]
    return {
        "det_classes": cls_w.shape[0] // NUM_ANCHORS,
        "classifier_out_ch": cls_w.shape[0],
        "regressor_out_ch": reg_w.shape[0],
        "seg_classes": seg_w.shape[0],
    }


def diff_by_group(sd, ref):
    """그룹별 상대 변화량. 파라미터와 BN 통계를 나눈다(헤드 BN 통계는 forward만 해도 바뀌므로)."""
    out = {}
    for g in GROUPS:
        acc = {"param": [0.0, 0.0, 0, 0], "bn_stat": [0.0, 0.0, 0, 0]}  # diff², ref², changed, total
        for k, b in ref.items():
            if k.split(".")[0] != g or k not in sd or not b.is_floating_point():
                continue
            kind = "bn_stat" if k.endswith(BN_BUFFERS) else "param"
            a = sd[k].float()
            if a.shape != b.shape:  # 클래스 수가 다른 헤드는 비교 불가
                acc[kind][2] += 1
                acc[kind][3] += 1
                continue
            d = (a - b.float())
            acc[kind][0] += float((d ** 2).sum())
            acc[kind][1] += float((b.float() ** 2).sum())
            acc[kind][2] += int(d.abs().max() > 1e-7)
            acc[kind][3] += 1
        out[g] = {kind: {"rel_change": (v[0] ** 0.5) / max(v[1] ** 0.5, 1e-12),
                         "changed_tensors": f"{v[2]}/{v[3]}"} for kind, v in acc.items() if v[3]}
    return out


def build_model(num_classes, seg_classes):
    Backbone = m.import_from_repo(HYB_REPO, "backbone", "HybridNetsBackbone")
    return Backbone(compound_coef=3, num_classes=num_classes, ratios=[(0.62, 1.58), (1.0, 1.0), (1.58, 0.62)],
                    scales=[2 ** 0, 2 ** 0.70, 2 ** 1.32], seg_classes=seg_classes - 1, backbone_name=None,
                    seg_mode="multiclass")


def preprocess(img):
    padded, crop = m.pad_to_stride(img, stride=STRIDE)
    return padded, crop, m.to_tensor(padded, rgb=True, imagenet_norm=True)


def collect_images(args):
    imgs = sorted(glob.glob(os.path.join(HYB_REPO, "demo", "image", "*.jpg")))
    frames = sorted(glob.glob(os.path.join(args.frames, "**", "*.jpg"), recursive=True))
    random.Random(args.seed).shuffle(frames)
    return [("demo", p) for p in imgs] + [("limo", p) for p in frames[:args.n_frames]]


def draw(img, dets, color, obj_list):
    for (x1, y1, x2, y2), c, s in zip(dets["rois"], dets["class_ids"], dets["scores"]):
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(img, p1, p2, color, 2)
        name = obj_list[int(c)] if int(c) < len(obj_list) else f"cls{int(c)}"
        cv2.putText(img, f"{name} {s:.2f}", (p1[0], max(p1[1] - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", nargs="*", default=None,
                    help="비교할 체크포인트 (기본: 공식 + outputs/train/hybridnets*/best_da.pt 전부)")
    ap.add_argument("--frames", default=os.path.join(m.ROOT, "dataset", "LimoPro_frames_20261027"))
    ap.add_argument("--n-frames", type=int, default=6, help="LimoPro 프레임 샘플 수")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(m.ROOT, "outputs", "check_hybridnets_det"))
    args = ap.parse_args()

    ckpts = args.ckpt if args.ckpt else sorted(glob.glob(os.path.join(m.ROOT, "outputs", "train", "hybridnets*", "best_da.pt")))
    ckpts = [OFFICIAL] + [c for c in ckpts if os.path.abspath(c) != os.path.abspath(OFFICIAL)]
    names = ["official" if c == OFFICIAL else os.path.basename(os.path.dirname(c)) for c in ckpts]
    os.makedirs(args.out, exist_ok=True)

    postprocess = m.import_from_repo(HYB_REPO, "utils.utils", "postprocess")
    BBoxTransform = m.import_from_repo(HYB_REPO, "utils.utils", "BBoxTransform")
    ClipBoxes = m.import_from_repo(HYB_REPO, "utils.utils", "ClipBoxes")
    regress_boxes, clip_boxes = BBoxTransform(), ClipBoxes()

    ref = load_state_dict(OFFICIAL)
    images = collect_images(args)
    canvases = [cv2.imread(p) for _, p in images]
    inputs = [preprocess(img) for img in canvases]
    canvases = [inp[0].copy() for inp in inputs]  # 박스 좌표가 패딩된 입력 기준이라 그 위에 그림

    summary = {"official_obj_list": OFFICIAL_OBJ_LIST, "conf": args.conf, "checkpoints": {}}
    print(f"이미지 {len(images)}장 (demo {sum(s == 'demo' for s, _ in images)}, limo {sum(s == 'limo' for s, _ in images)})")

    for ci, (name, path) in enumerate(zip(names, ckpts)):
        sd = load_state_dict(path)
        info = head_info(sd)
        obj_list = OFFICIAL_OBJ_LIST if info["det_classes"] == len(OFFICIAL_OBJ_LIST) else [f"cls{i}" for i in range(info["det_classes"])]
        entry = {"path": os.path.relpath(path, m.ROOT), **info, "diff_vs_official": diff_by_group(sd, ref)}

        model = build_model(info["det_classes"], info["seg_classes"])
        model.load_state_dict(sd)
        model.to(m.device).eval()

        per_src = {"demo": [], "limo": []}
        with torch.no_grad():
            for (src, _), (_, _, x), canvas in zip(images, inputs, canvases):
                _, regression, classification, anchors, _ = model(x)
                dets = postprocess(x, anchors, regression, classification, regress_boxes, clip_boxes,
                                   args.conf, args.iou)[0]
                per_src[src].append((len(dets["scores"]), float(classification.max())))
                draw(canvas, dets, COLORS[ci % len(COLORS)], obj_list)
        entry["detections"] = {
            src: {"images": len(v), "mean_boxes": float(np.mean([n for n, _ in v])) if v else 0.0,
                  "mean_max_score": float(np.mean([s for _, s in v])) if v else 0.0}
            for src, v in per_src.items()}
        summary["checkpoints"][name] = entry

        print(f"\n[{name}] {entry['path']}")
        print(f"  검출 클래스 {info['det_classes']}개 (classifier 출력 {info['classifier_out_ch']}ch)"
              f" / 분할 클래스 {info['seg_classes']}개 (bg+road+lane)")
        if name != "official":
            print("  공식 대비 상대 변화량 (param | bn_stat):")
            for g, d in entry["diff_vs_official"].items():
                p, b = d.get("param", {}), d.get("bn_stat", {})
                print(f"    {g:18s} {p.get('rel_change', 0):8.4f} ({p.get('changed_tensors', '-'):>7s})"
                      f" | {b.get('rel_change', 0):8.4f} ({b.get('changed_tensors', '-'):>7s})")
        for src, d in entry["detections"].items():
            print(f"  검출 [{src}] 평균 박스 {d['mean_boxes']:.2f}개, 평균 최고 점수 {d['mean_max_score']:.3f}"
                  f" (conf {args.conf})")
        del model
        if m.device.type == "cuda":
            torch.cuda.empty_cache()

    # 몽타주: 이미지마다 한 칸, 체크포인트별 박스 색 범례
    h, w = canvases[0].shape[:2]
    tiles = [cv2.resize(c, (w, h)) for c in canvases]
    cols = 3
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)])
    legend = np.full((28 * len(names) + 8, grid.shape[1], 3), 255, np.uint8)
    for i, n in enumerate(names):
        cv2.putText(legend, n, (10, 24 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS[i % len(COLORS)], 2)
    cv2.imwrite(os.path.join(args.out, "montage.jpg"), np.vstack([legend, grid]))
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {os.path.relpath(args.out, m.ROOT)}/summary.json, montage.jpg")


if __name__ == "__main__":
    main()
