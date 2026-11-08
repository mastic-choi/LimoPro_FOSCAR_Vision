#!/usr/bin/env python3
"""다른 PC에서 바로 학습할 수 있게 코드 + 모델 + 데이터셋을 zip 하나로 묶는다 (사용법: tools/rtx/README_RTX.txt).

작은 파일 1만 개를 USB에 그대로 복사하면 느려서 무압축 zip 하나로 만들고, USB에서 다시 읽어 CRC를 검사한다.

사용 예:
  .venv/Scripts/python -I scripts/make_rtx_bundle.py --data dataset/ego_v2 --out E:/2026AutoRace/LimoPro_train_bundle
  # 3차, H100 서버용 (대회장 조명 견본 + 1·2차 가중치 + README_H100.md, GPU 사용률 기본 100%)
  .venv/Scripts/python -I scripts/make_rtx_bundle.py --target h100 --data dataset/ego_v3 --extra-data dataset/venue_ref_20261106       --weights v1 v2 --out outputs/share
"""
import argparse
import json
import os
import time
import shutil
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOP = "LimoPro_train"
SCRIPTS = ("train_seg.py", "train_all.py", "train_watchdog.py", "make_model_compare_gif.py", "venue_aug.py")
RTX = ("run_train.sh", "run_train.bat", "requirements_rtx.txt", "README_RTX.txt")
# models/ 안에서 학습에 안 쓰는 것
SKIP_DIRS = {".git", "__pycache__", "images", "demo", "inference", "tutorial", "pictures", "toolkits"}
SKIP_EXT = {".mp4", ".gif", ".jpg", ".png", ".onnx"}
SKIP_WEIGHTS = {"medium.pth", "small.pth", "nano.pth"}
RUNS = {"v1": {"hybridnets": "hybridnets_seed0", "tln_large": "tln_large_seed0", "yolop_chbce": "yolop_chbce_seed0", "yolopv2": "yolopv2_seed0"},
        "v2": {"hybridnets": "hybridnets_v2_seed0", "tln_large": "tln_large_v2_seed0", "yolop": "yolop_v2_seed0", "yolopv2": "yolopv2_v2_seed0"}}


def walk(base, skip_dirs=()):
    for d, subs, files in os.walk(base):
        subs[:] = [s for s in subs if s not in skip_dirs]
        for f in files:
            yield os.path.join(d, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "dataset", "ego_v2"))
    ap.add_argument("--out", required=True, help="결과 폴더 (USB)")
    ap.add_argument("--extra-data", nargs="*", default=[], help="dataset/ 아래에 함께 넣을 폴더 (예: 대회장 조명 견본)")
    ap.add_argument("--weights", nargs="*", default=[], choices=list(RUNS), help="weights/에 넣을 이전 라운드 best_da.pt")
    ap.add_argument("--target", choices=["rtx", "h100"], default="rtx")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    zip_path = os.path.join(args.out, f"{TOP}.zip" if args.target == "rtx" else f"{TOP}_v3_h100.zip")
    tmp = zip_path + ".tmp"
    n = 0
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:
        for s in SCRIPTS:
            z.write(os.path.join(ROOT, "scripts", s), f"{TOP}/scripts/{s}")
        z.write(os.path.join(ROOT, "tools", "rtx", "run_train.py"), f"{TOP}/scripts/run_train.py")
        for s in RTX:  # git autocrlf로 CRLF가 돼도 Ubuntu에서 돌도록 .sh는 LF, .bat은 CRLF로 고정
            with open(os.path.join(ROOT, "tools", "rtx", s), "rb") as f:
                b = f.read().replace(b"\r\n", b"\n")
            z.writestr(f"{TOP}/{s}", b.replace(b"\n", b"\r\n") if s.endswith(".bat") else b)
        z.write(os.path.join(ROOT, "tools", "rtx", "CLAUDE_RTX.md"), f"{TOP}/CLAUDE.md")  # 동아리 PC의 Claude Code 지시
        models = os.path.join(ROOT, "models")
        for repo in ("HybridNets", "TwinLiteNetPlus", "YOLOP", "YOLOPv2"):
            for p in walk(os.path.join(models, repo), SKIP_DIRS):
                if os.path.splitext(p)[1].lower() in SKIP_EXT or os.path.basename(p) in SKIP_WEIGHTS:
                    continue
                z.write(p, f"{TOP}/models/{os.path.relpath(p, models)}".replace(os.sep, "/"))
                n += 1
        for d in [args.data] + args.extra_data:
            name = os.path.basename(os.path.normpath(d))
            for p in walk(d):
                z.write(p, f"{TOP}/dataset/{name}/{os.path.relpath(p, d)}".replace(os.sep, "/"))
                n += 1
        for rnd in args.weights:
            for key, run in RUNS[rnd].items():
                z.write(os.path.join(ROOT, "outputs", "train", run, "best_da.pt"), f"{TOP}/weights/{key}_{rnd}_best_da.pt")
        if args.target == "h100":
            z.write(os.path.join(ROOT, "tools", "rtx", "README_H100.md"), f"{TOP}/README.md")
        z.writestr(f"{TOP}/bundle.json", json.dumps({"target": args.target, "gpu_duty": 1.0 if args.target == "h100" else 0.9,
                                                      "data": os.path.basename(os.path.normpath(args.data)), "weights": args.weights,
                                                      "built": time.strftime("%Y-%m-%d %H:%M")}, ensure_ascii=False, indent=1))
    os.replace(tmp, zip_path)
    with zipfile.ZipFile(zip_path) as z:  # 다시 읽어서 검증
        bad = z.testzip()
    assert bad is None, f"zip 손상: {bad}"
    shutil.copy(os.path.join(ROOT, "tools", "rtx", "README_H100.md" if args.target == "h100" else "README_RTX.txt"), args.out)
    print(f"{zip_path}: {os.path.getsize(zip_path) / 1e9:.2f}GB, 모델/데이터 파일 {n}개, CRC 검사 통과")


if __name__ == "__main__":
    main()
