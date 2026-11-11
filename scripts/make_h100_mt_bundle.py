#!/usr/bin/env python3
"""H100 다중 작업 학습 묶음(LimoPro_mt_h100.zip)을 만든다. 사용법은 tools/h100_mt/README_H100.md.

담는 것: 학습·데이터 스크립트, HybridNets 코드(가중치 제외), 4차 차선 가중치, synth_signs_v2, cones_v1(TraCon),
        대회장 조명 견본, run.sh/README/requirements. FSOCO와 Klein 모델은 서버에서 받는다.
담지 않는 것: 트랙 도면 지도(팀 비공개), 원본 사진.
무압축 zip(사진은 이미 압축돼 있음) -> 다시 열어 CRC 검사.

사용 예:
  .venv/Scripts/python -I scripts/make_h100_mt_bundle.py --out outputs/share
"""
import argparse, os, time, zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOP = "LimoPro_mt"
SCRIPTS = ("train_multitask.py", "train_seg.py", "make_model_compare_gif.py", "venue_aug.py",
           "synth_klein_refine.py", "progress_report.py", "prepare_cones.py", "synth_preview.py",
           "v5_bag_colors.json")  # venue_aug v5가 읽는 색 표 (11/10 서버에서 빠졌던 파일)
SKIP_DIRS = {".git", "__pycache__", "images", "demo", "inference", "tutorial", "pictures", "toolkits", "weights"}
SKIP_EXT = {".mp4", ".gif", ".jpg", ".png", ".onnx", ".pth", ".pt"}


def walk(base, skip_dirs=()):
    for d, subs, files in os.walk(base):
        subs[:] = [s for s in subs if s not in skip_dirs]
        for f in files:
            yield os.path.join(d, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "share"))
    ap.add_argument("--weights", default=os.path.join(ROOT, "outputs", "train", "hybridnets_v4_seed0", "best_da.pt"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dst = os.path.join(a.out, "LimoPro_mt_h100.zip")
    items = []   # (원본, zip 안 경로)
    for f in ("run.sh", "README_H100.md", "requirements.txt"):
        items.append((os.path.join(ROOT, "tools", "h100_mt", f), f"{TOP}/{f}"))
    for f in SCRIPTS:
        items.append((os.path.join(ROOT, "scripts", f), f"{TOP}/scripts/{f}"))
    hb = os.path.join(ROOT, "models", "HybridNets")
    for p in walk(hb, SKIP_DIRS):
        if os.path.splitext(p)[1].lower() not in SKIP_EXT:
            items.append((p, f"{TOP}/models/HybridNets/" + os.path.relpath(p, hb).replace(os.sep, "/")))
    items.append((a.weights, f"{TOP}/weights/hybridnets_v4_best_da.pt"))
    for ds, skip in (("synth_signs_v2", {"_klein_tmp", "_klein_tmp_s0"}), ("cones_v1", set()), ("venue_ref_20261106", set())):
        base = os.path.join(ROOT, "dataset", ds)
        for p in walk(base, skip):
            items.append((p, f"{TOP}/dataset/{ds}/" + os.path.relpath(p, base).replace(os.sep, "/")))
    total = sum(os.path.getsize(s) for s, _ in items)
    print(f"{len(items)} files, {total / 1e9:.2f} GB -> {dst}", flush=True)
    t0 = time.time()
    with zipfile.ZipFile(dst + ".tmp", "w", zipfile.ZIP_STORED, allowZip64=True) as z:
        for k, (s, d) in enumerate(items):
            z.write(s, d)
            if (k + 1) % 5000 == 0:
                print(f"  {k + 1}/{len(items)}", flush=True)
    os.replace(dst + ".tmp", dst)
    with zipfile.ZipFile(dst) as z:
        bad = z.testzip()
    print(f"done {time.time() - t0:.0f}s, {os.path.getsize(dst) / 1e9:.2f} GB, CRC {'OK' if bad is None else 'BAD ' + bad}")


if __name__ == "__main__":
    main()
