#!/usr/bin/env python3
"""H100 다중 작업 v3 묶음(LimoPro_mt_v3_h100.zip)을 만든다. 사용법은 tools/h100_mt_v3/README_V3.md.

이미지는 담지 않는다: 서버에 v2 학습 때 풀어 둔 LimoPro_mt 폴더(dataset/synth_signs_v2 Klein 다듬기 끝, cones_v1+FSOCO,
venue_ref, models/HybridNets, .venv)를 그대로 쓰고, 그 위에 이 zip을 덮어 푼다.
담는 것: 학습 스크립트(v3 옵션 + 학습 중 모니터·케이블 증강), 시작 가중치(multitask-v2 best), run_v3.sh, README, SHA256SUMS.
v2 run.sh는 덮어쓰지 않도록 run_v3.sh로 넣는다. .sh는 LF 줄바꿈으로 바꿔 넣는다.

사용 예:
  .venv/Scripts/python -I scripts/make_h100_mt_v3_bundle.py --out outputs/share
"""
import argparse, hashlib, os, time, zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOP = "LimoPro_mt"
SCRIPTS = ("train_multitask.py", "train_seg.py", "make_model_compare_gif.py", "venue_aug.py", "v5_bag_colors.json",
           "mt_online_aug.py", "mt_calib.py", "mt_det_loss.py", "eval_mt_monitor.py", "preview_monitor_real.py")
V3 = os.path.join(ROOT, "tools", "h100_mt_v3")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "share"))
    ap.add_argument("--weights", default=os.path.join(ROOT, "outputs", "train", "hybridnets_mt_v2", "hybridnets_mt_v2_best.pt"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dst = os.path.join(a.out, "LimoPro_mt_v3_h100.zip")
    items = [(os.path.join(V3, "run.sh"), "run_v3.sh"), (os.path.join(V3, "README_V3.md"), "README_V3.md"),
             (os.path.join(V3, "requirements.txt"), "requirements.txt")]
    items += [(os.path.join(ROOT, "scripts", f), f"scripts/{f}") for f in SCRIPTS]
    items.append((a.weights, "weights/hybridnets_mt_v2_best.pt"))
    data = {}
    for s, d in items:
        b = open(s, "rb").read()
        if d.endswith(".sh"):
            b = b.replace(b"\r\n", b"\n")
        data[d] = b
    data["SHA256SUMS"] = "".join(f"{hashlib.sha256(b).hexdigest()}  {d}\n"
                                 for d, b in data.items() if d.startswith(("scripts/", "weights/"))).encode()
    print(f"{len(data)} files, {sum(map(len, data.values())) / 1e6:.1f} MB -> {dst}", flush=True)
    t0 = time.time()
    with zipfile.ZipFile(dst + ".tmp", "w", zipfile.ZIP_DEFLATED) as z:
        for d, b in data.items():
            zi = zipfile.ZipInfo(f"{TOP}/{d}", time.localtime()[:6])
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = (0o755 if d.endswith(".sh") else 0o644) << 16
            z.writestr(zi, b)
    os.replace(dst + ".tmp", dst)
    with zipfile.ZipFile(dst) as z:
        bad = z.testzip()
        names = z.namelist()
    assert not any(n.lower().endswith((".png", ".jpg")) for n in names)
    print("\n".join("  " + n for n in names))
    print(f"done {time.time() - t0:.0f}s, {os.path.getsize(dst) / 1e6:.1f} MB, CRC {'OK' if bad is None else 'BAD ' + bad}")


if __name__ == "__main__":
    main()
