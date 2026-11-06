"""Per-stage speed of all models through the real node. Run with no other load (Claude Code closed).

Camera already running:  python -m lane_perception.bench --seconds 60
Bag as input:            python -m lane_perception.bench --bag ~/bags/x --seconds 40 --allow-mismatch
"""
import argparse
import csv
import os
import re
import signal
import subprocess
import time

import numpy as np

STAGES = ["decode", "resize", "preprocess", "infer", "postprocess", "overlay", "publish", "total"]
WARMUP = 30
OUT = os.path.expanduser("~/lane_perception/bench")


def stop(proc):
    if proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(15)
        except subprocess.TimeoutExpired:
            proc.kill()


def summarize(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))[WARMUP:]
    if len(rows) < 10:
        return None
    wall = np.array([float(r["wall"]) for r in rows])
    res = {"frames": len(rows), "fps": (len(rows) - 1) / (wall[-1] - wall[0])}
    for s in STAGES:
        v = np.array([float(r[s]) for r in rows])
        res[s] = (v.mean(), np.percentile(v, 95))
    return res


def tegra(path):
    ram, gpu = [], []
    for line in open(path, errors="ignore"):
        m = re.search(r"RAM (\d+)/", line)
        g = re.search(r"GR3D_FREQ (\d+)%", line)
        if m:
            ram.append(int(m.group(1)))
        if g:
            gpu.append(int(g.group(1)))
    return (max(ram), float(np.mean(gpu))) if ram and gpu else (0, 0.0)


def run(model, a, stamp):
    base = os.path.join(OUT, f"{stamp}_{model}_{a.backend}")
    cmd = ["ros2", "launch", "lane_perception", "lane_perception.launch.py", f"model:={model}", f"backend:={a.backend}",
           f"allow_aspect_mismatch:={'true' if a.allow_mismatch else 'false'}", f"timing_csv:={base}.csv", f"publish_overlay:={'false' if a.no_overlay else 'true'}"]
    log = open(base + ".node.log", "w")
    node = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(180):
        if "ready:" in open(base + ".node.log", errors="ignore").read() or node.poll() is not None:
            break
        time.sleep(1)
    if node.poll() is not None:
        return None, "node exited, see " + base + ".node.log"
    idle = int(subprocess.check_output("free -m | awk 'NR==2{print $3}'", shell=True))
    with open(base + ".tegra.log", "w") as tf:
        teg = subprocess.Popen(["tegrastats", "--interval", "500"], stdout=tf, stderr=subprocess.DEVNULL)
        bag = None
        if a.bag:
            bag = subprocess.Popen(["ros2", "bag", "play", a.bag, "--topics", a.topic], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(a.seconds)
        for p in (bag, teg, node):
            if p is not None:
                stop(p)
    if not os.path.exists(base + ".csv"):
        return None, "no timing file"
    res = summarize(base + ".csv")
    if res is None:
        return None, "too few frames: is the camera topic publishing a 16:9 image?"
    res["ram_loaded"] = idle
    res["ram_peak"], res["gpu_util"] = tegra(base + ".tegra.log")
    return res, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["tln", "yolop", "hybridnets", "yolopv2"])
    ap.add_argument("--backend", default="trt")
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--bag", default="")
    ap.add_argument("--topic", default="/camera/color/image_raw")
    ap.add_argument("--allow-mismatch", action="store_true")
    ap.add_argument("--no-overlay", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    stamp = time.strftime("%m%d_%H%M%S")
    power = subprocess.run("nvpmodel -q | head -1", shell=True, capture_output=True, text=True).stdout.strip()
    lines = [f"# Speed {stamp}", "", f"- source: {a.bag or 'live ' + a.topic}", f"- backend: {a.backend}", f"- overlay: {'off' if a.no_overlay else 'on'}", f"- power: {power}",
             f"- {a.seconds:.0f}s per model, first {WARMUP} frames dropped", "- each cell: mean / p95 ms", "",
             "| model | fps | " + " | ".join(STAGES) + " | RAM peak MB | GPU % |", "|---|" + "---:|" * (len(STAGES) + 3)]
    for m in a.models:
        print("measuring", m, flush=True)
        res, err = run(m, a, stamp)
        if res is None:
            lines.append(f"| {m} | failed: {err} |")
            continue
        cells = " | ".join(f"{res[s][0]:.1f} / {res[s][1]:.1f}" for s in STAGES)
        lines.append(f"| {m} | {res['fps']:.1f} | {cells} | {res['ram_peak']} | {res['gpu_util']:.0f} |")
        time.sleep(5)
    path = os.path.join(OUT, f"{stamp}_summary.md")
    open(path, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("saved:", path)


if __name__ == "__main__":
    main()
