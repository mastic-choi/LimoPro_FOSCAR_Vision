#!/usr/bin/env python3
"""models/의 da/ll 모델 7개 추론 속도 벤치마크 (batch=1, 640x480 프레임).

모델 로더/전처리는 make_model_compare_gif.py의 것을 그대로 쓴다(비교 GIF와 같은 조건).
측정 항목:
  - forward: 전처리 끝난 입력 텐서로 model(x)만. GPU sync 포함 ms.
  - end2end: BGR 프레임 -> 전처리 -> forward -> argmax -> CPU 마스크(원본 해상도)까지.
             실차 파이프라인에서 체감하는 지연에 가깝다.
첫 호출 시간(first)도 따로 기록한다(ZLUDA 커널 컴파일/cudnn autotune 등이 들어감).

속도 측정이라 모델은 **하나씩 순서대로**, 각자 별도 프로세스에서 돈다(동시 실행 경쟁 X,
YOLOPv2가 바꾸는 TorchScript 전역 설정이 다른 모델에 안 새게).

사용 예:
  .venv/Scripts/python scripts/benchmark_models.py --frame D:/fine-tune/LimoPro_frames_20261023/frame_000200.png
  # Jetson 등 다른 장비에서도 같은 명령(models/ 준비 후) -> 결과 표를 그대로 비교
"""
import argparse
import datetime
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def timed(fn, sync, iters):
    ts = []
    for _ in range(iters):
        t = time.perf_counter()
        fn()
        sync()
        ts.append((time.perf_counter() - t) * 1000)
    return np.array(ts)


def bench_one(key, frame, warmup, iters):
    import cv2
    import torch

    import make_model_compare_gif as m

    sync = torch.cuda.synchronize if m.device.type == "cuda" else (lambda: None)
    img0 = cv2.imread(frame) if frame else np.random.randint(0, 255, (480, 640, 3), np.uint8)
    name, loader = m.MODELS[key]
    infer = loader()
    x = infer.prep(img0)

    def fwd():
        with torch.no_grad():
            infer.model(x)

    first = timed(fwd, sync, 1)[0]
    timed(fwd, sync, warmup)
    f = timed(fwd, sync, iters)
    timed(lambda: infer(img0), sync, max(3, warmup // 4))
    e = timed(lambda: infer(img0), sync, iters)
    return {
        "key": key, "name": name, "input": "x".join(map(str, reversed(x.shape[2:]))),
        "first_ms": first,
        "fwd_mean_ms": f.mean(), "fwd_p50_ms": np.median(f), "fwd_p90_ms": np.percentile(f, 90),
        "fwd_fps": 1000 / f.mean(),
        "e2e_mean_ms": e.mean(), "e2e_fps": 1000 / e.mean(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", help="벤치마크에 쓸 실제 프레임(없으면 랜덤 640x480)")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--models", nargs="*", help="일부만 (기본: 전부, README 표 순서)")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "outputs", "benchmarks"))
    ap.add_argument("--worker", help="(내부용) 이 모델 하나만 측정하고 JSON 한 줄 출력")
    args = ap.parse_args()

    if args.worker:
        r = bench_one(args.worker, args.frame, args.warmup, args.iters)
        print("RESULT " + json.dumps({k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in r.items()}))
        return

    import make_model_compare_gif as m

    keys = args.models or list(m.MODELS)
    dev = m.torch.cuda.get_device_name(0) if m.device.type == "cuda" else f"CPU ({platform.processor()})"
    print(f"device: {dev} | torch {m.torch.__version__} | warmup {args.warmup}, iters {args.iters}", flush=True)

    rows = []
    for key in keys:
        cmd = [sys.executable, "-u", os.path.abspath(__file__), "--worker", key,
               "--warmup", str(args.warmup), "--iters", str(args.iters)] + (["--frame", args.frame] if args.frame else [])
        t = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
        if p.returncode != 0 or line is None:
            print(f"{key}: 실패(exit {p.returncode})\n{(p.stdout + p.stderr)[-1500:]}", flush=True)
            continue
        r = json.loads(line[len("RESULT "):])
        rows.append(r)
        print(f"{r['name']:<28} fwd {r['fwd_mean_ms']:7.2f} ms ({r['fwd_fps']:6.1f} fps) | "
              f"e2e {r['e2e_mean_ms']:7.2f} ms ({r['e2e_fps']:6.1f} fps) | first {r['first_ms']:8.0f} ms "
              f"[{time.time() - t:.0f}s]", flush=True)

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.out_dir, exist_ok=True)
    md = [f"# 모델 추론 벤치마크 ({stamp})", "",
          f"- device: {dev}", f"- torch: {m.torch.__version__}, host: {platform.node()}",
          f"- frame: {args.frame or 'random 640x480'}, batch=1, fp32, warmup {args.warmup}, iters {args.iters}", "",
          "| 모델 | 입력 | forward ms (p50 / p90) | forward FPS | end2end ms | end2end FPS | 첫 호출 ms |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['name']} | {r['input']} | {r['fwd_mean_ms']:.2f} ({r['fwd_p50_ms']:.2f} / {r['fwd_p90_ms']:.2f}) | "
                  f"{r['fwd_fps']:.1f} | {r['e2e_mean_ms']:.2f} | {r['e2e_fps']:.1f} | {r['first_ms']:.0f} |")
    path = os.path.join(args.out_dir, f"benchmark_{platform.node()}_{stamp}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n" + "\n".join(md) + f"\n\n저장: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
