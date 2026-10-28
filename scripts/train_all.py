#!/usr/bin/env python3
"""앙상블 멤버를 순서대로 train_seg.py로 학습. 끊겨도 같은 명령을 다시 실행하면 이어서 한다.

- test.json이 있는 모델(끝까지 학습 + test 평가 완료)은 건너뜀
- 나머지는 train_seg.py가 last.pt에서 자동 재개 (매 epoch 저장)
- 프로세스가 비정상 종료하면 --retries 번까지 자동 재시도 (KMU 완료 검증 래퍼와 같은 역할)
- GPU 사용률은 GPU_DUTY 환경변수 그대로 (기본 0.9)
- 학습 로그는 <out>/<model>_seed<seed>/train.log 에 누적

사용 예 (PC가 꺼졌다 켜진 뒤에도 같은 명령):
  .venv/Scripts/python scripts/train_all.py --data dataset/ego_v1
  .venv/Scripts/python scripts/train_all.py --data dataset/ego_v1 --models tln_large yolop --seeds 0 1 2
  .venv/Scripts/python scripts/train_all.py --data dataset/ego_v1 --models yolop --suffix _chbce -- --loss channel
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ENSEMBLE = ["yolopv2", "tln_large", "yolop", "hybridnets"]  # make_model_compare_gif.ENSEMBLE (torch import 없이 쓰려고 복사)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--models", nargs="*", default=ENSEMBLE)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "train"))
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--suffix", default="", help="결과 폴더 이름 <model><suffix>_seed<seed> (같은 모델 다른 설정 비교용)")
    ap.add_argument("extra", nargs=argparse.REMAINDER, help="-- 뒤 인자는 train_seg.py로 그대로 (예: -- --epochs 30)")
    args = ap.parse_args()
    extra = [a for a in args.extra if a != "--"]

    for seed in args.seeds:
        for model in args.models:
            out = os.path.join(args.out, f"{model}{args.suffix}_seed{seed}")
            if os.path.exists(os.path.join(out, "test.json")):
                print(f"[skip] {model} seed {seed}: 완료됨", flush=True)
                continue
            os.makedirs(out, exist_ok=True)
            cmd = [sys.executable, os.path.join(HERE, "train_seg.py"), "--model", model, "--data", args.data,
                   "--seed", str(seed), "--out", out] + extra
            for attempt in range(1, args.retries + 1):
                print(f"[run] {model} seed {seed} (시도 {attempt}/{args.retries}) {time.strftime('%H:%M:%S')}", flush=True)
                with open(os.path.join(out, "train.log"), "a", encoding="utf-8") as log:
                    rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
                if rc == 0 and os.path.exists(os.path.join(out, "test.json")):
                    print(f"[done] {model} seed {seed}", flush=True)
                    break
                print(f"[fail] {model} seed {seed}: exit {rc}, train.log 확인", flush=True)
            else:
                print(f"[give up] {model} seed {seed}: {args.retries}번 실패, 다음 모델로", flush=True)


if __name__ == "__main__":
    main()
