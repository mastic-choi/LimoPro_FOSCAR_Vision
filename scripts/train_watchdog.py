#!/usr/bin/env python3
"""train_all.py 학습 감시. --interval(기본 10분)마다 점검하고 상태를 로그에 남긴다.

정상이면 계속 돌고, 아래 상황이면 이유를 출력하고 종료한다(Claude Code 백그라운드 작업으로 띄우면 종료가 곧 알림):
  exit 0  DONE     모든 모델 test.json 생성 (학습 끝)
  exit 2  DEAD     train_all.py 프로세스가 없음
  exit 3  GIVEUP   train_all.log에 [give up] (재시도 다 실패)
  exit 4  STALL    다음 epoch가 너무 오래 안 나옴 (max(--stall-min, 직전 epoch 시간 x 3), 첫 epoch는 컴파일 때문에 --first-epoch-min)
  exit 5  GPU_IDLE 첫 epoch 이후인데 GPU 사용률 0%가 2번 연속

감시하는 동안 Windows 절전을 막는다(SetThreadExecutionState). 화면 잠금/모니터 꺼짐은 학습과 무관.
Windows/Linux 공통. GPU 사용률은 nvidia-smi가 있으면 그것으로, 없으면 Windows 성능 카운터로 잰다.
--once: --interval 만큼 기다렸다 한 번 점검하고 종료(정상이어도 exit 0으로 보고) -> 주기 보고용.
        절전 방지/GPU_IDLE 연속 판정은 계속 도는 감시 쪽이 맡는다.

사용 예:
  .venv/Scripts/python scripts/train_watchdog.py --models tln_large yolop hybridnets yolopv2
"""
import argparse
import csv
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
TRAIN_SCRIPTS = ("run_train.py", "train_all.py", "train_seg.py")


def ps(cmd):
    r = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True)
    return r.stdout.strip()


def train_procs():
    """[(pid, cmdline)] for run_train.py / train_all.py / train_seg.py"""
    if platform.system() != "Windows":
        out = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True, text=True).stdout
        rows = [l.strip().split(None, 1) for l in out.splitlines() if l.strip()]
        return [(int(r[0]), r[1]) for r in rows if len(r) == 2 and any(k in r[1] for k in TRAIN_SCRIPTS) and "python" in r[1]]
    out = ps("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "ForEach-Object { \"$($_.ProcessId)`t$($_.CommandLine)\" }")
    rows = [l.split("\t", 1) for l in out.splitlines() if "\t" in l]
    return [(int(p), c) for p, c in rows if any(k in c for k in TRAIN_SCRIPTS)]


def gpu_util(pids):
    if not pids:
        return 0.0
    if shutil.which("nvidia-smi"):
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout.split()
        return float(out[0]) if out and out[0].replace(".", "").isdigit() else 0.0
    out = ps("(Get-Counter '\\GPU Engine(*engtype_3D)\\Utilization Percentage',"
             "'\\GPU Engine(*engtype_Compute*)\\Utilization Percentage' -SampleInterval 3 -ErrorAction SilentlyContinue)"
             ".CounterSamples | ForEach-Object { \"$($_.InstanceName)`t$($_.CookedValue)\" }")
    tot = 0.0
    for l in out.splitlines():
        name, _, v = l.partition("\t")
        m = re.search(r"pid_(\d+)_", name)
        if m and int(m.group(1)) in pids:
            tot += float(v or 0)
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["tln_large", "yolop", "hybridnets", "yolopv2"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0])
    ap.add_argument("--suffix", default="", help="train_all.py --suffix 와 같게")
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "train"))
    ap.add_argument("--interval", type=int, default=600)
    ap.add_argument("--stall-min", type=float, default=40, help="epoch 정체 판정 최소 분")
    ap.add_argument("--first-epoch-min", type=float, default=90, help="첫 epoch(ZLUDA 컴파일 포함) 정체 판정 분")
    ap.add_argument("--log", default=os.path.join(ROOT, "outputs", "train_watchdog.log"))
    ap.add_argument("--once", action="store_true", help="interval 뒤 한 번만 점검하고 종료 (주기 보고용)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    runs = [f"{m}{args.suffix}_seed{s}" for s in args.seeds for m in args.models]
    all_log = os.path.join(ROOT, "outputs", "train_all.log")
    giveups_at_start = open(all_log, encoding="utf-8", errors="replace").read().count("[give up]") if os.path.exists(all_log) else 0
    idle_count = 0

    def report(msg):
        line = f"{time.strftime('%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(args.log, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    if args.once:
        time.sleep(args.interval)
    while True:
        if platform.system() == "Windows":
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        done = [r for r in runs if os.path.exists(os.path.join(args.out, r, "test.json"))]
        if len(done) == len(runs):
            for r in runs:
                t = json.load(open(os.path.join(args.out, r, "test.json"), encoding="utf-8"))["test"]
                report(f"  {r}: test da {t['all']['da_iou']} ll {t['all']['ll_iou']} "
                       f"(커브 da {t['curve']['da_iou']} / 직진 da {t['straight']['da_iou']})")
            report("DONE 모든 모델 학습 완료")
            sys.exit(0)

        procs = train_procs()
        if not any("train_all.py" in c or "run_train.py" in c for _, c in procs):  # run_train.py는 모델 사이 잠깐 train_all.py가 없음
            report(f"DEAD 학습 프로세스 없음 (완료 {len(done)}/{len(runs)})")
            sys.exit(2)
        if os.path.exists(all_log) and open(all_log, encoding="utf-8", errors="replace").read().count("[give up]") > giveups_at_start:
            report("GIVEUP train_all.log에 [give up] 발생")
            sys.exit(3)

        seg = [(p, c) for p, c in procs if "train_seg.py" in c]
        m = re.search(r"--out (\S+)", seg[0][1]) if seg else None  # train_all.py가 항상 --out <.../run 이름> 을 넘김
        cur = os.path.basename(m.group(1).strip('"')) if m else next((r for r in runs if r not in done), "?")
        d = os.path.join(args.out, cur)
        epochs = []
        if os.path.exists(os.path.join(d, "log.csv")):
            with open(os.path.join(d, "log.csv"), encoding="utf-8") as f:
                epochs = list(csv.DictReader(f))
        ref = os.path.join(d, "log.csv") if epochs else os.path.join(d, "train.log")
        since = (time.time() - os.path.getmtime(ref)) / 60 if os.path.exists(ref) else 0
        last_sec = float(epochs[-1]["sec"]) if epochs else 0
        limit = max(args.stall_min, 3 * last_sec / 60) if epochs else args.first_epoch_min
        util = gpu_util({p for p, _ in seg})
        last = epochs[-1] if epochs else {}
        report(f"OK {cur} epoch {last.get('epoch', 0)} val da {last.get('val_da_iou', '-')} ll {last.get('val_ll_iou', '-')}"
               f" | 마지막 갱신 {since:.0f}분 전 (한도 {limit:.0f}) | GPU {util:.0f}% | 완료 {len(done)}/{len(runs)}")
        if since > limit:
            report(f"STALL {cur}: {since:.0f}분째 epoch 갱신 없음")
            sys.exit(4)
        idle_count = idle_count + 1 if (epochs and util < 1) else 0
        if idle_count >= 2:
            report(f"GPU_IDLE {cur}: GPU 0%가 2번 연속")
            sys.exit(5)
        if args.once:
            sys.exit(0)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
