#!/usr/bin/env python3
"""synth_klein_refine.py 진행 상태를 일정 간격으로 한 줄씩 출력한다 (Claude Monitor로 받아 대응용).

각 줄: 시간, done/total, kept/reverted/failed, 장당 초, 남은 분, 상태 태그
상태 태그
  OK        정상 진행
  STALL     간격 동안 done이 늘지 않음
  FAILRATE  최근 간격의 failed 비율이 30% 초과
  DEAD      진행 파일의 pid 프로세스가 없음(끝나지 않았는데)
  FINISHED  done == total 이고 current 없음 → 출력 후 종료
사용: python scripts/progress_report.py <data 폴더> [--interval 600]
"""
import argparse, json, os, subprocess, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("data")
ap.add_argument("--interval", type=int, default=600)
a = ap.parse_args()
pj = os.path.join(a.data, "klein_progress.json")


def alive(pid):
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True)
    return str(pid) in r.stdout


prev = None
while True:
    try:
        p = json.load(open(pj))
    except Exception as e:  # 쓰는 중이면 잠깐 뒤 다시
        time.sleep(5)
        try:
            p = json.load(open(pj))
        except Exception:
            print(f"{time.strftime('%H:%M:%S')} READERR {e}", flush=True)
            time.sleep(a.interval)
            continue
    tags = []
    finished = p["done"] >= p["total"] and not p.get("current")
    if finished:
        tags.append("FINISHED")
    else:
        if prev is not None and p["done"] <= prev["done"]:
            tags.append("STALL")
        if prev is not None:
            dd = p["done"] - prev["done"]
            df = p["failed"] - prev["failed"]
            if dd > 0 and df / dd > 0.3:
                tags.append("FAILRATE")
        if not alive(p["pid"]):
            tags.append("DEAD")
    print(f"{time.strftime('%H:%M:%S')} {p['done']}/{p['total']} kept={p['kept']} reverted={p['reverted']} "
          f"failed={p['failed']} {p['sec_per_img']}s/img eta={p['eta_min']}min "
          f"{' '.join(tags) or 'OK'}", flush=True)
    if finished or "DEAD" in tags:
        sys.exit(0)
    prev = p
    time.sleep(a.interval)
