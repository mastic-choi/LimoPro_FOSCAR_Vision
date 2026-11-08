#!/usr/bin/env python3
"""동아리 RTX PC용 학습 진입점 (run_train.sh / run_train.bat이 호출).

1. GPU 확인: CUDA를 못 쓰면 바로 중단 (CPU로 며칠 도는 것 방지)
2. 모델마다 실제 batch로 forward/backward 한 번 -> 메모리가 모자라면 batch를 절반으로 (8 -> 4 -> 2)
3. scripts/train_all.py로 순서대로 학습. 다시 실행하면 끝난 모델은 건너뛰고 last.pt에서 이어서
4. 끝나면 가져갈 결과만 results_<PC이름>.zip으로 묶음 (best_da.pt, best_ll.pt, log.csv, test.json, train.log)

3차부터: 대회장 조명 증강(--venue-ref-dir, 번들에 견본 폴더가 있으면 자동으로 켬), 대회장 test(test_venue) 채점,
이전 라운드 가중치에서 시작(--init v2가 기본, v1 | none, 번들의 weights/ 폴더). bundle.json의 gpu_duty가 GPU_DUTY 기본값.
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
KEEP = ("best_da.pt", "best_ll.pt", "log.csv", "test.json", "train.log")


def preflight(models, batch, allow_cpu):
    """학습과 GPU 메모리가 겹치지 않게 별도 프로세스로 호출됨. 모델별로 쓸 수 있는 batch를 출력."""
    sys.path.insert(0, HERE)
    import torch
    import train_seg as T
    import make_model_compare_gif as m
    print(f"torch {torch.__version__}, cuda {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}, "
              f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB", flush=True)
    elif not allow_cpu:
        sys.exit("CUDA를 쓸 수 없습니다. nvidia-smi로 드라이버를 확인하세요.")
    res = {}
    for name in models:
        model, _, _, fwd, _ = T.ADAPTERS[name](m)
        model = model.to(m.device).train()
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-9)  # 옵티마이저 상태 메모리까지 포함
        n = batch
        while True:
            try:
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                x = torch.zeros(n, 3, 384, 640, device=m.device)
                t = torch.zeros(n, 360, 640, device=m.device)
                pda, pll = fwd(model, x)
                loss = T.seg_loss(T.crop(pda), t) + T.seg_loss(T.crop(pll), t)
                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
                del x, t, pda, pll, loss
                if torch.cuda.is_available():
                    used = torch.cuda.memory_reserved() / torch.cuda.get_device_properties(0).total_memory
                    if used > 0.9:  # 실제 학습은 다른 프로그램과 메모리를 나눠 쓰므로 여유를 둠
                        raise RuntimeError(f"out of memory margin: 예약 {used:.0%}")
                break
            except RuntimeError as e:  # TorchScript(YOLOPv2)는 OOM이 torch.jit.Error로 올라옴 -> 메시지로 판별
                if "out of memory" not in str(e).lower():
                    raise
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                if n <= 2:
                    sys.exit(f"{name}: batch 2도 GPU 메모리가 모자랍니다.")
                n //= 2
                print(f"{name}: 메모리 부족 -> batch {n}", flush=True)
        peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        print(f"{name}: batch {n} 통과 (GPU 최대 {peak:.1f}GB)", flush=True)
        res[name] = n
        del model, opt
        torch.cuda.empty_cache()
    print("PREFLIGHT " + json.dumps(res), flush=True)


def keep_awake():
    if platform.system() == "Windows":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["hybridnets", "tln_large", "yolop", "yolopv2"],
                    help="이 순서로 학습. 시간이 모자라면 앞쪽 모델만 끝나도 됨")
    ap.add_argument("--data", default=os.path.join(ROOT, "dataset", "ego_v3"))
    ap.add_argument("--suffix", default="_v3")
    ref = os.path.join(ROOT, "dataset", "venue_ref_20261106")
    ap.add_argument("--venue-ref-dir", default=ref if os.path.isdir(ref) else None, help="대회장 조명 증강 견본 폴더 (none이면 끔)")
    ap.add_argument("--venue-p", type=float, default=0.7, help="대회장 조명 증강 적용 비율")
    ap.add_argument("--init", choices=["none", "v1", "v2"], default="v2", help="시작 가중치: v2 = 2차 best_da에서 이어서 (기본), v1 = 1차, none = BDD100K 사전학습")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--allow-cpu", action="store_true", help="동작 시험용")
    ap.add_argument("--preflight-only", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.preflight_only:
        return preflight(args.models, args.batch, args.allow_cpu)

    assert os.path.isdir(os.path.join(args.data, "train", "images")), f"데이터셋이 없습니다: {args.data}"
    bj = os.path.join(ROOT, "bundle.json")
    if os.path.exists(bj) and "GPU_DUTY" not in os.environ:
        with open(bj, encoding="utf-8") as f:
            os.environ["GPU_DUTY"] = str(json.load(f).get("gpu_duty", 0.9))
    print(f"GPU_DUTY {os.environ.get('GPU_DUTY', '0.9')}, 데이터 {args.data}, 대회장 조명 증강 {args.venue_ref_dir}, 시작 가중치 {args.init}", flush=True)
    keep_awake()
    env = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1")
    cmd = [sys.executable, os.path.abspath(__file__), "--preflight-only", "--batch", str(args.batch), "--models", *args.models]
    p = subprocess.run(cmd + (["--allow-cpu"] if args.allow_cpu else []), cwd=ROOT, stdout=subprocess.PIPE, text=True,
                       encoding="utf-8", errors="replace", env=env)
    print(p.stdout, flush=True)
    if p.returncode:
        print(f"사전 점검 실패 (종료 코드 {p.returncode}). GPU 드라이버와 torch 설치를 확인하세요.", flush=True)
        sys.exit(1)
    batches = json.loads(next(l for l in p.stdout.splitlines() if l.startswith("PREFLIGHT"))[10:])

    out = os.path.join(ROOT, "outputs", "train")
    t0 = time.time()
    for name in args.models:
        extra = []
        if args.venue_ref_dir and args.venue_ref_dir != "none":
            extra += ["--venue-ref-dir", args.venue_ref_dir, "--venue-p", str(args.venue_p)]
        if args.init != "none":
            w = os.path.join(ROOT, "weights", f"{'yolop_chbce' if name == 'yolop' and args.init == 'v1' else name}_{args.init}_best_da.pt")
            assert os.path.exists(w), f"시작 가중치가 없습니다: {w}"
            extra += ["--init", w]
        rc = subprocess.call([sys.executable, os.path.join(HERE, "train_all.py"), "--data", args.data, "--models", name,
                              "--suffix", args.suffix, "--out", out, "--seeds", *map(str, args.seeds), "--",
                              "--batch", str(batches[name]), "--epochs", str(args.epochs), "--workers", str(args.workers),
                              *(["--loss", "channel"] if name == "yolop" else []), *extra],  # yolop: 1차에서 채널별 BCE가 더 좋았음
                             cwd=ROOT, env=env)
        if rc:
            print(f"{name}: train_all.py 비정상 종료 (종료 코드 {rc})", flush=True)
            sys.exit(1)

    zip_path = os.path.join(ROOT, f"results_{platform.node() or 'rtx'}.zip")
    summary, done = [], 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as z:
        for name in args.models:
            for seed in args.seeds:
                d = f"{name}{args.suffix}_seed{seed}"
                for k in KEEP:
                    f = os.path.join(out, d, k)
                    if os.path.exists(f):
                        z.write(f, os.path.join(d, k))
                tj = os.path.join(out, d, "test.json")
                if os.path.exists(tj):
                    with open(tj, encoding="utf-8") as f:
                        j = json.load(f)
                    t = j["test"]["all"]; v = j.get("test_venue", {}).get("all")
                    summary.append(f"{d}: test da {t['da_iou']:.4f}, ll {t['ll_iou']:.4f}"
                                   + (f" | 대회장 test da {v['da_iou']:.4f}, ll {v['ll_iou']:.4f}" if v else ""))
                    done += 1
                else:
                    summary.append(f"{d}: 실패 ({os.path.join(out, d, 'train.log')} 확인)")
    print(f"\n{time.time() - t0:.0f}초 걸림\n" + "\n".join(summary) + f"\n결과: {zip_path}", flush=True)
    if done < len(args.models) * len(args.seeds):
        sys.exit("일부 모델 학습 실패. 같은 명령을 다시 실행하면 이어서 합니다.")


if __name__ == "__main__":
    main()
