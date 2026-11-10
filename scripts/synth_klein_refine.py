#!/usr/bin/env python3
"""synth_signs.py 결과 일부를 FLUX 3 Klein(편집 모델)로 다듬어 실물 느낌을 더한다.

- 대상: 박스 높이가 충분한 이미지 중 풀(venue/general)과 클래스를 고르게 N장
- 가장 큰 물체 중심으로 정사각 크롭(최대 512, 16의 배수) → Klein에 참조 이미지로 넣고
  "모양·위치는 그대로, 실물처럼만" 지시 → 물체 주변(박스 확장)만 부드럽게 되돌려 붙임
- 'slow'(서행, 글자)는 판면 안쪽을 원본 그대로 보호
- 검사: 라벨 박스 안 색이 너무 바뀌면(평균 절대차 > --max-diff) 원본 유지(reverted)
- GPU 쉬는 시간: 한 장 처리 시간 x --rest 만큼 쉬어서 평균 사용률을 낮춘다(기본 0.25 → 평균 약 80%)
- 원본은 <data>/images_orig/로 옮겨 두고, 결과는 <data>/images/에 같은 이름으로 덮어쓴다
- 이어하기 지원: <data>/klein_refine*.csv에 있는 이미지는 건너뜀
- 진행 상태: <data>/klein_progress.json (done/kept/reverted/failed, 속도, 남은 시간)
- 병렬(H100 등): --nshard K --shard i 로 프로세스 K개를 띄우면 picked를 나눠 맡음(기록 파일은 shard마다 따로)
- sd-cli/모델 위치: 환경변수 KLEIN_SD_CLI, KLEIN_MODEL_DIR (기본은 이 PC의 C:/Users/Admin/QwenImage)
  KLEIN_MODEL_DIR 안: diffusion_models/FLUX-3-Image-Klein-AD-Q5_K.gguf, text_encoders/Qwen3-VL-4B-Instruct-Q8_0.gguf,
  vae/flux3_ae.safetensors. KLEIN_OFFLOAD=0 이면 --offload-to-cpu를 끔(VRAM 넉넉할 때 빠름)

사용 예:
  .venv/Scripts/python scripts/synth_klein_refine.py --data dataset/synth_signs_v1 --n 1000
"""
import argparse, csv, json, os, random, shutil, subprocess, time, zlib
import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QW = r"C:\Users\Admin\QwenImage"
SD = os.environ.get("KLEIN_SD_CLI", os.path.join(QW, "bin", "sd-cli.exe"))
MD = os.environ.get("KLEIN_MODEL_DIR", os.path.join(QW, "models"))
MODEL = dict(diffusion=os.path.join(MD, "diffusion_models", "FLUX-3-Image-Klein-AD-Q5_K.gguf"),
             llm=os.path.join(MD, "text_encoders", "Qwen3-VL-4B-Instruct-Q8_0.gguf"),
             vae=os.path.join(MD, "vae", "flux3_ae.safetensors"))
OFFLOAD = ["--offload-to-cpu"] if os.environ.get("KLEIN_OFFLOAD", "1") != "0" else []
PROMPT = ("Make the small sign (or traffic light) and its thin pole look like a real physical object photographed "
          "by this same low-quality camera: matching indoor lighting, color cast, soft contact shadow on the floor, "
          "slight blur and noise. Keep the sign's design, symbols, letters, colors, size and position exactly the same. "
          "Do not change anything else in the photo.")
MAIN = {"pedestrian", "sign_A", "sign_B", "bump", "slow", "parking", "traffic_light"}


def read_labels(path, classes, W, H):
    out = []
    for line in open(path):
        c, cx, cy, w, h = line.split()
        cx, cy, w, h = float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
        out.append((classes[int(c)], cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return out


def img_path(base, sub, name):
    for ext in (".jpg", ".png"):
        p = os.path.join(base, sub, name + ext)
        if os.path.exists(p):
            return p
    return os.path.join(base, sub, name + ".jpg")


def pick_images(data, base, split, classes, n, min_h, seed):
    rows = list(csv.DictReader(open(os.path.join(data, "meta.csv"), encoding="utf-8")))
    if split:
        rows = [r for r in rows if r.get("split") == split]
    buckets, size = {}, None
    for r in rows:
        if r["objects"] == "[]":
            continue
        if "width" in r:
            W, H = int(r["width"]), int(r["height"])
        else:   # v2(ego_v4 기반)는 크기가 모두 같다 → 한 번만 읽음
            if size is None:
                size = cv2.imread(img_path(base, "images", r["image"])).shape[:2]
            H, W = size
        labs = read_labels(os.path.join(base, "labels", r["image"] + ".txt"), classes, W, H)
        mains = [l for l in labs if l[0] in MAIN and (l[4] - l[2]) >= min_h * H / 720]
        if not mains:
            continue
        big = max(mains, key=lambda l: (l[3] - l[1]) * (l[4] - l[2]))
        key = big[0]
        if key == "traffic_light":   # 신호등은 켜진 불 색으로 나눈다
            lamps = [l[0] for l in labs if l[0].startswith("light_")]
            key = lamps[0] if lamps else key
        buckets.setdefault((r["pool"], key), []).append(r["image"])
    rnd = random.Random(seed)
    for v in buckets.values():
        rnd.shuffle(v)
    keys = sorted(buckets)
    picked = []
    while len(picked) < n and any(buckets[k] for k in keys):
        for k in keys:
            if buckets[k] and len(picked) < n:
                picked.append(buckets[k].pop())
    return picked


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="dataset/synth_signs_v1")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--min-h", type=float, default=40, help="대상 물체 최소 박스 높이(px, 720 기준)")
    ap.add_argument("--rest", type=float, default=0.25, help="처리 시간 대비 쉬는 비율(GPU 평균 사용률 제한)")
    ap.add_argument("--max-diff", type=float, default=40, help="라벨 박스 안 평균 색 변화 허용치")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="", help="v2처럼 split 폴더가 있으면 그 이름(예: train)")
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    a = ap.parse_args()
    if a.n <= 0:
        a.n = 10 ** 9   # 0 = 조건에 맞는 이미지 전부
    os.chdir(ROOT)
    data = a.data
    base = os.path.join(data, a.split) if a.split else data   # images/labels/결과 기록 위치
    classes = open(os.path.join(data, "classes.txt")).read().split()
    os.makedirs(os.path.join(base, "images_orig"), exist_ok=True)
    sfx = f"_s{a.shard}" if a.nshard > 1 else ""
    tmp = os.path.abspath(os.path.join(base, "_klein_tmp" + sfx))
    os.makedirs(tmp, exist_ok=True)
    logp = os.path.join(base, f"klein_refine{sfx}.csv")
    done = set()
    for f in os.listdir(base):
        if f.startswith("klein_refine") and f.endswith(".csv"):
            done |= {r["image"] for r in csv.DictReader(open(os.path.join(base, f), encoding="utf-8"))}
    picked = pick_images(data, base, a.split, classes, a.n, a.min_h, a.seed)
    picked = [p for k, p in enumerate(picked) if k % a.nshard == a.shard]
    todo = [p for p in picked if p not in done]
    print(f"picked {len(picked)}, already done {len(picked) - len(todo)}, todo {len(todo)}", flush=True)

    new = not os.path.exists(logp)
    fh = open(logp, "a", newline="", encoding="utf-8")
    w = csv.writer(fh)
    if new:
        w.writerow(["image", "status", "crop", "max_box_diff", "sec"])
    stat = {"kept": 0, "reverted": 0, "failed": 0}
    if not new:
        for r in csv.DictReader(open(logp, encoding="utf-8")):
            stat[r["status"]] = stat.get(r["status"], 0) + 1
    t_start, times = time.time(), []

    def progress(cur):
        el = time.time() - t_start
        rate = len(times) / el if el > 0 else 0
        json.dump({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "total": len(picked),
                   "done": len(picked) - len(todo) + len(times), **stat,
                   "sec_per_img": round(el / max(1, len(times)), 1),
                   "eta_min": round((len(todo) - len(times)) / rate / 60, 1) if rate else None,
                   "current": cur, "pid": os.getpid()},
                  open(os.path.join(base, f"klein_progress{sfx}.json"), "w"), indent=1)

    for name in todo:
        progress(name)
        t0 = time.time()
        ip = img_path(base, "images", name)
        op = os.path.join(base, "images_orig", os.path.basename(ip))
        if not os.path.exists(op):
            shutil.copy2(ip, op)
        img = cv2.imread(op)
        H, W = img.shape[:2]
        labs = read_labels(os.path.join(base, "labels", name + ".txt"), classes, W, H)
        mains = [l for l in labs if l[0] in MAIN]
        big = max(mains, key=lambda l: (l[3] - l[1]) * (l[4] - l[2]))
        cs = min(512, H, W) // 16 * 16
        cx, cy = (big[1] + big[3]) / 2, (big[2] + big[4]) / 2
        X0 = int(np.clip(cx - cs / 2, 0, W - cs))
        Y0 = int(np.clip(cy - cs / 2 + cs * 0.1, 0, H - cs))   # 지주·그림자가 보이게 살짝 아래로
        crop = img[Y0:Y0 + cs, X0:X0 + cs]
        cp, rp = os.path.join(tmp, "in.png"), os.path.join(tmp, "out.png")
        cv2.imwrite(cp, crop)
        if os.path.exists(rp):
            os.remove(rp)
        cmd = [SD, "--diffusion-model", MODEL["diffusion"], "--vae", MODEL["vae"], "--llm", MODEL["llm"],
               "-r", cp, "-p", PROMPT, "--steps", str(a.steps), "--cfg-scale", "1.0", "--sampling-method", "euler",
               "-W", str(cs), "-H", str(cs), "--seed", str(zlib.crc32(name.encode()) % 100000), "-o", rp,
               *OFFLOAD, "--vae-tiling", "--flash-attn", "auto,vae=on"]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=os.path.dirname(SD))
            ok = r.returncode == 0 and os.path.exists(rp)
            if not ok:
                print(name, "sd-cli rc", r.returncode, (r.stderr or r.stdout)[-400:], flush=True)
        except subprocess.TimeoutExpired:
            ok = False
        sec = time.time() - t0
        if not ok:
            stat["failed"] += 1
            w.writerow([name, "failed", f"{X0},{Y0},{cs}", "", round(sec, 1)])
            fh.flush()
            times.append(sec)
            time.sleep(sec * a.rest)
            continue
        q = cv2.imread(rp)
        if q.shape[:2] != (cs, cs):
            q = cv2.resize(q, (cs, cs), interpolation=cv2.INTER_AREA)
        # 붙일 영역: 크롭 안 물체 박스를 넓히고 아래(지주·그림자)로 늘린다
        m = np.zeros((cs, cs), np.float32)
        keep = np.zeros((cs, cs), np.float32)
        for c, x0, y0, x1, y1 in mains:
            bw, bh = x1 - x0, y1 - y0
            ex0, ey0 = int(x0 - X0 - 0.3 * bw), int(y0 - Y0 - 0.3 * bh)
            ex1, ey1 = int(x1 - X0 + 0.3 * bw), int(y1 - Y0 + 1.2 * bh)
            cv2.rectangle(m, (ex0, ey0), (ex1, ey1), 1.0, -1)
            if c == "slow":
                cv2.rectangle(keep, (int(x0 - X0 + 0.12 * bw), int(y0 - Y0 + 0.12 * bh)),
                              (int(x1 - X0 - 0.12 * bw), int(y1 - Y0 - 0.12 * bh)), 1.0, -1)
        llp = os.path.join(base, "ll_masks", os.path.basename(ip)[:-4] + ".png")
        if os.path.exists(llp):   # 차선 선은 원본 유지(분할 마스크와 어긋나지 않게)
            llm = cv2.dilate((cv2.imread(llp, 0) > 0).astype(np.uint8), np.ones((5, 5), np.uint8))
            keep = np.maximum(keep, llm[Y0:Y0 + cs, X0:X0 + cs].astype(np.float32))
        m = cv2.GaussianBlur(m * (1 - keep), (0, 0), 3)[..., None]
        out = img.copy().astype(np.float32)
        reg = out[Y0:Y0 + cs, X0:X0 + cs]
        out[Y0:Y0 + cs, X0:X0 + cs] = q * m + reg * (1 - m)
        out = np.clip(out, 0, 255).astype(np.uint8)
        # 검사: 모든 라벨 박스 안 평균 색 변화
        diffs = []
        for c, x0, y0, x1, y1 in labs:
            xs0, ys0, xs1, ys1 = int(max(0, x0)), int(max(0, y0)), int(min(W, x1)), int(min(H, y1))
            if xs1 - xs0 < 2 or ys1 - ys0 < 2:
                continue
            d = np.abs(out[ys0:ys1, xs0:xs1].astype(np.float32) - img[ys0:ys1, xs0:xs1].astype(np.float32))
            diffs.append(float(np.abs(d.reshape(-1, 3).mean(0)).max()) if c.startswith("light_") else float(d.mean()))
        md = max(diffs) if diffs else 0.0
        status = "kept" if md <= a.max_diff else "reverted"
        if status == "kept":
            cv2.imwrite(ip, out, [cv2.IMWRITE_JPEG_QUALITY, 92])
        else:   # 되돌린 결과는 검토용으로 남긴다
            os.makedirs(os.path.join(base, "klein_reverted"), exist_ok=True)
            cv2.imwrite(os.path.join(base, "klein_reverted", name + ".jpg"), out, [cv2.IMWRITE_JPEG_QUALITY, 92])
        stat[status] += 1
        w.writerow([name, status, f"{X0},{Y0},{cs}", round(md, 1), round(sec, 1)])
        fh.flush()
        times.append(sec)
        time.sleep(sec * a.rest)
    progress(None)
    print("finished", stat, flush=True)


if __name__ == "__main__":
    main()
