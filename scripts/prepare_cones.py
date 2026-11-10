#!/usr/bin/env python3
"""공개 라바콘 데이터셋(TraCon, FSOCO)을 YOLO 형식 + 우리 클래스 번호(cone=11)로 통일한다.

입력 (dataset/cones_raw/, 받은 그대로 두고 읽기만 함)
  tracon/TraCon_dataset/{images,labels}/{train,val,test}/   YOLO, 클래스 0 = cone
      출처 https://github.com/ikatsamenis/Cone-Detection (540장, 콘 947개, 라이선스 표기 없음·논문 인용 요청)
  fsoco/ (fsoco_bounding_boxes_train.zip 압축 해제)        Supervisely JSON, 팀별 폴더
      출처 https://fsoco.github.io/fsoco-dataset/ (11,572장, 라이선스 표기 없음)
      클래스(파랑/노랑/작은 주황/큰 주황/기타 콘)는 전부 cone 하나로 합친다.
      FSOCO 이미지 가장자리의 워터마크 테두리(--fsoco-border px)는 잘라내고 박스도 옮긴다.

출력 dataset/cones_v1/
  images/{train,val,test}/<src>_<name>.jpg   (긴 변 --max-side 이하로 줄임)
  labels/{train,val,test}/<src>_<name>.txt   (11 cx cy w h)
  meta.csv (image, split, source, orig_path, boxes), counts.json
  classes.txt (synth_signs_v1과 같은 순서 + cone)

사용 예:
  .venv/Scripts/python scripts/prepare_cones.py --tracon
  .venv/Scripts/python scripts/prepare_cones.py --fsoco --fsoco-border 140
"""
import argparse, csv, glob, json, os, re, zlib
from multiprocessing import Pool

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSES = ["pedestrian", "sign_A", "sign_B", "bump", "slow", "parking",
           "traffic_light", "light_red", "light_yellow", "light_green", "limo", "cone"]
CONE = CLASSES.index("cone")


def safe(name):
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", name)


def write_one(args):
    src_img, boxes, split, out_name, out, max_side, crop = args
    im = cv2.imread(src_img, cv2.IMREAD_COLOR)
    if im is None:
        return None
    if crop:
        b = crop
        H0, W0 = im.shape[:2]
        im = im[b:H0 - b, b:W0 - b]
        boxes = [(x0 - b, y0 - b, x1 - b, y1 - b) for x0, y0, x1, y1 in boxes]
    H, W = im.shape[:2]
    s = min(1.0, max_side / max(H, W))
    if s < 1.0:
        im = cv2.resize(im, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA)
    lines = []
    for x0, y0, x1, y1 in boxes:   # 원본 픽셀 좌표 → 잘라낸 뒤 범위로 자르고 정규화
        x0, x1 = max(0, min(x0, x1)), min(W, max(x0, x1))
        y0, y1 = max(0, min(y0, y1)), min(H, max(y0, y1))
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        lines.append(f"{CONE} {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}")
    cv2.imwrite(os.path.join(out, "images", split, out_name + ".jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
    with open(os.path.join(out, "labels", split, out_name + ".txt"), "w") as fh:
        fh.write("\n".join(lines) + ("\n" if lines else ""))
    return out_name, split, len(lines)


def tracon_jobs(raw, out, max_side):
    base = os.path.join(raw, "tracon", "TraCon_dataset")
    jobs = []
    for split in ["train", "val", "test"]:
        for ip in sorted(glob.glob(os.path.join(base, "images", split, "*"))):
            stem = os.path.splitext(os.path.basename(ip))[0]
            lp = os.path.join(base, "labels", split, stem + ".txt")
            im = cv2.imread(ip)
            if im is None:
                continue
            H, W = im.shape[:2]
            boxes = []
            if os.path.exists(lp):
                for line in open(lp):
                    p = line.split()
                    if len(p) != 5:
                        continue
                    cx, cy, w, h = (float(v) for v in p[1:])
                    boxes.append(((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H))
            jobs.append((ip, boxes, split, "tracon_" + safe(stem), out, max_side, 0))
    return jobs


def fsoco_jobs(raw, out, max_side, border, val_ratio):
    base = os.path.join(raw, "fsoco")
    jobs, cls_seen = [], {}
    for jp in sorted(glob.glob(os.path.join(base, "**", "ann", "*.json"), recursive=True)):
        team_dir = os.path.dirname(os.path.dirname(jp))
        img_name = os.path.basename(jp)[:-5]
        ip = os.path.join(team_dir, "img", img_name)
        if not os.path.exists(ip):
            continue
        ann = json.load(open(jp, encoding="utf-8"))
        boxes = []
        for ob in ann.get("objects", []):
            cls_seen[ob.get("classTitle")] = cls_seen.get(ob.get("classTitle"), 0) + 1
            if ob.get("geometryType") != "rectangle":
                continue
            (x0, y0), (x1, y1) = ob["points"]["exterior"]
            boxes.append((x0, y0, x1, y1))
        team = os.path.basename(team_dir)
        name = "fsoco_" + safe(team) + "_" + safe(os.path.splitext(img_name)[0])
        # 팀 단위로 train/val을 나눠 같은 팀 사진이 양쪽에 섞이지 않게
        split = "val" if (zlib.crc32(team.encode()) % 1000) / 1000 < val_ratio else "train"
        jobs.append((ip, boxes, split, name, out, max_side, border))
    print("fsoco classTitle counts:", cls_seen)
    return jobs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="dataset/cones_raw")
    ap.add_argument("--out", default="dataset/cones_v1")
    ap.add_argument("--tracon", action="store_true")
    ap.add_argument("--fsoco", action="store_true")
    ap.add_argument("--fsoco-border", type=int, default=140, help="FSOCO 워터마크 테두리 폭(px)")
    ap.add_argument("--fsoco-val", type=float, default=0.1)
    ap.add_argument("--max-side", type=int, default=1280)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    os.chdir(ROOT)
    for sp in ["train", "val", "test"]:
        os.makedirs(os.path.join(a.out, "images", sp), exist_ok=True)
        os.makedirs(os.path.join(a.out, "labels", sp), exist_ok=True)
    with open(os.path.join(a.out, "classes.txt"), "w") as fh:
        fh.write("\n".join(CLASSES) + "\n")

    jobs = []
    if a.tracon:
        jobs += tracon_jobs(a.raw, a.out, a.max_side)
    if a.fsoco:
        jobs += fsoco_jobs(a.raw, a.out, a.max_side, a.fsoco_border, a.fsoco_val)
    print(f"{len(jobs)} images to write")
    res = []
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(write_one, jobs, chunksize=8)):
            if r:
                res.append(r)
            if (i + 1) % 1000 == 0:
                print(f"{i + 1}/{len(jobs)}", flush=True)

    # meta.csv는 소스별로 갱신(다른 소스 행은 유지)
    mp = os.path.join(a.out, "meta.csv")
    srcs = {r[0].split("_")[0] for r in res}
    old = []
    if os.path.exists(mp):
        old = [r for r in csv.DictReader(open(mp, encoding="utf-8")) if r["image"].split("_")[0] not in srcs]
    with open(mp, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "split", "source", "boxes"])
        for r in old:
            w.writerow([r["image"], r["split"], r["source"], r["boxes"]])
        for name, split, n in sorted(res):
            w.writerow([name, split, name.split("_")[0], n])
    cnt = {}
    for r in csv.DictReader(open(mp, encoding="utf-8")):
        k = f"{r['source']}/{r['split']}"
        c = cnt.setdefault(k, {"images": 0, "boxes": 0, "empty": 0})
        c["images"] += 1
        c["boxes"] += int(r["boxes"])
        c["empty"] += int(r["boxes"]) == 0
    json.dump(cnt, open(os.path.join(a.out, "counts.json"), "w"), indent=1)
    for k, v in sorted(cnt.items()):
        print(k, v)


if __name__ == "__main__":
    main()
