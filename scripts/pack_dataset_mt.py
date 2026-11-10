#!/usr/bin/env python3
"""Klein 이 끝난 학습 데이터를 다른 사람에게 보낼 수 있게 따로 복사하고 zip 으로 묶는다.

사용: python3 -I pack_dataset.py <LimoPro_mt 폴더> <출력 폴더>
"""
import csv, glob, json, os, shutil, sys, zipfile, collections

src_root, out = sys.argv[1], sys.argv[2]
src = os.path.join(src_root, "dataset")
name = os.path.basename(out.rstrip("/"))
assert os.path.exists(os.path.join(src, "synth_signs_v2/train/.klein_done")), "Klein 이 아직 안 끝남"
assert not os.path.exists(out), f"이미 있음: {out}"

SKIP_TRAIN = {"images_orig", "klein_reverted", "klein_refine.log"}


def ignore(d, names):
    rel = os.path.relpath(d, src)
    drop = {n for n in names if n.startswith("_klein_tmp") or n.startswith("klein_progress") or n == ".klein_done"}
    if rel == os.path.join("synth_signs_v2", "train"):
        drop |= SKIP_TRAIN & set(names)
    return drop


for d in ("synth_signs_v2", "cones_v1", "venue_ref_20261106"):
    shutil.copytree(os.path.join(src, d), os.path.join(out, d), ignore=ignore)

# Klein 통계
kc = collections.Counter()
for f in glob.glob(os.path.join(out, "synth_signs_v2/train/klein_refine*.csv")):
    for r in csv.DictReader(open(f, encoding="utf-8")):
        kc[r["status"]] += 1

sc = json.load(open(os.path.join(out, "synth_signs_v2/counts.json"), encoding="utf-8"))
cc = json.load(open(os.path.join(out, "cones_v1/counts.json"), encoding="utf-8"))
classes = open(os.path.join(out, "synth_signs_v2/classes.txt"), encoding="utf-8").read().split()

def n_img(split):
    return len(os.listdir(os.path.join(out, "synth_signs_v2", split, "images")))

seg_rows = "\n".join(
    f"| `{s}` | {n_img(s):,} | " + ("O" if os.path.isdir(os.path.join(out, 'synth_signs_v2', s, 'sl_masks')) else "") + " |"
    for s in ("train", "val", "test", "test_venue", "test_bag"))
cone_rows = "\n".join(f"| `{k}` | {v['images']:,} | {v['boxes']:,} |" for k, v in sorted(cc.items()))
cls_rows = "\n".join(f"| {i} | `{c}` |" for i, c in enumerate(classes))

readme = f"""# {name}

LIMO Pro 다중 작업 인식 모델(HybridNets)용 학습 데이터입니다.
차선 분할(주행 영역·차선·흰 가로선)과 객체 검출(표지판·신호등·라바콘 등 12종)을 같이 학습합니다.

## 폴더 구성

| 폴더 | 내용 |
|---|---|
| `synth_signs_v2/` | 차선 학습 사진(ego_v4)에 표지판·신호등을 합성한 데이터. 분할 마스크와 검출 박스가 같이 있음 |
| `cones_v1/` | 라바콘 검출 데이터 (TraCon + FSOCO). 클래스는 `cone` 하나 |
| `venue_ref_20261106/` | 대회장 조명 증강용 색 견본 사진 (11/06 대회장) |

## synth_signs_v2

| split | 사진 | 흰 가로선 라벨 |
|---|---|---|
{seg_rows}

split 폴더 안:

| 폴더 | 내용 |
|---|---|
| `images/` | 입력 사진 |
| `labels/` | 검출 박스 (YOLO 형식: `클래스 cx cy w h`, 0~1 정규화) |
| `da_masks/` | 주행 영역 마스크 |
| `ll_masks/` | 차선 마스크 |
| `sl_masks/`, `sl_valid/` | 흰 가로선(정지선·미션 시작선) 마스크와 라벨을 믿을 픽셀 (train, test_venue만) |

- `train/images/`의 합성 표지판 사진 중 일부는 FLUX 3 Klein 이미지 편집 모델로 실물처럼 다듬은 것입니다.
  다듬은 결과를 쓴 사진 {kc.get('kept', 0):,}장, 색이 너무 바뀌어 원본으로 둔 사진 {kc.get('reverted', 0):,}장.
  사진별 기록은 `train/klein_refine*.csv`에 있습니다.
- `meta.csv`: 사진마다 합성한 물체 정보 (클래스, 거리, 위치, 가림 정도)
- `counts.json`: split·클래스별 개수
- `stopline_labels.csv`: 흰 가로선 라벨 상태

## cones_v1

| 출처/split | 사진 | 박스 |
|---|---|---|
{cone_rows}

- 같은 `images/`, `labels/` 구조(YOLO 형식)이고 클래스 번호는 아래 표와 같이 `cone` = 11입니다.
- FSOCO는 사진 테두리의 워터마크(140px)를 잘라 내고 박스를 옮겼습니다. 콘 5종(파랑·노랑·주황·큰 주황·모름)을 `cone` 하나로 합쳤습니다.
- 출처: TraCon, FSOCO(https://fsoco.github.io/fsoco-dataset/, Kaggle 사본 `persello/fsoco-boxes`). FSOCO 원본은 라이선스 표기가 없고, 팀 내부 학습용으로만 씁니다.

## 검출 클래스 (12)

| 번호 | 클래스 |
|---|---|
{cls_rows}

`traffic_light`는 신호등 몸체, `light_red`·`light_yellow`·`light_green`은 켜진 불입니다.
"""
open(os.path.join(out, "README.md"), "w", encoding="utf-8").write(readme)

# zip (사진은 이미 압축돼 있어서 STORED)
zp = out.rstrip("/") + ".zip"
base = os.path.dirname(out.rstrip("/"))
n = 0
with zipfile.ZipFile(zp + ".part", "w", zipfile.ZIP_STORED, allowZip64=True) as z:
    for dp, dns, fns in os.walk(out):
        dns.sort()
        for fn in sorted(fns):
            p = os.path.join(dp, fn)
            z.write(p, os.path.relpath(p, base))
            n += 1
os.replace(zp + ".part", zp)
assert zipfile.ZipFile(zp).testzip() is None
print(f"files {n}, zip {os.path.getsize(zp)/1e9:.2f} GB -> {zp}")
print(f"klein {dict(kc)}")
