#!/usr/bin/env python3
"""H100 multitask v4 학습용 데이터셋 폴더 LimoPro_dataset_mt_v2/ 를 만든다 (하드링크라 디스크를 거의 안 씀).

  synth_signs_v2/    LimoPro_dataset_mt_v1 의 합성 데이터 + 정지선 검수 반영(sl_labels/synth: 새로 쓴 sl_masks·sl_valid,
                     remove_existing.txt 의 기존 라벨 지움). 바꾸는 파일은 링크를 끊고 새로 써서 원본은 그대로.
  bag_mt_v1/         USB 실차 bag pseudo-label, 사람 검수 (train, test_bag1116). excluded·review 는 안 넣음
  real_tl_v1/        실물 신호등 사진 (박스만)
  real_fork_v1/      실물 Ø80 mm A·B 표지판 M7 주행 (박스만, 빈 프레임 포함)
  venue_ref_20261106/ 대회장 조명 견본 (증강용)
사용: python build_dataset_v2.py <v3real 묶음 dataset 폴더> <LimoPro_dataset_mt_v2 작업 폴더> <출력 폴더>
"""
import os, shutil, sys

src_v1, work, out = [os.path.abspath(x) for x in sys.argv[1:4]]
if os.path.exists(out):
    sys.exit(f"이미 있음: {out}")


def link_tree(a, b, skip=()):
    for root, dirs, files in os.walk(a):
        rel = os.path.relpath(root, a)
        if any(rel == s or rel.startswith(s + os.sep) for s in skip):
            continue
        os.makedirs(os.path.join(b, rel), exist_ok=True)
        for f in files:
            os.link(os.path.join(root, f), os.path.join(b, rel, f))


def replace(src, dst):   # 하드링크를 끊고 새 파일로 (원본 보호)
    if os.path.exists(dst):
        os.remove(dst)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


link_tree(os.path.join(src_v1, "synth_signs_v2"), os.path.join(out, "synth_signs_v2"))
n_new = n_rm = 0
sl = os.path.join(work, "sl_labels", "synth")
for sp in sorted(os.listdir(sl)):
    for sub in ("sl_masks", "sl_valid"):
        d = os.path.join(sl, sp, sub)
        if os.path.isdir(d):
            for f in os.listdir(d):
                replace(os.path.join(d, f), os.path.join(out, "synth_signs_v2", sp, sub, f)); n_new += sub == "sl_masks"
for line in open(os.path.join(sl, "remove_existing.txt")):
    if line.strip():
        sp, stem = line.strip().split("/")
        for sub in ("sl_masks", "sl_valid"):
            p = os.path.join(out, "synth_signs_v2", sp, sub, stem + ".png")
            if os.path.exists(p):
                os.remove(p); n_rm += sub == "sl_masks"
link_tree(os.path.join(work, "bag_mt_v1"), os.path.join(out, "bag_mt_v1"), skip=("excluded", "review"))
for f in ("pseudo_meta.jsonl",):
    p = os.path.join(out, "bag_mt_v1", f)
    if os.path.exists(p): os.remove(p)
link_tree(os.path.join(src_v1, "real_tl_v1"), os.path.join(out, "real_tl_v1"))
link_tree(os.path.join(work, "real_fork_v1"), os.path.join(out, "real_fork_v1"))
link_tree(os.path.join(src_v1, "venue_ref_20261106"), os.path.join(out, "venue_ref_20261106"))


def cnt(d, sub="images"):
    p = os.path.join(out, d)
    return {sp: len(os.listdir(os.path.join(p, sp, sub))) for sp in sorted(os.listdir(p)) if os.path.isdir(os.path.join(p, sp, sub))}


sl_cnt = {sp: len(os.listdir(os.path.join(out, "synth_signs_v2", sp, "sl_masks"))) for sp in sorted(os.listdir(os.path.join(out, "synth_signs_v2")))
          if os.path.isdir(os.path.join(out, "synth_signs_v2", sp, "sl_masks"))}
print("synth 이미지", cnt("synth_signs_v2"), "| 정지선 라벨", sl_cnt, f"(새로 {n_new}, 기존 지움 {n_rm})")
print("bag", cnt("bag_mt_v1"), "| real_tl", cnt("real_tl_v1", ".") if False else os.listdir(os.path.join(out, "real_tl_v1", "images")),
      "| real_fork", {sp: len(os.listdir(os.path.join(out, "real_fork_v1", "images", sp))) for sp in os.listdir(os.path.join(out, "real_fork_v1", "images"))})
