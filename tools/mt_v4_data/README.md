# mt_v4_data — hybridnets_mt_v4 학습 데이터(LimoPro_dataset_mt_v2) 만들기 (2026-11-16)

순서와 결과는 [`docs/MULTITASK_V4_TRAINING.md`](../../docs/MULTITASK_V4_TRAINING.md). 실행 환경: rosbags 는 갈림길 분류기 venv, 모델 추론은 v3real 묶음 venv.
`fork_crop_rule.py`·`fork_real_*.py` 는 `fork-sign-crop-cls` 브랜치의 `scripts/` 에 있음.

| 파일 | 하는 일 |
|---|---|
| `bag_frames.py` | rosbag 압축 영상 → 0.5 s 마다, 멈춘 장면 빼고 원본 JPEG + frames.csv |
| `annotate_motion.py` | 프레임마다 /odom 속도, /imu 기울기 |
| `pseudo_label_mt.py` | multitask-v3real 로 da·ll·박스 pseudo-label, 갈림길 A/B 는 분류기 + bag 글자, 멈춤·기울어짐·불확실 프레임 빼기 |
| `review_bag_mt.py` | 사람 검수 웹 앱 (일반: 7863, 정지선: 7864) + 반영 (`apply`) |
| `review_overlay.py` | 확인용 오버레이 격자 |
| `sl_candidates.py` / `sl_apply.py` / `sl_fill_pseudo.py` | 정지선 후보(모든 데이터셋) → 사람 판정 반영 → 검수 안 된 bag 프레임 pseudo |
| `real_fork_det_labels.py` | 실물 A·B 주행 SAM2 박스 → 박스만 데이터 (다른 표지판류 오탐은 배경) |
| `build_dataset_v2.py` | 합친 데이터셋 폴더 (하드링크, 원본 보호) |
