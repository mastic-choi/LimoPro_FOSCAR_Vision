# 11-18 대회장 정리: DA 주행 검사, 대회장 자동 라벨, v6 학습, 분류기 v0.3, 학습 속도(`--compile`)

작성 2026-11-18, 정리 2026-11-19.

## 요약
- 릴리즈 전에 **대회장 bag 전체로 DA 주행 검사**(`tools/mt_v4_data/bag_da_gate.py`)를 하도록 했습니다. 시험셋 점수만으로는 대회장에서 DA 가 끝까지 유지되는지 알 수 없기 때문입니다.
- 11-18 대회장 bag 을 **자동 라벨**해서 996 장을 만들었습니다(`tools/mt_v4_data/auto_label_v4.py`).
- `multitask-v6`: v4 에서 이어 학습하고 위 자동 라벨을 더했습니다. 11-18 bag 11 개 모두 DA 주행 검사를 통과해 릴리즈했습니다.
- 갈림길 분류기 `fork-sign-v0.3` (사전 릴리즈): 대회장 A·B 를 학습했고, 원 바깥을 항상 가립니다.

## 1. 갈림길 분류기 fork-sign-v0.3
- 릴리즈: https://github.com/mastic-choi/LimoPro_FOSCAR_Vision/releases/tag/fork-sign-v0.3
- 대회장 표지판은 **양면(앞 A, 뒤 B)** 이라, 크롭마다 사람이 글자를 고르게 했습니다(`tools/mt_v4_data/review_fork_crops.py`, 1 = A, 2 = B, 3 = 버림).
- **원 바깥 항상 가리기** (`train_fork_cls.py --post-mask 1.0`): 대회장 LED 벽이나 반대쪽 면 원판 같은 배경으로 글자를 맞히는 지름길을 막습니다. 가리기는 ONNX 안에 들어 있어서(`export_fork_cls.py`) 로봇 입력은 그대로입니다. OpenCV 4.5.4 dnn 에서 일치를 확인했습니다(최대 차이 4e-7).
- 좌우 뒤집기는 쓰지 않습니다(글자도 뒤집힘).
- 대회장 '없음' 오탐: v0.2 35% → v0.3 4%. 지름 45px(1280 기준) 이상에서 반대 글자 0. 약점은 45px 미만의 작은 B 입니다. 로봇에서는 45px 이상일 때만 판단을 쌓고 여러 프레임 투표로 정하기를 권합니다(팀 레포 미반영).

## 2. DA 주행 검사 `tools/mt_v4_data/bag_da_gate.py`
후보 모델과 지금 차량 모델(기준)을 bag 전체 프레임에 넣고, bag 마다 아래를 봅니다.

| 항목 | 통과 조건 |
|---|---|
| ① 통째로 꺼짐 (기준은 도로를 보는데 후보는 DA·LL ≈ 0) | 0 프레임 |
| ② 가장 긴 연속 꺼짐 | 0.3 s 미만 |
| ③ 차 앞이 비어 보임 (화면 맨 아래 띠 DA < 20%) | 기준 + 5 프레임 이하 |
| ④ 흔들림 (이웃 프레임 DA IoU 5% 분위) | 기준 − 0.05 이상 |

- 손볼 점: ① 은 차를 들고 있는 장면도 셉니다(v4 도 rawcam 15 프레임). 주행 구간만 세도록 고칠 예정입니다.
- 사용: `python tools/mt_v4_data/bag_da_gate.py --cand <후보 best.pt> --base <v4 best.pt> --bundle <LimoPro_mt_v4> --stride 2 <bag 프레임 폴더>...` (프레임: `bag_fork_frames.py --all`)
- 차량에 올라간 모델 버전은 bag 에 기록된 `/ar/lane/sl_mask`(차량 모델 출력)를 후보 모델 출력과 비교해 확인할 수 있습니다(`tools/mt_v4_data/which_model_in_bag.py`).

남은 일: 검수 앱(`review_bag_mt.py`)의 'da·ll 비움' 버튼은 "학습 제외"가 아니라 "빈 정답으로 학습"입니다. 버튼을 뜻이 분명하게 바꾸고("도로 없음(빈 정답으로 학습)" / "분할 학습 제외"), 모델이 도로를 보는데 '도로 없음'을 누르면 경고하도록 할 예정입니다. **아직 안 했습니다.**

## 3. 11-18 bag 자동 라벨 `tools/mt_v4_data/auto_label_v4.py`
- **DA·LL = v4 예측 그대로**입니다. v4 가 먼 곳까지 끝까지 칠하기 때문입니다. ego lane(`ego_lane_label.py`: v4 차선에 좌우 직선을 맞추고 그 사이를 채움)은 확인용으로만 씁니다.
- 주행 중 프레임을 0.25 s 간격으로 고르고, 차가 서 있으면 2 s 에 1 장만 씁니다.

| 판정 | 장수 |
|---|---:|
| 직진 자동 통과 (v4 vs ego IoU ≥ 0.85) | 649 |
| 커브 자동 통과 (앞뒤 0.1 s 프레임과 DA IoU ≥ 0.80) | 592 |
| 검수 대기 (직진인데 ego 와 다름) | 9 (모두 3차선, 학습 안 씀) |
| 버림 (흔들림 152, DA < 10% 13) | 165 |

- **빈 DA 정답은 자동으로 만들지 않습니다.** 신호등 구간은 115 장이 통과했습니다.
- 3차선(미션 7) 245 장은 뺐습니다. v4 가 장애물 LIMO 위까지 DA 로 칠하기 때문입니다. → 학습 996 장.
- 박스: v4 검출 0.5 이상은 정답, 0.2~0.5 는 무시 박스(-2). 정지선은 라벨 없음(학습 안 함).

## 4. v6 학습
- 실행: `tools/h100_mt_v4/run_v6.sh` (이 PC: `/home/guest6/Desktop/smoony/autorace2026/competition/LimoPro_mt_v4/run_v6.sh`)
- v4 대비 바뀐 점:
  - 시작 가중치 **v4 best.pt**
  - 대회장 rawcam 차선 라벨(사람 검수) 192 장 ×3. 빈 DA 정답은 뺐습니다.
  - `bag_mt_venue1118` 996 장 ×1 추가
  - 갈림길 박스 SAM 2.1 (14px 까지), 바닥 반사 증강 `--aug-glare-p 0.3`
  - lr 1e-4(검출 머리·BiFPN ×10), batch 32, bf16, 최대 150 epoch, val loss 12 epoch(0.005) 조기 종료
- **학습에 안 쓴 bag**: `arall_1118_135742` (신호등까지 감), `ar4_1118_135516` (주차) → 릴리즈 전 DA 주행 검사용
- 학습 사진 15,991 장, epoch 당 약 170 s. **epoch 53 에서 멈춤, val loss 최저 epoch 41** 을 올렸습니다(약 2.6 시간).
- 결과: 11-18 bag 11 개 모두 DA 주행 검사 통과(통째로 꺼짐 0), 같은 test 비교는 [`multitask-v6` 릴리즈 노트](https://github.com/mastic-choi/LimoPro_FOSCAR_Vision/releases/tag/multitask-v6)에 있습니다.

## 5. 학습 속도와 `--compile`
**측정** (v6 학습 중, H100 PCIe):
- GPU 사용률(0.5 s 간격 10 번)은 49~100%, 평균 약 75%입니다. 전력 230 W / 350 W.
- 데이터 대기 1~1.5% → 데이터 로딩 병목이 아닙니다.
- 학습 속도 96.6 장/s (v4 93 장/s).

**100% 가 안 되는 이유**: HybridNets(EfficientNet-B3, batch 32, 640×384, bf16)는 H100 에 비해 작습니다. GPU 계산 하나하나가 금방 끝나고, 파이썬이 다음 계산을 넘기는 사이에 빈틈이 생깁니다.

**`--compile`** (`train_multitask.py` 에 이미 있는 옵션: 모델 forward 를 `torch.compile`):
- 빈틈(파이썬·커널 실행 부담)을 줄일 수 있습니다. **학습 설정(하이퍼파라미터)은 바뀌지 않습니다.**
- 대가: 첫 epoch 에 컴파일 시간이 들고, 실행 중인 학습에는 다시 시작해야 적용됩니다. 그래서 v6 에는 쓰지 않았습니다.
- 효과는 아직 재 보지 않았습니다. **다음 학습 전에 `--max-steps` 로 짧게 켠 것과 끈 것의 장/s 를 비교**하고, 빨라지면 그때부터 씁니다.
- batch 를 키우는 것도 GPU 를 더 채우지만 학습률이 함께 바뀌는 설정 변경이라, 따로 합의한 뒤에만 합니다.

## 이번에 추가·수정한 코드
| 파일 | 내용 |
|---|---|
| `tools/mt_v4_data/bag_da_gate.py` | 릴리즈 전 DA 주행 검사 |
| `tools/mt_v4_data/which_model_in_bag.py` | bag 의 차량 출력으로 차에 올라간 모델 버전 찾기 |
| `tools/mt_v4_data/auto_label_v4.py` | 대회장 bag 자동 라벨 (v4 DA, ego lane 확인) |
| `tools/mt_v4_data/ego_lane_label.py` | 직진 ego lane 맞추기 |
| `tools/mt_v4_data/bag_fork_frames.py` | bag → 프레임 (구간·전체, 원본 영상 bag 도) |
| `tools/mt_v4_data/merge_sam_fork.py` | SAM 추적 박스로 갈림길 박스 바꾸기 (14px 이상) |
| `tools/mt_v4_data/review_fork_crops.py` | 갈림길 크롭 글자 검수 앱 (1 A / 2 B / 3 버림) |
| `scripts/fork_real_sam2_refine.py` | SAM 박스 다듬기 (뒤 표지판 빼기) |
| `scripts/train_fork_cls.py` | `--post-mask` (원 바깥 항상 가리기) |
| `scripts/export_fork_cls.py` | 가리기를 ONNX 안에 (OpenCV 4.5.4 dnn 호환) |
| `scripts/train_multitask.py` | `--aug-glare-p` (바닥 반사 증강) |
| `scripts/floor_glare.py` | 바닥 반사 증강 (선 옆 차로에 곧은 흰 띠) |
| `tools/h100_mt_v4/run_v6.sh` | v6 학습 명령 기록 |
| 그 밖 | `bag_frames.py`(원본 영상 bag), `pseudo_label_mt.py`(`--bag-sign-map`), `real_fork_det_labels.py`(`--min-sign-px`), `fork_real_sam2_track.py`(LETTER·BACK) |

## 이 PC 의 데이터 위치
- 11-18 bag: `/home/guest6/Desktop/smoony/autorace2026/competition/rosbags/`
- bag 프레임: `/home/guest6/Desktop/smoony/autorace2026/competition/label_work/fullbag1118/`
- 자동 라벨: `/home/guest6/Desktop/smoony/autorace2026/competition/label_work/datasets/bag_mt_venue1118/`
- 비교 결과(그림·JSON): `/home/guest6/Desktop/smoony/autorace2026/competition/LimoPro_mt_v4/outputs/eval_same/`
