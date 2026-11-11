# LimoPro FOSCAR Vision

> LIMO PRO 카메라 한 장으로 **차선·주행영역·표지판·신호등·라바콘**을 한 번에 보는 다중 작업(multi-task) 모델. AutoRace 2026 트랙에 맞춰 파인튜닝했습니다.

![대회장 test 다중 작업 검출](outputs/train_mt_v2/montages/montage_test_venue.jpg)

*대회장 조명 test 프레임입니다. 색 박스가 모델 예측(검출), 흰 얇은 박스가 정답입니다. 분할(차선·주행영역)은 같은 모델이 함께 냅니다.*

최신 릴리즈는 **[`multitask-v2`](https://github.com/mastic-choi/LimoPro_FOSCAR_Vision/releases/tag/multitask-v2)** (HybridNets 기반)이고, 다음 버전 **v3**를 학습 준비 중입니다.
상세 실험 기록(1~4차 차선 모델, 데이터·학습 설정, 한계 등)은 [`docs/HISTORY.md`](docs/HISTORY.md)에 있습니다.

## 1. 주요 기능

- **한 모델, 네 가지 출력** — HybridNets 하나로 아래를 동시에 냅니다.
  - **da (주행가능영역):** 지금 달리는 내 차로만. 3차선·갈림길도 내 차로만 칠합니다.
  - **ll (차선):** 차선 위치
  - **흰 가로선:** 정지선·횡단보도 같은 흰 가로선 (분할 4번째 채널)
  - **검출 12종:** pedestrian, sign_A, sign_B, bump, slow, parking, traffic_light, light_red/yellow/green, limo, cone
- **트랙·카메라 맞춤** — BDD100K 공개 모델을 LIMO PRO 영상과 대회 트랙(10m×6m, 차선 폭 550mm)에 맞게 파인튜닝.
- **대회장 조명 증강** — 빨강·파랑 색 조명을 학습 사진에 입혀, 색 조명에서도 주행영역이 번지지 않게 함.
- **1차 추론 코드 그대로** — v2는 v1과 구조·입력·출력이 같아 추론 코드에 가중치만 바꿔 끼우면 됩니다.

## 2. 모델 다운로드

| 릴리즈 | 내용 |
|---|---|
| **[`multitask-v2`](https://github.com/mastic-choi/LimoPro_FOSCAR_Vision/releases/tag/multitask-v2)** | **다중 작업 2차 HybridNets (권장·최신).** 검출 mAP50 0.90~0.95, A/B 구분 84~92% |
| [`multitask-v2-skip`](https://github.com/mastic-choi/LimoPro_FOSCAR_Vision/releases/tag/multitask-v2-skip) | 참고용: v2 + 백본 지름길(실험 B). 성능은 v2와 비슷, 추론에 `forward_mt` 필요 |
| [`multitask-v1`](https://github.com/mastic-choi/LimoPro_FOSCAR_Vision/releases/tag/multitask-v1) | 다중 작업 1차 + 학습 데이터셋 + 학습 코드 |
| [`ego-lane-v4`](https://github.com/mastic-choi/LimoPro_FOSCAR_Vision/releases/tag/ego-lane-v4) | 차선 전용 4차 HybridNets (실차 코너·이탈 프레임 보강) |
| [`ego-lane-v3`](https://github.com/mastic-choi/LimoPro_FOSCAR_Vision/releases/tag/ego-lane-v3) | 차선 전용 3차 모델 4개 (HybridNets · TwinLiteNet+ · YOLOP · YOLOPv2) |
| [`ego-lane-v2`](https://github.com/mastic-choi/LimoPro_FOSCAR_Vision/releases/tag/ego-lane-v2) · [`ego-lane-v1`](https://github.com/mastic-choi/LimoPro_FOSCAR_Vision/releases/tag/ego-lane-v1) | 차선 전용 2차 · 1차 |

```bash
# 최신 다중 작업 가중치 받기
gh release download multitask-v2 -p "hybridnets_mt_v2_best.pt"
```

- `hybridnets_mt_v2_best.pt`는 state_dict(epoch 14)입니다. 분할 4채널(배경/도로/차선/흰 가로선), 검출 12클래스.
- **전처리:** 1280×720 → 640×360 리사이즈 → 위아래 12px 패딩(값 114), RGB + ImageNet 정규화.
- 차선 전용 모델의 모델별 전처리와 사전학습 가중치는 [`models/README.md`](models/README.md)를 보세요.

## 3. 성능 (multitask-v2)

점수는 시드 1개, best.pt(epoch 14) 기준입니다. test의 표지판·신호등은 실제 프레임에 합성한 것이라 실제 표지판 성능은 아직 확인 전입니다.

| 지표 | 1차 | **2차 (v2)** |
|---|---:|---:|
| 검출 mAP50 (기존 / 대회장 / bag) | 0.651 / 0.714 / 0.751 | **0.895 / 0.937 / 0.952** |
| A/B 구분 — A·B 점수만 비교 (기존 / 대회장 / bag) | 58 / 57 / 54% | **84 / 84 / 92%** |
| 차선 da / ll — 기존 test | 0.952 / 0.819 | 0.955 / 0.816 |
| 차선 da / ll — 대회장 test | 0.934 / 0.493 | 0.935 / 0.486 |
| 차선 da / ll — 실차 test_bag | 0.981 / 0.820 | 0.981 / 0.826 |
| 흰 가로선 IoU — 대회장 test | 0.578 | **0.604** |

검출 AP50 (대회장 test): pedestrian 0.99 · bump 0.90 · sign_A 0.84 · sign_B 0.88 · parking 0.91 · slow 1.00 · traffic_light 0.99 · light red/yellow/green 0.95 / 1.00 / 0.97.

- 26px 이상 표지판은 A/B를 96~100% 맞히고, 18px 미만(멀리 있는 표지판)은 61~83%입니다.
- 전체 표와 그림은 [`outputs/train_mt_v2/README.md`](outputs/train_mt_v2/README.md), 원인 분석은 [`docs/MULTITASK_DETECTION_ANALYSIS.md`](docs/MULTITASK_DETECTION_ANALYSIS.md)에 있습니다.

## 4. Quick start

학습·추론 코드는 HybridNets를 함께 두어야 돕니다(`models/README.md`). 경로는 Windows + `.venv` 기준입니다.

**추론 / 시각화** — test 사진에 검출 박스를 그려 몽타주를 만듭니다.

```bash
.venv/Scripts/python scripts/eval_mt_montage.py . hybridnets_mt_v2_best.pt outputs/preview "multitask-v2"
```

**이어서 학습 (v1 → v2 재현)** — 1차 best.pt에서 검출 머리·BiFPN lr 10배, 검출 손실 2배로 15 epoch.

```bash
.venv/Scripts/python scripts/train_multitask.py \
  --seg-data dataset/synth_signs_v2 --det-data dataset/cones_v1 \
  --venue-ref-dir dataset/venue_ref_20261106 \
  --init outputs/train/hybridnets_mt_v1_seed0/best.pt \
  --head-lr-mult 10 --det-weight 2 --warmup-steps 500 --epochs 15 \
  --amp --batch 32 --workers 16 --out outputs/train/hybridnets_mt_v2
# 실험 A(=v2)·B를 한 번에 돌리고 비교표까지: bash scripts/run_mt_exp_ab.sh
```

**H100 서버 학습 (다음 버전 v3)** — 묶음 zip을 만들어 서버에 올리고 실행합니다(이미지는 서버에 둔 v2 데이터 재사용).

```bash
# 이 PC: 코드 + 시작 가중치(v2 best)만 담은 번들 생성
.venv/Scripts/python -I scripts/make_h100_mt_v3_bundle.py --out outputs/share

# 서버: v2 폴더 위에 덮어 풀고 실행
unzip -o LimoPro_mt_v3_h100.zip && cd LimoPro_mt
bash run_v3.sh check    # 데이터·가중치·SHA256 확인
bash run_v3.sh          # 학습 → 모니터 시험 → results_v3_<서버이름>.zip
```

자세한 서버 실행법은 [`tools/h100_mt_v3/README_V3.md`](tools/h100_mt_v3/README_V3.md), v3 설계는 [`docs/MULTITASK_V3_TRAINING.md`](docs/MULTITASK_V3_TRAINING.md)를 보세요.

## 5. 로드맵 (multitask-v3, 학습 준비 중)

v2 best에서 이어서 학습하며, 아직 결과는 없습니다. 계획은 [`docs/MULTITASK_V3_TRAINING.md`](docs/MULTITASK_V3_TRAINING.md)에 있습니다.

- **신호등 뒤 방해 모니터 증강** — 학습 중 신호등 뒤에 단색/땡땡이 화면을 그려, 모니터 위 원을 가짜 불빛으로 잡는 오탐을 줄입니다(라벨 없는 음성 샘플).
- **학습 후 temperature 보정** — 학습 중 보정 손실 대신, 끝난 뒤 val로 클래스별 temperature를 맞춰(Guo et al. 2017) `calib.json`에 저장.
- **val loss 조기 종료 + 차선 보호** — 최대 120 epoch, val loss 기준 조기 종료. 차선 IoU가 나빠진 epoch은 best로 고르지 않음.
- **작은 박스 무시, A/B 오버샘플링** — √넓이 14px 미만 무시(불빛 제외), A/B/주차가 든 이미지를 2배 자주 추출.

## 6. 문서

- [`docs/HISTORY.md`](docs/HISTORY.md) — 전체 실험 기록(차선 1~4차, 데이터·학습 설정, 결과·한계, 저장소 구성)
- [`outputs/train_mt_v2/README.md`](outputs/train_mt_v2/README.md) — 다중 작업 2차 결과 전체
- [`docs/MULTITASK_DETECTION_ANALYSIS.md`](docs/MULTITASK_DETECTION_ANALYSIS.md) — 표지판 구분 실패 원인 분석(A/B 언더피팅)
- [`docs/MULTITASK_V3_TRAINING.md`](docs/MULTITASK_V3_TRAINING.md) · [`tools/h100_mt_v3/README_V3.md`](tools/h100_mt_v3/README_V3.md) — v3 설계와 서버 실행법
- [`docs/ENSEMBLE_FINETUNE.md`](docs/ENSEMBLE_FINETUNE.md) — 차선 파인튜닝 방법과 결과
- [`docs/LIMO_DEPLOY_NOTES.md`](docs/LIMO_DEPLOY_NOTES.md) — LIMO PRO 사양과 배포 전 확인 사항

## 7. 사용한 모델 · 라이선스 · 크레딧

- 기반 모델: [HybridNets](https://github.com/datvuthanh/HybridNets), [TwinLiteNetPlus](https://github.com/chequanghuy/TwinLiteNetPlus), [YOLOP](https://github.com/hustvl/YOLOP), [YOLOPv2](https://github.com/CAIC-AD/YOLOPv2) — 모두 MIT License.
- 라바콘 공개 데이터: TraCon, FSOCO.
- AutoRace 2026 참가팀 **FOSCAR** — [smoony0226/autorace2026](https://github.com/smoony0226/autorace2026).
