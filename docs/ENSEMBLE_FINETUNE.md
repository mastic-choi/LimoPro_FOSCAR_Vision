# 앙상블 멤버 ego lane 파인튜닝 (2026-10-28)

공개 모델(BDD100K 사전학습)은 da를 "도로 전체"로 칠한다. 우리가 필요한 건 **지금 달리는 차선(ego lane)**뿐이라,
ego lane pseudo-label([`EGO_LANE_PSEUDO_LABEL.md`](EGO_LANE_PSEUDO_LABEL.md))을 사람이 검수한 데이터로
앙상블 멤버 4개를 각각 파인튜닝했다.

## 0. 한눈에 보기

```
LimoPro_frames_20261027 (1280x720, 8465장)
  │  Drive 중복본·정지 중복 제거                         -> 8232장           (dataset/README.md)
  ▼
ego lane pseudo-label (4모델 앙상블 + HSV + 기하 추적 + SAM2) -> OK 3286 / FAIL 4946
  │  사람 검수: 0~4929번 OK 1994장 중 257장 반려          -> 검증된 OK 1737장
  ▼
build_train_dataset.py: 시간 구간 분할 + 비슷한 프레임 제거 + 커브/직진 1:1
  │                                                     -> train 950 / val 97 / test 268
  ▼
train_seg.py x 4 모델 (train_all.py 순차 실행, train_watchdog.py 감시)
```

## 1. 데이터

### 1.1 원본과 pseudo-label
- 원본: `dataset/LimoPro_frames_20261027/session_183234` — Orbbec Dabai DC1, 1280×720 jpg, 약 8.6Hz, 16분 주행.
  정지 상태 중복 프레임은 제거(바로 앞 프레임과 평균 차이 < 2.5). 자세한 건 `dataset/README.md`.
- pseudo-label: `scripts/make_ego_lane_labels.py --no-gif --save-labels` (프로세스 2개 병렬, GPU 99%로 약 55분).
  입력이 가로 640보다 크면 비율 유지로 640×360으로 줄여서 처리·저장한다.

| 판정 | 장수 | 비율 |
|---|---|---|
| OK | 3286 | 40% |
| FAIL: 좌/우 차선 한쪽 누락 | 1829 | 22% |
| FAIL: 추적 끊김 | 1735 | 21% |
| FAIL: 추적 행 부족 | 1382 | 17% |

- 앙상블 멤버는 4개: YOLOPv2, TwinLiteNet+ Large, YOLOP, HybridNets.
  TwinLiteNet+ N/S/M/L은 같은 구조의 크기 차이라 Large만 남겼다(4개를 다 넣으면 이 계열 혼자 7표 중 과반).

### 1.2 사람 검수
- `scripts/split_ego_labels.py`로 ok/ fail/ 분리 → `scripts/review_ok_labels.py make`로 [원본 | 오버레이] 이미지 생성
  → 사진 앱에서 넘기며 틀린 OK를 삭제 → `review_ok_labels.py apply`로 ok → fail 이동(`rejected.csv`).
- 1차 검수: frame 0~4929의 OK 1994장 중 **257장(13%) 반려**, 검증된 OK **1737장**. 4930번 이후 OK 1292장은 미검수.

### 1.3 학습 데이터셋 (`dataset/ego_v1`, `scripts/build_train_dataset.py`)
- **시간 구간 단위 분할**: 연속 프레임은 서로 거의 같아서 무작위로 나누면 test 점수가 부풀려진다.
  - test = frame 4300~4929 전체(자연 비율 그대로), val = 2000~2299, 두 구간 앞뒤 30프레임은 버퍼로 제외.
- train만 줄인다: 마지막으로 남긴 프레임과 dHash 해밍거리 < 6이면 버림 → 커브/직진 1:1.
  - 커브 판정: ego da 중심선을 아래→위로 2차 피팅, |휨| ≥ 1.0 또는 |기울기| ≥ 0.4. 실제 커브와 차가 비스듬히 선 장면이 같이 잡힌다.

| split | 장수 | 커브 |
|---|---|---|
| train | 950 | 475 (+ 직진 475) |
| val | 97 | 23 |
| test | 268 | 84 |

> 주의: test의 커브는 자동 라벨이 **성공한** 완만한 커브뿐이다. 차선이 한쪽만 보이는 급커브(FAIL)는 아직 평가에 없다.

## 2. 학습 방식 (`scripts/train_seg.py`)

공통 코드 하나에 모델별 어댑터를 끼운다. 어댑터는 각 모델 출력을 **da/ll 전경 확률**로 바꾼다
(앙상블 추론 `make_model_compare_gif.py`의 `probs()`와 같은 정의).

| 모델 | 파라미터 | 입력 | 출력 → 확률 | LR |
|---|---|---|---|---|
| TwinLiteNet+ Large | 1.94M | RGB /255 | 2ch logit → softmax[1] | 5e-5 |
| YOLOP | 7.9M | BGR ImageNet 정규화 | 채널별 sigmoid 2ch → c1/(c0+c1) | 1e-4 |
| HybridNets | 12.8M | RGB ImageNet 정규화 | 3ch logit → da = road+lane, ll = lane | 1e-4 |
| YOLOPv2 | 38.9M | RGB /255 | da 2ch → c1/(c0+c1), ll 1ch 그대로 | 1e-5 |

- **입력**: 640×360 이미지를 위아래 12px 패딩해 640×384. 라벨은 640×360 그대로, 손실/평가 때 출력 높이 12:-12만 사용
  (TwinLiteNetPlus 공식 BDD100K 학습과 같은 배치).
- **손실**: BCE + soft Dice, `da + 2.0 × ll` (KMU에서 ll 가중치 2.0이 효과 있었음). AdamW, poly LR(power 1.5), batch 8, 최대 40 epoch.
- **증강**(학습 중 매번 무작위, 파일 수는 그대로 950):
  - 기하: 좌우 반전 50%, 회전 ±5°·확대 0.9~1.1·이동 ±5% 50%
  - 색: HSV 지터 50%
  - 빛: 과노출(밝기 ×1.6~2.6) 15%, 위쪽 조명 번짐 25%, 바닥 반사 띠 20%
  - 블러(셋 중 하나): 방향 모션 블러 9~25px 15%, 전진 줌 블러 4~10% 15%, 가우시안 10%
  - 샘플: `outputs/augmentation_samples.jpg`
- **조기 종료**: val (da + ll) IoU가 8 epoch 동안 0.001 이상 안 오르면 멈추고 test로.
- **저장/재개**: 매 epoch `last.pt`(임시 파일에 쓴 뒤 교체 → 전원이 꺼져도 안 깨짐). 다시 실행하면 자동으로 이어서.
  `best_da.pt` / `best_ll.pt`는 val 기준 최고점. 끝나면 `test.json`(best_da 가중치, 전체/커브/직진 IoU).

### 2.1 YOLOPv2 (TorchScript) 학습 가능하게 만들기
학습 코드가 공개되지 않아 TorchScript 가중치를 직접 학습한다. 두 가지를 고쳤다(`load_yolopv2_trainable`).
1. 배포본이라 파라미터가 `requires_grad=False`로 저장돼 있음 → 켬.
2. `batch_norm`에 `cudnn_enabled=True`가 상수로 박혀 있어 ZLUDA에서 `CUDNN_STATUS_EXECUTION_FAILED`.
   추론용 로더처럼 `torch.jit.freeze`하면 가중치가 상수가 돼 학습이 안 됨 → 그래프를 inline한 뒤 batch_norm 8번째 입력만 False로 교체.
- 그래프의 batch_norm은 추적 당시 eval 모드로 고정이라 통계는 안 바뀌고 가중치만 학습된다. 검출 헤드 파라미터 72개는 손실에 안 쓰여 기울기 없음(정상).

### 2.2 실행과 감시
- `scripts/train_all.py`: 모델을 순서대로 돌림. `test.json`이 있으면 건너뛰고, 비정상 종료 시 3번까지 재시도.
  PC가 꺼졌다 켜져도 **같은 명령을 다시 실행**하면 끝난 모델은 건너뛰고 멈춘 모델은 이어서 학습한다.
- `scripts/train_watchdog.py`: 상태를 점검해 `outputs/train_watchdog*.log`에 남긴다. 감시 중에는 Windows 절전을 막는다
  (화면 잠금은 학습과 무관). DONE / DEAD / GIVEUP / STALL / GPU_IDLE이면 이유를 출력하고 종료 → Claude Code 알림으로 조치.
  `--once`는 10분 뒤 한 번 점검하고 종료(주기 보고용).
- GPU 사용률 상한은 `GPU_DUTY`(기본 0.9). 장시간 100% 부하로 PC가 꺼진 적이 있어서다.
- ZLUDA는 처음 보는 연산(특히 역전파)을 CPU로 컴파일한다. 모델마다 첫 epoch가 길고 그동안 GPU 0%가 정상
  (TwinLiteNet+ 첫 epoch 273초 → 이후 105초).

## 3. 결과 (test 268장, 학습에 안 쓴 시간 구간)

| 모델 | epoch (best) | 전체 da | 전체 ll | 커브 da | 커브 ll | 직진 da | 직진 ll | epoch당 |
|---|---|---|---|---|---|---|---|---|
| TwinLiteNet+ Large | 40 (–) | 0.954 | 0.784 | 0.918 | 0.731 | 0.970 | 0.806 | 105초 |
| YOLOP | 35 조기 종료 (27) | 0.925 | 0.723 | 0.894 | 0.670 | 0.940 | 0.745 | 83초 |
| HybridNets | 학습 중 | | | | | | | 64초 |
| YOLOPv2 | 대기 | | | | | | | |
| YOLOP 채널별 BCE | 대기 | | | | | | | |

- 모든 모델에서 **커브가 직진보다 약하다**(TwinLiteNet+ da 0.92 vs 0.97).
- TwinLiteNet+는 val 최고점(da 0.960 / ll 0.791)과 test가 거의 같아 특정 구간 과적합 흔적은 없다.
- YOLOP가 가장 낮고 손실도 3.3 근처에서 다른 모델(0.5~0.6)보다 높게 머물렀다. 원인 가설은 §4.

## 4. 관찰 / 다음 할 일

- **YOLOP 출력과 손실의 불일치(가설)**: YOLOP는 중간층은 Hardswish, 출력층만 채널별 sigmoid 2ch다.
  우리는 전경/(배경+전경) 비율로 확률을 만들어 학습했는데, 두 채널이 함께 포화되면 이 비율을 통한 기울기가 거의 사라진다.
  → YOLOP 원래 방식대로 **채널별 BCE**(`train_seg.py --loss channel`)로 재학습해 비교한다:
  `train_all.py --data dataset/ego_v1 --models yolop --suffix _chbce -- --loss channel`
- 4개 학습이 끝나면: 파인튜닝한 4개로 앙상블을 다시 구성해 test에서 단일 모델과 비교.
- 급커브(FAIL 4946장)는 아직 학습/평가에 없다. 파인튜닝 앙상블로 FAIL 프레임 pseudo-label → 모델 간 의견이 갈리는 프레임만 사람 검수 → 학습 데이터 추가(반복).
- 4930번 이후 OK 1292장 검수가 남아 있다.

### 4.1 다음 루프: 파인튜닝 앙상블 pseudo-label → 사람 검수
1. `make_finetuned_pseudo_labels.py --eval-test` 로 앙상블이 단일 모델보다 나은지 test에서 확인.
2. `make_finetuned_pseudo_labels.py --labels ... --reviewed-max 4929 --out outputs/pseudo_ft_v1`
   - 대상: fail/ 전부 + 4929번 이후 미검수 OK. val/test 구간(+버퍼)은 제외(급커브 사람 GT로 평가에 쓸 것).
   - da = 멤버 확률 평균 ≥ 0.5, ll = 과반 투표, 저장 da = da ∪ ll. `meta.csv`의 `da_agree`(멤버 쌍별 IoU 평균) 오름차순 = 검수 우선순위.
3. 검수: `tools/ego_label_review_{mac,radeon,nvidia}.ipynb` (GPU 설정 셀만 다르고 나머지 동일).
   - 노트북은 SAM2만 로드한다(모델 4개 불필요). gradio는 5.x 고정 — 6.x는 huggingface-hub ≥ 1.16을 요구해 transformers와 충돌.
   - 라데온 버전은 이 PC(Windows ZLUDA)에서 버튼 흐름 + SAM(2.4초/회) + 웹 앱 기동까지 확인. 맥/NVIDIA 버전은 문법 검사만 함.
4. 검수 결과(`<out>_review/`)를 학습 데이터에 추가 → 재학습.
- 자기참조 주의: 이 모델들 라벨을 사람이 검수만 해서 만든 GT로 같은 모델을 평가하면 점수가 부풀려진다. 급커브 test GT는 SAM으로 직접 만들 것.
- 배포 후보는 속도가 중요하므로 LIMO PRO(Jetson)에서 `scripts/benchmark_models.py`로 실측 필요.

## 5. 재현 명령

```bash
# 1) pseudo-label (반으로 나눠 2개 병렬 실행 후 폴더 합치기)
GPU_DUTY=0.99 .venv/Scripts/python scripts/make_ego_lane_labels.py --frames dataset/LimoPro_frames_20261027/session_183234 \
    --no-gif --save-labels outputs/ego_labels_20261027_183234
.venv/Scripts/python scripts/split_ego_labels.py outputs/ego_labels_20261027_183234

# 2) 사람 검수
.venv/Scripts/python scripts/review_ok_labels.py make  outputs/ego_labels_20261027_183234
#    (ok_review/ 에서 틀린 이미지 삭제)
.venv/Scripts/python scripts/review_ok_labels.py apply outputs/ego_labels_20261027_183234

# 3) 학습 데이터셋
.venv/Scripts/python -I scripts/build_train_dataset.py outputs/ego_labels_20261027_183234 dataset/ego_v1 --reviewed-max 4929

# 4) 학습 (끊기면 같은 명령 다시) + 감시
.venv/Scripts/python scripts/train_all.py --data dataset/ego_v1 --models tln_large yolop hybridnets yolopv2
.venv/Scripts/python scripts/train_watchdog.py --models tln_large yolop hybridnets yolopv2
```
