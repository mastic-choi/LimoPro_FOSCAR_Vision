# LimoPro 다중 작업 v3 + 실물 신호등 추가 학습 (RTX 3070 Ti PC)

> **2026-11-13 foscar PC에서 실행한 결과:** epoch당 358초(아래 20~30분 추정보다 훨씬 빠름), 조기 종료 없이 30 epoch에 약 3시간, GPU 메모리 약 6.2GB(batch 8). 결과는 [`outputs/train_mt_v3real/README.md`](../../outputs/train_mt_v3real/README.md), 가중치는 릴리즈 [`multitask-v3real`](https://github.com/mastic-choi/LimoPro_FOSCAR_Vision/releases/tag/multitask-v3real)에 있습니다.
> 이 폴더의 `run.sh`, `requirements.txt`, 이 README는 번들 `LimoPro_mt_v3real_rtx.zip`에 들어 있던 것입니다. 번들의 `requirements.txt`에는 `six`가 빠져 있어 첫 실행이 `ModuleNotFoundError: No module named 'six'`로 실패했고, 여기에는 추가해 두었습니다.

multitask-v3에서 이어서, 로봇 카메라로 찍은 **실물 신호등 사진**을 섞어 짧게 더 학습합니다.
v3는 합성 신호등으로만 배워서, 실물 신호등의 노란불·초록불을 거의 못 잡았습니다(1.2m에서 노랑 0.14, 초록 0.18점).

## 준비물 (USB에서 SSD로 복사)

| 파일 | 내용 |
|---|---|
| `LimoPro_mt_v3real_rtx.zip` | 코드, 시작 가중치(multitask-v3), 실물 신호등 데이터(`dataset/real_tl_v1`), 실행 스크립트 |
| `LimoPro_dataset_mt_v1.zip` (5.1GB) | v3 학습에 쓴 합성 데이터. `run.sh`가 필요한 부분만 풂 (콘은 안 풂) |

**두 zip을 SSD의 같은 폴더에 두세요.** USB에서 바로 학습하지 않습니다. 이 USB는 exFAT이라 실행 권한이 없으니 스크립트는 `bash run.sh`로 실행합니다.

## 실행

```bash
mkdir -p ~/LimoPro_v3real && cp /media/$USER/*/2026AutoRace/LimoPro_v3real_rtx/*.zip ~/LimoPro_v3real/
cd ~/LimoPro_v3real
unzip -q LimoPro_mt_v3real_rtx.zip        # unzip이 없으면: python3 -m zipfile -e LimoPro_mt_v3real_rtx.zip .
cd LimoPro_mt_rtx
bash run.sh check                         # venv 만들기, 데이터 풀기, 데이터·가중치·GPU 확인 (처음엔 10분 안팎)
tmux new -s limo                          # 창을 닫아도 계속 돌게
bash run.sh                               # 학습 -> 실물 신호등 비교 -> 모니터 시험 -> 결과 zip
```

- 끊기면 `bash run.sh`를 다시 실행하면 `last.pt`부터 이어서 합니다.
- **GPU 메모리:** batch 8로 시작하고, 메모리 부족으로 죽으면 4, 2로 줄여서 자동으로 이어 갑니다. 이 PC 기록상 HybridNets는 batch 4에서 약 5.1GB를 썼습니다.
- **시간(추정):** v3는 H100에서 epoch당 약 175초였습니다. 이 PC에서는 epoch당 20~30분으로 보고, 최대 30 epoch이지만 val loss가 4 epoch 동안 안 내려가면 멈추니 보통 3~6시간입니다.
- **결과:** `LimoPro_mt_rtx/results_v3real_<PC이름>.zip`. 이 파일만 USB에 담아 오면 됩니다.

## 이번 학습 설정

| 항목 | 값 | 이유 |
|---|---|---|
| 시작 가중치 | multitask-v3 (`weights/hybridnets_mt_v3_best.pt`, 릴리스와 같은 파일) | A/B 구분·모니터 방해 대응은 v3 것을 그대로 가져감 |
| 실물 신호등 | train 961장 x2, val 163장, test 171장 + 1.2m 12장 | 한 epoch의 약 17%. 분할 라벨이 없어 **차선 손실에는 안 들어감** |
| 콘 | 안 씀 (`--det-data none`) | 미션 8은 라이다로 처리. 출력 12클래스는 그대로라 TensorRT·C++ 노드 변경 없음. **cone 출력은 쓰지 말 것** |
| lr | 5e-5 (검출 머리·BiFPN은 x10), 30 epoch에 맞춰 줄어듦 | 이어서 하는 짧은 학습 |
| BatchNorm | 통계를 v3 값으로 고정 (`--freeze-bn`) | v3는 batch 32로 학습. 여기서는 4~8이라 통계가 흔들리면 차선이 나빠질 수 있음 |
| 차선 보호 | val da+ll IoU가 v3보다 0.01 넘게 낮은 epoch은 `best.pt`로 안 고름 | 차선을 지키기 위해 |
| 방해 모니터 | 신호등 뒤 모니터를 확률 0.4로 그림 (**실물 사진에도**), 그중 절반은 땡땡이 | 실물 데이터에는 모니터 장면이 없어서 |
| 그 밖 | v3와 같음 (작은 박스 14px 무시, A/B 2배, 대회장 조명 증강) | |

값은 환경변수로 바꿉니다(`run.sh` 맨 위). 예: `EPOCHS=40 REAL_REPEAT=3 bash run.sh`, 다른 시작 가중치는 `INIT=경로`.

## 결과에서 볼 것

1. **`eval_real/eval_real.txt`**: 실물 신호등에서 v3와 학습 결과 후보(`best`, `best_val_loss`, `best_score`)를 나란히 비교합니다.
   점수 0.3, 0.5 기준으로 몸체·빨강·노랑·초록 재현율, 다른 색으로 잡은 수, 오탐 수를 냅니다. 학습 전 v3 값은 아래와 같습니다.

   | v3 | 몸체 / 빨강 / 노랑 / 초록, 점수 ≥ 0.3 | 점수 ≥ 0.5 |
   |---|---|---|
   | test 171장 (대부분 0.4~0.8m) | 100 / 96 / 96 / 93% | 98 / 82 / 71 / **22%** |
   | test_1m2 12장 (약 1.2m) | 100 / 100 / **0 / 0%** | 83 / 100 / **0 / 0%** |

2. **`test.json`의 차선 IoU**: v3는 da / ll이 기존 test 0.952 / 0.802, 대회장 test 0.932 / 0.483입니다. 이보다 눈에 띄게 낮으면 `candidates`에서 다른 후보를 고릅니다.
3. **`eval_monitor/`**: 합성 신호등 뒤에 모니터를 그렸을 때 가짜 불빛 수 (v3: 땡땡이 모니터 앞 가짜 불빛 4개).

## 공용 PC 주의 (이 PC 학습환경 노트)

- `.bashrc`, 드라이버, 시스템 패키지, 다른 사람의 폴더·conda 환경은 바꾸지 않습니다. `run.sh`는 자기 폴더의 `.venv`만 씁니다.
- 시작 전에 `nvidia-smi`로 다른 사람이 GPU를 쓰고 있는지 확인합니다.
- Python은 `~/anaconda3/bin/python3.12`를 자동으로 고릅니다(기본 python3는 3.8이라 안 됨).
- 처음 실행할 때 인터넷이 필요합니다: PyTorch 등 설치(pip 캐시가 있으면 빠름), HybridNets가 모델을 만들 때 받는 `efficientnet-b3` 가중치(47MB, 한 번 받으면 캐시).
