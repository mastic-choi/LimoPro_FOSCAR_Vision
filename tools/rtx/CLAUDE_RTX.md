# LimoPro ego lane 2차 학습 (동아리 RTX PC)

사용자가 학습 시작을 요청하면 아래 절차대로 진행한다. 사용자와는 한국어로 대화하고, 파일 위치는 절대경로로 알린다.

이 코드는 라데온(ZLUDA) PC에서 개발됐고 이 PC는 NVIDIA다. ZLUDA 관련 코드는 GPU 이름에 ZLUDA가 있을 때만 동작하므로 여기서는 cuDNN을 그대로 쓴다. 코드와 문서의 ZLUDA 설명을 이 PC에 적용하지 말고, cuDNN을 끄지 않는다.

## 시작

1. `nvidia-smi`로 NVIDIA GPU를 확인하고, 다른 프로세스가 GPU를 쓰고 있으면 사용자에게 묻는다.
2. 이 폴더가 USB 위에 있으면 멈추고 SSD로 옮기라고 안내한다.
3. 학습을 **백그라운드 작업**으로 시작한다(`<outputs>`는 이 폴더의 `outputs`). 먼저 `outputs` 폴더를 만든다.
   - Ubuntu: `bash run_train.sh > outputs/train_all.log 2>&1`
   - Windows: `cmd /c run_train.bat > outputs\train_all.log 2>&1` (PowerShell이면 `.\run_train.bat *> outputs\train_all.log`)
   - 사용자가 모델을 지정하면 `--models`를 붙인다. 기본은 `hybridnets tln_large yolop yolopv2` 순서.
4. 첫 실행은 PyTorch 설치로 몇 분 걸린다. `outputs/train_all.log`에 `[run]`이 나오면 시작 성공이다. 그전에 프로세스가 끝나면 로그를 보고 고친다.
5. `[run]`을 확인한 뒤 아래 감시 2개를 **각각 백그라운드 작업으로** 띄우고, 학습 시작과 GPU, 모델별 batch를 보고한다.

## 10분 감시

`<py>`는 Ubuntu `.venv/bin/python`, Windows `.venv\Scripts\python`. `--models`는 학습과 똑같이 맞춘다.

- **상시 감시** (문제가 생기면 종료해서 알림):
  `<py> scripts/train_watchdog.py --suffix _v2 --models hybridnets tln_large yolop yolopv2 --first-epoch-min 60`
- **10분 보고** (10분 뒤 한 번 점검하고 종료):
  `<py> scripts/train_watchdog.py --once --suffix _v2 --models hybridnets tln_large yolop yolopv2 --first-epoch-min 60`

백그라운드 작업은 끝나면 Claude에게 알림이 오므로 sleep으로 기다리거나 로그를 폴링하지 않는다.
`nohup`, `&`, `disown`으로 띄우면 알림이 오지 않으니 쓰지 않는다.

10분 보고 작업이 끝날 때마다:
1. 출력 줄(모델, epoch, val da/ll, GPU, 완료 수)을 사용자에게 짧게 보고한다.
2. 학습이 진행 중이면 10분 보고 작업을 다시 띄운다. 이 반복을 멈추지 않는다.

## 상시 감시가 끝났을 때

| exit | 뜻 | 조치 |
|---|---|---|
| 0 DONE | 모두 완료 | 아래 "끝나면"으로 |
| 2 DEAD | 학습 프로세스 없음 | `outputs/train/<run>/train.log` 끝을 보고 원인을 찾은 뒤, 시작과 같은 명령으로 재실행(자동 이어하기). 상시 감시도 다시 띄움 |
| 3 GIVEUP | 한 모델이 3번 연속 실패 | train.log의 오류를 보고한다. 학습 중 메모리 부족이면 `--batch`를 절반으로 해서 재실행하고, 코드 문제면 고쳐서 재실행 |
| 4 STALL | epoch 갱신 없음 | 프로세스와 GPU 상태 확인 후, 멈췄으면 학습 프로세스만 종료하고 재실행 |
| 5 GPU_IDLE | GPU 사용률 0% 연속 | 프로세스와 `nvidia-smi` 확인 후, 멈췄으면 STALL처럼 처리 |

조치 후 무엇을 했는지 보고한다. 같은 원인으로 두 번 실패하면 더 시도하지 말고 사용자에게 묻는다.

## 지킬 것

- GPU 사용률은 기본 90% 제한(`GPU_DUTY`)을 유지한다. 사용자가 요청할 때만 바꾼다.
- `--fresh`를 쓰지 않고 `outputs/train`, `last.pt`, `dataset/`을 지우지 않는다. 재실행하면 이어서 학습된다.
- 학습 중에 GPU를 쓰는 다른 작업을 돌리지 않는다.
- 하이퍼파라미터(LR, epoch, 증강, 데이터)를 사용자 동의 없이 바꾸지 않는다. batch는 시작 때 메모리에 맞춰 자동으로 정해지며, 학습 중 메모리 부족일 때만 낮춘다.
- 동아리 PC의 시스템 설정, 드라이버, 다른 사람의 파일을 건드릴 일이 생기면 먼저 묻는다.

## 끝나면

1. `results_<PC이름>.zip`이 생겼는지 확인한다. 없으면 시작과 같은 명령을 한 번 더 실행한다(끝난 모델은 건너뛰고 zip을 만든다).
2. 모델별 test da/ll, 커브/직진 da, epoch 수, 학습 시간을 표로 보고한다. 1차 결과와 비교한다.

| 1차 | test da | test ll | 커브 da | 커브 ll |
|---|---:|---:|---:|---:|
| HybridNets | 0.967 | 0.809 | 0.937 | 0.760 |
| TwinLiteNet+ Large | 0.954 | 0.784 | 0.918 | 0.731 |
| YOLOP (채널별 BCE) | 0.940 | 0.749 | 0.906 | 0.692 |
| YOLOPv2 | 0.962 | 0.810 | 0.938 | 0.761 |

3. 2차 학습 데이터에는 옆 차로까지 칠한 라벨이 일부 섞여 있고 test 정답은 내 차로만 칠해져 있다. 점수가 내려가도 라벨 기준 차이일 수 있으니 함께 알린다.
4. zip의 절대경로를 알리고 USB에 복사할지 묻는다.
