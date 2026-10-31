LimoPro ego lane 2차 학습 (NVIDIA RTX PC)

[Claude로 실행]
LimoPro_train 폴더에서 claude를 켜고 "CLAUDE.md대로 학습 시작해줘"라고 하면 된다.
Claude가 학습을 돌리고 10분마다 상태를 보고한다. 창을 닫으면 감시가 멈춘다.

[직접 실행]
1. LimoPro_train.zip을 PC의 SSD로 복사한 뒤 압축 해제. USB에서 바로 돌리지 말 것.
2. 인터넷 연결 확인 (첫 실행 때 PyTorch 등 약 3GB 다운로드).
3. Ubuntu:  cd LimoPro_train && bash run_train.sh
   Windows: LimoPro_train 폴더에서 run_train.bat 더블클릭

기본으로 hybridnets, tln_large, yolop, yolopv2를 이 순서로 학습한다. 일부만 하려면:
   bash run_train.sh --models hybridnets tln_large

[끊겼을 때]
같은 명령을 다시 실행하면 끝난 모델은 건너뛰고 마지막 epoch부터 이어서 한다.

[GPU 사용률]
기본 90% 제한. 제한을 풀려면:
   Ubuntu:  GPU_DUTY=1.0 bash run_train.sh
   Windows: set GPU_DUTY=1.0 후 run_train.bat

[GPU 메모리]
시작할 때 batch 8을 시험해 보고 모자라면 4, 2로 자동으로 낮춘다.

[결과]
끝나면 LimoPro_train/results_<PC이름>.zip이 생긴다. 이 파일을 USB에 담아 오면 된다.
진행 기록: outputs/train/<모델>_v2_seed0/train.log, log.csv

[문제가 생기면]
- "CUDA를 쓸 수 없습니다": nvidia-smi가 되는지 확인. 드라이버 설치/업데이트 필요.
- Ubuntu에서 venv 오류: sudo apt install python3-venv
- 설치가 중간에 끊김: .venv 폴더를 지우고 다시 실행.
