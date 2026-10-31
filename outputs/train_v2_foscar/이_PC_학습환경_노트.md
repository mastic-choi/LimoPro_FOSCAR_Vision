# 이 PC(foscar)에서 학습할 때 알아야 할 것

작성일 2026-10-31. LimoPro 2차 학습을 준비하면서 이 PC에서 직접 실행하고 측정한 내용이다.
드라이버나 환경이 바뀌면 숫자는 다시 확인한다. 아직 확인하지 못한 것은 10번에 따로 적었다.

## 1. 먼저 읽을 것

1. **기본 Python이 3.8이다.** 최신 PyTorch가 설치되지 않는다. `~/anaconda3/bin/python3.12`로 venv를 만든다.
2. **GPU 메모리는 실제로 약 6GB만 믿는다.** PyTorch가 쓸 수 있는 총량은 7.65GiB(7,834MiB)이고, 데스크톱이 0.5~2GB를 쓴다. 이 값은 실행 중에도 변한다.
3. **batch를 정할 때는 2 step 이상 돌리고, GPU 전체 기준으로 잰다.** 1 step만 재면 실제보다 훨씬 낮게 나올 수 있다.
4. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`를 켠다.** 같은 모델에서 메모리를 최대 740MiB 아꼈다.
5. **학습은 SSD에서 돌린다.** USB는 보관과 전달에만 쓴다.
6. **학습 전에 `unset PYTHONPATH LD_LIBRARY_PATH`를 한다.** `.bashrc`가 ROS와 CUDA 12.1 경로를 올려 둔다.
7. **공용 PC다.** `.bashrc`, 드라이버, 시스템 패키지, 다른 사람의 폴더와 conda 환경은 묻지 않고 바꾸지 않는다.

## 2. 사양

| 항목 | 값 |
|---|---|
| OS | Ubuntu 20.04.6, 커널 5.15.0-139 |
| 시스템 라이브러리 | glibc 2.31, GLIBCXX 3.4.28, gcc 9.4.0, cmake 3.16.3 |
| CPU | AMD Ryzen 7 7800X3D (8코어 16스레드) |
| RAM | 30GiB, swap 2GiB, `/dev/shm` 16GB |
| GPU | RTX 3070 Ti, nvidia-smi 기준 8,192MiB, PyTorch 기준 7.65GiB |
| 드라이버 | 560.35.05 (nvidia-smi 표시 CUDA 12.6), 전력 한도 320W |
| 시스템 CUDA | `/usr/local/cuda-11.3`, `cuda-12`, `cuda-12.1` (PyTorch 휠에는 필요 없음) |
| 작업 디스크 | `/` nvme1n1p2, 1.8TB ext4, 남은 공간 1.1TB |
| 다른 디스크 | nvme0n1p2, 931GB ext4, 마운트 안 됨. 내용과 소유자 미확인이므로 건드리지 않는다 |
| 기타 | 호스트명과 계정 모두 `foscar`, Docker 28.1.1, nvidia-container-toolkit 1.20.1 |

glibc가 2.31이라 `manylinux_2_28` 휠은 설치되지만, 더 높은 glibc를 요구하는 휠은 설치되지 않는다.

## 3. Python과 패키지

| 위치 | 버전 | 비고 |
|---|---|---|
| `/usr/bin/python3` | 3.8.10 | PATH 기본값. numpy 1.26 이상 설치 불가 |
| `~/anaconda3/bin/python3.12` | 3.12.7 | **새 학습에 사용** |
| `~/miniconda3/bin/python` | 3.13.2 | conda 기본 환경 |
| miniconda envs | 3.7~3.9 | `yolop`, `vad`, `suyang` 등 기존 프로젝트용. 건드리지 않는다 |

- `python3.10`, `python3.11`, `python3.12` 명령은 PATH에 없다. 이름으로 찾는 스크립트는 3.8을 고르게 되므로 버전까지 확인해야 한다.
- conda는 자동으로 켜지지 않는다.
- venv는 Anaconda의 Python을 참조한다. Anaconda를 지우면 venv도 깨진다.

**동작을 확인한 조합:**

```
Python 3.12.7
torch 2.11.0+cu128, torchvision 0.26.0 (cuDNN 9.19.0)
numpy 1.26.4, opencv-python-headless 4.10.0.84, pillow 11.0.0, scipy 1.15.3
timm 0.5.4, pandas 2.3.1, matplotlib 3.10.0
```

- 드라이버가 CUDA 12.6으로 표시돼도 `cu128` 빌드는 정상 동작한다. PyTorch는 휠에 포함된 CUDA 런타임을 쓴다.
- venv 하나가 약 7.2GB다. pip 캐시는 `~/.cache/pip`(17GB)에 있어 같은 패키지의 재설치는 빠르다.
- 사전학습 가중치를 코드에서 내려받는 모델은 첫 실행에 인터넷이 필요하다. 예: HybridNets는 `efficientnet-b3-5fb5a3c3.pth`(47MB)를 `~/.cache/torch/hub/checkpoints`에 받는다.

## 4. 환경변수

`.bashrc`가 터미널마다 다음을 올린다. Claude Code의 Bash 환경에도 똑같이 들어온다.

```
PYTHONPATH      = SCMC_2025/devel, morai_ws/devel, /opt/ros/noetic 의 python3/dist-packages
LD_LIBRARY_PATH = SCMC_2025/devel/lib, morai_ws/devel/lib, /opt/ros/noetic/lib, /usr/local/cuda-12.1/lib64
CUDA_HOME       = /usr/local/cuda-12.1
```

측정 결과, **지금은 이 값들 때문에 실제로 깨지는 일은 없다.** venv의 torch는 `.bashrc` 환경에서도 자기 CUDA 라이브러리를 읽었고, ROS 쪽에 학습 코드나 패키지와 겹치는 모듈 이름도 없었다.

그래도 지우는 이유는 PYTHONPATH가 venv의 패키지보다 **앞에** 오기 때문이다. 누가 ROS 워크스페이스에 `utils`나 `model` 같은 이름의 패키지를 빌드하면 학습 코드의 import가 그쪽으로 넘어갈 수 있다. `.bashrc`는 다른 프로젝트가 쓰므로 고치지 말고, 학습 시작 스크립트에서만 지운다.

## 5. GPU 메모리

### 쓸 수 있는 양

- nvidia-smi 총량은 8,192MiB지만 PyTorch가 쓸 수 있는 것은 **7,834MiB(7.65GiB)** 다.
- 데스크톱 사용량은 **528~2,031MiB** 를 오갔다. `gnome-shell` 하나가 같은 세션에서 1,065MiB에서 140MiB로 바뀌었다. 원인은 확인하지 못했으니 시작할 때의 값을 믿지 않는다.
- 실제 학습은 사전 점검보다 최대 150MiB 정도 더 쓴다.

### 모델별 측정값

입력 640×384, AdamW, 학습 프로세스가 쓴 양(nvidia-smi 증가분)이다.

| 모델 | batch | expandable_segments 켬 | 기본 할당기 |
|---|---:|---:|---:|
| HybridNets | 8 | 메모리 부족 | 메모리 부족 |
| HybridNets | 4 | 5,107MiB | 5,437MiB |
| TwinLiteNet+ Large | 8 | 5,291MiB | 6,031MiB |
| TwinLiteNet+ Large | 4 | 미측정 | 3,219MiB |
| YOLOP | 8 | 4,977MiB | 5,433MiB |
| YOLOP | 4 | 미측정 | 2,851MiB |
| YOLOPv2 | 8 | 5,897MiB (실제 학습 6,020MiB) | 6,289MiB |
| YOLOPv2 | 4 | 3,237MiB | 미측정 |

### 얻은 교훈

- **1 step만 재면 틀린다.** YOLOPv2 batch 8을 기본 할당기로 쟀을 때 첫 step 뒤 5,260MiB였는데 둘째 step 뒤 7,634MiB였다. `expandable_segments`를 켠 상태에서는 3~4 step에서 더 늘지 않았다.
- **`memory_reserved() / total`은 다른 프로그램을 빼먹는다.** `torch.cuda.mem_get_info()`로 GPU 전체를 재야 한다.
- **현재 예약량 대신 최대 예약량을 쓴다.** YOLOPv2는 step이 끝나면 예약량이 줄어든다.
- **OOM 뒤 정리는 `except` 밖에서 한다.** `except` 안에서는 예외가 실패한 시도의 텐서를 붙잡고 있어 `empty_cache()`가 비우지 못한다. 그러면 다음 batch 측정값이 부풀려진다.
- **TorchScript 모델의 OOM은 `torch.jit.Error`로 올라온다.** 예외 종류 대신 메시지에서 `out of memory`를 찾아 판별한다.

### batch 자동 결정 코드

아래 코드를 4개 모델 전부에서 실행해 확인했다.

```python
def find_batch(opt, step, start=8, limit=0.95):
    """step(n): batch n으로 forward/backward/opt.step/zero_grad를 한 번 수행"""
    n = start
    while True:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            step(n)
            step(n)  # 2 step 필수: 첫 step은 메모리를 적게 잡힘
            free, total = torch.cuda.mem_get_info()  # 기기 전체 기준 (데스크톱 포함)
            used = (total - free - torch.cuda.memory_reserved() + torch.cuda.max_memory_reserved()) / total
            ok = used <= limit
        except RuntimeError as e:  # TorchScript의 OOM은 torch.jit.Error로 올라와서 메시지로 판별
            if "out of memory" not in str(e).lower():
                raise
            ok = False
            opt.zero_grad(set_to_none=True)
        if ok:
            return n
        if n <= 1:
            raise SystemExit("batch 1도 메모리 부족")
        n //= 2
```

`limit`을 0.95로 둔 것은 2026-10-31 사용자 결정이다. 이 값에서는 여유가 적으므로 다음을 지킨다.

- 학습 중에는 브라우저 등 GPU를 쓰는 프로그램을 새로 열지 않는다.
- 메모리 부족으로 죽어도 체크포인트에서 이어서 돌릴 수 있게 작성한다. batch를 절반으로 낮춰 재실행할 방법도 만들어 둔다.

## 6. 속도와 발열

160장으로 잠깐 돌린 시간을 전체 데이터(3,625장)로 환산한 **대략적인 추정치**다. 초 단위로 잰 짧은 시험이라 오차가 크다.

| 모델 | batch | step당 | epoch당 | 40 epoch |
|---|---:|---:|---:|---:|
| HybridNets | 4 | 약 0.17초 | 약 2.6분 | 약 1.7시간 |
| TwinLiteNet+ Large | 8 | 약 0.25초 | 약 1.9분 | 약 1.3시간 |
| YOLOP | 8 | 약 0.18초 | 약 1.4분 | 약 0.9시간 |
| YOLOPv2 | 8 | 약 0.27초 | 약 2분 | 약 1.4시간 |

- 조건은 GPU 사용률 제한 90%(`GPU_DUTY`), DataLoader worker 4개다. worker 수는 튜닝하지 않았다.
- **실제 학습 기록(2026-10-31):** train 3,709장 기준 epoch당 HybridNets(batch 4) 156초, TwinLiteNet+ Large 114초, YOLOP 72초, YOLOPv2 111초(나머지 batch 8). 4개 모델 합계 4시간 37분 동안 중단이나 메모리 부족이 없었다.
- 긴 학습 중 확인할 때마다 65~68°C, 약 250W였다. 연속 기록이 아니라 그때그때 확인한 값이다.
- `GPU_DUTY` 제한은 개발 PC(라데온)가 100% 부하에서 꺼진 적이 있어서 생긴 것이고, 이 PC에서 같은 문제가 확인된 것은 아니다.
- 자동 절전은 꺼져 있다(`sleep-inactive-ac-type='nothing'`). `systemd-inhibit`도 정상 동작한다.

**1차 학습(개발 PC)과 비교.** 1차는 RX 9070 XT + ZLUDA, 전 모델 batch 8, train 950장의 실제 기록이다(`LimoPro_FOSCAR_Vision-main.zip`의 `docs/ENSEMBLE_FINETUNE.md`). 2차는 이 PC, train 3,625장 기준 추정치이고 1차와 같은 epoch에서 멈춘다고 가정했다.

| 모델 | 1차 시간 (epoch) | 1차 epoch당 | 2차 epoch당 (batch) | 2차 시간 |
|---|---:|---:|---:|---:|
| TwinLiteNet+ Large | 73분 (40) | 105초 | 약 113초 (8) | 약 75분 |
| YOLOP 채널별 BCE | 49분 (35) | 83초 | 약 82초 (8) | 약 48분 |
| HybridNets | 35분 (32) | 64초 | 약 154초 (4) | 약 82분 |
| YOLOPv2 | 87분 (38) | 136초 | 약 122초 (8) | 약 77분 |
| 합계 | 약 4.1시간 | | | 약 4.7시간 |

이미지 수가 3.8배인데 시간은 비슷하다. 이 PC가 초당 처리하는 이미지가 1차의 1.6~4.2배이기 때문이다. HybridNets만 batch가 4로 줄어 시간이 늘어난다.

## 7. USB와 파일 경로

- USB는 SanDisk Cruzer Blade 58.5GB이고 `/media/foscar/새 볼륨`에 연결된다. 2026-10-31에 NTFS에서 **exFAT**로 바뀌었다. 형식은 `lsblk -f`로 다시 확인한다.
- `.Spotlight-V100`, `.fseventsd`는 macOS에 꽂았을 때 생기는 시스템 폴더로, 학습과 관계없다.
- **경로에 공백과 한글이 있다.** 공백 없는 경로를 전제로 한 코드가 깨진다. 예: `train_watchdog.py`의 `--out (\S+)`. SSD의 작업 폴더는 공백 없는 경로로 만든다.
- **파일 이름 제한이 있다.** `:`가 들어간 이름은 만들 수 없음을 확인했다(exFAT 규칙상 `* ? " < > | \`도 불가). 시각을 파일 이름에 넣을 때 `:`를 쓰지 않는다.
- **심볼릭 링크를 만들 수 없다.** `ln -s`가 `Operation not permitted`로 실패한다. 링크가 든 데이터셋은 zip으로 묶거나 SSD에서 다룬다.
- **파일 권한이 저장되지 않는다.** 모든 파일이 `rwxr-xr-x`로 보이고 `chmod`도 효과가 없다. 스크립트는 `bash script.sh`로 실행한다.
- 작업 폴더는 `/home/foscar/2026AutoRace/`를 쓴다. 1.5GB, 12,112개 파일 zip을 USB(NTFS일 때)에서 SSD로 푸는 데 43초 걸렸다.
- 뽑기 전에 파일 관리자에서 꺼내기를 하거나 `sync` 후 `umount` 한다.
- 결과 크기: 4개 모델의 `best_da.pt`, `best_ll.pt`와 로그를 묶은 zip이 약 500MB다.

## 8. Windows나 다른 PC에서 만든 코드를 옮길 때

- [ ] 줄바꿈 확인: `grep -c $'\r' run.sh`. 0이 아니면 bash가 깨진다.
- [ ] 경로 확인: `.venv/Scripts/python`은 `.venv/bin/python`으로, `py -3.10`이나 `powershell` 호출도 바꾼다.
- [ ] 대소문자 확인: Windows에서 통하던 파일 이름과 import가 여기서는 실패할 수 있다.
- [ ] DataLoader는 여기서 fork로 뜬다. `cv2.setNumThreads(0)`을 두고, worker 안에서 CUDA를 쓰지 않는다.
- [ ] 다른 GPU 전용 코드는 GPU 이름으로 분기됐는지 확인한다. 이 PC는 NVIDIA이므로 ZLUDA 관련 설명을 적용하지 않고 cuDNN을 끄지 않는다.
- [ ] `torch.load`는 기본이 `weights_only=True`다. 옵티마이저 상태 등이 들어간 체크포인트는 `weights_only=False`가 필요할 수 있다.
- [ ] `float(loss)`는 경고를 낸다. 동작에는 문제없지만 `loss.item()`을 쓰는 편이 깔끔하다.

## 9. 새 학습 코드 시작점

```bash
#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
unset PYTHONPATH LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
if [ ! -f .venv/.ready ]; then
  "$HOME/anaconda3/bin/python3.12" -m venv .venv
  .venv/bin/python -m pip install -U pip
  .venv/bin/python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
  .venv/bin/python -m pip install -r requirements.txt
  touch .venv/.ready
fi
exec systemd-inhibit --what=sleep:idle --who=train --why=training .venv/bin/python train.py "$@"
```

**코드에 넣을 것**

- 시작할 때 CUDA를 확인하고, 없으면 바로 중단한다.
- 5번의 batch 자동 결정 코드.
- 매 epoch 체크포인트를 임시 파일에 쓴 뒤 `os.replace`로 교체하고, 재실행하면 이어서 돌게 한다.
- 실패 시 자동 재시도와 조기 종료.
- 작은 데이터(160장, 2 epoch)로 학습, 이어하기, 평가, 결과 저장까지 먼저 시험한다. 이번에도 이 시험에서 메모리 문제를 찾았다.

**Claude Code로 돌릴 때**

- 학습은 백그라운드 작업으로 띄운다. `nohup`, `&`, `disown`으로 띄우면 끝났을 때 알림이 오지 않는다.
- 창을 닫으면 감시가 멈춘다.

## 10. 아직 확인하지 못한 것

- `gnome-shell`의 GPU 메모리가 크게 변하는 이유
- batch 4와 8의 정확도 차이
- 혼합 정밀도(AMP)로 메모리를 줄이는 효과
- 데스크톱을 끄고 학습할 때 확보되는 메모리
- Docker 컨테이너 안에서의 GPU 학습
- Python 3.10, 3.11 조합 (이 PC에 없어 시험하지 못함)

## 11. LimoPro 2차 학습에 적용한 내용

- **학습 결과:** 2026-10-31 완료. 보고서와 가중치는 USB의 `2026AutoRace/LimoPro_2차_학습결과/`에 있다.
- **수정본 위치:** `/home/foscar/2026AutoRace/LimoPro_train` (`.venv` 설치 완료). 이 USB의 `LimoPro_train_bundle/LimoPro_train.zip`은 **수정 전 원본**이다.
- **시작 방법:** 수정본 폴더에서 `mkdir -p outputs && bash run_train.sh > outputs/train_all.log 2>&1`
- **`run_train.sh`:** Python 버전 확인, Anaconda 3.12 사용, 환경변수 정리, `expandable_segments` 켜기
- **`scripts/run_train.py`:** 사전 점검을 2 step, GPU 전체, 최대 예약량 기준으로 변경. 메모리 기준 90%에서 95%로 상향. OOM 뒤 정리 위치 수정
- **`CLAUDE.md`:** 이 PC 메모 추가
- **학습 데이터:** `dataset/ego_v2_ys_split`, train 3,709 / val 327 / test 542장. `ego_v2`에 최윤성 검수 803장을 더한 뒤 val/test를 재분배했다. 6번의 2차 시간은 train 3,625장 기준이고, 지금과 거의 같다.
- **검수 결과 합치기:** `scripts/merge_review.py`. 형식 오류, val/test 구간 침범, 기존 프레임과 겹침이 있으면 중단한다. 학습 중에는 데이터 폴더를 바꾸지 않는다.
- **재분배:** `scripts/resplit_dataset.py`. 연속 프레임은 서로 거의 같으므로 무작위로 나누지 않고 시간 구간 단위로 나눈다. 구간 앞뒤 30프레임은 학습에서 뺀다. 1차 비교를 위해 기존 val/test는 그대로 두고 어려운 장면 구간을 더했다.
- **1차와 비교:** `test.json`의 `v1`, `v1_curve` 그룹이 1차와 같은 test 268장이다. `all`은 1차 표와 직접 비교하지 않는다.
- **학습 설정:** LR, epoch, 증강은 바꾸지 않았다. `GPU_DUTY`도 90% 그대로다.
- **1차와 다른 점:** 1차는 전 모델 batch 8이었다. 이 PC에서 HybridNets만 메모리 때문에 batch 4이므로, 1차 점수와 비교할 때 조건이 다르다.
- **1차 기록과 전체 코드:** USB의 `LimoPro_FOSCAR_Vision-main.zip`. 학습 방식과 결과는 `docs/ENSEMBLE_FINETUNE.md`에 있다.

사전 점검이 고른 batch는 다음과 같다. 작은 데이터로 학습, 이어하기, 평가, 결과 zip 생성, 감시 스크립트까지 확인했다.

| HybridNets | TwinLiteNet+ Large | YOLOP | YOLOPv2 |
|---:|---:|---:|---:|
| 4 | 8 | 8 | 8 |

**YOLOPv2 batch 8의 조건**

- 시작할 때 데스크톱이 약 1.5GB를 넘으면 자동으로 batch 4로 내려간다.
- 학습 중 데스크톱이 약 1.8GB를 넘으면 메모리 부족으로 멈출 수 있다. 그때는 `bash run_train.sh --batch 4`로 재실행하면 이어서 돈다.
