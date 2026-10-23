# models/

공개 da(주행가능영역)/ll(차선) 모델 레포와 사전학습 가중치(모두 BDD100K 학습, 파인튜닝 없음).
이 README를 뺀 나머지는 `.gitignore` 대상이라, 새로 clone하면 아래 명령으로 다시 받아야 한다.

성능 수치는 BDD100K 기준(각 논문/레포 보고값).

| GIF 칸 | 모델 | 파라미터 | da mIoU | ll IoU | 레포 | 가중치 (`models/` 기준) |
|---|---|---|---|---|---|---|
| 1행 2칸 | YOLOPv2 | 38.9M | 93.2 | 27.3 | [CAIC-AD/YOLOPv2](https://github.com/CAIC-AD/YOLOPv2) | `YOLOPv2/data/weights/yolopv2.pt` (TorchScript) |
| 1행 3칸 | TwinLiteNet+ Large | 1.94M | 92.9 | **34.2** | [chequanghuy/TwinLiteNetPlus](https://github.com/chequanghuy/TwinLiteNetPlus) | `TwinLiteNetPlus/pretrained/large.pth` |
| 1행 4칸 | TwinLiteNet+ Medium | 0.48M | 92.0 | 32.3 | 〃 | `TwinLiteNetPlus/pretrained/medium.pth` |
| 2행 1칸 | YOLOP | 7.9M | 91.5 | 26.2 | [hustvl/YOLOP](https://github.com/hustvl/YOLOP) | `YOLOP/weights/End-to-end.pth` (레포에 포함) |
| 2행 2칸 | TwinLiteNet+ Small | 0.12M | 90.6 | 29.3 | 〃 TwinLiteNetPlus | `TwinLiteNetPlus/pretrained/small.pth` |
| 2행 3칸 | HybridNets | 12.8M | 90.5 | 31.6 | [datvuthanh/HybridNets](https://github.com/datvuthanh/HybridNets) | `HybridNets/weights/hybridnets.pth` |
| 2행 4칸 | TwinLiteNet+ Nano | 0.03M | 87.3 | 23.3 | 〃 TwinLiteNetPlus | `TwinLiteNetPlus/pretrained/nano.pth` |

"GIF 칸"은 `scripts/make_model_compare_gif.py`가 만드는 4×2 비교 GIF
(`outputs/montages/model_compare.gif`)에서의 위치다. 1행 1칸은 원본 프레임.

## 추론 설정 (`scripts/make_model_compare_gif.py`)

전처리는 각 레포 공식 demo를 따른다. LimoPro 카메라 프레임(640×480)은 `letterbox(640, auto=True)`에서
패딩 없이 그대로 들어간다.

| 모델 | 입력 크기 | 전처리 | 특이사항 |
|---|---|---|---|
| YOLOPv2 | 640×480 | RGB, /255 | TorchScript. ZLUDA용으로 `torch.jit.freeze` + batch_norm의 `cudnn_enabled` 상수를 False로 교체 + JIT fuser(NNC) 끔 |
| TwinLiteNet+ (4종) | 640×480 | RGB, /255 | Medium은 TwinLiteNet-KMU가 파인튜닝 베이스로 쓴 모델 |
| YOLOP | 640×480 | **BGR**, ImageNet 정규화 | 공식 demo가 BGR 그대로 넣음 |
| HybridNets | **640×512** (위아래 16px 패딩) | RGB, ImageNet 정규화 | BiFPN이 1/128까지 내려가서 128 배수 필요(공식 demo의 32 배수 패딩으론 640×480에서 터짐). 출력 3클래스(bg/road/lane) argmax |

## 실행 기록 (2026-10-24, RX 9070 XT + ZLUDA, `frame_000085~000433` 349장)

| 모델 | 소요 시간 | 조건 |
|---|---|---|
| HybridNets | 23s | 단독 재실행 |
| YOLOPv2 | 41s | 컴파일 캐시된 뒤 재실행 |
| TwinLiteNet+ Nano / Small / Medium / Large | 85 / 86 / 88 / 93s | 7개 동시 실행 |
| YOLOP | 171s | 7개 동시 실행, 첫 실행(컴파일 포함) |

> ⚠️ ZLUDA 첫 커널 컴파일, PNG 저장, 동시 실행 경쟁이 섞인 수치라 **모델 추론 속도 비교용이 아니다.**
> 실제 FPS는 별도 벤치마크로, 특히 배포 대상(LIMO PRO의 Jetson)에서 직접 재야 한다.

육안 관찰(첫인상): 노란 실선은 대부분 잡지만, 흰 횡단보도 선은 거의 모든 모델이 차선으로 못 잡는다.
TwinLiteNet+ Large/Medium/Nano는 화면 맨 아래 가장자리에 빨간(ll) 줄이 생긴다.

## 다운로드 (레포 루트에서, Git Bash)

```bash
cd models
for r in CAIC-AD/YOLOPv2 chequanghuy/TwinLiteNetPlus hustvl/YOLOP datvuthanh/HybridNets; do
  git clone --depth 1 https://github.com/$r
done
curl -L -o YOLOPv2/data/weights/yolopv2.pt https://github.com/CAIC-AD/YOLOPv2/releases/download/V0.0.1/yolopv2.pt
curl -L --create-dirs -o HybridNets/weights/hybridnets.pth https://github.com/datvuthanh/HybridNets/releases/download/v1.0/hybridnets.pth
pip install gdown
(cd TwinLiteNetPlus && python -m gdown --folder https://drive.google.com/drive/folders/1EqBzUw0b17aEumZmWYrGZmbx_XJqU-vz -O pretrained)
```

## 파이썬 환경

ZLUDA용으로 패치된 전역 torch(2.1.2+cu118)를 그대로 쓰도록 `--system-site-packages` venv를 만든다.

```bash
python -m venv --system-site-packages .venv
.venv/Scripts/python -m pip install "timm==0.5.4" efficientnet_pytorch yacs prefetch_generator tensorboardX webcolors pretrainedmodels
```

`timm`은 0.5.4로 고정해야 한다(HybridNets가 `timm.models.layers.activations`를 import함).
