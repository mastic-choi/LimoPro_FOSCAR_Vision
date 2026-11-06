# limo_deploy/

LIMO PRO(Jetson)에서 차선 인지를 돌리는 TensorRT FP16 엔진과 ROS 2 패키지.

## 엔진 (`engines/`)

| 파일 | 모델 | 크기 |
|---|---|---|
| `hybridnets_fp16.engine` | HybridNets | 30 MB |
| `tln_fp16.engine` | TwinLiteNet+ Large | 6 MB |
| `yolop_fp16.engine` | YOLOP | 13 MB |
| `yolopv2_fp16.engine` | YOLOPv2 | 53 MB |

- 가중치: 파인튜닝 결과 `ego-lane-v2` (`*_v2_best_da.pt`)
- 입력 640×384, 출력 `da_prob`, `ll_prob`. 크롭은 어댑터(`lane_perception/adapters.py`)가 처리
- 빌드 환경: Jetson Orin Nano 8GB, JetPack 6 (L4T 36.4.7), TensorRT 10.3.0, CUDA 12.x, 2026-11-03
- 무결성: `engines/SHA256SUMS.txt`, 빌드 기록: `engines/*.build.log`

> ⚠️ TensorRT 엔진은 GPU 종류와 TensorRT 버전에 묶인다. 다른 보드나 버전에서는 로드되지 않으니
> 그 환경에서 `export_trt.py`로 다시 빌드한다.

## ROS 2 패키지 (`lane_perception/`)

ROS 2 Humble ament_python 패키지. 차의 `~/foscar_ws/src/lane_perception` 과 같은 내용이다.

| 파일 | 역할 |
|---|---|
| `lane_perception/node.py` | 인지 노드: 카메라 → `/perception/da_mask`, `/perception/ll_mask` (mono8) |
| `lane_perception/adapters.py` | 모델별 전처리·추론·후처리 (TensorRT / PyTorch) |
| `lane_perception/export_trt.py` | `.pt` → ONNX → TensorRT FP16 엔진 빌드 |
| `lane_perception/viz_node.py` | 주행 모니터: 카메라를 BEV로 펼쳐 da, 생성 경로, Ld, 목표점 표시 (브라우저 MJPEG) |
| `lane_perception/bench.py`, `evaluate.py`, `verify.py` | 속도 측정, 정확도 평가, 엔진·pt 출력 비교 |
| `launch/`, `config/` | 카메라, 인지 노드 실행 설정 |

코드는 엔진과 가중치를 `~/lane_perception/engines`, `~/lane_perception/release/ego-lane-v2` 에서 찾는다.
차에 설치하려면:

```bash
mkdir -p ~/lane_perception/engines && cp engines/*_fp16.engine ~/lane_perception/engines/
cp -r lane_perception ~/foscar_ws/src/
cd ~/foscar_ws && colcon build --symlink-install --packages-select lane_perception
ros2 launch lane_perception lane_perception.launch.py model:=hybridnets   # yolop | yolopv2 | hybridnets | tln
```

엔진 다시 빌드 (torch 가 있는 환경, 예: `dl_env`):

```bash
python -m lane_perception.export_trt --model yolopv2
```

자세한 실행 방법과 측정값은 `LANE_PERCEPTION_USAGE.md` 를 본다.
