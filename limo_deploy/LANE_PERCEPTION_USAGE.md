# Ego lane 인지 노드 사용 정리

- 작성일: 2026-11-04
- 환경: LIMO PRO, Jetson Orin Nano 8GB, ROS2 Humble
- 패키지: `~/foscar_ws/src/lane_perception`

## 1. 빠른 실행 (배시 명령)

`~/.bash_aliases`에 저장되어 있습니다. 새 터미널을 열고 시작 메뉴에서 번호를 고른 뒤 실행합니다. 이미 열려 있던 터미널에서는 `source ~/.bash_aliases`를 먼저 실행합니다.

| 터미널 | 명령 | 동작 |
|---|---|---|
| 1 | `lane_cam` | 카메라 (1280×720, 컬러만) |
| 2 | `lane_node` | 인지 노드 (기본 모델 `yolop`) |
| 3 | `lane_viz` | 시각화 창 |

종료는 각 터미널에서 `Ctrl+C`입니다.

| 명령 | 동작 |
|---|---|
| `lane_node tln` | `tln`으로 실행 |
| `lane_node hybridnets` | `hybridnets`로 실행 |
| `lane_node yolopv2` | `yolopv2`로 실행 |
| `lane_node yolop publish_overlay:=true` | `/perception/overlay` 토픽도 출력 |
| `lane_node yolop backend:=pt` | FP16 엔진 대신 `.pt`로 실행 |
| `lane_hz` | 마스크 출력 속도 확인 |
| `lane_build` | 패키지 다시 빌드 (새 파일 추가, `setup.py` 변경 시) |
| `lane_help` | 명령 목록 보기 |

`.py` 수정은 다시 빌드하지 않아도 바로 반영됩니다.

## 2. 실행 명령 원문

배시 명령이 실제로 실행하는 내용입니다. 일반 터미널에서는 `.bashrc`가 ROS와 `wego_ws`를 이미 불러오지만, 아래처럼 직접 `source`해도 됩니다.

**공통 환경**

```bash
source /opt/ros/humble/setup.bash
source ~/foscar_ws/install/setup.bash
```

**카메라**

```bash
ros2 launch lane_perception camera.launch.py
```

`camera.launch.py` 없이 드라이버 launch를 직접 실행하는 명령입니다.

```bash
ros2 launch orbbec_camera astra_stereo_u3.launch.py \
  enable_depth:=false enable_ir:=false \
  enable_point_cloud:=false enable_colored_point_cloud:=false \
  color_width:=1280 color_height:=720 color_fps:=30 color_format:=MJPG
```

**인지 노드**

```bash
ros2 launch lane_perception lane_perception.launch.py model:=yolop
```

**시각화 노드**

```bash
ros2 run lane_perception lane_viz_node
```

**출력 속도 확인**

```bash
ros2 topic hz /perception/da_mask
```

**패키지 빌드**

```bash
cd ~/foscar_ws
colcon build --symlink-install --packages-select lane_perception
```

**트랙 rosbag 녹화**

```bash
ros2 bag record -o ~/bags/track_1280x720 /camera/color/image_raw /camera/color/camera_info
```

raw 영상은 1분에 약 4.8GB입니다.

**FP16 변환**

```bash
~/miniconda3/envs/dl_env/bin/python -m lane_perception.export_trt --model yolop
```

**속도 측정 (카메라를 켠 상태에서)**

```bash
~/miniconda3/envs/dl_env/bin/python -m lane_perception.bench --seconds 60 --no-overlay
```

## 3. 카메라 launch를 따로 만든 이유

모델은 **1280×720을 640×360으로 줄인 이미지**로 학습했습니다. 그런데 평소 쓰는 `astra_stereo_u3.launch.py`의 기본값은 640×480(4:3)입니다.

`camera.launch.py`는 기존 launch를 그대로 불러오면서 값만 지정합니다. 그래서 긴 명령을 줄이면서, 같은 launch를 쓰는 기존 코드에는 영향을 주지 않습니다.

| 설정 | 값 | 이유 |
|---|---|---|
| 드라이버 launch | `astra_stereo_u3.launch.py` | `teleop_launch.py`도 쓰는 launch |
| 컬러 | 1280×720, 30fps, MJPG | 학습 원본과 같은 해상도 |
| 깊이, IR, 포인트클라우드 | 끔 | 차선 인식에 필요 없음, 1280×720과 동시 사용은 시험 안 함 |

깊이가 필요하면 `lane_cam enable_depth:=true`로 켤 수 있지만, 이 조합은 시험하지 않았습니다. 카메라 fps는 `lane_cam color_fps:=15`처럼 바꿀 수 있습니다.

## 4. 알아둬야 할 것

### 입력 해상도

- **노드는 16:9 입력만 처리합니다.** 640×480이 들어오면 에러를 찍고 프레임을 건너뜁니다.
- **처리 흐름:** 1280×720 → 640×360 축소 → 위아래 12px 회색 패딩 → 640×384 추론 → 패딩 제거 → 640×360 마스크
- **640×360을 바로 받지 않는 이유:** 현재 USB 2.0 연결에서는 컬러 640×360이 지원되지 않고, 1280×720이 학습 원본과 같습니다.
- **지원되는 컬러 프로필 (USB 2.0):** 1280×720과 640×480 (MJPG, YUYV)

### 카메라

- **`dabai_dcl.py`는 쓰지 않습니다.** 기본값으로는 지원되지 않는 형식을 요청해 종료되고, 1280×720을 지정하면 없는 가속도 센서를 찾으며 카메라를 계속 재시작합니다.
- **USB 2.0으로 연결되어 있습니다.** 깊이 영상을 함께 쓰면 대역폭이 부족할 수 있습니다.


### 통합 launch

`teleop_launch.py`는 실행 인자로 해상도를 넘길 수 있어 파일을 고치지 않아도 됩니다. **아직 실행해 보지 않았고**, 차량 구동부(`limo_base`)도 함께 켜지므로 안전한 상태에서 실행해야 합니다.

```bash
ros2 launch wego teleop_launch.py color_width:=1280 color_height:=720 color_fps:=30 color_format:=MJPG \
  enable_depth:=false enable_ir:=false enable_point_cloud:=false enable_colored_point_cloud:=false
```

### 네트워크

`.bashrc`에 `ROS_LOCALHOST_ONLY=1`이 있어 다른 PC에서는 토픽이 보이지 않습니다.

### 시각화 노드

`lane_viz`는 CPU를 많이 씁니다(코어 하나의 약 88%). 표시 속도는 약 16~20fps입니다. 속도를 잴 때와 주행할 때는 끕니다.

## 5. 토픽 정보

| 토픽 | 타입 | 인코딩 | 크기 | 내용 | 출력 조건 |
|---|---|---|---|---|---|
| `/camera/color/image_raw` | `sensor_msgs/Image` | `rgb8` | 1280×720 | 인지 노드의 입력 (약 28~29Hz) | `lane_cam` 실행 시 |
| `/perception/da_mask` | `sensor_msgs/Image` | `mono8` (0/255) | 640×360 | 주행 가능 영역 (내 차로) | 항상 |
| `/perception/ll_mask` | `sensor_msgs/Image` | `mono8` (0/255) | 640×360 | 차선 | 항상 |
| `/perception/overlay` | `sensor_msgs/Image` | `bgr8` | 640×360 | 영상에 da(초록), ll(빨강)을 합성 | `publish_overlay:=true`일 때만 |

- **타임스탬프:** 마스크의 `header.stamp`와 `frame_id`는 입력 이미지와 같습니다.
- **임계값:** `da_thresh`, `ll_thresh` 모두 학습 평가와 같은 0.5입니다.
- **프레임 처리:** 최신 프레임만 처리하고, 밀린 프레임은 버립니다(구독 큐 1).


### 인지 노드 파라미터

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `model` | `yolopv2` (launch), `yolop` (`lane_node`) | `tln`, `yolop`, `hybridnets`, `yolopv2` |
| `backend` | `trt` | `trt` (FP16 엔진), `pt` |
| `weight` | 빈 값 | 비워 두면 모델과 backend에 맞는 기본 경로 사용 |
| `device` | `cuda` | `cuda`, `cpu` (`pt`에서만 적용) |
| `input_topic` | `/camera/color/image_raw` | 입력 토픽 |
| `da_thresh`, `ll_thresh` | `0.5` | 마스크 임계값 |
| `publish_overlay` | `false` | overlay 토픽 출력 |
| `allow_aspect_mismatch` | `false` | 16:9가 아닌 입력을 눌러서 처리 (동작 확인용) |
| `timing_csv` | 빈 값 | 구간별 시간을 CSV로 저장 |

launch 인자로 넘길 수 있는 값은 `model`, `backend`, `publish_overlay`, `allow_aspect_mismatch`, `timing_csv`입니다. 나머지는 `config/lane_perception.yaml`에서 바꿉니다.

## 6. FP16 변환 결과

엔진 위치: `~/lane_perception/engines/<모델>_fp16.engine`

| 모델 | 빌드 시간 | 엔진 크기 | 건너뛴 후보 | `.pt`와의 프레임별 IoU 차이 최대 (기준 0.005) | 판정 |
|---|---:|---:|---|---:|---|
| `tln` | 약 8분 | 6.1MB | 0건 | 0.0039 | 통과 |
| `yolop` | 약 8분 | 12.8MB | 0건 | 0.0125 | **기준 초과** |
| `hybridnets` | 약 15분 | 30.2MB | 0건 | 0.0036 | 통과 |
| `yolopv2` | 약 11분 | 53.1MB | 2건 이상 (로그 일부만 남음) | 0.0138 | **기준 초과** |

- **비교 방법:** 같은 이미지 31장에서 `.pt`와 FP16의 마스크 IoU를 계산했습니다.
- **비교 이미지의 한계:** 기존 4:3 bag을 눌러서 만든 이미지입니다. 기준 초과 모델은 트랙 16:9 영상으로 다시 재야 합니다.
- **`yolopv2` 기준 초과:** 31장 중 2장이고, 확률 차이는 최대 0.03 미만입니다. 0.5 근처 픽셀이 뒤집힌 것으로 보입니다.
- **`yolop` 기준 초과:** 초과 장수와 원인은 확인하지 않았습니다.
- **어댑터 검증:** 4종 모두 학습 코드와 31장 전부 픽셀 차이 0입니다.
- **건너뛴 후보:** 빌드 중 약 240MB 할당 실패로 시험하지 못한 구현 후보입니다. 속도에만 영향이 있을 수 있고, 원인은 확정하지 못했습니다.
- **다시 빌드해야 할 때:** 가중치, 입력 크기, TensorRT 버전이 바뀌거나 다른 종류의 보드로 옮길 때입니다.

## 7. FPS 측정 결과

**조건:** FP16, 15W, 실시간 카메라 1280×720(약 29Hz), 컬러만, Claude Code 켠 상태, `lane_viz` 끔, 모델별 60초(처음 30프레임 제외)

**표 읽는 법:** `A / B`는 **평균 / p95**이고 단위는 ms입니다. p95는 프레임의 95%가 이 시간 안에 끝났다는 뜻입니다. 20Hz 제어 주기에 맞추려면 전체 시간이 50ms 안에 들어야 합니다.

### overlay 끔 (기본 설정)

| 모델 | fps | 변환 | 리사이즈 | 전처리 | **추론** | 후처리 | 발행 | **전체** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `yolop` | **21.8** | 0.9 / 1.1 | 0.6 / 0.7 | 13.9 / 14.2 | **23.4 / 23.6** | 2.2 / 2.4 | 1.0 / 1.1 | **42.1 / 42.6** |
| `yolopv2` | **20.3** | 1.0 / 1.3 | 0.6 / 0.7 | 6.8 / 8.1 | **29.2 / 29.5** | 2.1 / 2.5 | 1.0 / 1.2 | **40.7 / 42.5** |
| `hybridnets` | **19.9** | 0.9 / 1.2 | 0.5 / 0.6 | 11.1 / 11.7 | **32.1 / 32.3** | 1.9 / 2.2 | 0.9 / 1.1 | **47.3 / 48.4** |
| `tln` | **17.9** | 0.9 / 1.3 | 0.6 / 0.7 | 4.9 / 5.4 | **43.2 / 43.4** | 2.0 / 2.3 | 1.0 / 1.1 | **52.5 / 53.3** |

### overlay 켬

| 모델 | fps | 변환 | 리사이즈 | 전처리 | **추론** | 후처리 | overlay | 발행 | **전체** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `yolop` | **19.2** | 0.9 / 1.2 | 0.6 / 0.7 | 12.5 / 14.2 | **23.5 / 23.7** | 2.1 / 2.5 | 7.4 / 8.7 | 1.6 / 1.9 | **48.7 / 51.8** |
| `yolopv2` | **18.8** | 0.9 / 1.2 | 0.6 / 0.7 | 6.8 / 8.3 | **29.3 / 29.5** | 2.1 / 2.5 | 4.9 / 7.3 | 1.6 / 2.0 | **46.3 / 49.6** |
| `hybridnets` | **15.7** | 0.9 / 1.1 | 0.5 / 0.6 | 9.9 / 11.0 | **36.6 / 37.3** | 1.8 / 2.1 | 9.6 / 10.7 | 1.4 / 1.6 | **60.7 / 63.1** |
| `tln` | **15.5** | 0.9 / 1.4 | 0.6 / 0.7 | 4.8 / 6.0 | **44.3 / 52.5** | 1.9 / 2.2 | 6.4 / 7.3 | 1.5 / 1.7 | **60.4 / 69.1** |

### 구간 설명

| 구간 | 하는 일 |
|---|---|
| 변환 | ROS 메시지를 이미지 배열로 바꾸고 BGR로 변환 |
| 리사이즈 | 1280×720을 640×360으로 축소 |
| 전처리 | 패딩, 색 순서 변경, 정규화 후 GPU로 전달 |
| 추론 | GPU에서 FP16 엔진 실행 |
| 후처리 | 패딩을 잘라내고 확률을 마스크로 변환 |
| overlay | 영상에 da와 ll을 합성 |
| 발행 | 결과를 ROS 토픽으로 전송 |
| 전체 | 프레임 1장의 노드 처리 시간 (변환~발행의 합) |

### 결과의 한계

- **YOLOv8m 미포함:** 함께 돌리면 fps가 더 낮아집니다.
- **책상 앞 영상:** 차를 세워 두고 잰 값입니다. 정확도와는 무관합니다.
- **가끔 생기는 지연:** 처리 간격 최댓값이 70~134ms까지 나옵니다. 원인은 확인하지 않았습니다.
- **원인을 확인하지 못한 값:** `yolopv2`는 전체 시간이 가장 짧은데 fps는 `yolop`보다 낮고, `hybridnets`는 overlay를 끄자 추론 시간도 줄었습니다.
- **Claude Code 영향:** 영향을 준 흔적은 보이지 않지만, 끄고 비교하지는 않았습니다.

원본 결과: `~/lane_perception/bench/1103_233948_summary.md` (overlay 켬), `~/lane_perception/bench/1103_234708_summary.md` (overlay 끔)

## 8. 파일 위치

| 위치 | 내용 |
|---|---|
| `~/foscar_ws/src/lane_perception/lane_perception/node.py` | 인지 노드 |
| `~/foscar_ws/src/lane_perception/lane_perception/viz_node.py` | 시각화 노드 |
| `~/foscar_ws/src/lane_perception/lane_perception/adapters.py` | 모델 4종의 전처리, 추론, 후처리 |
| `~/foscar_ws/src/lane_perception/lane_perception/export_trt.py` | ONNX 변환과 FP16 빌드 |
| `~/foscar_ws/src/lane_perception/lane_perception/verify.py` | 학습 코드, `.pt`, FP16 결과 비교 |
| `~/foscar_ws/src/lane_perception/lane_perception/bench.py` | 구간별 속도 측정 |
| `~/foscar_ws/src/lane_perception/launch/camera.launch.py` | 카메라 launch |
| `~/foscar_ws/src/lane_perception/launch/lane_perception.launch.py` | 인지 노드 launch |
| `~/foscar_ws/src/lane_perception/config/lane_perception.yaml` | 인지 노드 파라미터 |
| `~/lane_perception/engines/` | FP16 엔진과 ONNX |
| `~/lane_perception/release/` | 릴리스 가중치와 학습 코드 |
| `~/lane_perception/third_party/` | 원본 모델 저장소, 추가 패키지, `COMMITS.txt` |
| `~/lane_perception/bench/` | 속도 측정 결과 |
| `~/.bash_aliases` | `lane_` 명령 (`lane_perception commands` 구간) |
