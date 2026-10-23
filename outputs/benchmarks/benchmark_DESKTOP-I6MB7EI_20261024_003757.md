# 모델 추론 벤치마크 (20261024_003757)

- device: AMD Radeon RX 9070 XT [ZLUDA]
- torch: 2.1.2+cu118, host: DESKTOP-I6MB7EI
- frame: D:/fine-tune/LimoPro_frames_20261023/frame_000200.png, batch=1, fp32, warmup 20, iters 200

| 모델 | 입력 | forward ms (p50 / p90) | forward FPS | end2end ms | end2end FPS | 첫 호출 ms |
|---|---|---|---|---|---|---|
| YOLOPv2 (38.9M) | 640x480 | 67.10 (66.01 / 72.51) | 14.9 | 76.67 | 13.0 | 11158 |
| TwinLiteNet+ Large (1.94M) | 640x480 | 25.99 (25.91 / 26.40) | 38.5 | 29.15 | 34.3 | 377 |
| TwinLiteNet+ Medium (0.48M) | 640x480 | 11.14 (10.38 / 14.41) | 89.7 | 16.51 | 60.6 | 369 |
| YOLOP (7.9M) | 640x480 | 16.33 (15.82 / 17.73) | 61.2 | 28.30 | 35.3 | 437 |
| TwinLiteNet+ Small (0.12M) | 640x480 | 9.00 (8.82 / 9.52) | 111.1 | 12.82 | 78.0 | 378 |
| HybridNets (12.8M) | 640x512 | 38.71 (38.31 / 41.07) | 25.8 | 46.47 | 21.5 | 476 |
| TwinLiteNet+ Nano (0.03M) | 640x480 | 7.41 (7.28 / 7.74) | 134.9 | 11.14 | 89.8 | 362 |
