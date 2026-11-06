"""경로 설정. 기본값은 동아리 RTX PC 기준이고, 환경변수로 바꿀 수 있다.

  TRACK_LABEL_WORK  지도(map/)와 결과(results/)를 두는 작업 폴더
  TRACK_LABEL_RAW   위치를 찾을 raw 사진 폴더 (*.jpg, 1280x720)
  LIMO_TRAIN_ROOT   LimoPro_train 폴더 (scripts/train_seg.py, 2차 가중치, 학습 데이터셋)
  HYBRIDNETS_CKPT   비교·검증에 쓸 HybridNets 가중치
"""
import os
WORK = os.environ.get("TRACK_LABEL_WORK", "/home/foscar/20261106")
RAW_IMAGES = os.environ.get("TRACK_LABEL_RAW", "/media/foscar/80BD-730A/dataset_raw/20261106/session_180809/images")
LIMO_TRAIN = os.environ.get("LIMO_TRAIN_ROOT", "/home/foscar/2026AutoRace/LimoPro_train")
HYBRIDNETS_CKPT = os.environ.get("HYBRIDNETS_CKPT", LIMO_TRAIN + "/outputs/train/hybridnets_v2_seed0/best_da.pt")
TRAIN_DATA = os.environ.get("TRACK_LABEL_TRAIN_DATA", LIMO_TRAIN + "/dataset/ego_v2_ys_split/train")
