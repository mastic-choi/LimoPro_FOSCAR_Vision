#!/usr/bin/env bash
# Ubuntu + NVIDIA RTX: 환경 설치 후 학습. 끊겨도 같은 명령을 다시 실행하면 이어서 한다.
#   bash run_train.sh                 (기본: hybridnets tln_large yolop yolopv2 순서)
#   bash run_train.sh --models hybridnets
set -e
cd "$(dirname "$0")"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
if [ ! -f .venv/.ready ]; then
  PY=""
  for c in python3.10 python3.11 python3.12 python3; do
    if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
  done
  [ -n "$PY" ] || { echo "python3이 없습니다: sudo apt install python3 python3-venv"; exit 1; }
  "$PY" -m venv .venv || { echo "venv 실패: sudo apt install python3-venv"; exit 1; }
  .venv/bin/python -m pip install -U pip
  .venv/bin/python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url "$TORCH_INDEX"
  .venv/bin/python -m pip install -r requirements_rtx.txt
  touch .venv/.ready
fi
# 절전 방지. 권한이 없어 실패하는 환경(SSH 등)에서는 그냥 실행
if command -v systemd-inhibit >/dev/null 2>&1 && systemd-inhibit --what=sleep:idle --who=LimoPro --why=check true >/dev/null 2>&1; then
  exec systemd-inhibit --what=sleep:idle --who=LimoPro --why=training .venv/bin/python scripts/run_train.py "$@"
fi
exec .venv/bin/python scripts/run_train.py "$@"
