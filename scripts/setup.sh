#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
MODE="${1:-gpu}"
case "$MODE" in gpu|client|download) ;; *) echo "参数应为 gpu、client 或 download"; exit 1 ;; esac
if [[ "$MODE" == gpu && "$(uname -s)" != Linux ]]; then
  echo "GPU 服务和训练请在 Linux NVIDIA 服务器安装，Mac 请运行 setup-client"
  exit 1
fi
EXTRAS="$MODE,test"
[[ "$MODE" != gpu ]] || EXTRAS="gpu,asr,test"
[[ "$MODE" != download ]] || EXTRAS="test"
if command -v uv >/dev/null 2>&1; then
  [[ -x "$PYTHON" ]] || uv venv --python "${VOICE_PYTHON_VERSION:-3.11}" "$PROJECT_ROOT/.venv"
  uv pip install --python "$PYTHON" -e ".[${EXTRAS}]"
else
  VOICE_PYTHON="${VOICE_PYTHON:-python3.11}"
  command -v "$VOICE_PYTHON" >/dev/null 2>&1 || { echo "请安装 Python 3.11 或 uv"; exit 1; }
  [[ -x "$PYTHON" ]] || "$VOICE_PYTHON" -m venv "$PROJECT_ROOT/.venv"
  "$PYTHON" -m pip install --upgrade pip
  "$PYTHON" -m pip install -e ".[${EXTRAS}]"
fi
if [[ "$MODE" == gpu ]]; then
  if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "语速调整需要FFmpeg，请安装后使用：sudo apt-get install -y ffmpeg"
  fi
  "$PYTHON" -m voice_lab.doctor --gpu
else
  "$PYTHON" -m voice_lab.doctor
fi
