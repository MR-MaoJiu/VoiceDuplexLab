#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
for ARGUMENT in "$@"; do
  if [[ "$ARGUMENT" == --download-only ]]; then
    exec bash scripts/prefetch.sh "$@"
  fi
done
[[ "$(uname -s)" == Linux ]] || { echo "请将项目复制到 Linux NVIDIA 服务器后执行训练入口"; exit 1; }
ensure_uv
bash scripts/setup.sh gpu
exec "$PYTHON" -m voice_lab.pipeline "$@"
