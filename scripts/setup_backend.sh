#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
[[ "$(uname -s)" == Linux ]] || { echo "后台模型的 vLLM 服务需要 Linux NVIDIA 服务器"; exit 1; }
command -v uv >/dev/null 2>&1 || { echo "独立后台环境需要 uv，请先安装 uv"; exit 1; }
[[ -x .venv-backend/bin/python ]] || uv venv --python 3.11 .venv-backend
uv pip install --python .venv-backend/bin/python 'vllm==0.11.0'
echo "后台模型环境已安装，请运行 ./quickstart.sh backend"
