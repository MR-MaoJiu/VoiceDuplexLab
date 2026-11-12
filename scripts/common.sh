#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/models/huggingface}"
export VOICE_CONFIG="${VOICE_CONFIG:-$PROJECT_ROOT/configs/default.yaml}"
PYTHON="$PROJECT_ROOT/.venv/bin/python"
require_environment() {
  if [[ ! -x "$PYTHON" ]]; then
    echo "请先运行 ./quickstart.sh setup-gpu 或 ./quickstart.sh setup-client"
    exit 1
  fi
}

# 下载缓存随项目放到数据盘，便于无卡模式准备后复用。
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PROJECT_ROOT/.cache/uv}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-180}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-180}"
ensure_uv() {
  if ! command -v uv >/dev/null 2>&1; then
    if [[ ! -x "$PROJECT_ROOT/.bootstrap/bin/uv" ]]; then
      if python3 -m pip --version >/dev/null 2>&1; then
        python3 -m pip install --target "$PROJECT_ROOT/.bootstrap" 'uv==0.9.10'
      else
        python3 "$PROJECT_ROOT/scripts/bootstrap_uv.py"
      fi
    fi
    export PATH="$PROJECT_ROOT/.bootstrap/bin:$PATH"
  fi
}


# 默认启动优先加载已验收配置，显式指定的配置仍由调用者控制。
select_runtime_config() {
  if [[ "$VOICE_CONFIG" == "$PROJECT_ROOT/configs/default.yaml" ]]; then
    for PROFILE in repair-ready.yaml bundle-runtime.yaml repair-candidate.yaml; do
      if [[ -f "$PROJECT_ROOT/configs/$PROFILE" ]]; then
        export VOICE_CONFIG="$PROJECT_ROOT/configs/$PROFILE"
        break
      fi
    done
  fi
}
