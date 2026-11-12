#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
select_runtime_config
if [[ "${BACKEND_ENGINE:-vllm}" == transformers ]]; then
  require_environment
  export CUDA_VISIBLE_DEVICES="${BACKEND_GPU:-0}"
  exec "$PYTHON" -m voice_lab.hf_backend --config "$VOICE_CONFIG" --port "${BACKEND_PORT:-8001}" "$@"
fi
[[ -x .venv-backend/bin/vllm ]] || { echo "请先运行 ./quickstart.sh setup-backend"; exit 1; }
export CUDA_VISIBLE_DEVICES="${BACKEND_GPU:-0}"
BACKEND_MODEL="${BACKEND_MODEL:-Qwen/Qwen3-8B}"
REVISION_ARGS=()
if [[ -n "${BACKEND_REVISION:-}" ]]; then
  REVISION_ARGS=(--revision "$BACKEND_REVISION")
elif [[ "$BACKEND_MODEL" == Qwen/Qwen3-8B ]]; then
  REVISION_ARGS=(--revision b968826d9c46dd6066d109eabc6255188de91218)
fi
exec .venv-backend/bin/vllm serve "$BACKEND_MODEL" "${REVISION_ARGS[@]}" \
  --host 127.0.0.1 --port "${BACKEND_PORT:-8001}" \
  --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser deepseek_r1 \
  --gpu-memory-utilization "${BACKEND_MEMORY_FRACTION:-0.35}" \
  --max-model-len "${BACKEND_MAX_LENGTH:-8192}" "$@"
