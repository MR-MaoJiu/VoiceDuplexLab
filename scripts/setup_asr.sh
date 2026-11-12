#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
require_environment
if command -v uv >/dev/null 2>&1; then
  uv pip install --python "$PYTHON" -e '.[asr]'
else
  "$PYTHON" -m pip install -e '.[asr]'
fi
exec "$PYTHON" -m voice_lab.asr "$@"
