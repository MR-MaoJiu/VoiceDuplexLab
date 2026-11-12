#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
ensure_uv
bash scripts/setup.sh download
exec "$PYTHON" -m voice_lab.pipeline --download-only "$@"
