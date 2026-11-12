#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
require_environment
[[ $# -gt 0 ]] || { echo "用法 ./quickstart.sh prepare data/train.jsonl"; exit 1; }
"$PYTHON" -m voice_lab.validate "$1"
exec "$PYTHON" -m voice_lab.prepare "$@"
