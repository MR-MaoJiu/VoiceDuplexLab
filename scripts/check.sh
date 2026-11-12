#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
require_environment
"$PYTHON" -m compileall -q voice_lab
"$PYTHON" -m pytest -q tests
"$PYTHON" -m voice_lab.doctor "$@"
