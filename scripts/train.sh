#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
require_environment
exec "$PYTHON" -m voice_lab.train "$@"
