#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
select_runtime_config
require_environment
exec "$PYTHON" -m voice_lab.server "$@"
