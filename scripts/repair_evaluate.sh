#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
require_environment
CONFIG="${VOICE_CONFIG}"
if [[ "$CONFIG" == "$PROJECT_ROOT/configs/default.yaml" ]]; then
  for PROFILE in repair-ready.yaml repair-candidate.yaml bundle-runtime.yaml; do
    if [[ -f "configs/$PROFILE" ]]; then
      CONFIG="$PROJECT_ROOT/configs/$PROFILE"
      break
    fi
  done
fi
ADAPTER="$("$PYTHON" - "$CONFIG" <<'PY'
import sys
from voice_lab.config import load_config
print(load_config(sys.argv[1])['model'].get('adapter') or 'none')
PY
)"
exec "$PYTHON" -m voice_lab.dialogue_quality --config "$CONFIG" \
  --data "${REPAIR_DATA:-data/dialogue-repair-v5}" --adapter "$ADAPTER" \
  --output "${REPAIR_REPORT:-runs/repair-quality-local}" "$@"
