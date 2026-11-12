#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
require_environment
if [[ "$VOICE_CONFIG" == "$PROJECT_ROOT/configs/default.yaml" ]]; then
  if [[ -f configs/resources-local.yaml ]]; then
    export VOICE_CONFIG="$PROJECT_ROOT/configs/resources-local.yaml"
  elif [[ -f configs/bundle-runtime.yaml ]]; then
    export VOICE_CONFIG="$PROJECT_ROOT/configs/bundle-runtime.yaml"
  fi
fi
TARGET_STEPS="${1:-100}"
JOB=data/dialogue-repair-v5
"$PYTHON" -m voice_lab.repair_pipeline --config "$VOICE_CONFIG" --output "$JOB"
CONFIG="$JOB/config.yaml"
"$PYTHON" -m voice_lab.prepare "$JOB/eval.jsonl" --config "$CONFIG" --output "$JOB/eval_prepared" --resume
RESUME_ARGS=()
CHECKPOINT="$("$PYTHON" - <<'PY'
from voice_lab.pipeline import training_checkpoint
checkpoint,_=training_checkpoint('runs/dialogue-repair-v5',0)
if checkpoint: print(checkpoint)
PY
)"
if [[ -n "$CHECKPOINT" ]]; then
  RESUME_ARGS=(--resume "$CHECKPOINT")
fi
SAVED_STEP="$("$PYTHON" - <<'PY'
import json
from pathlib import Path
path=Path('runs/dialogue-repair-v5/progress.json')
print(json.loads(path.read_text())['step'] if path.is_file() else 0)
PY
)"
if [[ "$SAVED_STEP" -ge "$TARGET_STEPS" ]]; then
  echo "已保存第$SAVED_STEP步，达到本次目标。"
  exit 0
fi
exec "$PYTHON" -m voice_lab.train --config "$CONFIG" --manifest "$JOB/train.jsonl" \
  --eval-dir "$JOB/eval_prepared" "${RESUME_ARGS[@]}" --max-steps "$TARGET_STEPS" --max-seconds 2400
