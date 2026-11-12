#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
require_environment
"$PYTHON" -m voice_lab.formal_data
FORMAL_CONFIG="$PROJECT_ROOT/configs/formal-budget.yaml"
if [[ ! -f "$FORMAL_CONFIG" ]]; then
  "$PYTHON" - <<'PYTHON'
import os
from pathlib import Path
from voice_lab.config import ROOT, load_config
from voice_lab.pipeline import save_yaml
config=load_config(ROOT / 'configs/resources-local.yaml')
config['backend']['weights_path']=str(ROOT / 'models/Qwen3-8B')
config['data']={'raw_dir':str(ROOT / 'data/raw/smoothconv-full-2ch')}
config['training'].update(save_every=25,keep_checkpoints=3,min_free_gib=10,
                          gradient_accumulation=4,instruction_weight=32,
                          compile_llm_forward=False,eval_every=100)
shared=Path('/root/gpufree-share')
backup=os.environ.get('VOICE_CHECKPOINT_BACKUP_DIR')
if backup or shared.is_dir():
    config['training']['backup_dir']=backup or str(shared/'VoiceDuplexLab-formal-checkpoints')
save_yaml(ROOT/'configs/formal-budget.yaml',config)
PYTHON
fi
if "$PYTHON" - <<'PYTHON'
from voice_lab.config import ROOT,load_config
from voice_lab.model import model_fingerprint
from voice_lab.prepare import compiler_version
import json
path=ROOT/'runs/compiler_benchmark.json'
value=json.loads(path.read_text()) if path.is_file() else {}
config=load_config(ROOT/'configs/formal-budget.yaml')
valid=value.get('compiler')==compiler_version(config) and value.get('fingerprint')==model_fingerprint(config)
raise SystemExit(0 if valid else 3)
PYTHON
then
  :
else
  "$PYTHON" -m voice_lab.compiler_benchmark data/formal/full-corpus-v1/real_train.jsonl --config "$FORMAL_CONFIG"
fi
"$PYTHON" - <<'PYTHON'
import json
from voice_lab.config import ROOT,load_config
from voice_lab.pipeline import save_yaml
report=json.loads((ROOT/'runs/compiler_benchmark.json').read_text())
path=ROOT/'configs/formal-budget.yaml'
config=load_config(path)
config['training']['compile_llm_forward']=not report['identical']
save_yaml(path,config)
print('特征编译使用快速路径' if report['identical'] else '一致性检查未通过，继续使用原始编译路径',flush=True)
PYTHON
exec "$PYTHON" -m voice_lab.pipeline --config "$FORMAL_CONFIG" \
  --corpus-dir data/formal/full-corpus-v1 --instructions 136 --max-windows 0 \
  --max-steps 0 --epochs 2 --eval-samples 32 --lazy --max-seconds 7200 "$@"
