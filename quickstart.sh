#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"
COMMAND="${1:-help}"
[[ $# -eq 0 ]] || shift
case "$COMMAND" in
  restore-resources) exec bash scripts/restore_resources.sh "$@" ;;
  prefetch) exec bash scripts/prefetch.sh "$@" ;;
  smoke-train) exec bash scripts/smoke_training.sh "$@" ;;
  repair-evaluate) exec bash scripts/repair_evaluate.sh "$@" ;;
  repair-train) exec bash scripts/repair_training.sh "$@" ;;
  expression-data) exec bash scripts/expression_training.sh data "$@" ;;
  expression-prepare) exec bash scripts/expression_training.sh prepare "$@" ;;
  expression-train) exec bash scripts/expression_training.sh train "$@" ;;
  native-prepare|native-train)
    source scripts/common.sh
    require_environment
    exec "$PYTHON" -m voice_lab.train_native "${COMMAND#native-}" "$@" ;;
  emotion-resources)
    source scripts/common.sh
    require_environment
    exec "$PYTHON" -m scripts.download_emotion_resources "$@" ;;
  vocoder-train) source scripts/common.sh; require_environment; exec "$PYTHON" -m voice_lab.train_vocoder "$@" ;;
  formal-train) exec bash scripts/formal_training.sh "$@" ;;
  oneclick) exec bash scripts/oneclick.sh "$@" ;;
  download) exec bash scripts/download.sh "$@" ;;
  setup-gpu) exec bash scripts/setup.sh gpu "$@" ;;
  setup-asr) exec bash scripts/setup_asr.sh "$@" ;;
  setup-client) exec bash scripts/setup.sh client "$@" ;;
  setup-backend) exec bash scripts/setup_backend.sh "$@" ;;
  backend) exec bash scripts/backend.sh "$@" ;;
  backend-transformers) export BACKEND_ENGINE=transformers; exec bash scripts/backend.sh "$@" ;;
  web|serve) exec bash scripts/serve.sh "$@" ;;
  client) exec bash scripts/client.sh "$@" ;;
  upload) exec bash scripts/upload_voice.sh "$@" ;;
  prepare) exec bash scripts/prepare.sh "$@" ;;
  train) exec bash scripts/train.sh "$@" ;;
  check) exec bash scripts/check.sh "$@" ;;
  *) cat <<'HELP'
VoiceDuplexLab 快速启动

历史数据链路实验       ./quickstart.sh oneclick
恢复已预上传的资源     ./quickstart.sh restore-resources /root/gpufree-share
无卡模式下载全部素材   ./quickstart.sh prefetch
已有环境仅下载素材     ./quickstart.sh download
服务器安装             ./quickstart.sh setup-gpu
安装本地客户端         ./quickstart.sh setup-client
检查环境与回归测试     ./quickstart.sh check --gpu
安装独立后台模型环境   ./quickstart.sh setup-backend
启动后台模型           ./quickstart.sh backend
复用训练环境启动后台   ./quickstart.sh backend-transformers
启动语音服务及网页     ./quickstart.sh web
网页地址               http://127.0.0.1:8765
上传参考音色           ./quickstart.sh upload voices/alice.wav
开始语音对话           ./quickstart.sh client --voice 音色ID
准备训练特征           ./quickstart.sh prepare data/train.jsonl
检查未训练问题与连续对话 ./quickstart.sh repair-evaluate
重新构建正常对话监督   ./quickstart.sh repair-train [目标累计步数]
准备情绪与哭笑录音     ./quickstart.sh expression-data --archive SMIIP_NV.zip
编译原生声学监督       ./quickstart.sh expression-prepare --config 配置 --manifest 清单 --prepared 特征目录
编译联合训练输入       ./quickstart.sh native-prepare --config 配置 --manifest 清单 --prepared 特征目录
联合训练与恢复         ./quickstart.sh native-train --config 配置 --prepared 特征目录 --output 输出目录 [--resume 断点]
下载补充情绪语句       ./quickstart.sh emotion-resources --output data/emotion/RAVDESS
训练或恢复声学适配器   ./quickstart.sh expression-train --config 配置 --prepared 特征目录 --output 输出目录
真实GPU训练验证        ./quickstart.sh smoke-train
正式训练及断点继续     ./quickstart.sh formal-train [--max-seconds 7200]
单步训练检查           ./quickstart.sh train --max-steps 1
开始或恢复训练         ./quickstart.sh train [--resume runs/duplex-lora/step-0000001]

完整步骤、多用户配置和数据格式请阅读 README.md。
HELP
    [[ "$COMMAND" == help ]] || exit 1 ;;
esac
