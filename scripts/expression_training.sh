#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
require_environment
MODE="${1:-help}"
[[ $# -eq 0 ]] || shift
case "$MODE" in
  data) exec "$PYTHON" -m voice_lab.emotion_data "$@" ;;
  prepare|train) exec "$PYTHON" -m voice_lab.train_expression "$MODE" "$@" ;;
  *) echo "用法：./quickstart.sh expression-data [--archive SMIIP_NV.zip --output 数据目录]"
     echo "准备：./quickstart.sh expression-prepare --config 配置 --manifest 清单 --prepared 特征目录"
     echo "训练：./quickstart.sh expression-train --config 配置 --prepared 特征目录 --output 输出目录 [--resume 断点]"
     exit 1 ;;
esac
