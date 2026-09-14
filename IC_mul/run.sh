#!/usr/bin/env bash
# 启动 workflow_by_code.py 的封装脚本
# 必须在 qlib_me 环境运行；强制 fork 启动方式避免 numpy 2.0 下多进程报错。
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"

PY=/home/fei/miniconda3/envs/qlib_me/bin/python
SCRIPT=workflow_by_code.py
LOG=/tmp/wf_run.log

# 默认参数（命令行可覆盖，例如 ./run.sh --baseline 或 ./run.sh --turnover-weight 0）
ARGS=(
  --train 2017-01-01,2023-12-31
  --valid 2024-01-01,2024-12-31
  --test  2025-01-01,2026-06-30
  --cost 0.002
  --turnover-weight 1.0
  --rounds 1000
  --early-stop 100
  --topk 50
  --n-drop 5
  "$@"
)

echo "启动: $PY $SCRIPT ${ARGS[*]}"
echo "日志: $LOG"

JOBLIB_START_METHOD=fork LOKY_START_METHOD=fork \
  nohup "$PY" "$SCRIPT" "${ARGS[@]}" > "$LOG" 2>&1 &

echo "PID=$!"
echo "查看进度: tail -f 40 $LOG"
