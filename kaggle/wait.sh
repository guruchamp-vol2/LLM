#!/usr/bin/env bash
# Wait for a Kaggle kernel to finish: prints status changes, exits when complete/error.
# Usage: bash kaggle/wait.sh guruchampvol2/llm-tpu-smoke
set -a; . /c/Users/dhruv/.kaggle/.env; set +a
KG=/c/Users/dhruv/.venvs/kaggle/Scripts/kaggle.exe
last=""
while true; do
  s=$($KG kernels status "$1" 2>&1 | grep -oiE 'status "?[a-z_.]+' | tail -1)
  [ "$s" != "$last" ] && echo "$(date +%H:%M:%S) $s" && last="$s"
  echo "$s" | grep -qiE 'complete|error|cancel' && break
  sleep 30
done
