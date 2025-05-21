#!/bin/bash
args=()
for arg in "$@"; do
  if [[ "$arg" != "-fdebug-default-version=4" ]]; then
    args+=("$arg")
  fi
done
# 執行實際的 C 編譯器，使用過濾後的參數
# 確保 /usr/bin/cc 是您系統上實際的 C 編譯器路徑
exec /usr/bin/cc "${args[@]}"
