#!/usr/bin/env bash
# oral-cut 转录 —— 把媒体文件转成 SRT（内部稿）+ words.json（逐字时间戳）
#
# 用法:
#   transcribe.sh [--srt|--txt] <媒体文件...>
#
# 适用环境: Linux / macOS / WSL 调 Windows 解释器
# Windows 解释器 / 引擎路径从 config.json 或环境变量读取（见 config.example.json）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG="${ORAL_CUT_CONFIG:-$ROOT_DIR/config.json}"

# ---- 读配置（有 jq 用 jq，没有就退回环境变量） --------------------------------
cfg() {  # cfg <key> <default>
  local key="$1" def="${2:-}"
  if [[ -f "$CONFIG" ]] && command -v jq >/dev/null 2>&1; then
    local v
    v="$(jq -r --arg k "$key" '.[$k] // empty' "$CONFIG" 2>/dev/null || true)"
    [[ -n "$v" ]] && { echo "$v"; return; }
  fi
  echo "$def"
}

PY_BIN="${ORAL_CUT_PYTHON:-$(cfg python "$(command -v python3 || echo python3)")}"
ENGINE="${ORAL_CUT_ENGINE:-$(cfg engine "$ROOT_DIR/batch_transcriber.py")}"
FMT="--srt"

[[ -f "$ENGINE" ]] || { echo "[ERR] 找不到转录引擎: $ENGINE" >&2; exit 1; }

# 是否把路径转成 Windows 形式（WSL 下调用 Windows 解释器时需要）
NEED_WINPATH=0
case "$PY_BIN" in
  *.exe|/mnt/*) NEED_WINPATH=1 ;;
esac

to_engine_path() {
  if [[ "$NEED_WINPATH" == "1" ]]; then
    if command -v wslpath >/dev/null 2>&1; then
      wslpath -w "$1"
    else
      echo "[ERR] 路径需要转换成 Windows 形式，但缺少 wslpath" >&2
      exit 1
    fi
  else
    echo "$1"
  fi
}

FILES=()
for arg in "$@"; do
  case "$arg" in
    --srt|--txt|-s|-t) FMT="$arg" ;;
    *) FILES+=("$arg") ;;
  esac
done

if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "用法: transcribe.sh [--srt|--txt] <媒体文件...>" >&2
  exit 1
fi

CONVERTED=()
for f in "${FILES[@]}"; do
  [[ -f "$f" ]] || { echo "[ERR] 文件不存在: $f" >&2; exit 1; }
  CONVERTED+=("$(to_engine_path "$f")")
done

for f in "${CONVERTED[@]}"; do
  echo "[..] 输入: $f"
done

# 引擎的 cwd 必须在引擎所在目录：模型按相对路径 './models/qwen3_asr' 查找。
# 若引擎是 Windows 进程，cwd 不能是 UNC 路径 —— 所以先 cd 到引擎目录的 WSL 对应路径。
cd "$ROOT_DIR"

exec "$PY_BIN" "$ENGINE" "$FMT" --output-mode 2 "${CONVERTED[@]}"
