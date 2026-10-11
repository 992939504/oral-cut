#!/usr/bin/env bash
# oral-cut 首次运行引导：虚拟环境 → 依赖 → 探测系统 → 下载模型
#
# 用法:
#   bash scripts/setup.sh              # 交互，每步先问过你
#   bash scripts/setup.sh --yes        # 跳过确认（CI / 已明确授权）
#   bash scripts/setup.sh --size small # 直接指定档位
#
# 原则：**不默默装东西。** 每一步都先说清楚做什么、占多少空间、放哪里。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
VENV="$ROOT_DIR/.venv"
PY="$VENV/bin/python"
PIP="$VENV/bin/pip"

ASSUME_YES=0
if [[ "${1:-}" == "--yes" ]]; then ASSUME_YES=1; fi

ask() {
  [[ "$ASSUME_YES" == "1" ]] && return 0
  printf "%s [y/N]: " "$1"
  read -r _ans || return 1
  case "$_ans" in [yY]|[yY][eE][sS]) return 0 ;; *) return 1 ;; esac
}

step() {
  echo
  echo "────────────────────────────────────────────────────────"
  echo "$1"
  echo "────────────────────────────────────────────────────────"
}

echo "════════════════════════════════════════════════════════"
echo " oral-cut 首次运行引导"
echo "════════════════════════════════════════════════════════"
echo "仓库根目录: $ROOT_DIR"

# ---------------------------------------------------------------- 0. 现状
step "[0/4] 检查当前状态"
[ -d "$VENV" ]                       && echo "  · 虚拟环境 : 已有" || echo "  · 虚拟环境 : 无"
"$PY" -c "import qwen_asr" 2>/dev/null && echo "  · 引擎依赖 : 已装" || echo "  · 引擎依赖 : 未装"
[ -d "$ROOT_DIR/models/qwen3_asr" ]  && echo "  · 模型     : 已有" || echo "  · 模型     : 无"
if command -v ffmpeg >/dev/null 2>&1; then
  echo "  · ffmpeg   : $(ffmpeg -version 2>/dev/null | head -1 | awk '{print $3}')"
else
  echo "  · ffmpeg   : ⚠ 未找到"
fi

if [ ! -d "$VENV" ] && python3 -c "import qwen_asr" 2>/dev/null; then
  echo
  echo "  ⚠ 你的系统 Python 里已经装了引擎依赖。本脚本仍会建独立虚拟环境——"
  echo "    不往系统环境里再塞东西，将来删掉 .venv/ 就干净了。"
fi

# ---------------------------------------------------------------- ffmpeg
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo
  echo "[!] 缺 ffmpeg —— 转录和剪切都靠它。先装好再重跑本脚本："
  echo "      Debian/Ubuntu : sudo apt install ffmpeg"
  echo "      macOS         : brew install ffmpeg"
  echo "      Windows       : 下载后把 ffmpeg.exe（和 ffprobe.exe）放进 PATH"
  exit 1
fi

# ---------------------------------------------------------------- 1. venv
step "[1/4] 本地虚拟环境"
if [ -d "$VENV" ]; then
  echo "已存在，跳过：$VENV"
else
  echo "将创建一个独立的虚拟环境，不动你系统里的 Python："
  echo "    $VENV"
  echo "将来不想要了，直接删掉这个目录就行。"
  ask "可以吗？" || { echo "已取消，什么都没装。"; exit 0; }
  python3 -m venv "$VENV"
  echo "✓ 已创建"
fi

# ---------------------------------------------------------------- 2. 依赖
step "[2/4] 安装依赖"
if "$PY" -c "import qwen_asr, torch" 2>/dev/null; then
  echo "依赖已就绪，跳过。"
else
  echo "将安装 requirements.txt —— 含 torch，体积较大，可能要几分钟。"
  echo "    $PIP install -r $ROOT_DIR/requirements.txt"
  ask "开始安装？" || { echo "已取消，什么都没装。"; exit 0; }
  "$PIP" install --upgrade pip >/dev/null
  "$PIP" install -r "$ROOT_DIR/requirements.txt"
  echo "✓ 依赖装好"
fi

# ---------------------------------------------------------------- 3. 探测
step "[3/4] 探测系统 → 推荐模型"
GPU="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || true)"
if [ -n "$GPU" ]; then
  echo "  GPU      : $GPU"
else
  echo "  GPU      : 无独显（或未装 nvidia-smi）"
fi
echo "  磁盘余量 : $(df -h "$ROOT_DIR" | awk 'NR==2 {print $4}') 可用"

# ---------------------------------------------------------------- 4. 模型
step "[4/4] 下载模型"
echo "接下来会列出要下载的模型、体积、目标路径，等你确认后才真正下载（默认 ModelScope）。"
echo

if [[ "$ASSUME_YES" == "1" ]]; then
  exec "$PY" "$ROOT_DIR/scripts/setup_models.py" --yes "${@:2}"
else
  exec "$PY" "$ROOT_DIR/scripts/setup_models.py" "$@"
fi
