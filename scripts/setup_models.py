#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
setup_models.py —— 下载 oral-cut 所需的语音识别模型

设计原则：**不默默下载几个 G 的东西。**
先列出要下的模型、体积、目标路径，等用户确认，再动手。

用法：
  python setup_models.py                    # 交互：列清单 → 等确认 → 下载
  python setup_models.py --size small       # CPU 档（Qwen3-ASR-0.6B）
  python setup_models.py --size full        # GPU 档（Qwen3-ASR-1.7B）
  python setup_models.py --yes              # 跳过确认（CI / 已明确授权）
  python setup_models.py --dry-run          # 只检测和列清单，不下载
  python setup_models.py --source hf        # 用 Hugging Face 而不是 ModelScope
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

# ---- 模型目录名（引擎按相对路径 models/qwen3_asr 查找，不要改） ----------------
MODELS_SUBDIR = os.path.join("models", "qwen3_asr")

# ---- 可选档位 ----------------------------------------------------------------
# 体积为估算值（bf16 权重 + tokenizer 等附属文件），用于给用户判断下载量。
TIERS = {
    "full": [
        ("Qwen/Qwen3-ASR-1.7B", "Qwen3-ASR-1.7B", 4.7),
        ("Qwen/Qwen3-ForcedAligner-0.6B", "Qwen3-ForcedAligner-0.6B", 1.8),
    ],
    "small": [
        ("Qwen/Qwen3-ASR-0.6B", "Qwen3-ASR-0.6B", 1.9),
        ("Qwen/Qwen3-ForcedAligner-0.6B", "Qwen3-ForcedAligner-0.6B", 1.8),
    ],
}

TIER_DESC = {
    "full": "GPU 档 —— 有 NVIDIA 显卡（8G+ 显存）时精度最高",
    "small": "CPU 档 —— 无显卡或显存紧张时使用，仍是逐字时间戳",
}

# 判断一个模型目录是否已完整下载：至少要有权重文件和 config
REQUIRED_HINTS = ("config.json",)


def human(n_gb: float) -> str:
    return f"{n_gb:.1f} GB"


def already_downloaded(target_dir: str) -> bool:
    """粗略判断模型是否已下载完整（有权重 + config）。"""
    if not os.path.isdir(target_dir):
        return False
    entries = os.listdir(target_dir)
    has_config = any(h in entries for h in REQUIRED_HINTS)
    has_weight = any(
        e.endswith(".safetensors") or e.endswith(".bin") or e.endswith(".gguf")
        for e in entries
    )
    # 有些模型权重在子目录，再兜一层
    if not has_weight:
        for e in entries:
            sub = os.path.join(target_dir, e)
            if os.path.isdir(sub) and any(
                f.endswith((".safetensors", ".bin", ".gguf")) for f in os.listdir(sub)
            ):
                has_weight = True
                break
    return has_config and has_weight


def detect_gpu() -> str | None:
    """尽力探测 GPU，仅用于给出建议（不强制）。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=8,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip().splitlines()[0]
    except Exception:
        pass
    return None


def detect_tool(source: str) -> tuple[bool, str]:
    """检查下载工具是否可用，返回 (可用, 安装命令)。"""
    if source == "modelscope":
        if shutil.which("modelscope"):
            return True, ""
        # 也允许以模块方式调用
        try:
            subprocess.run([sys.executable, "-c", "import modelscope"],
                           capture_output=True, timeout=20, check=True)
            return True, ""
        except Exception:
            return False, f"{sys.executable} -m pip install -U modelscope"
    else:
        if shutil.which("huggingface-cli") or shutil.which("hf"):
            return True, ""
        try:
            subprocess.run([sys.executable, "-c", "import huggingface_hub"],
                           capture_output=True, timeout=20, check=True)
            return True, ""
        except Exception:
            return False, f'{sys.executable} -m pip install -U "huggingface_hub[cli]"'


def download(repo_id: str, local_dir: str, source: str) -> bool:
    """下载一个模型到 local_dir。返回是否成功。"""
    os.makedirs(local_dir, exist_ok=True)
    if source == "modelscope":
        cmd = [sys.executable, "-m", "modelscope", "download",
               "--model", repo_id, "--local_dir", local_dir]
    else:
        cli = shutil.which("huggingface-cli") or shutil.which("hf")
        if cli:
            cmd = [cli, "download", repo_id, "--local-dir", local_dir]
        else:
            cmd = [sys.executable, "-m", "huggingface_hub.commands.huggingface_cli",
                   "download", repo_id, "--local-dir", local_dir]

    print(f"\n[..] 下载 {repo_id}\n     → {local_dir}")
    try:
        proc = subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n[!] 已取消")
        return False
    if proc.returncode != 0:
        print(f"[ERR] 下载失败（退出码 {proc.returncode}）")
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(
        description="下载 oral-cut 所需的语音识别模型（下载前会先让你确认）")
    ap.add_argument("--size", choices=sorted(TIERS), default=None,
                    help="档位：full（GPU）/ small（CPU）。不指定则交互选择")
    ap.add_argument("--dir", default=None,
                    help=f"模型根目录（默认脚本同级 {MODELS_SUBDIR}）")
    ap.add_argument("--source", choices=["modelscope", "hf"], default="modelscope",
                    help="下载源，默认 modelscope（国内快）")
    ap.add_argument("--yes", action="store_true", help="跳过确认")
    ap.add_argument("--dry-run", action="store_true", help="只列清单，不下载")
    args = ap.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    # 脚本可能在 scripts/ 下，模型要落到仓库根的 models/qwen3_asr
    root = os.path.dirname(base_dir.rstrip("/\\")) or base_dir
    models_root = args.dir or os.path.join(root, MODELS_SUBDIR)

    # ---- 1. 选档位 ----
    size = args.size
    if size is None:
        gpu = detect_gpu()
        rec = "full" if gpu else "small"
        print("=" * 64)
        print("oral-cut 模型安装")
        print("=" * 64)
        print(f"\n检测到显卡: {gpu if gpu else '无（或未安装 nvidia-smi）'}")
        print(f"推荐档位  : {rec} —— {TIER_DESC[rec]}\n")
        for key in ("full", "small"):
            print(f"  {key:<6} {TIER_DESC[key]}")
        print()
        ans = input(f"选择档位 [回车={rec}]: ").strip() or rec
        size = ans if ans in TIERS else rec

    wanted = TIERS[size]

    # ---- 2. 检查已下载 / 待下载 ----
    todo, done = [], []
    for repo_id, dirname, gb in wanted:
        target = os.path.join(models_root, dirname)
        (done if already_downloaded(target) else todo).append((repo_id, dirname, gb, target))

    # ---- 3. 列清单 ----
    print("\n" + "=" * 64)
    print(f"档位: {size} —— {TIER_DESC[size]}")
    print(f"模型目录: {models_root}")
    print("=" * 64)

    if done:
        print("\n已就绪（跳过）:")
        for repo_id, dirname, gb, target in done:
            print(f"  ✓ {dirname:<32} {human(gb):>8}")

    if not todo:
        print("\n所有模型都已就绪，无需下载。")
        return 0

    total = sum(gb for _, _, gb, _ in todo)
    print(f"\n将要下载 ({len(todo)} 个模型，合计约 {human(total)}):")
    for repo_id, dirname, gb, target in todo:
        print(f"  · {repo_id}")
        print(f"      → {target}")
        print(f"      约 {human(gb)}")

    # ---- 4. 确认 ----
    if not args.yes and not args.dry_run:
        print()
        print("-" * 64)
        print(f"即将占用磁盘约 {human(total)}，且需要一段时间下载。")
        try:
            answer = input("确认下载？[y/N]: ").strip().lower()
        except EOFError:
            answer = ""
        if answer not in ("y", "yes"):
            print("已取消，未下载任何文件。")
            return 0

    if args.dry_run:
        print("\n[dry-run] 未实际下载。")
        return 0

    # ---- 5. 检查工具 ----
    ok, install_cmd = detect_tool(args.source)
    if not ok:
        print(f"\n[ERR] 缺少下载工具（{args.source}）。请先安装：\n  {install_cmd}")
        return 1

    # ---- 6. 下载 ----
    failed = []
    for repo_id, dirname, gb, target in todo:
        if not download(repo_id, target, args.source):
            failed.append(repo_id)

    # ---- 7. 校验 ----
    print("\n" + "=" * 64)
    print("校验")
    print("=" * 64)
    all_ok = True
    for repo_id, dirname, gb, target in todo:
        if repo_id in failed:
            print(f"  ✗ {dirname}  下载失败")
            all_ok = False
        elif already_downloaded(target):
            print(f"  ✓ {dirname}")
        else:
            print(f"  ? {dirname}  文件不完整，建议重跑本脚本")
            all_ok = False

    if all_ok:
        print("\n完成。下一步：\n  bash scripts/transcribe.sh 你的视频.mp4")
        return 0
    print("\n有模型未就绪，请重跑本脚本补齐。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
