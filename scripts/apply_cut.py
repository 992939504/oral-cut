#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apply_cut.py —— 按已确认的剪辑清单执行剪切（音画一起剪）

用法:
  python apply_cut.py <plan.json> --media 原片.mp4 [--out 目录] [--suffix cut-v1]

铁律（见 docs/rules.md 规则 1）:
  · 输出永远是**新文件名**（-cut-v1、-cut-v2…），**永不覆盖**
  · plan.json 里 "confirmed": false 时**拒绝执行** —— 内容删减权在人
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys


def media_duration(media: str) -> float:
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
           "-of", "default=nw=1:nk=1", media]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    return float(p.stdout.strip())


def keep_segments(cuts: list[dict], duration: float,
                  min_keep: float = 0.25) -> list[tuple[float, float]]:
    """剪点 → 保留段（区间补集）。太短的保留段并进相邻剪点，避免碎片。"""
    spans = sorted((c["start"], c["end"]) for c in cuts if c["end"] > c["start"])
    keep, cursor = [], 0.0
    for s, e in spans:
        s = max(s, cursor)
        if s - cursor >= min_keep:
            keep.append((cursor, s))
        cursor = max(cursor, e)
    if duration - cursor >= min_keep:
        keep.append((cursor, duration))
    return keep


def unique_path(base_dir: str, stem: str, suffix: str, ext: str) -> str:
    """永不覆盖：同名就往上加序号。"""
    cand = os.path.join(base_dir, f"{stem}-{suffix}{ext}")
    n = 2
    while os.path.exists(cand):
        cand = os.path.join(base_dir, f"{stem}-{suffix}-r{n}{ext}")
        n += 1
    return cand


def build_filter(segs: list[tuple[float, float]]) -> str:
    """构造 filter_complex：每段 trim，然后 concat（音画同步）。"""
    parts = []
    pairs = []
    for i, (s, e) in enumerate(segs):
        parts.append(
            f"[0:v]trim=start={s:.3f}:end={e:.3f},setpts=PTS-STARTPTS[v{i}]"
        )
        parts.append(
            f"[0:a]atrim=start={s:.3f}:end={e:.3f},asetpts=PTS-STARTPTS[a{i}]"
        )
        pairs.append(f"[v{i}][a{i}]")
    # 注意：[vN][aN] 串后直接接 concat，中间不能加 ';'
    chain = ";".join(parts) + ";" + "".join(pairs) + f"concat=n={len(segs)}:v=1:a=1[outv][outa]"
    return chain


def main() -> int:
    ap = argparse.ArgumentParser(description="按确认过的清单执行剪切")
    ap.add_argument("plan", help="plan_cut.py 产出的清单 JSON")
    ap.add_argument("--media", required=True, help="原始媒体文件")
    ap.add_argument("--out", default=None, help="输出目录（默认为原片所在目录）")
    ap.add_argument("--suffix", default="cut-v1", help="输出文件后缀，默认 cut-v1")
    ap.add_argument("--dry-run", action="store_true", help="只打印将要执行的命令")
    args = ap.parse_args()

    if not os.path.isfile(args.plan):
        print(f"[ERR] 找不到清单: {args.plan}", file=sys.stderr)
        return 1
    if not os.path.isfile(args.media):
        print(f"[ERR] 找不到媒体: {args.media}", file=sys.stderr)
        return 1

    with open(args.plan, encoding="utf-8") as f:
        plan = json.load(f)

    if not plan.get("confirmed"):
        print("[拒绝] 清单里 \"confirmed\" 不是 true。", file=sys.stderr)
        print("       内容删减权在人手里 —— 确认后把该字段改成 true 再跑。", file=sys.stderr)
        print("       见 docs/rules.md 规则 0。", file=sys.stderr)
        return 2

    cuts = plan.get("cuts", [])
    warned = [c for c in cuts if c.get("level_verdict") == "possible_speech"]
    if warned:
        print(f"[!] 有 {len(warned)} 处标记为「可能是漏识别」：")
        for c in warned:
            print(f"      {c['start']:.2f}–{c['end']:.2f}  {c['text']}")
        print("    建议从清单里剔除这些再执行。")

    duration = media_duration(args.media)
    segs = keep_segments(cuts, duration)

    if not segs:
        print("[ERR] 保留段为空 —— 清单把整条片子都剪了？检查一下。", file=sys.stderr)
        return 1

    total_keep = sum(e - s for s, e in segs)
    print(f"[..] 原片 {duration:.2f}s → 保留 {len(segs)} 段、共 {total_keep:.2f}s"
          f"（剪掉 {duration - total_keep:.2f}s）")

    src_dir = os.path.dirname(os.path.abspath(args.media))
    out_dir = args.out or src_dir
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.media))[0]
    out_path = unique_path(out_dir, stem, args.suffix, ".mp4")

    cmd = ["ffmpeg", "-y", "-v", "error", "-i", args.media,
           "-filter_complex", build_filter(segs),
           "-map", "[outv]", "-map", "[outa]",
           "-c:v", "libx264", "-preset", "medium", "-crf", "20",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
           out_path]

    if args.dry_run:
        print("\n[dry-run] 将执行:")
        print(" ".join(f'"{c}"' if " " in c else c for c in cmd))
        return 0

    print(f"[..] 输出: {out_path}")
    p = subprocess.run(cmd)
    if p.returncode != 0:
        print(f"[ERR] ffmpeg 失败（退出码 {p.returncode}）", file=sys.stderr)
        return 1

    print(f"\n[ok] 完成: {out_path}")
    print("     下一步（必做）: 对成片重跑一遍识别，与源稿逐段比对，确认没吞字。")
    print("     见 docs/rules.md 规则 6。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
