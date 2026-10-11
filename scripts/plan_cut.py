#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plan_cut.py —— 从逐字时间戳算出剪点清单（只算，不剪）

用法:
  python plan_cut.py <words.json> [--media 原片.mp4] [--silence 0.9] [--out plan.json]
  python plan_cut.py video.words.json --media video.mp4 --out video.plan.json

产出:
  · 控制台：可读清单（起始 → 结束 ｜ 内容 ｜ 原因）
  · JSON：机器可读，交给 apply_cut.py 执行

⚠️ 本脚本**只出清单，不剪**。剪点必须经人确认后才交给 apply_cut.py。
   内容删减权在人手里——见 docs/rules.md 规则 0。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys


# ----------------------------------------------------------------------------
# 工具
# ----------------------------------------------------------------------------

def fmt(t: float) -> str:
    """秒 → m:ss.xx"""
    m, s = divmod(max(t, 0.0), 60)
    return f"{int(m)}:{s:05.2f}"


def load_words(path: str) -> list[dict]:
    """读逐字时间戳。兼容裸数组和契约格式 {language_code, words}。"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("words", [])
    if not isinstance(data, list) or not data:
        raise ValueError(f"没读到逐字时间戳: {path}")

    words = []
    for w in data:
        try:
            text = str(w["text"])
            start = float(w["start"])
            end = float(w["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end <= start:
            continue
        words.append({"text": text, "start": start, "end": end})
    words.sort(key=lambda x: x["start"])
    return words


def probe_mean_volume(media: str, start: float, end: float,
                      timeout: int = 90) -> float | None:
    """量一段区间的 mean_volume(dB)。失败返回 None。"""
    if end <= start:
        return None
    cmd = ["ffmpeg", "-v", "info", "-ss", f"{start:.3f}",
           "-t", f"{end - start:.3f}", "-i", media,
           "-af", "volumedetect", "-f", "null", "-"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except Exception:
        return None
    m = re.search(r"mean_volume:\s*(-?[\d.]+)\s*dB", p.stderr)
    return float(m.group(1)) if m else None


def media_duration(media: str) -> float | None:
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
           "-of", "default=nw=1:nk=1", media]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return float(p.stdout.strip())
    except Exception:
        return None


# ----------------------------------------------------------------------------
# 找剪点
# ----------------------------------------------------------------------------

def find_gap_cuts(words: list[dict], min_gap: float, media: str | None) -> list[dict]:
    """相邻字间隙 ≥ min_gap 的候选剪点。

    主判据 = 字级间隙（天然不落在字上）。边界就是「前字的 end」到「后字的 start」，
    所以不需要额外的收缩步骤 —— 这正是它比 silencedetect 硬的地方。
    """
    cuts = []
    for a, b in zip(words, words[1:]):
        gap = b["start"] - a["end"]
        if gap < min_gap:
            continue
        cuts.append({
            "type": "silence",
            "start": round(a["end"], 3),
            "end": round(b["start"], 3),
            "dur": round(gap, 3),
            "text": f"『{a['text']}』与『{b['text']}』之间",
            "reason": f"字间隙 {gap:.2f}s ≥ {min_gap}s",
        })
    return cuts


def find_edge_cuts(words: list[dict], min_gap: float,
                   media: str | None) -> list[dict]:
    """片头 / 片尾静音。"""
    cuts = []
    head = words[0]["start"]
    if head >= min_gap:
        cuts.append({
            "type": "head", "start": 0.0, "end": round(head, 3),
            "dur": round(head, 3), "text": "（片头静音）",
            "reason": f"开头静音 {head:.2f}s ≥ {min_gap}s",
        })
    dur = media_duration(media) if media else None
    tail_start = words[-1]["end"]
    if dur and (dur - tail_start) >= min_gap:
        cuts.append({
            "type": "tail", "start": round(tail_start, 3), "end": round(dur, 3),
            "dur": round(dur - tail_start, 3), "text": "（片尾静音）",
            "reason": f"结尾静音 {dur - tail_start:.2f}s ≥ {min_gap}s",
        })
    return cuts


def find_repeat_hints(words: list[dict], max_len: int = 12) -> list[dict]:
    """疑似重复（口吃 / 卡壳 / 说了两遍）。

    机械特征，**仅供参考**：只找相邻两段完全相同的字序列。
    真伪判断归人 —— 带情绪的第一遍可能是该留的那一遍（规则 7）。
    """
    n = len(words)
    hints = []
    i = 0
    while i < n:
        matched = False
        for L in range(min(max_len, n - i), 1, -1):
            seg = [w["text"] for w in words[i:i + L]]
            nxt = [w["text"] for w in words[i + L:i + 2 * L]]
            if len(nxt) == L and seg == nxt:
                hints.append({
                    "type": "repeat_hint",
                    "start": round(words[i]["start"], 3),
                    "end": round(words[i + L - 1]["end"], 3),
                    "dur": round(words[i + L - 1]["end"] - words[i]["start"], 3),
                    "text": "".join(seg),
                    "reason": f"相邻两段完全相同（{L} 字），疑似重说",
                })
                i += L
                matched = True
                break
        i += 1 if not matched else 0
        if matched:
            continue
    return hints


# ----------------------------------------------------------------------------
# 电平交叉验证
# ----------------------------------------------------------------------------

def cross_check(cuts: list[dict], words: list[dict], media: str | None,
                margin_db: float = 6.0) -> None:
    """就地给每个剪点加 level 结论。

    说话段电平取全部字覆盖区间的 mean（近似：整片），剪点区间单独量。
    剪点电平若不显著低于语音 → 标 possible_speech（可能是识别漏掉的话，不剪）。
    """
    if not media:
        return

    # 语音段参考电平：取若干字密集处采样
    speech_samples = []
    for w in words:
        if w["end"] - w["start"] >= 0.15:
            v = probe_mean_volume(media, w["start"], w["end"])
            if v is not None:
                speech_samples.append(v)
        if len(speech_samples) >= 12:
            break
    speech_mean = sum(speech_samples) / len(speech_samples) if speech_samples else None

    for c in cuts:
        if c["type"] in ("head", "tail") or c["dur"] < 0.15:
            continue
        lv = probe_mean_volume(media, c["start"], c["end"])
        c["level_db"] = None if lv is None else round(lv, 1)
        if lv is None or speech_mean is None:
            c["level_verdict"] = "unknown"
            continue
        if lv >= speech_mean - margin_db:
            c["level_verdict"] = "possible_speech"
            c["reason"] += f"；⚠ 电平 {lv:.1f}dB 接近语音({speech_mean:.1f}dB)，可能是漏识别，建议不剪"
        else:
            c["level_verdict"] = "quiet"
            c["reason"] += f"；电平 {lv:.1f}dB 明显低于语音，确认静音"


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="算出剪辑清单（只算，不剪）")
    ap.add_argument("words", help="逐字时间戳文件（*.words.json）")
    ap.add_argument("--media", default=None, help="原始媒体文件（用于电平交叉验证 / 片尾时长）")
    ap.add_argument("--silence", type=float, default=0.9,
                    help="静音剪点阈值秒数，默认 0.9")
    ap.add_argument("--out", default=None, help="清单输出路径（默认 <words>.plan.json）")
    ap.add_argument("--no-level", action="store_true", help="跳过电平交叉验证（快很多）")
    ap.add_argument("--repeats", action="store_true", help="附带疑似重复提示")
    args = ap.parse_args()

    if not os.path.isfile(args.words):
        print(f"[ERR] 找不到: {args.words}", file=sys.stderr)
        return 1
    if args.media and not os.path.isfile(args.media):
        print(f"[ERR] 找不到媒体文件: {args.media}", file=sys.stderr)
        return 1

    words = load_words(args.words)
    print(f"[..] 读到 {len(words)} 个字，覆盖 {fmt(words[0]['start'])} → {fmt(words[-1]['end'])}")

    cuts = find_edge_cuts(words, args.silence, args.media)
    cuts += find_gap_cuts(words, args.silence, args.media)

    if not args.no_level and args.media:
        print(f"[..] 电平交叉验证（{len(cuts)} 处候选，需要一点时间）…")
        cross_check(cuts, words, args.media)

    hints = find_repeat_hints(words) if args.repeats else []

    cuts.sort(key=lambda c: c["start"])

    # ---- 打印清单 ----
    print()
    print("=" * 72)
    print(f"剪辑清单（静音阈值 {args.silence}s）—— {len(cuts)} 条")
    print("=" * 72)
    if not cuts:
        print("（没有发现可剪点）")
    for i, c in enumerate(cuts, 1):
        flag = "  ⚠ 需人确认" if c.get("level_verdict") == "possible_speech" else ""
        print(f"{i:>3}. {fmt(c['start'])} → {fmt(c['end'])}  ({c['dur']:.2f}s)")
        print(f"     {c['text']}{flag}")
        print(f"     {c['reason']}")

    if hints:
        print()
        print("-" * 72)
        print(f"疑似重复（仅供参考，不是剪点）—— {len(hints)} 条")
        print("-" * 72)
        for i, h in enumerate(hints, 1):
            print(f"{i:>3}. {fmt(h['start'])} → {fmt(h['end'])}  「{h['text']}」")

    print()
    print("⚠  以上只是**清单**。内容删减权在人 —— 确认后交给 apply_cut.py 执行。")

    # ---- 写 JSON ----
    out = args.out or (os.path.splitext(args.words)[0] + ".plan.json")
    payload = {
        "source_words": os.path.abspath(args.words),
        "media": os.path.abspath(args.media) if args.media else None,
        "silence_sec": args.silence,
        "confirmed": False,          # apply_cut.py 要求它为 true 才动手
        "cuts": cuts,
        "repeat_hints": hints,
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n[ok] 清单已写入: {out}")
    print("     要执行剪切：把该文件里 \"confirmed\" 改成 true，再跑 apply_cut.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
