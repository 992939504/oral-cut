#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qwen3-ASR 批量转录工具 (CLI)
将视频/音频文件批量转录为 TXT 或 SRT 格式。
支持两种输出模式：原目录输出 或 日期文件夹集中输出。

用法:
  python batch_transcriber.py [--txt|--srt] <文件1> [文件2] ...
  python batch_transcriber.py --txt *.mp4
  python batch_transcriber.py --srt video.mp4 audio.wav
  python batch_transcriber.py --single video.mp4      # 强制整段投喂
  python batch_transcriber.py --chunk-sec 60 *.mp4    # 强制分段，段长 60 秒
  (拖放文件到 transcribe.bat)

长时间音频（自动，默认）:
  按音频时长自动切换：TXT 模式 ≤600 秒单段投喂（历史行为，阈值见 _SINGLE_PASS_MAX_SECONDS）；
  SRT 模式阈值收紧至 300 秒（Forced Aligner 官方上限，见 _SRT_ALIGNER_MAX_SECONDS）。
  超过阈值则启用脚本内部分段转录（实测 qwen-asr transformers 后端不按
  MAX_ASR_INPUT_SECONDS 切分长音频，会整段投喂导致 KV cache OOM）：
  ffmpeg 切段 → 逐段转录 → TXT 重叠去重 / SRT 时间戳偏移合并。
  分段长度与批次按 GPU 显存自适应（_auto_limits）：如 8GB 卡用 120s 段长，
  避免库默认 1200s 段长（40GB+ 大显存配置）在小显存卡上 OOM。
  时长探测失败时回退单段模式（SRT 回退分段）。SRT 模式依赖 Forced Aligner
  （每次只能对齐 5 分钟以内语音），长音频必须走分段模式。

依赖检查:
  pip install torch qwen-asr
  ffmpeg 需在 PATH 中（ffprobe 随 ffmpeg 附带，用于时长探测）
"""

import os
import re
import sys
import glob
import json
import math
import shutil
import types
import subprocess
import tempfile
import threading
import time
from datetime import date
from typing import List, Tuple, Optional

try:
    import torch
except ImportError:
    print("[ERR] 缺少 PyTorch: pip install torch")
    sys.exit(1)

try:
    from qwen_asr import Qwen3ASRModel
    from qwen_asr.inference import utils as _asr_utils
except ImportError:
    print("[ERR] 缺少 qwen-asr: pip install qwen-asr")
    sys.exit(1)

# ---- 长时间音频模式（自动切换） ----
# 客观限制：模型 LLM 上下文 64k token（max_position_embeddings=65536，约 50 token/秒音频），
# 整段投喂硬上限约 15~20 分钟；官方 Forced Aligner 每次只能对齐 5 分钟以内语音。
# 实测 qwen-asr transformers 后端不按 MAX_ASR_INPUT_SECONDS 切分长音频（整段投喂），
# 因此长音频改由脚本内 ffmpeg 分段：切段→逐段转录→TXT 重叠去重/SRT 时间戳偏移合并。
# 自动模式：时长 ≤ _SINGLE_PASS_MAX_SECONDS 单段投喂（保持历史行为），超过则分段。
_SINGLE_PASS_MAX_SECONDS = 600
# Forced Aligner 官方支持上限：每次只能对齐 5 分钟（300s）以内语音。
# SRT 模式的自动切换阈值不得超过此值（TXT 模式无此限制，仍用 600s）。
_SRT_ALIGNER_MAX_SECONDS = 300
# 模型上下文安全段长上限：64k token ≈ 1310s 音频（50 token/s），
# 留出生成文本余量后取 1200s；TXT 分段模式的段长超过此值会警告。
_CONTEXT_MAX_SECONDS = 1200
# 单段投喂时临时设置的库常量：设为超大值以禁用库内分段（整段一次投喂）
_SINGLE_PASS_ASR_SECONDS = 86400
# 库默认分段长度（导入时捕获，分段模式恢复用；None 表示库无此属性）
_ASR_LIB_DEFAULT_MAX_SECONDS = getattr(_asr_utils, 'MAX_ASR_INPUT_SECONDS', None)
# 运行时由引擎按显存计算的自适应值（initialize 时填充，main/_configure_segmentation 使用）
_EFFECTIVE_SINGLE_MAX = None   # TXT 单段投喂上限（秒）
_EFFECTIVE_CHUNK_SECONDS = None  # 分段模式段长（秒）

# 文件扩展名常量
VIDEO_EXTS: set = {'.mp4', '.mov', '.avi', '.mkv', '.flv', '.wmv', '.webm'}
AUDIO_EXTS: set = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac', '.wma'}
ALL_EXTS: set = VIDEO_EXTS | AUDIO_EXTS


def is_media_file(path: str) -> bool:
    """是否为支持的媒体文件（扩展名合法且文件存在）"""
    return (os.path.splitext(path)[1].lower() in ALL_EXTS
            and os.path.isfile(path))


def ask_choice(prompt: str, choices: dict, default: Optional[int] = None) -> int:
    """交互式输入数字选项；choices: {选项值: 描述}。
    Ctrl+C/EOF 时退出程序；default 指定空输入时的回退值（None 则继续询问）。"""
    while True:
        try:
            c = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(1)
        if c.isdigit() and int(c) in choices:
            return int(c)
        if default is not None and c == '':
            return default
        print("  请输入 " + " 或 ".join(str(k) for k in choices))


def probe_duration(path: str) -> Optional[float]:
    """用 ffprobe 探测媒体文件时长（秒）；失败返回 None（ffprobe 随 ffmpeg 附带）"""
    try:
        r = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', path],
            capture_output=True, text=True, errors='replace', timeout=60)
        if r.returncode != 0 or not r.stdout.strip():
            return None
        return float(r.stdout.strip())
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def _fmt_duration(sec: Optional[float]) -> str:
    """格式化时长显示（h:mm:ss 或 m:ss）；None 表示未知"""
    if sec is None:
        return "时长未知"
    s = round(sec)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _auto_limits(vram_gb: float) -> Tuple[int, int, int]:
    """按显存自适应返回 (TXT单段上限秒, 分段长度秒, 推理批次)。

    估算口径：权重 ~3.4GB、运行时常量 ~1.7GB，KV cache 约 112KB/秒音频
    （50 token/s × 28层 × 8kv头 × 128维 × 2字节 × K/V），留 15% 余量。
    库默认分段 1200s 是按 40GB+ 大显存设计的，8GB 卡必须缩小段长/批次。"""
    if vram_gb >= 40:
        return 600, 1200, 4
    if vram_gb >= 24:
        return 600, 600, 4
    if vram_gb >= 16:
        return 600, 300, 4
    if vram_gb >= 12:
        return 600, 300, 2
    if vram_gb >= 8:
        return 450, 120, 2
    return 300, 60, 1



def expand_input_paths(paths: List[str]) -> List[str]:
    """将文件、目录和通配符统一展开为待处理的媒体文件列表。"""
    expanded = []
    seen = set()

    for raw_path in paths:
        matches = glob.glob(os.path.expanduser(raw_path))
        candidates = matches if matches else [os.path.expanduser(raw_path)]

        for candidate in candidates:
            if os.path.isdir(candidate):
                # 拖入文件夹时处理其下全部媒体文件；递归可覆盖分目录保存的视频。
                directory_files = []
                for root, dirs, names in os.walk(candidate):
                    dirs.sort(key=str.lower)
                    for name in sorted(names, key=str.lower):
                        item_path = os.path.join(root, name)
                        if is_media_file(item_path):
                            directory_files.append(item_path)
                items = directory_files
            else:
                items = [candidate]

            for item in items:
                key = os.path.normcase(os.path.abspath(item))
                if key not in seen:
                    seen.add(key)
                    expanded.append(item)

    return expanded


def _fix_pad_token_id(gen_cfg):
    """确保 pad_token_id 不与任何 eos_token_id 相同，防止生成卡死"""
    eos = gen_cfg.eos_token_id
    if isinstance(eos, int):
        eos_list = [eos]
    elif eos is None:
        eos_list = []
    else:
        eos_list = list(eos)
    if gen_cfg.pad_token_id is None or gen_cfg.pad_token_id in eos_list:
        # Qwen3 tokenizer 的 eos_token_id 为 151643/151645，pad_token_id
        # 若与 eos 相同会导致 beam search 提前终止、生成卡死在空白循环。
        # 151644 是 eos 之间的安全值，源于 HuggingFace tokenizer_config.json。
        # See: https://huggingface.co/Qwen/Qwen3-1.7B/blob/main/tokenizer_config.json
        gen_cfg.pad_token_id = 151644
# 库缺少 MAX_ASR_INPUT_SECONDS 属性时只提示一次（避免每文件重复刷屏）
_WARNED_SEG_ATTR = False


def _configure_segmentation(single_pass: bool, chunk_sec: Optional[int]) -> None:
    """统一禁用库内部分段（MAX_ASR_INPUT_SECONDS=86400）。

    长音频不再依赖库内部分段（实测 transformers 后端不按该常量切分），
    改由 _transcribe_chunked 脚本内手动分段，因此单次调用一律整段投喂，
    避免不同库版本行为不一致。"""
    global _WARNED_SEG_ATTR
    try:
        if _asr_utils.MAX_ASR_INPUT_SECONDS != _SINGLE_PASS_ASR_SECONDS:
            _asr_utils.MAX_ASR_INPUT_SECONDS = _SINGLE_PASS_ASR_SECONDS
    except AttributeError:
        if not _WARNED_SEG_ATTR:
            _WARNED_SEG_ATTR = True
            print("[WARN] 库缺少 MAX_ASR_INPUT_SECONDS，可能无法禁用库内分段")


# ---- 脚本内手动分段（Plan B：库内部分段在 transformers 后端不生效）----
_CHUNK_OVERLAP_SECONDS = 5  # TXT 分段重叠（秒），用于边界文本去重；SRT 不重叠（逐条合并）


def _merge_chunk_text(prev: str, cur: str) -> str:
    """拼接相邻段文本，去除重叠区域造成的重复（后缀-前缀精确匹配）"""
    if not prev:
        return cur
    max_n = min(len(prev), len(cur), _CHUNK_OVERLAP_SECONDS * 12)
    for n in range(max_n, 3, -1):  # 最小匹配 4 字符
        if prev[-n:] == cur[:n]:
            return prev + cur[n:]
    return prev + cur


def _split_audio_chunks(audio_path: str, chunk_sec: int, overlap_sec: float,
                        tmp_dir: str) -> Optional[List[Tuple[str, float]]]:
    """ffmpeg 切分音频为重叠片段，返回 [(片段路径, 起始秒), ...]；失败返回 None。
    片段时长均 ≤ chunk_sec，最后一段可能更短。"""
    dur = probe_duration(audio_path)
    if dur is None:
        return None
    if dur <= chunk_sec:
        return [(audio_path, 0.0)]
    overlap_sec = min(overlap_sec, max(0, chunk_sec - 2))
    step = chunk_sec - overlap_sec
    n = math.ceil(dur / step)
    chunks = []
    for i in range(n):
        start = i * step
        out = os.path.join(tmp_dir, f"chunk_{i:04d}.wav")
        cmd = ['ffmpeg', '-v', 'error', '-ss', f'{start:.3f}',
               '-t', f'{chunk_sec}', '-i', audio_path,
               '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', '-y', out]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True,
                               errors='replace', timeout=600)
        except OSError:
            print("[ERR] ffmpeg 不可用，无法分段")
            return None
        if r.returncode != 0:
            print(f"[ERR] 音频切段失败（第 {i+1}/{n} 段）: {r.stderr[:200]}")
            return None
        chunks.append((out, start))
    return chunks


# ========================================================================
# Qwen3-ASR 引擎封装
# ========================================================================
class Qwen3ASREngine:
    _model = None
    _aligner_loaded = False
    _lock = threading.RLock()

    @classmethod
    def _model_dir(cls) -> str:
        return os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "models", "qwen3_asr"
        )

    @classmethod
    def is_available(cls) -> bool:
        path = os.path.join(cls._model_dir(), "Qwen3-ASR-1.7B")
        return os.path.exists(path)

    @classmethod
    def initialize(cls, need_timestamps: bool = False) -> bool:
        with cls._lock:
            if cls._model is not None:
                # 模型已加载，如需 aligner 但尚未加载则补载（统一走锁内单一路径）
                if need_timestamps and not cls._aligner_loaded:
                    cls._load_aligner()
                return True
            d = cls._model_dir()
            asr_path = os.path.join(d, "Qwen3-ASR-1.7B")
            aligner_path = os.path.join(d, "Qwen3-ForcedAligner-0.6B")
            if not os.path.exists(asr_path):
                print(f"[ERR] 模型目录不存在: {asr_path}")
                return False
            try:
                # 按显存自适应：库默认分段 1200s × batch4 是 40GB+ 大显存配置，
                # 小显存卡必须缩小段长/批次，否则 KV cache 直接 OOM
                global _EFFECTIVE_SINGLE_MAX, _EFFECTIVE_CHUNK_SECONDS
                vram_gb = None
                try:
                    vram_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
                except Exception:
                    pass
                if vram_gb is not None:
                    _EFFECTIVE_SINGLE_MAX, _EFFECTIVE_CHUNK_SECONDS, batch = _auto_limits(vram_gb)
                else:
                    # 无法读取显存时用 8GB 档的保守值
                    _EFFECTIVE_SINGLE_MAX, _EFFECTIVE_CHUNK_SECONDS, batch = 450, 120, 2
                kwargs = dict(
                    dtype=torch.bfloat16,
                    device_map="cuda:0",
                    max_inference_batch_size=batch,  # 分段模式下每批 chunk 数，防长音频 OOM
                    max_new_tokens=8192,
                )
                # 只有需要时间戳时才加载 Forced Aligner
                if need_timestamps and os.path.exists(aligner_path):
                    kwargs["forced_aligner"] = aligner_path
                    kwargs["forced_aligner_kwargs"] = dict(
                        dtype=torch.bfloat16, device_map="cuda:0"
                    )
                print("[..] 加载 Qwen3-ASR 模型中...", end=" ", flush=True)
                t0 = time.time()
                cls._model = Qwen3ASRModel.from_pretrained(asr_path, **kwargs)
                # 加载成功后才标记 aligner 就绪，避免失败时残留脏状态
                if need_timestamps and os.path.exists(aligner_path):
                    cls._aligner_loaded = True
                if cls._model is not None:
                    for cfg in (cls._model.model.generation_config,
                                cls._model.model.thinker.generation_config):
                        _fix_pad_token_id(cfg)
                print(f"完成 ({time.time()-t0:.1f}s)")
                if vram_gb is not None:
                    print(f"[..] 显存 {vram_gb:.1f}GB → 自适应：单段 ≤{_EFFECTIVE_SINGLE_MAX}s，",
                          f"分段 {_EFFECTIVE_CHUNK_SECONDS}s，批次 {batch}",
                          f"（库默认分段 {_ASR_LIB_DEFAULT_MAX_SECONDS}s）")
                else:
                    print(f"[..] 无法读取显存，使用保守配置：单段 ≤{_EFFECTIVE_SINGLE_MAX}s，",
                          f"分段 {_EFFECTIVE_CHUNK_SECONDS}s，批次 {batch}")
                return True
            except Exception as e:
                print(f"\n[ERR] 模型加载失败: {e}")
                return False

    @classmethod
    def _load_aligner(cls) -> bool:
        """延迟加载 Forced Aligner（SRT 模式首次调用时触发）"""
        if cls._aligner_loaded or cls._model is None:
            return cls._aligner_loaded
        with cls._lock:
            if cls._aligner_loaded:
                return True
            aligner_path = os.path.join(cls._model_dir(), "Qwen3-ForcedAligner-0.6B")
            if not os.path.exists(aligner_path):
                print("[WARN] Forced Aligner 目录不存在，无法生成时间戳")
                return False
            try:
                print("[..] 加载 Forced Aligner...", end=" ", flush=True)
                t0 = time.time()
                cls._model.load_forced_aligner(
                    aligner_path,
                    forced_aligner_kwargs=dict(
                        dtype=torch.bfloat16, device_map="cuda:0"
                    ),
                )
                cls._aligner_loaded = True
                print(f"完成 ({time.time()-t0:.1f}s)")
                return True
            except Exception as e:
                # load_forced_aligner 可能不是公开 API，回退方案：重新加载整个模型
                print(f"\n[WARN] 延迟加载 aligner 失败: {e}")
                print("[..] 将重新加载模型（含 aligner）...")
                cls._model = None
                cls._aligner_loaded = False
                return cls.initialize(need_timestamps=True)

    @classmethod
    def transcribe(cls, audio_path: str, language: Optional[str] = None,
                   return_time_stamps: bool = True, single_pass: bool = True,
                   chunk_sec: Optional[int] = None) -> Optional[object]:
        with cls._lock:
            if cls._model is None and not cls.initialize(need_timestamps=return_time_stamps):
                return None
            # 如需时间戳但 aligner 未加载，尝试延迟加载
            if return_time_stamps and not cls._aligner_loaded:
                cls._load_aligner()
        # 统一禁用库内分段（长音频由 _transcribe_chunked 手动分段）
        _configure_segmentation(single_pass, chunk_sec)
        if single_pass:
            return cls._transcribe_single(audio_path, language, return_time_stamps)
        return cls._transcribe_chunked(audio_path, language, return_time_stamps, chunk_sec)

    @classmethod
    def _transcribe_single(cls, audio_path: str, language: Optional[str],
                           return_time_stamps: bool) -> Optional[object]:
        """整段投喂一次转录（短音频）；带 30s 心跳提示"""
        hb_stop = threading.Event()
        def _heartbeat():
            t0 = time.time()
            while not hb_stop.wait(30):
                print(f"  [..] 转录进行中（已用时 {time.time()-t0:.0f}s）...", flush=True)
        hb = threading.Thread(target=_heartbeat, daemon=True)
        hb.start()
        try:
            results = cls._model.transcribe(
                audio=audio_path,
                language=language,
                return_time_stamps=return_time_stamps,
            )
            return results[0] if results and len(results) > 0 else None
        except Exception as e:
            print(f"[ERR] 转录失败: {e}")
            return None
        finally:
            hb_stop.set()

    @classmethod
    def _transcribe_chunked(cls, audio_path: str, language: Optional[str],
                            return_time_stamps: bool,
                            chunk_sec: Optional[int]) -> Optional[object]:
        """脚本内手动分段转录：ffmpeg 切段 → 逐段转录 → 合并。

        TXT：段间 5s 重叠，按后缀-前缀匹配去重后拼接；
        SRT：段间不重叠，时间戳加段偏移后逐条合并（aligner 每段 ≤ 5 分钟）。"""
        chunk_len = chunk_sec if chunk_sec is not None else (
            _EFFECTIVE_CHUNK_SECONDS or _ASR_LIB_DEFAULT_MAX_SECONDS or 120)
        if return_time_stamps:
            chunk_len = min(chunk_len, _SRT_ALIGNER_MAX_SECONDS)  # aligner 硬上限
        tmp_dir = tempfile.mkdtemp(prefix='asr_chunks_')
        try:
            overlap = 0.0 if return_time_stamps else _CHUNK_OVERLAP_SECONDS
            chunks = _split_audio_chunks(audio_path, chunk_len, overlap, tmp_dir)
            if chunks is None:
                return None
            n = len(chunks)
            if n == 1 and chunks[0][1] == 0.0:
                # 整段未超过段长，直接单次转录
                return cls._transcribe_single(audio_path, language, return_time_stamps)
            print(f"[..] 分段转录：{n} 段 × {chunk_len}s...", flush=True)
            texts = []
            all_ts = []
            for i, (path, start) in enumerate(chunks, 1):
                try:
                    results = cls._model.transcribe(
                        audio=path, language=language,
                        return_time_stamps=return_time_stamps,
                    )
                except Exception as e:
                    print(f"[ERR] 第 {i}/{n} 段转录失败: {e}")
                    return None
                if not results or len(results) == 0:
                    print(f"[ERR] 第 {i}/{n} 段无结果")
                    return None
                r = results[0]
                text = getattr(r, 'text', '') or ''
                if return_time_stamps:
                    for ts in (getattr(r, 'time_stamps', None) or []):
                        seg_text = (getattr(ts, 'text', '') or '').strip()
                        if not seg_text:
                            continue
                        all_ts.append(types.SimpleNamespace(
                            text=seg_text,
                            start_time=start + getattr(ts, 'start_time', 0),
                            end_time=start + getattr(ts, 'end_time', 0),
                        ))
                else:
                    texts.append(text)
                print(f"  [..] 段 {i}/{n} 完成", flush=True)
            if return_time_stamps:
                return types.SimpleNamespace(
                    text=''.join(t.text for t in all_ts), time_stamps=all_ts)
            merged = ''
            for t in texts:
                merged = _merge_chunk_text(merged, t)
            return types.SimpleNamespace(text=merged, time_stamps=None)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


# ========================================================================


def _fmt_srt_time(sec: float) -> str:
    """将秒数转换为 SRT 时间格式 HH:MM:SS,mmm（毫秒进位安全，避免 ,1000）"""
    total_ms = round(sec * 1000)
    h, rem = divmod(total_ms, 3600000)
    m, rem = divmod(rem, 60000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _write_words_sidecar(out_path: str, timestamps) -> None:
    """把字符级（中文逐字／西文逐词）时间戳另存为 <name>.words.json。

    格式与 VidMuse / HyperFrames 的 transcript.json 契约对齐：
        [{"text": "大", "start": 0.16, "end": 0.32}, ...]

    纯附加产物：不改变 SRT/TXT 的既有行为；写失败只警告，不中断转录。
    """
    words = []
    for ts in timestamps:
        seg_text = (getattr(ts, 'text', '') or '').strip()
        if not seg_text:
            continue
        words.append({
            "text": seg_text,
            "start": round(float(getattr(ts, 'start_time', 0) or 0), 3),
            "end": round(float(getattr(ts, 'end_time', 0) or 0), 3),
        })
    if not words:
        return
    words_path = os.path.splitext(out_path)[0] + '.words.json'
    try:
        with open(words_path, 'w', encoding='utf-8') as f:
            json.dump(words, f, ensure_ascii=False, indent=1)
        print(f"  [..] 已写入 {os.path.basename(words_path)}"
              f"（{len(words)} 条逐字时间戳）", flush=True)
    except OSError as e:
        print(f"  [WARN] 逐字时间戳 sidecar 写入失败: {e}", flush=True)


# ========================================================================
# 文件处理
# ========================================================================
def extract_audio(video_path: str, out_wav: str) -> Tuple[bool, str]:
    """用 ffmpeg 从视频提取 16kHz 单声道 WAV，返回 (成功, 错误信息)"""
    cmd = [
        'ffmpeg', '-i', video_path, '-vn',
        '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
        '-y', out_wav
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           errors='replace', timeout=1800)
        if r.returncode != 0:
            return False, r.stderr[:200]
        return True, ""
    except FileNotFoundError:
        return False, "ffmpeg 未安装或不在 PATH 中"
    except subprocess.TimeoutExpired:
        return False, "ffmpeg 处理超时 (1800s)"


def transcribe_one(audio_path: str, fmt: str, out_path: str,
                   language: Optional[str] = None, single_pass: bool = True,
                   chunk_sec: Optional[int] = None) -> Tuple[bool, str]:
    """转录音频，按 fmt (txt/srt) 写出到 out_path，返回 (bool, str)
    
    TXT 模式跳过耗时的 forced alignment，仅获取文本。
    SRT 模式需要时间戳，会运行 forced alignment。
    single_pass: True=整段投喂（短音频），False=库内分段（长音频）。
    chunk_sec: 分段模式下的段长（秒），None 使用库默认。
    """
    use_ts = (fmt == 'srt')
    result = Qwen3ASREngine.transcribe(
        audio_path, language=language, return_time_stamps=use_ts,
        single_pass=single_pass, chunk_sec=chunk_sec)
    if result is None:
        return False, "转录失败"

    text = getattr(result, 'text', '') or ''

    if fmt == 'txt':
        if not text:
            return False, "转录结果为空"
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"  [..] 已写入 {os.path.basename(out_path)}", flush=True)
        return True, os.path.basename(out_path)

    if fmt == 'srt':
        timestamps = getattr(result, 'time_stamps', None)
        if not timestamps:
            if text:
                # 单行化文本避免换行符破坏 SRT 格式规范
                print("[WARN] 未获取到时间戳，已写入占位 SRT（全文压在 0~1 秒）", flush=True)
                text_clean = text.replace('\r', '').replace('\n', ' ')
                with open(out_path, 'w', encoding='utf-8') as f:
                    f.write(f"1\n00:00:00,000 --> 00:00:01,000\n{text_clean}\n")
                return True, os.path.basename(out_path)
            return False, "未获取到时间戳，无法生成 SRT"

        # 另存逐字时间戳 sidecar（纯附加产物，不影响下面 SRT 的既有行为）
        _write_words_sidecar(out_path, timestamps)

        # 合并字符级时间戳为短语级 SRT 条目
        merged = []
        for ts in timestamps:
            seg_text = (getattr(ts, 'text', '') or '').strip()
            if not seg_text:
                continue
            start = getattr(ts, 'start_time', 0)
            end = getattr(ts, 'end_time', 0)
            if not merged:
                merged.append([seg_text, start, end])
            else:
                prev_text, prev_start, prev_end = merged[-1]
                gap = start - prev_end
                # 间隔 < 0.3s 或上一段文本过短时合并
                if gap < 0.3 or len(prev_text) < 3:
                    merged[-1][0] = prev_text + seg_text
                    merged[-1][2] = end
                else:
                    merged.append([seg_text, start, end])

        with open(out_path, 'w', encoding='utf-8') as f:
            for i, (seg_text, seg_start, seg_end) in enumerate(merged):
                f.write(f"{i+1}\n")
                f.write(f"{_fmt_srt_time(seg_start)} --> {_fmt_srt_time(seg_end)}\n")
                f.write(f"{seg_text}\n\n")
        return True, os.path.basename(out_path)

    return False, "未知格式"


def process_file(filepath: str, fmt: str, out_dir: Optional[str] = None,
                 single_pass: bool = True,
                 chunk_sec: Optional[int] = None) -> Tuple[bool, str]:
    """处理单个文件，返回 (成功, 消息)
    
    out_dir: 若指定则输出到此目录，否则输出到文件所在目录
    single_pass: True=整段投喂（短音频），False=库内分段（长音频）。
    chunk_sec: 分段模式下的段长（秒），None 使用库默认。
    """
    ext = os.path.splitext(filepath)[1].lower()
    if not is_media_file(filepath):
        if ext not in ALL_EXTS:
            return False, f"不支持格式 {ext}"
        return False, "文件不存在"

    actual_dir = out_dir if out_dir is not None else os.path.dirname(os.path.abspath(filepath))
    base = os.path.splitext(os.path.basename(filepath))[0]
    suffix = '.srt' if fmt == 'srt' else '.txt'
    out_path = os.path.join(actual_dir, base + suffix)
    # 日期文件夹模式下，同名文件自动编号避免覆盖
    if out_dir is not None:
        counter = 1
        orig_base = base
        while os.path.exists(out_path):
            base = f"{orig_base}_{counter}"
            out_path = os.path.join(actual_dir, base + suffix)
            counter += 1
    audio_in = filepath
    tmp_wav = None

    try:
        if ext in VIDEO_EXTS:
            fd, tmp_wav = tempfile.mkstemp(suffix='.wav')
            os.close(fd)
            ok, err_msg = extract_audio(filepath, tmp_wav)
            if not ok:
                return False, f"音频提取失败: {err_msg}"
            audio_in = tmp_wav

        ok, msg = transcribe_one(audio_in, fmt, out_path,
                                 single_pass=single_pass, chunk_sec=chunk_sec)
        if ok:
            size = os.path.getsize(out_path)
            return True, f"已生成 {msg} ({size:,} 字节)"
        return False, msg

    except Exception as e:
        return False, f"异常: {e}"
    finally:
        if tmp_wav and os.path.exists(tmp_wav):
            try:
                os.remove(tmp_wav)
            except OSError:
                pass


# ========================================================================
# 主入口
# ========================================================================


def parse_args(argv: List[str]) -> Tuple[Optional[str], List[str], int,
                                 Optional[bool], Optional[int]]:
    """解析命令行参数，返回 (fmt, files, output_mode, single_pass, chunk_sec)
    
    fmt: 'txt' | 'srt' | None (表示需要交互选择)
    output_mode: 1=日期文件夹, 2=原目录 (默认)
    single_pass: None=按时长自动 | True=强制单段(--single) | False=强制分段
    chunk_sec: 分段段长（秒，--chunk-sec），与 --single 互斥
    """
    fmt = None
    files = []
    output_mode = 2
    single_pass = None
    chunk_sec = None

    i = 1
    while i < len(argv):
        a = argv[i]
        if a in ('--txt', '-t'):
            fmt = 'txt'
        elif a in ('--srt', '-s'):
            fmt = 'srt'
        elif a == '--output-mode':
            i += 1
            if i >= len(argv):
                print("[ERR] --output-mode 缺少参数 (1 或 2)")
                sys.exit(1)
            try:
                output_mode = int(argv[i])
            except ValueError:
                print(f"[ERR] --output-mode 参数无效: {argv[i]}，必须为 1 或 2")
                sys.exit(1)
            if output_mode not in (1, 2):
                print("[ERR] --output-mode 只能为 1 或 2")
                sys.exit(1)
        elif a == '--single':
            single_pass = True
        elif a == '--chunk-sec':
            i += 1
            if i >= len(argv):
                print("[ERR] --chunk-sec 缺少参数 (秒数)")
                sys.exit(1)
            try:
                chunk_sec = int(argv[i])
            except ValueError:
                print(f"[ERR] --chunk-sec 参数无效: {argv[i]}，必须为正整数")
                sys.exit(1)
            if chunk_sec <= 0:
                print("[ERR] --chunk-sec 必须为正整数")
                sys.exit(1)
        elif a.startswith('-'):
            print(f"[ERR] 未知参数: {a}")
            sys.exit(1)
        else:
            files.append(a)
        i += 1

    if single_pass is True and chunk_sec is not None:
        print("[ERR] --single 与 --chunk-sec 不能同时使用")
        sys.exit(1)

    return fmt, files, output_mode, single_pass, chunk_sec


def resolve_file_paths(raw: str) -> List[str]:
    """解析用户拖入/粘贴的文件路径字符串，处理引号包裹的路径"""
    # 匹配引号包裹的路径或空格分隔的路径
    pattern = r'"([^"]+)"|([^"\s]+)'
    matches = re.findall(pattern, raw)
    paths = [m[0] or m[1] for m in matches]
    return paths


def main():
    fmt, files, output_mode, single_pass, chunk_sec = parse_args(sys.argv)
    
    interactive_mode = False
    # 如果没有命令行文件参数，进入交互拖入模式
    if not files:
        interactive_mode = True
        print("=" * 50)
        print("  Qwen3-ASR 批量转录工具")
        print("=" * 50)
        print()
        print("  请把视频/音频文件拖到这个窗口里，然后按回车")
        print("  (支持格式: mp4, mov, avi, mkv, wav, mp3, ...)")
        print()
        try:
            raw = input("  文件路径: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(1)
        print()
        
        if not raw:
            print("[ERR] 没有输入文件路径")
            sys.exit(1)
        
        files = resolve_file_paths(raw)
        if not files:
            print("[ERR] 没有可处理的文件")
            sys.exit(1)
        
        # 交互模式也支持 --txt/--srt 选择
        if fmt is None:
            fmt = 'txt' if ask_choice("  输出格式 (1=TXT, 2=SRT，回车=TXT): ", {1: 'TXT', 2: 'SRT'}, default=1) == 1 else 'srt'
        # 交互模式：多文件时询问输出目录方式
        if len(files) > 1:
            output_mode = ask_choice(
                "  输出目录方式 (1=日期文件夹, 2=原目录): ",
                {1: '日期文件夹', 2: '原目录'})
            print()

    # 将文件夹、通配符和具体文件统一展开，并过滤有效文件。
    files = expand_input_paths(files)
    valid = []
    for f in files:
        if not is_media_file(f):
            ext = os.path.splitext(f)[1].lower()
            if ext not in ALL_EXTS:
                print(f"  [WARN] 不支持格式 {ext}: {f}")
            else:
                print(f"  [WARN] 文件不存在: {f}")
        else:
            valid.append(f)

    if not valid:
        print("[ERR] 没有可处理的文件")
        sys.exit(1)

    n = len(valid)
    print(f"共 {n} 个文件待处理:")
    for f in valid:
        print(f"  {os.path.basename(f)}")
    print()

    # 如果命令行模式未指定格式，在这里选择
    if fmt is None:
        fmt = 'txt' if ask_choice("输出格式 (1=TXT, 2=SRT，回车=TXT): ", {1: 'TXT', 2: 'SRT'}, default=1) == 1 else 'srt'
    # 命令行多文件且未指定 --output-mode 时询问
    if not interactive_mode and len(valid) > 1 and output_mode == 2 and '--output-mode' not in sys.argv:
        output_mode = ask_choice(
            "输出目录方式 (1=日期文件夹, 2=原目录，回车=原目录): ",
            {1: '日期文件夹', 2: '原目录'}, default=2)

    print(f"\n格式: {fmt.upper()}")
    print("=" * 50)

    # 输出目录模式
    out_dir = None
    if output_mode == 1:
        base_dir = os.path.dirname(os.path.abspath(valid[0]))
        today_str = date.today().isoformat()  # YYYY-MM-DD
        out_dir = os.path.join(base_dir, today_str)
        os.makedirs(out_dir, exist_ok=True)
        print(f"输出目录: {out_dir}")
        print("=" * 50)

    # 加载模型
    print("[..] 检测模型...", end=" ", flush=True)
    if not Qwen3ASREngine.is_available():
        print("\n[ERR] 模型不可用: models/qwen3_asr/Qwen3-ASR-1.7B/")
        sys.exit(1)
    print("就绪")

    if not Qwen3ASREngine.initialize(need_timestamps=(fmt == 'srt')):
        print("[ERR] 模型加载失败")
        sys.exit(1)

    # 逐个处理（自动模式：≤阈值单段投喂，>阈值库内分段）
    ok_count, fail_count = 0, 0
    for i, fp in enumerate(valid, 1):
        name = os.path.basename(fp)
        dur = probe_duration(fp)
        if single_pass is not None:
            sp = single_pass          # --single / --chunk-sec 强制指定
        elif chunk_sec is not None:
            sp = False                # --chunk-sec 强制分段
        elif dur is None:
            if fmt == 'srt':
                # SRT 模式探测失败时回退分段：单段会撞 Forced Aligner 5 分钟上限，
                # 分段（库默认段长）对任意时长都安全
                print(f"  [WARN] 无法探测时长，SRT 模式按分段处理: {name}")
                sp = False
            else:
                print(f"  [WARN] 无法探测时长，按单段模式处理: {name}")
                sp = True
        else:
            # 自动阈值：TXT 受显存自适应单段上限约束（≤600s）；SRT 再叠加 Forced Aligner 5 分钟上限
            limit = min(_EFFECTIVE_SINGLE_MAX or _SINGLE_PASS_MAX_SECONDS,
                        _SRT_ALIGNER_MAX_SECONDS if fmt == 'srt'
                        else _SINGLE_PASS_MAX_SECONDS)
            sp = dur <= limit
        mode_label = "单段模式" if sp else "分段模式"
        est = ""
        if not sp and dur is not None:
            # 段长：--chunk-sec 显式值 > 显存自适应值 > 库默认
            chunk_len = chunk_sec if chunk_sec is not None else (
                _EFFECTIVE_CHUNK_SECONDS or _ASR_LIB_DEFAULT_MAX_SECONDS)
            if chunk_len:
                est = f"，约 {math.ceil(dur / chunk_len)} 段"
        print(f"\n[{i}/{n}] {name} ({_fmt_duration(dur)}, {mode_label}{est})", flush=True)
        if fmt == 'srt':
            if sp and dur is not None and dur > _SRT_ALIGNER_MAX_SECONDS:
                print(f"  [WARN] 音频 {_fmt_duration(dur)} 超过 Forced Aligner 上限",
                      f" {_SRT_ALIGNER_MAX_SECONDS}s，SRT 时间戳可能失败")
            elif not sp and chunk_sec is not None and chunk_sec > _SRT_ALIGNER_MAX_SECONDS:
                print(f"  [WARN] 段长 {chunk_sec}s 超过 Forced Aligner 上限",
                      f" {_SRT_ALIGNER_MAX_SECONDS}s，SRT 时间戳可能失败")
        if not sp and chunk_sec is not None:
            hard = _CONTEXT_MAX_SECONDS
            soft = _EFFECTIVE_CHUNK_SECONDS or hard
            if chunk_sec > hard:
                print(f"  [WARN] 段长 {chunk_sec}s 超过模型上下文安全上限",
                      f" {hard}s，转录可能失败")
            elif chunk_sec > soft:
                print(f"  [WARN] 段长 {chunk_sec}s 超过显存自适应建议值 {soft}s，",
                      "可能显存不足（OOM）")
        t_start = time.time()
        succ, msg = process_file(fp, fmt, out_dir=out_dir,
                                 single_pass=sp, chunk_sec=chunk_sec)
        elapsed = time.time() - t_start
        if succ:
            ok_count += 1
            print(f"  [OK] {msg}  ({elapsed:.1f}s)", flush=True)
        else:
            fail_count += 1
            print(f"  [ERR] {msg}", flush=True)

    # 汇总
    print()
    print("=" * 50)
    print(f"  完成: 成功 {ok_count} | 失败 {fail_count} | 共 {n}")
    print("=" * 50)
    sys.exit(0 if fail_count == 0 else 1)


if __name__ == '__main__':
    main()
