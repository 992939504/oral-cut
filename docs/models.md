# 模型清单

全部**本地运行**，不需要任何云端 API。引擎是 [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)（Qwen 官方开源，**Apache-2.0**）。

---

## 一、先搞清楚：这个项目的瓶颈在哪

不是速度，是**字级时间戳**。

整套剪辑规则（剪点吸附、静音间隙判定）都建立在"每个字什么时候开始、什么时候结束"上面。

所以选模型的第一问不是"哪个快"，是——

> **它给不给逐字时间戳？**

给不了，再快也用不了。

---

## 二、推荐档位

### 档位 A · GPU 主力（NVIDIA 8G+ 显存）

| 模型 | 体积 | 逐字时间戳 |
|---|---|---|
| `Qwen/Qwen3-ASR-1.7B` | 约 4.2 GB | ✅ |
| `Qwen/Qwen3-ForcedAligner-0.6B` | 约 1.8 GB | ✅（负责对齐） |

**选择理由**：中文强，官方支持 timestamp prediction。两人组合 = 识别 + 强制对齐，逐字精度最高。

### 档位 B · CPU / 小显存 ← **推荐给大多数人**

| 模型 | 体积 | 逐字时间戳 |
|---|---|---|
| `Qwen/Qwen3-ASR-0.6B` | 约 1.2 GB | ✅ |
| `Qwen/Qwen3-ForcedAligner-0.6B` | 约 1.8 GB | ✅ |

**为什么不是更快的 SenseVoice？**

因为 **SenseVoice 只给句级时间戳**。它快得多（27 分钟中文十几秒），但剪点吸附做不了——而剪点吸附是这个项目的全部价值。

0.6B 跑 CPU 会慢（一条 3 分钟视频大概几分钟），但**同一条管线、同样的字级精度**，不用维护两套逻辑。

### 档位 C · 极端低配（树莓派级）

| 方案 | 体积 | 逐字时间戳 |
|---|---|---|
| `sherpa-onnx` + `Paraformer-zh` | 约 300 MB | ✅ |
| `sherpa-onnx` + `SenseVoice-Small` | 约 830 MB | ❌ 句级 |

[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)（Apache-2.0）纯 CPU、完全离线，能塞进嵌入式设备。
Paraformer 系列支持时间戳；SenseVoice 只有句级——**后者只能用来做纯转录，不能用本项目的剪辑功能**。

### 附 · 方言支持（Qwen3-ASR 的隐藏优势）

Qwen3-ASR 官方支持 **22 种汉语方言**：

> 安徽、东北、福建、甘肃、贵州、河北、河南、湖北、湖南、江西、宁夏、山东、陕西、**山西**、四川、天津、云南、浙江、粤语（香港口音）、粤语（广东口音）、吴语、闽南语

如果你的口音落在这些范围里，识别质量会明显好于通用模型——这是它比 Whisper 更适合中文口播的核心原因之一。

> 注意：**ForcedAligner 只支持 11 种语言**（中、英、粤、法、德、意、日、韩、葡、俄、西），不覆盖全部方言。方言语音的对齐质量可能弱于识别质量。

---

## 三、横向对比

| 模型 | 参数 | 逐字时间戳 | 标点 | 中文 | 相对速度 | 许可 |
|---|---|---|---|---|---|---|
| **Qwen3-ASR-1.7B** | 1.7B | ✅ 原生 | 依赖后处理 | 强 | 慢（GPU 可接受） | Apache-2.0 |
| **Qwen3-ASR-0.6B** | 0.6B | ✅ 原生 | 依赖后处理 | 强 | 中 | Apache-2.0 |
| SenseVoice-Small | ~230M | ❌ 句级 | ✅ 自带 | 强（中文/粤语优） | **极快** | 待核 |
| Paraformer-zh | ~220M | ✅ | ✅ | 强 | 快 | Apache-2.0 |
| Whisper (large-v3) | 1.5B | ✅ 词级 | ✅ | 一般 | 慢 | MIT |
| FireRedASR | 更大 | ✅ | — | 强 | 慢 | 待核 |

> Whisper 列在这里只为完整性。**中文口播场景下它不是首选**：中文标点和断句都弱于上面几个。

---

## 四、下载

官方给了两条路，国内推荐 ModelScope：

```bash
# ModelScope（国内快）
pip install -U modelscope
modelscope download --model Qwen/Qwen3-ASR-1.7B --local_dir ./models/qwen3_asr/Qwen3-ASR-1.7B
modelscope download --model Qwen/Qwen3-ASR-0.6B --local_dir ./models/qwen3_asr/Qwen3-ASR-0.6B
modelscope download --model Qwen/Qwen3-ForcedAligner-0.6B --local_dir ./models/qwen3_asr/Qwen3-ForcedAligner-0.6B

# Hugging Face
pip install -U "huggingface_hub[cli]"
huggingface-cli download Qwen/Qwen3-ASR-1.7B --local-dir ./models/qwen3_asr/Qwen3-ASR-1.7B
```

或者用本项目的脚本（会先列清单等确认）：

```bash
python scripts/setup_models.py --size full    # GPU 档
python scripts/setup_models.py --size small   # CPU 档
```

---

## 五、目录约定

引擎按**相对脚本目录**找模型：

```
<安装目录>/
├── batch_transcriber.py
└── models/
    └── qwen3_asr/
        ├── Qwen3-ASR-1.7B/            # 或 Qwen3-ASR-0.6B
        └── Qwen3-ForcedAligner-0.6B/
```

模型放对位置，引擎自己就能找到，不需要额外配置。

---

## 六、显存自适应

引擎会读 GPU 显存自动调分段长度（`_auto_limits`）：

- 库默认是 1200s 段长 × batch 4，那是 40GB+ 大显存配置
- 小显存卡（8GB）必须缩小到 120s 段长，否则 KV cache 直接 OOM
- 显存读不到时用 8GB 档的保守值

SRT 模式下 ForcedAligner 每次只能对齐 5 分钟以内语音，长音频自动分段。
