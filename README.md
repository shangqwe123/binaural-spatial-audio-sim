# Binaural Spatial Audio Simulator

一个用于研究和离线数据构建的简化双耳空间音频渲染基线，采用 Pyroomacoustics RIR 与 ARI SOFA HRIR。它参考了 MiDashengLM-Spatial 描述的数据合成思路，**不是其官方实现或复现**。

## 功能

- 将单声道干声渲染为双耳音频，支持静止或线性移动声源。
- `hrtf` 模式使用 HRIR 和 `1/r` 距离增益；`room` 模式先模拟单通道房间响应（距离衰减只来自 RIR），再进行双耳化。
- HRTF 快照同时受时间间隔和声源视角变化限制；RIR 快照同时受时间间隔和位置位移限制，并限制声源到听者距离的变化。静止声源只用一个快照。
- 输出双耳 WAV、包含空间参数的 JSON，以及基础 ITD/ILD 诊断 CSV。
- 批量生成：语音（Emilia）和音效（AudioCaps）→ 带方向、距离、房间、运动标签和文字描述的双耳数据集，见"批量生成数据集"。

## 安装

建议使用 Python 3.10–3.12 和虚拟环境。基础 HRTF 模式：

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

需要房间混响时，再安装可选依赖：

```bash
python -m pip install -r requirements-room.txt
```

首次运行会从 ARI 下载默认 SOFA 文件到 `data/hrtf_nh2.sofa`（约 3 MB）：

```bash
python render.py --download-hrtf --output outputs/hrtf_demo.wav
```

也可以先手动下载 ARI SOFA 文件，并用 `--sofa PATH` 指定位置。SOFA 数据按 ARI 数据库自己的使用条款分发；提交代码时不会把它包含进仓库。

## 单条渲染

HRTF-only 试听，不需要 Pyroomacoustics：

```bash
python render.py --input path/to/mono.wav --mode hrtf --start 4 4 1.5 --end 4 1 1.5 --listener 2 2.5 1.5 --room 6 5 3 --output outputs/moving_hrtf.wav
```

加上房间响应：

```bash
python render.py --input path/to/mono.wav --mode room --room 12 10 4 --listener 6 5 2 --start 9 8 2 --end 9 2 2 --absorption 0.3 --order 5 --rir-step-ms 100 --max-rir-distance-m 0.2 --step-ms 20 --max-angle-step-deg 1 --output outputs/large_room.wav
```

把 `--start` 和 `--end` 设为同一坐标即可渲染固定声源。输入音频必须是单声道；脚本会重采样到 SOFA 文件的采样率，不会静默混合多声道输入。

默认采样控制：

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `--step-ms` | 20 ms | 相邻 HRTF 快照的最大时间间隔 |
| `--max-angle-step-deg` | 1° | 相邻 HRTF 快照的最大视角变化 |
| `--rir-step-ms` | 100 ms | 相邻 RIR 快照的最大时间间隔 |
| `--max-rir-distance-m` | 0.2 m | 相邻 RIR 快照的最大声源位移 |
| `--max-rir-radial-m` | 0.02 m | 相邻 RIR 快照之间声源到听者距离的最大变化 |

静止声源只生成一个 HRTF 快照和一个 RIR 快照。收紧这些阈值会增加计算量；批量训练前建议先试听并抽查位置标签。

相邻 RIR 快照之间做线性过渡时，两者直达声的到达时间不同，叠加后会产生梳状滤波（音色发虚）。声源朝听者靠近或远离时最明显，绕听者转动时几乎没有。默认的 `--max-rir-radial-m 0.02` 只在距离变化大的轨迹上增加快照数（3 m 内直线靠近的 5 s 片段：RIR 51→111 个，耗时约 3.8→4.8 s）；设为 0 可关闭。

## 批量生成数据集

数据集生成用 `scene.py` 引擎（不需要 Pyroomacoustics），和上面的 `room` 模式不同：

- 早期反射：直达声 + 二阶以内的镜像声源，**每条反射用它自己到达方向的 HRIR**；
- 后期混响：左右耳不相关的衰减噪声（扩散场，两耳频谱幅度相同、相位独立），带 HRTF 扩散场平均音色，三个频带衰减速度不同；能量按 Sabine 临界距离设定（在临界距离处直达声和全部反射声一样大；显式算出的早期反射能量从后期混响里扣除，避免重复计算）。HRIR 按插值后实际使用的扩散场能量归一化，直达声和混响用同一个参考；
- 开放环境：直达声 + 地面反射，没有混响。

### 1. 准备源数据清单

```bash
# Emilia：解压后每条语音是 .json + .mp3
python prepare_sources.py emilia --root path/to/Emilia --output sources/speech.jsonl
# AudioCaps：官方 CSV + 已下载的音频（按 youtube_id / Y<youtube_id> / audiocap_id 匹配文件名）
python prepare_sources.py audiocaps --csv train.csv val.csv test.csv --audio-dir path/to/audio --output sources/sfx.jsonl
```

清单每行是 `{"audio", "type": "speech"|"sfx", "text", "group", ...}`，也可以自己写。`group` 决定 train/validation/test 划分（Emilia 按说话人，AudioCaps 按视频）。

- Emilia 默认只保留 2–20 s、DNSMOS ≥ 3.2 的片段。Emilia 是真实场景录音，可能自带混响，房间标签会因此偏"大"，必要时提高 `--min-dnsmos`。
- AudioCaps 描述里有 and / while / with 等连接词的视为多声源，默认丢弃（`--keep-multi` 保留）。这是粗过滤，大规模使用前建议再用 CLAP 或大模型复查。立体声会转成单声道。

### 2. 生成

```bash
python generate_dataset.py --sources sources/speech.jsonl sources/sfx.jsonl --output-dir generated --variants 8 --workers 16
python check_dataset.py generated/manifest.jsonl
```

单进程下 5 s 片段：开放环境约 0.6 s，房间内移动声源约 1–1.5 s。`--resume` 跳过已生成的条目；`--limit N` 只取前 N 条源做试跑。

### 场景采样规则

**平行数据**：每条源音频生成 `--variants` 条（默认 8）不同空间版本，内容完全相同，只有空间属性不同，同一组的 `set_id` 相同。组内各属性均匀轮换：n=8 时 8 个方向各 1 条、4 种场景各 2 条、5 种运动（静止、圆弧、直线、走近、走远）至少各 1 条，远近各半。小房间不放远处和走近/走远，冲突时和其他条交换场景。同一组总在同一个 train/validation/test 划分里。


| 属性 | 取值 |
| --- | --- |
| 场景 | 小房间（3–5×3–4×2.5–3 m，RT60 0.2–0.4 s）、中房间（6–10×5–8×3–3.5 m，0.4–0.7 s）、大厅（15–30×12–20×6–10 m，0.9–1.6 s）、开放环境 |
| 方向 | 8 个扇区（正前、左前、左、左后、正后、右后、右、右前），声源落在扇区中心 ±15° 以内 |
| 距离 | 近 1.0–1.5 m，远 3–5 m；小房间只有近 |
| 高度 | 仰角 ±10°，描述中不提 |
| 运动 | 组内均匀轮换：静止；圆弧（距离不变，跨 2–4 个扇区）；直线（跨 2 个扇区，如左前→右前，中点会近一些，但不低于距离档下限的 75%）；靠近/远离（方向不变，1.0–1.25 m ↔ 4–5 m，小房间没有） |
| 运动时间 | "停—动—停"：在声源有声的区间内随机放一段 2–4 s 的移动，之前停在起点、之后停在终点；manifest 记录 `move_start_s` / `move_end_s` |
| 听者 | 在房间内随机位置，高度 1.2–1.7 m；整条轨迹离墙至少 0.3 m |

- 响度：渲染前把干声有效部分的 RMS 统一到 `--target-dbfs`（默认 -30 dBFS），渲染后**不做归一化**，只在峰值超过 0.99 时整体衰减（记录在 `clip_gain`）。所以远近、房间大小都会体现在音量上。
- 每类运动只改变一个属性，标签在运动全程有效。
- `generate_dataset.py` 里的 `allowed()` 预留给"内容和场景搭配"的规则（例如车辆只放户外），目前全部允许。
- 随机种子由 `--seed`、源数据集和源 ID、变体编号决定；增删其他文件、移动数据目录都不影响已有样本。

### 输出

`<split>/<id>.wav`（24-bit 双耳）和 `manifest.jsonl`。每行包含：

- `source`：类型、文本（转写或描述）、分组；
- `room`：类型、尺寸、RT60、听者位置；
- `spatial`：`motion`（static / arc / line / radial）、起点和终点的扇区、距离档和精确角度距离、`via_sector`（圆弧经过的扇区）、`rotation`、`radial`（approach / recede）、10 Hz 轨迹 `[t, 方位, 仰角, 距离]`、最近距离；
- `captions`：中英文模板描述，例如"在一个小房间里，声音：a dog barks，声音在近处从左前方经右前方绕到右后方。"。模板描述比较呆板，建议之后用大模型改写成多种说法，结构化标签保留作真值；
- `dry_duration_s`：干声时长，之后的部分是传播延迟和混响尾音。

`check_dataset.py` 检查侧向声源（离正前/正后 30° 以上）的方向线索是否和标签一致：在声源有声的第一秒和最后一秒，用 GCC-PHAT（200 Hz–8 kHz）估计耳间时间差，同时看宽带耳间强度差；另外列出各房间、距离档的平均电平。

在 2 条测试音频 × 40 个场景上的结果：开放环境、中房间、大厅基本 100% 一致；小房间较弱（语音静止声源时间差约 60%、强度差约 90%）。小房间墙近，反射在直达声后 1–2 ms 内到达，混响能量也比直达声大 5 dB 以上，和真实小房间一样难定位。控制实验（语音固定在正左/正右 1.25 m，每类房间 20 次）中，时间差方向正确率：开放 20/20、小房间 17/20、中房间 20/20、大厅 20/20。

## 单条渲染的输出和坐标

- 坐标系：`+x` 前方、`+y` 左侧、`+z` 上方；听者朝向 `+x`；左侧方位角为正。
- WAV：24-bit PCM 双耳音频，通道顺序为左、右。全片使用同一个防削波增益。
- JSON：渲染参数、轨迹节点及位置/方向/距离标签。
- CSV：每 50 ms 估计的宽带 ITD/ILD，适合作为基础诊断，不是感知质量分数。
- 标签按干声发声时间记录；渲染输出还包含传播/滤波延迟和尾音。

## 方法范围与局限

`hrtf` 模式对干声进行时变 HRIR 卷积，并按 `1/r` 调整幅度；不模拟精确传播延迟。`room` 模式使用 Pyroomacoustics 在头部中心模拟一个全向接收点，再按声源直达方向应用双耳 HRIR。它没有给每条反射路径分别应用方向相关的 HRIR，因此属于简化的 RIR-HRIR 串联，不是完整 BRIR 仿真。

HRTF 插值先按 HRIR 起点（峰值以下 -20 dB 的第一个采样）对齐，再对球面邻域内的频谱和延迟加权；这会平滑实测方向细节。模型假设听者头部固定、声源按 XYZ 直线匀速移动，不精确模拟多普勒、近场 HRTF、遮挡或复杂晚期混响。宽带 ITD/ILD 受信号内容和混响影响，不能单独代表方向定位准确率。

## Web Audio 对比

把本项目的渲染结果和浏览器内置 HRTF（`PannerNode`）在同一轨迹上对比试听：

```bash
python render.py --output outputs/hrtf_left_to_right_smooth.wav
python webaudio_server.py
```

然后打开 `http://127.0.0.1:8766/webaudio_compare.html`。

## 验证

```bash
python -m unittest -v
```

## 参考

- [MiDashengLM-Spatial 论文](https://arxiv.org/html/2610.11156)
- [Pyroomacoustics 文档](https://pyroomacoustics.readthedocs.io/en/stable/pyroomacoustics.room.html)
- [ARI HRTF 数据库](https://www.oeaw.ac.at/isf/outreach/software/hrtf-database)
- [SOFA HRTF 下载入口](https://sofacoustics.org/data/database/ari/)
