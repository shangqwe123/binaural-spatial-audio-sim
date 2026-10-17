# Binaural Spatial Audio Simulator

一个用于研究和离线数据构建的简化双耳空间音频渲染基线，采用 Pyroomacoustics RIR 与 ARI SOFA HRIR。它参考了 MiDashengLM-Spatial 描述的数据合成思路，**不是其官方实现或复现**。

## 功能

- 将单声道干声渲染为双耳音频，支持静止或线性移动声源。
- `hrtf` 模式使用 HRIR 和 `1/r` 距离增益；`room` 模式先模拟单通道房间响应（距离衰减只来自 RIR），再进行双耳化。
- HRTF 快照同时受时间间隔和声源视角变化限制；RIR 快照同时受时间间隔和位置位移限制，并限制声源到听者距离的变化。静止声源只用一个快照。
- 输出双耳 WAV、包含空间参数的 JSON，以及基础 ITD/ILD 诊断 CSV。
- 批量生成入口按声源组固定 train/validation/test 划分，并记录 manifest。

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

## 批量生成

将有权使用的单声道 WAV、FLAC 或 OGG 文件放入输入目录。建议使用 `speaker_id/clip.wav` 结构，使同一说话人的文件进入同一个数据 split：

```bash
python generate_dataset.py --input-dir path/to/mono_audio --output-dir generated --mode room --variants 8 --rir-step-ms 100 --max-rir-distance-m 0.2 --step-ms 20 --max-angle-step-deg 1 --seed 20261013
```

输出包含 `train/`、`validation/`、`test/` 下的双耳 WAV、`manifest.jsonl` 和 `errors.json`。同一个输入文件的变体保持在同一 split。输入音频的来源和许可由使用者负责确认。

轨迹采样规则：

- 方位角覆盖全周（前、后、左、右），仰角 -10°–25°，半径 1.2 m 起。
- 每 4 个变体中 1 个是静止声源，其余是绕听者的直线弦，方位跨度 40°–150°。
- 轨迹全程到听者的距离不小于 `--min-distance`（默认 1.0 m），端点距墙至少 0.1 m；不满足就重新采样。
- `motion` 标签为 `static`、`azimuth_decreasing`（俯视顺时针，在前方即从左到右）或 `azimuth_increasing`。
- 随机种子由 `--seed`、输入文件相对路径和变体编号决定，增删其他文件不影响已有样本。
- manifest 中 `audio` 路径相对于输出目录；一个输入文件的所有变体都成功后才写入 manifest。

## 输出和坐标

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
