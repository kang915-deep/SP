# Qwen3-VL 版 VEGA 训练

## 四张 A800 一键启动

clone 后进入仓库根目录，把下面两个参数换成**已挂载、可写的数据盘目录**和**分配给你的四个物理 GPU 编号**。例如数据盘是 `/mnt/data`，卡号是 0、1、2、3：

```bash
bash start.sh /mnt/data 0,1,2,3
```

命令返回后台 PID 和可直接复制的 `tail -f` 命令。训练数据、模型权重、断点和日志保存在指定数据盘的 `qwen3-vega/` 下。默认也在这里安装软件。如果挂载盘不支持符号链接（例如部分 BOS 挂载），脚本会把 Python 环境、依赖缓存和编译临时文件放到当前用户的本地缓存目录，并在日志的 `Software runtime` 一行打印位置。也可以在启动命令前设置 `VEGA_RUNTIME_DIR=/本地盘上的专用目录`，选择空间足够且支持符号链接的 Linux 文件系统。

服务器需要 Linux、可用的 NVIDIA 驱动和 CUDA 构建环境、Python 3 与 pip，以及 git、curl、flock，并能访问 GitHub、软件源和数据源。训练媒体压缩包约 290 GiB，还要存解压媒体、权重、特征缓存和四份完整模型/Adam 断点，建议为整套工作预留 **约 1.5 TiB**。

1. **准备。** 自动配置 Python、CUDA 版 PyTorch 和项目依赖，获取固定版本的 VEGA 源码，下载并校验 Qwen、Wan、SP 文本编码器、298,027 条训练 QA 所需媒体和完整 VSI-Bench。
2. **测速。** 比较每卡 micro-batch 1、2 与语言层全重算、隔层重算、关闭重算的六种组合。每组跑相同的 5 个真实优化步，有效 batch 固定 64。第 1 步预热，用后 4 步的整机 QA/s 排名。只在每张卡采样到的最小剩余显存均达到 **8 GiB** 的配置里选最快的一组。CUDA OOM 的候选会记录并跳过，其他错误停止并写日志。测速覆盖的是开头这批样本，更长样本仍可能用更多显存。
3. **训练与评测。** 用获选配置从原始权重开始正式训练，四次验证均保存完整断点。选中 VEGA 权重后冻结底座，自动训练新抽的 16K SP，最后在完整 5,130 题上输出加 SP 前后的成对成绩。

| 想看什么 | 示例数据盘上的位置 |
|---|---|
| 实时进度、错误信息 | `/mnt/data/qwen3-vega/logs/launch-*.log` |
| 六组测速和最终选择 | `/mnt/data/qwen3-vega/speed_selection.json`，单组日志位于 `logs/sweep-*/` |
| 最终分数、提升、改好/改坏题数 | `/mnt/data/qwen3-vega/results.json`，其中 `detailed_results` 指向逐题产物所在的运行目录 |

**中断后重跑同一条命令。** 同一工作目录最多启动一个一键任务。已有完整训练断点时从最近一次继续；尚未留下完整断点时保留旧输出，另开一次正式运行；底座已选好则直接接后续 SP 和评测。

这份代码用 Qwen3-VL-8B-Instruct 替换 VEGA 的语言视觉底座。视频均匀采样为 4–8 帧，冻结的 Qwen 视觉塔和 Wan2.1-T2V-1.3B 分别编码同一帧。Wan 特征经 patch merger 对齐图像 token，再通过 VEGA 的逐 token 门控残差融合进入 Qwen3。Qwen3 自带的 DeepStack 分支保留。正式训练更新语言模型、输出头和新建的融合层，用助手答案 token 的交叉熵监督，每条 QA 等权。

## 1. 准备环境和权重

在 Linux 单机上运行。单卡验证使用 Python 3.12、PyTorch 2.8.0+cu128、torchvision 0.23 和 CUDA 12.8。以下命令从已安装对应 CUDA 版 PyTorch 的环境开始。创建新的工作目录，模型、媒体和训练输出均写入本地指定目录。

```bash
git clone https://github.com/kang915-deep/SP.git
cd SP
python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r qwen3_vega/requirements.txt
mkdir -p third_party
git clone --branch spatial-reasoning https://github.com/H-EmbodVis/VEGA-3D.git third_party/VEGA-3D
git -C third_party/VEGA-3D checkout --detach ca64583b3f961849a6a0a282a3c16918321adf87
export VEGA_SRC="$PWD/third_party/VEGA-3D/src"
export PYTHONPATH="$VEGA_SRC${PYTHONPATH:+:$PYTHONPATH}"
export WORK=/path/to/new/work-directory   # 改为自己的大容量盘目录
mkdir -p "$WORK"
python qwen3_vega/scripts/download_models.py --output "$WORK/assets" --source modelscope
```

权重下载也支持 `--source hf`。脚本校验文件长度和摘要。官方 VEGA 源码提供 Wan 编码、固定提示嵌入和特征融合。环境缺少 FlashAttention 扩展时，这套单帧全窗口 Wan 使用 PyTorch SDPA。

## 2. 先在一张分配好的卡上验证流程

```bash
python qwen3_vega/scripts/make_smoke_fixture.py --output "$WORK/fixture"
CUDA_VISIBLE_DEVICES=0 bash qwen3_vega/scripts/smoke.sh \
  "$WORK/assets" "$WORK/fixture" "$WORK/runs/smoke"
```

这个可公开复现的样例会生成 3 条训练 QA 和 2 条验证 QA，同时覆盖 MP4 和帧目录两种输入。完整 Qwen3 与 Wan 参与前向，单卡更新最后一个语言层和融合部分，节约 Adam 断点空间。脚本训练两步后退出，在新进程中恢复模型和 Adam，完成第三步，按最低验证 CE 导出权重，再从导出文件重新计算验证 CE。输出目录须为新路径。

## 3. 重建正式训练集

```bash
python qwen3_vega/scripts/prepare_data.py --output "$WORK/data" --stage annotations
python qwen3_vega/scripts/prepare_data.py --output "$WORK/data" --stage media
```

第一步从固定版本的公开标注中按原实验的行号划分取样。仓库内的压缩索引只保存行的划分编号。脚本检查七类纯视频题，规范化选择题答案，再加入原来的 LLaVA-Hound 样本。每个划分都校验条数以及 ID、媒体、题型、对话的联合摘要。重新运行能得到相同的训练、验证和确认集。

原实验先排除已留作验证、确认及测试的场景，再按视频、题型、题干和选项去重，得到 242,329 条合格 VSI 视频 QA。按“数据源 × 题型”比例用最大余数法分配 234,277 个名额，各组内用固定 seed 0 排序并按场景轮流取样。发布的行号索引重现这一次抽样的成员，独立于运行机器和媒体的本地路径。

| 划分 | QA 数 | 用途 |
|---|---:|---|
| 训练 | 298,027 | VSI-590K 七类视频题 234,277 条，加 LLaVA-Hound 63,750 条 |
| 验证 | 2,036 | 四个训练进度点评估与选模 |
| 确认 | 2,049 | 独立留出 |

七类题为物体计数、物体绝对距离、物体大小、房间大小、物体相对距离、相对方向和出现顺序。第二步校验 6 个 VSI 压缩包和 16 个 Hound 媒体分片，只解出所选样本需要的文件，核对覆盖后生成 `media_ready.json`。压缩包合计约 **290 GiB**，另需解压媒体、约 22 GB 初始模型以及训练断点空间。可用 `--archives-dir /另一块盘/archives` 放置下载缓存。访问 Hugging Face 时可通过 `HF_ENDPOINT` 指定镜像。

## 4. 在 4 × A800 80 GB 上正式训练

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3     # 填写实际分配的四张卡
export OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
bash qwen3_vega/scripts/train_4gpu.sh \
  --qwen "$WORK/assets/qwen" --wan "$WORK/assets/wan" \
  --data "$WORK/data" --output "$WORK/runs/full01" --micro-batch 1
```

| 配置 | 值 |
|---|---|
| 训练预算 | 1 epoch，4,657 个优化步，有效 batch 64，末步 43 条有效 QA |
| 优化 | BF16，ZeRO-2，GPU fused AdamW，峰值学习率 1e-5，3% warmup 后余弦衰减，梯度裁剪 1 |
| 四卡并行 | 每卡 micro-batch 1 时累积 16 次，每卡 2 时累积 8 次 |
| 验证与保存 | 第 1,165、2,329、3,493、4,657 步验证并保存，最低验证 CE 胜出，同分取较早步 |

冻结的视觉塔驻留 GPU，语言层使用梯度检查点，数据加载与梯度通信可重叠。每一步在 `metrics.jsonl` 中记录有效 QA/s、训练 CE 和 rank 0 的显存峰值。四卡实机上可同时运行以下命令查看每卡负载。

```bash
nvidia-smi --id="$CUDA_VISIBLE_DEVICES" --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total --format=csv -l 2
```

**四卡先测速度。** 在明确分配的四张卡上，用相同训练样本和有效 batch 64 比较下列两次运行。测速模式执行完整前向、反向、通信和 AdamW 更新。第 1 步预热，其后 4 步计时。每组独立生成 `throughput.json`，记录整机 QA/s、各卡 PyTorch 显存峰值，以及每 0.5 秒采样的设备平均利用率和最小剩余显存。设备读数包含同卡的其他进程。测速结束即退出，省去大断点写盘及下游任务。

```bash
for stride in 1 2; do
  bash qwen3_vega/scripts/train_4gpu.sh \
    --qwen "$WORK/assets/qwen" --wan "$WORK/assets/wan" --data "$WORK/data" \
    --output "$WORK/runs/bench-m1-c$stride" --micro-batch 1 \
    --checkpoint-every "$stride" --benchmark-steps 5
done
```

`--checkpoint-every 1` 在反向传播时重算全部 36 个语言层的中间结果，这是兼容旧实验的默认设置。`2` 只重算隔层，`0` 保留所有语言层的中间结果。这里的参数控制显存，与保存训练断点的间隔无关。如果四卡实测有足够余量，可以再测 `0` 或将训练 micro-batch 调到 2，使用新的输出目录。按 QA/s 以及长样本下最紧张那张卡的余量选正式配置，把获选参数加在第 4 节的训练命令后。验证、存盘和数据准备时间应另外计入整轮用时。

单 A800 80 GB 上的对照使用同两条真实 QA、完整视觉编码、82.33 亿参数前反向，每种配置预热后交错测三次。全层、隔层、关闭重算分别为 0.442、0.454、0.470 QA/s，PyTorch 峰值分别为 39.4、45.2、50.9 GiB。该探测逐条读取媒体，未分配完整 Adam 状态或执行多卡通信。四卡正式吞吐以目标机器的报告为准。

每 250 步默认额外保存模型与 Adam，全程保留可达约 2 TiB。空间有限时可传 `--save-every 4657`，仅保留四个验证点的完整断点。可选 `--cpu-adam` 用主机内存换显存，首次使用需要本机 C++ 编译环境。

意外中断后，选输出目录中最新的 `step-XXXXXX.json`，在原启动命令末尾加上例如 `--resume step-000250`。保持原世界大小、micro-batch、优化器位置和数据。新运行和不同配置使用各自的输出目录。

## 5. 自动接训 16K SP，评测完整 VSI-Bench

**第 4 节的四卡命令包含整个流程。** VEGA 完成一轮并选定权重后，脚本导出该检查点，冻结底座，训练 SP，然后输出同一底座加 SP 前后的完整 VSI 成绩。中途的 `--stop-after` 吞吐测试不会触发后续阶段。

| 阶段 | 设置与产物 |
|---|---|
| 取样 | 从底座实际使用的 234,277 条 VSI 视频 QA 中，用 seed 42 按来源和题型比例重新抽 16,000 条。训练样本与底座重合。验证 2,036 条、确认 2,049 条沿用原独立划分。这批训练题与旧 Qwen2.5 版 VEGA 的历史 16K 分开记录。 |
| 训练 SP | 按媒体分给已指定的四张卡提特征，SP 小头在第一张卡训练。原回答和候选由入选的 Qwen3-VEGA 产生。使用候选与上下文隐藏状态、候选概率、数值关系，以及独立从 16 帧视频提取的 Wan block 20 记忆。 |
| 选模 | 宽 128，FP32 AdamW，batch 32，utility KL 温度 0.25。只在验证集校准每类改答门槛，按视频等权得分选 epoch，同分优先少改坏。最少 10 轮，按平台降学习率，最多 160 轮。若到上限仍未满足平台判据，报告中标记 `adequacy_passed=false`。 |
| 完整 VSI | 底座每个视频均匀取最多 32 帧，贪心解码，最多生成 16 个 token。SP 从同一个原回答决定保留或改答，额外 Wan 记忆使用 16 帧。未校准的题型保留原答。冻结选模及门槛后，对 5,130 题成对计分。 |

正式训练成功后，结果位于 `$WORK/runs/full01/sp16k_vsi/`。

- `results.json` 包含 VEGA 分数、VEGA＋SP 分数、差值、改好/改坏题数和各题型结果。总分按官方八类宏平均，方向题三个难度先平均。
- `benchmark_predictions.json` 和 `per_question_scores.json` 保存逐题的原答、SP 答案及得分变化。数值题按相对误差获得部分分。
- `export/` 是选中的 VEGA 基座，`sp.pt` 和 `sp_selection.json` 是配套 SP 权重、改答门槛与训练曲线。`confirm_results.json` 是独立确认集结果。

VSI-Bench 默认从官方固定版本下载，3 个视频 ZIP 合计约 5.73 GB，解压还需要约一份视频的空间。SP 首次生成固定文本嵌入还需要 Wan 的 UMT5 权重，可复用已有的官方 Wan 目录。另留空间保存按视频分片的候选与 Wan 特征。仅想提前下载 VSI 时可以单独执行下面的命令。[官方数据源](https://huggingface.co/datasets/nyu-visionx/VSI-Bench/tree/3a6a1cf9ac54c403509b15260897cc6ddc41b408)

```bash
# 正式训练开始后，或者提前在单独的缓存目录下载
python qwen3_vega/sp/prepare.py --output "$WORK/vsi-cache"
```

若 VEGA 已在旧版本上训练完成，或下载、提特征等后续步骤意外中断，可在原机器上直接执行下列命令。它复用原检查点、已完成的视频特征以及最近完整的 SP 训练轮次。指定预下载目录时会复用该目录中的视频和标注。

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VSI_BENCH_DIR="$WORK/vsi-cache"   # 可选，供预下载和多个实验共用
python qwen3_vega/sp/pipeline.py --output "$WORK/runs/full01" \
  --qwen "$WORK/assets/qwen" --wan "$WORK/assets/wan" --data "$WORK/data"
```

## 6. 单独导出并复核验证 Loss

```bash
python qwen3_vega/export.py --run "$WORK/runs/full01" --output "$WORK/artifacts/best"
CUDA_VISIBLE_DEVICES=0 python qwen3_vega/check_export.py \
  --qwen "$WORK/assets/qwen" --wan "$WORK/assets/wan" \
  --data "$WORK/data" --export "$WORK/artifacts/best"
```

导出目录含 `vega.safetensors` 和记录选模步数、验证 CE、底座名称及划分摘要的 `config.json`。重载同时需要原始 Qwen3 和 Wan 目录。这里复核的是训练验证集上的答案 CE。完整 VSI-Bench 问答分数属于下游评测。

## 实测范围

SP 接续流程已在一张 A800 80 GB 上验收。实际重建了 16,000 / 2,036 / 2,049 条划分，训练集 ID 摘要为 `f2ba5018a37c056a885d356266a55749be06af63e0c5593efdbd98d18524622c`。官方完整 VSI 下载覆盖 5,130 题、288 个视频。用小样本导出的 VEGA 权重、正式 Wan 提示和真实计数及方向题跑通了特征提取、两轮 SP 更新、重载和 32 帧成对计分。另用测试预测对照原计分实现，5,130 个逐题分数与八类宏平均一致。正式四卡模型的完整对比成绩须等训练与评测结束后读取。

在一张 A800 PCIe 80 GB 上，另取真实训练划分中的 3 条 QA（覆盖 VSI 视频及 Hound 帧目录）和验证划分中的 2 条 QA，完成三步、第二步 Adam 断点恢复以及第三步选模。更新参数量为 234,923,778。选中验证 CE 和导出后重新计算的 CE 均为 **1.81635201**。此数值用于核对权重重载。公开标注重建已核对全部三个划分。整套媒体下载、四卡通信、四卡正式训练吞吐和完整 298,027 条训练仍需在目标机器上实测。

另用两条真实 QA 合批，对正式配置的 **8,232,712,706** 个可训练参数做了一次反向传播。第 1 和第 36 个语言层、输出头以及 Wan merger 均得到有限非零梯度。此次探测未分配完整 Adam 状态，PyTorch 峰值为 38.05 GiB。两条 QA 在 BF16 下合批与单条计算的 CE 最大差为 0.0382，因此正式验证和导出复核均固定每卡逐条计算，可独立调整训练 micro-batch。
