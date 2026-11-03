# Qwen3-VL 版 VEGA 训练

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

四卡首次调优可在独立输出目录以 `--stop-after 3` 比较 `--micro-batch 1` 和 `2`，保持有效 batch 64，按稳定步 QA/s 和四卡显存余量选正式配置。两次短测也会保存完整断点，需要预留磁盘。每 250 步默认额外保存模型与 Adam，全程保留可达约 2 TiB。空间有限时可传 `--save-every 4657`，仅保留四个验证点的完整断点。可选 `--cpu-adam` 用主机内存换显存，首次使用需要本机 C++ 编译环境。

意外中断后，选输出目录中最新的 `step-XXXXXX.json`，在原启动命令末尾加上例如 `--resume step-000250`。保持原世界大小、micro-batch、优化器位置和数据。新运行和不同配置使用各自的输出目录。

## 5. 导出和复核

```bash
python qwen3_vega/export.py --run "$WORK/runs/full01" --output "$WORK/artifacts/best"
CUDA_VISIBLE_DEVICES=0 python qwen3_vega/check_export.py \
  --qwen "$WORK/assets/qwen" --wan "$WORK/assets/wan" \
  --data "$WORK/data" --export "$WORK/artifacts/best"
```

导出目录含 `vega.safetensors` 和记录选模步数、验证 CE、底座名称及划分摘要的 `config.json`。重载同时需要原始 Qwen3 和 Wan 目录。这里复核的是训练验证集上的答案 CE。完整 VSI-Bench 问答分数属于下游评测。

## 实测范围

在一张 A800 PCIe 80 GB 上，另取真实训练划分中的 3 条 QA（覆盖 VSI 视频及 Hound 帧目录）和验证划分中的 2 条 QA，完成三步、第二步 Adam 断点恢复以及第三步选模。更新参数量为 234,923,778。选中验证 CE 和导出后重新计算的 CE 均为 **1.83678967**。此数值用于核对权重重载。公开标注重建已核对全部三个划分。整套媒体下载、四卡通信、四卡正式训练吞吐和完整 298,027 条训练仍需在目标机器上实测。

另用两条真实 QA 合批，对正式配置的 **8,232,712,706** 个可训练参数做了一次反向传播。第 1 和第 36 个语言层、输出头以及 Wan merger 均得到有限非零梯度。此次探测未分配完整 Adam 状态，PyTorch 峰值为 38.05 GiB。两条 QA 在 BF16 下合批与单条计算的 CE 最大差为 0.0382，因此正式验证和导出复核均固定每卡逐条计算，可独立调整训练 micro-batch。
