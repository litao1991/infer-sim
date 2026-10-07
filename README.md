# InferenceSim v1.1

基于 HBMSim 与 HBFSim 的大模型推理系统仿真器。支持 Dense/MoE、GPU/NPU
参数配置、TP/PP/DP/EP、HBF 分层存储、单机与跨机柜互联。主机只执行事件和
元数据计算，不加载模型权重、不执行实际神经网络计算。

## 快速运行

```sh
git clone https://github.com/litao1991/infer-sim.git
cd infer-sim
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-yaml.txt
.venv/bin/python tools/bootstrap_deps.py
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build build --target infer_sim -j 2
.venv/bin/python run.py examples/moe-small.json --output reports/moe.json
.venv/bin/python run.py examples/storage-hbf-hbf.yaml --output reports/storage-both.json
ctest --test-dir build --output-on-failure
```

需要 C++20、CMake 和 Python 3.9+；JSON 场景只需要 Python 标准库。
YAML 支持使用可选的 PyYAML 依赖，安装与使用见下一节。
`dependencies.lock.json` 固定 HBMSim/HBFSim 提交版本。初始化脚本将源码下载到
忽略提交的 `.deps/`，并应用仓库附带的 `patches/hbfsim-shared-hbm.patch`，使权重
Fill、KV 和激活访问共享同一 HBM 控制器。脚本可重复运行；不会重置已有依赖修改。
也可通过 `-DHBFSIM_SOURCE_DIR=... -DHBMSIM_SOURCE_DIR=...` 指定自己的本地源码，
其中 HBFSim 必须提供 `ResidencyManager::try_submit_hbm`。详见
[依赖版本与补丁](docs/DEPENDENCIES.md)。

## 用 YAML 切换硬件架构

已支持 `.json`、`.yaml` 和 `.yml`，统一使用相同的参数校验和仿真核心。
同一模型可以通过独立配置切换硬件与并行放置：

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-yaml.txt

# 本项目当前已安装好上述独立环境，可以直接运行。
.venv/bin/python run.py examples/moe-small.yaml \
  --architecture configs/hardware/gpu-8card-crossrack.yaml \
  --output reports/yaml-crossrack.json

.venv/bin/python run.py examples/moe-small.yaml \
  --architecture configs/hardware/npu-2card.yaml \
  --output reports/yaml-npu.json

.venv/bin/python tools/experiments.py sweep examples/moe-small.yaml \
  --architecture configs/hardware/gpu-single.yaml \
  --grid examples/sweep-small.yaml --output reports/yaml-sweep
```

现成的硬件配置：

| 配置 | 架构 |
| --- | --- |
| [gpu-single.yaml](configs/hardware/gpu-single.yaml) | 单卡 GPU |
| [gpu-8card-nvswitch.yaml](configs/hardware/gpu-8card-nvswitch.yaml) | 单机 8 卡，NVSwitch |
| [gpu-8card-crossrack.yaml](configs/hardware/gpu-8card-crossrack.yaml) | 2 机柜 × 每柜 2 节点 × 每节点 2 卡 |
| [npu-2card.yaml](configs/hardware/npu-2card.yaml) | 2 卡 NPU，包含 SRAM/DMA 参数 |

`--architecture` 替换完整的 `hardware` 和可选 `parallel` 段，避免切换设备后残留旧
字段。模型、请求和服务参数取自场景文件。四个配置均可搭配提供的小型 Dense/MoE
场景；数值用于本地功能验证，不代表某款真实芯片的实测规格。所有设备当前共用
一套算力/容量参数，尚不支持同一场景中每张卡采用不同 GPU/NPU 型号。

场景、参数扫描表、校准输入与 profile 都可用 YAML；报告仍统一输出 JSON。
完整字段与计时边界见 [硬件配置说明](docs/HARDWARE_CONFIG.md)。

## 主流模型预设

`--model` 从独立 YAML/JSON 加载模型，可以与 `--architecture` 一起使用。
已提供 10 个预设：Llama 3.1 8B/70B、Qwen2.5 7B/32B、Qwen3 8B/32B、
Qwen3 30B-A3B/235B-A22B、Mixtral 8x7B/8x22B。

```sh
.venv/bin/python run.py examples/model-evaluation.yaml \
  --model configs/models/qwen3-30b-a3b.yaml \
  --mode fast --output reports/qwen3-30b.json

# 从本地官方配置导入，不需要模型权重
.venv/bin/python tools/import_model.py configs/models/upstream/qwen3-30b-a3b.json \
  --model-id Qwen/Qwen3-30B-A3B --output /tmp/qwen3-imported.yaml
```

真实尺寸模型使用快速估算，小尺寸模型用于详细事件验证。预设记录来源与原始维度，
报告标记修改项；支持 Qwen3 独立的 `head_dim`。当前统计 Transformer 层矩阵，
词嵌入、输出头等完整模型开销未包含；DeepSeek 的 MLA 尚不支持。
支持列表、配置组合与精度边界见 [模型配置说明](docs/MODEL_CONFIG.md)。

## 权重与 KV 独立放置

支持权重、KV 分别选择 HBM 或 HBF 后备，包含全部四种组合，以及按层/专家的
HBM 常驻覆盖。KV 支持按块增量写、脏块写回、回迁、LRU 或近期块优先保留，
并单独报告权重和 KV 的流量、等待与容量。

```sh
.venv/bin/python run.py examples/storage-hbf-hbf.yaml \
  --output reports/storage-both.json --trace-out reports/storage-both-trace.json
.venv/bin/python run.py examples/storage-mixed-moe.yaml \
  --output reports/storage-mixed.json
```

HBF 数据经有界 HBM 缓存访问；物理后备使用有限的追加日志，当前没有 GC/擦除
回收，耗尽时明确报错。配置字段、7 个可运行示例与计时范围见
[存储放置说明](docs/STORAGE_CONFIG.md)。

## 已实现

| 版本 | 功能 |
| --- | --- |
| v0.1–0.2 | 版本化 JSON、解析估算、统一皮秒事件时钟、确定性的执行依赖图。 |
| v0.3–0.4 | HBMSim 时序、访问背压、连续批处理、分块 Prefill、逐 token Decode、KV 容量准入。 |
| v0.5–0.6 | HBF Fill/驻留/淘汰/预取/显式回写；TP Ring AllReduce、PP 流水、DP 副本、NVLink/NVSwitch。 |
| v0.7 | 不重复 top-k 路由、路由倾斜、专家放置、EP、双向 All-to-All、Grouped GEMM、共享专家和拖尾统计。 |
| v0.8 | 设备→节点→机柜拓扑、NIC 竞争、跨机柜链路、分层 AllReduce、物理放置扫描和各层流量/等待。 |
| v0.9 | GPU/NPU 共用执行图；GEMM/Attention/Grouped GEMM 独立参数，SRAM/DMA、HBM2/3/4 后端配置。 |
| v1.0 | 计算服务校准、留出集误差、外部推理框架测量接口、参数扫描、微基准场景、分布/瓶颈/关键路径报告。 |
| v1.1 | 权重/KV 独立 HBM/HBF 放置、层/专家覆盖、KV 分页/脏写回/回迁、近期块策略、独立容量和搬运统计。 |

## 示例与实验

百炼公开 trace 的流式统计、确定性窗口抽样和权重/KV 放置评估见
[百炼 trace 评估](docs/BAILIAN_EVAL.md)。原始 JSONL 与详细报告保存在忽略提交的
`data/bailian/` 和 `reports/bailian/`，避免将大文件推送到仓库。

```sh
# 顺序执行 8 种 top-k × 精度 × 集合通信算法组合
python3 tools/experiments.py sweep examples/moe-small.json \
  --grid examples/sweep-small.json --output reports/sweep

# 6 个 prompt 长度 × batch 大小的可重复仿真微基准
python3 tools/experiments.py microbench examples/dense-small.json --output reports/microbench

# 验证校准流程；此文件明确是合成数据，不是硬件实测
python3 tools/experiments.py calibrate examples/calibration-synthetic.json \
  --output reports/synthetic-profile.json
python3 run.py examples/npu-calibration-small.json \
  --profile reports/synthetic-profile.json --output reports/calibrated-synthetic.json

# 对接真实框架：adapter 的协议见 docs/CALIBRATION.md
python3 tools/reference_runner.py reports/prediction.json --output reports/reference.json \
  --adapter python3 /path/to/your_framework_adapter.py
python3 tools/experiments.py compare reports/prediction.json reports/reference.json \
  --output reports/compared.json
```

扫描支持 `model.*`、`hardware.*`、`parallel.*`、`serving.*`、`storage.*`；
同时改变设备数和并行度时，也需相应设置 `placement` 和 `device_nodes`。
失败组合写入汇总且命令返回非零，便于自动化检查。每次最多 256 个组合。

## 配置约定

- `config_version`: `"1.0"`；旧例子未填写时自动补齐。
- `model`: `kind` 为 `dense`/`moe`；基本维度为 `layers`、`hidden`、`heads`、
  `kv_heads`、`head_dim`、`ffn`、`dtype_bytes`。`head_dim` 省略时为 `hidden / heads`。
  MoE 增加 `experts`、`top_k`、
  `shared_experts`、`routing_seed`、`expert_weights`。
- `parallel`: `tp * pp * dp * ep = devices`；`pp` 整除层数。
  `expert_placement[e]` 指定专家的 EP lane；`placement` 将逻辑设备映射到物理设备。
- `hardware`: 每卡算力/效率/容量，`kind`、`hbm_profile`、互联、节点/机柜编号，
  可选算子效率。NPU 还需 `sram_bytes`、`dma_tile_bytes`、`dma_GBps`、`dma_latency_ps`。
- `serving`: `max_batch_tokens`、`max_seqs`、`prefill_chunk_tokens`、`kv_block_tokens`。
- `storage`: `weights`、`kv` 独立选择 `hbm`/`hbf`，固定 HBM 层/专家、权重缓存、
  KV 缓存、逻辑准入容量和物理 HBF 容量；旧 `hbf` 开关继续表示权重后备。
  服务中权重只读；`writeback_layer` 保留为服务结束后的权重回写验证。
- `requests`: 按到达时间排序的 `arrival_ps`、`prompt_tokens`、`output_tokens`。

详细公式、并行映射、计时边界见 [MODEL.md](docs/MODEL.md)。

## 报告和准确性边界

每份报告包含解析后的配置及 SHA-256、源码与可执行文件指纹、依赖版本、校准来源
和误差状态。详细模式还包含 TTFT/TPOT 分布、每专家路由数、每卡计算时间、各网络
层字节数/排队时间、SRAM/DMA 统计、执行依赖图及关键路径。`--trace-out` 导出
Chrome/Perfetto 时间线；报告时间单位为 **ps**，`1 µs = 1,000,000 ps`。

这是具有明确粒度的性能仿真器：使用算子级服务模型、粗粒度 HBM 访问和消息级网络
模型，不模拟 GPU/NPU 指令、数值结果或网络包。详细模式由 `hbm_profile` 决定
HBM 时序；`hbm_bandwidth_GBps` 用于快速估算。NPU 的 DMA 是串行分析模型。
FP8/INT8 等精度只通过 `dtype_bytes` 和对应算力/校准参数描述，不预测数值精度。

提供的校准样例均为 **synthetic**。没有真实测量时 `prediction_error.status`
明确为 `unmeasured`，不能把自测通过视为绝对预测误差达标。真实 GPU/NPU 的
独立校准需要相应设备的微基准和推理框架测量数据。

详细模式限制为最多 16 卡、16 层、128 个请求、1,024 个总 token，每层完整模型
权重最多 128 KiB、HBF 初始化最多 1 MiB。大型模型使用 `--mode fast`，只输出
解析估算与容量，不生成延迟分布或网络竞争预测。
显式存储配置另限制每类缓存最多 1 MiB、HBF 物理容量最多 8 MiB、KV 页最多
64 KiB、保守总块访问数最多 8,192 次，以控制本地事件规模。

[验证说明](docs/VALIDATION.md) · [本次结果](reports/VALIDATION_REPORT.md) · [校准协议](docs/CALIBRATION.md)
