# 主流模型配置

模型、硬件架构与请求负载可独立配置，支持 YAML 和 JSON。模型预设包含结构维度，
不包含权重；运行时不会下载或分配实际参数张量。

## 直接运行

在 `infer-sim` 目录中使用已安装 PyYAML 的项目环境：

```sh
.venv/bin/python run.py examples/model-evaluation.yaml \
  --model configs/models/qwen3-30b-a3b.yaml \
  --mode fast --output reports/qwen3-30b.json
```

`model-evaluation.yaml` 提供一条 512-token 输入、128-token 输出的负载，以及假设的
8 × 80 GiB GPU、TP=8 配置。算力与带宽数值用于演示，没有对应产品的实测校准。
实际尺寸模型使用 `--mode fast`；它给出层内参数量、计算量、KV/权重容量和计算时间
下界，不生成 TTFT/TPOT 或通信竞争结果。事件级验证继续使用小尺寸模型。

也可同时替换架构：

```sh
.venv/bin/python run.py examples/model-evaluation.yaml \
  --model configs/models/llama-3.1-8b.yaml \
  --architecture configs/hardware/gpu-8card-crossrack.yaml \
  --mode fast --output reports/llama-crossrack-capacity.json
```

已有 `configs/hardware/` 预设是小模型功能验证配置，HBM 容量只有 1 MiB；直接配真实
模型会得到容量不足的结果。评估实际设备时应复制该文件，填写目标硬件的每卡容量、
算力、带宽与拓扑。字段说明见 [HARDWARE_CONFIG.md](HARDWARE_CONFIG.md)。

加载顺序：场景 → `--model` 完整替换模型段 → `--architecture` 完整替换硬件及
可选并行段 → 统一校验。服务调度与请求取自场景。并行度仍需满足层数、隐藏维度、
FFN 维度等约束；例如切换到 Dense 后 EP 必须为 1，MoE 自定义专家放置需匹配专家数。
模型来源信息会随预设替换，避免沿用前一个模型的标签。

## 内置预设

表中 F 为 Dense FFN 或单个 MoE 专家的中间维度。均使用 2 字节权重/KV；名称中的
参数量是发布者的模型名称，仿真报告的参数统计只包含已建模的 Transformer 层矩阵。

| 配置文件（`configs/models/`） | 类型 | 层数 | H | Q heads / KV heads | head_dim | F | 专家数 / top-k |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- |
| `llama-3.1-8b.yaml` | Dense | 32 | 4096 | 32 / 8 | 128 | 14336 | — |
| `llama-3.1-70b.yaml` | Dense | 80 | 8192 | 64 / 8 | 128 | 28672 | — |
| `qwen2.5-7b.yaml` | Dense | 28 | 3584 | 28 / 4 | 128 | 18944 | — |
| `qwen2.5-32b.yaml` | Dense | 64 | 5120 | 40 / 8 | 128 | 27648 | — |
| `qwen3-8b.yaml` | Dense | 36 | 4096 | 32 / 8 | 128 | 12288 | — |
| `qwen3-32b.yaml` | Dense | 64 | 5120 | 64 / 8 | 128 | 25600 | — |
| `qwen3-30b-a3b.yaml` | MoE | 48 | 2048 | 32 / 4 | 128 | 768 | 128 / 8 |
| `qwen3-235b-a22b.yaml` | MoE | 94 | 4096 | 64 / 4 | 128 | 1536 | 128 / 8 |
| `mixtral-8x7b.yaml` | MoE | 32 | 4096 | 32 / 8 | 128 | 14336 | 8 / 2 |
| `mixtral-8x22b.yaml` | MoE | 56 | 6144 | 48 / 8 | 128 | 16384 | 8 / 2 |

Qwen 与 Mixtral 维度来自发布者的 `config.json`，快照保存于
`configs/models/upstream/`。每份预设记录原始文件 SHA-256、仓库提交和固定版本 URL。
例如 [Qwen3-30B-A3B 官方配置](https://huggingface.co/Qwen/Qwen3-30B-A3B/resolve/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39/config.json)
及 [Mixtral-8x22B 官方配置](https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1/resolve/cc88a6cc19fbd17d9f1c0ee0b0d70a748dce698d/config.json)。
Llama 维度采用作者论文 [The Llama 3 Herd of Models，表 3](https://arxiv.org/html/2407.21783v3#S3.T3)；
未下载需要授权的 checkpoint 配置，其 2 字节精度为明确标注的仿真假设。

## 自定义模型 YAML

最小预设只需要 `model` 段。下面与内置 Qwen3-30B-A3B 的已建模结构一致：

```yaml
model:
  kind: moe
  layers: 48
  hidden: 2048
  heads: 32
  kv_heads: 4
  head_dim: 128
  ffn: 768
  dtype_bytes: 2
  experts: 128
  top_k: 8
  shared_experts: 0
  routing_seed: 1
```

- `head_dim` 可独立于 `hidden / heads` 设置；省略时沿用旧版的整除推导。
  Qwen3 的部分模型必须显式指定，不能按隐藏维度反推。
- `ffn` 是单个专家的维度；总专家数与每 token 激活专家数分别为 `experts`、`top_k`。
- `shared_experts` 和 `expert_weights` 用于抽象共享专家与路由倾斜。预设未加载模型
  学得的路由器；详细模式使用可重复的合成路由。
- Dense 使用 `kind: dense`，不需要专家字段。
- `dtype_bytes` 当前同时影响权重、KV 与激活；修改它不等价于完整实现某种量化方法，
  量化尺度/零点、反量化成本和精度损失尚未建模，算力也需相应配置或校准。

可选 `model_metadata` 保存身份、来源与 `reference_dimensions`。改变原始维度后，
报告的 `model_accounting.overrides_from_preset` 会列出变更，便于识别缩小模型或精度扫描。

## 从本地 Hugging Face 配置导入

先准备已取得的本地 `config.json`，只需配置文件：

```sh
.venv/bin/python tools/import_model.py \
  configs/models/upstream/qwen3-30b-a3b.json \
  --model-id Qwen/Qwen3-30B-A3B \
  --output /tmp/qwen3-imported.yaml

.venv/bin/python run.py examples/model-evaluation.yaml \
  --model /tmp/qwen3-imported.yaml --mode fast
```

可选 `--source-url`、`--revision` 记录数据出处。输出 `.json` 时 Python 标准库即可运行；
YAML 需要 `requirements-yaml.txt`。导入器支持 `llama`、`qwen2`、`qwen3`、`qwen3_moe`、
`mixtral` 的受支持结构；校验精度、SwiGLU、attention 和专家维度，不执行远程模型代码。

当前明确拒绝 MLA、异构逐层布局、启用的滑动窗口、量化 checkpoint 元数据与未知模型
类型。**DeepSeek-V3/R1 的 MLA 和混合 Dense/MoE 层尚未接入**，不能通过改模型名称或
把维度填进普通 GQA 获得可信结果。仓库中的 DeepSeek-V3 配置仅用于拒绝路径测试。

## 方案扫描

扫描入口同样接受 `--model` 和 `--architecture`：

```sh
.venv/bin/python tools/experiments.py sweep examples/model-evaluation.yaml \
  --model configs/models/qwen3-30b-a3b.yaml --mode fast \
  --grid examples/sweep-small.yaml --output reports/qwen3-sweep
```

该扫描表包含 top-k、字节宽度与通信算法组合。快速模式只改变解析工作量，不执行
通信算法；要比较分层集合通信和网络竞争，使用缩小结构的详细模式。

## 报告的解释范围

`model_parameters`、`active_parameters_per_token` 与 `weight_bytes_total` 当前只统计
Transformer 层内投影、FFN/专家与路由门控矩阵；不包含词嵌入、LM 输出头、归一化参数、
投影 bias，也不单独计激活函数、RoPE 与采样计算。Qwen3 的 Q/K norm 未单独计时。
因此这些值会小于发布者完整 checkpoint 的参数量，`hbm_capacity_feasible` 也只判断
已建模权重、KV 和临时空间，不能作为完整 checkpoint 可部署的保证。

TP 采用理想分片模型，尚未模拟真实推理框架在 TP 大于 KV head 数时的 KV head 复制。
在这类布局下，每卡 KV 容量可能被低估；选择部署方案时需要加入框架的复制规则。
报告的 `model_accounting` 会附带该假设、统计范围、来源和配置覆盖项。

快速模式的计算时间是理想设备利用率下界；内存估算不含批次复用，MoE 保守读取全部
专家；二者均不能直接解释成实际请求延迟。实际性能校准见 [CALIBRATION.md](CALIBRATION.md)。
