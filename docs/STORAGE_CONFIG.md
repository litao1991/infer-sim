# 权重与 KV 的 HBM/HBF 放置（v1.1）

## 支持范围

权重和 KV 可以独立选择 HBM 或 HBF，支持四种组合。HBF 表示后备存储：计算前
通过 HBM 缓存访问数据。权重缓存与 KV 缓存分别预留容量，同一卡上的固定权重、
缓存 Fill、KV 读写、激活与回写都通过同一个 HBMSim 控制器，保留队列和带宽争用。

| 权重 | KV | 小模型示例 |
| --- | --- | --- |
| HBM | HBM | `examples/storage-hbm-hbm.yaml` |
| HBF → HBM 缓存 | HBM | `examples/storage-hbf-hbm.yaml` |
| HBM | HBF → HBM 缓存 | `examples/storage-hbm-hbf.yaml` |
| HBF → HBM 缓存 | HBF → HBM 缓存 | `examples/storage-hbf-hbf.yaml` |
| 按层、专家混合 | 按层混合 | `examples/storage-mixed-moe.yaml` |
| HBM | 最近 KV 块优先保留 | `examples/storage-hot-kv.yaml` |
| HBM | 干净块优先淘汰、未封口脏块优先保留 | `examples/storage-write-aware.yaml` |
| HBF → HBM 缓存 | HBF → HBM 缓存，NPU/EP | `examples/storage-npu-moe.yaml` |

## 直接运行

在项目目录中：

```sh
.venv/bin/python run.py examples/storage-hbf-hbf.yaml \
  --output reports/storage-both.json --trace-out reports/storage-both-trace.json

.venv/bin/python run.py examples/storage-mixed-moe.yaml \
  --architecture configs/hardware/npu-2card.yaml \
  --output reports/storage-mixed-npu.json
```

示例为 1/2 卡、2 层的小模型，不加载实际权重或 KV 张量。模型预设与架构预设仍可
通过 `--model`、`--architecture` 组合；真实尺寸模型使用 `--mode fast`，只检查解析
工作量与容量，不能预测 KV 换出/回迁次数或追加日志耗尽时间。

## 配置字段

下面是两者使用 HBF 的可运行小模型存储段：

```yaml
storage:
  weights: hbf
  kv: hbf

  cache_bytes_per_device: 8192      # HBM 中的权重缓存预留
  slot_bytes: 8192                 # 每个权重缓存槽的大小
  prefetch: false                  # 权重下一层预取

  kv_cache_bytes_per_device: 4096  # HBM 中的 KV 缓存预留
  kv_capacity_bytes_per_device: 65536 # 每卡允许存活的 offload KV 页容量
  hbf_capacity_bytes: 1048576      # 全场景共享的 HBF 物理追加空间，含权重后备
  kv_cache_policy: lru
```

所有容量单位为 **字节**。`kv_capacity_bytes_per_device` 是逻辑准入预算，包括当前
在 HBM 缓存中的 offload KV 页；它不是额外的 HBM 预留，也不是 HBF 的物理容量。
完成请求会归还逻辑预算、释放缓存槽。`hbf_capacity_bytes` 是整个仿真共享的 HBF
容量；DP 权重副本、所有卡的 KV 脏版本都会使用它。

| 字段 | 含义与约束 |
| --- | --- |
| `weights` | `hbm` 或 `hbf`。未设置时继承旧 `hbf` 开关；旧开关也未设置时默认 HBM。 |
| `kv` | `hbm` 或 `hbf`，默认 HBM。 |
| `weight_hbm_layers` | 指定常驻 HBM 的层编号，从 0 开始；该层基础权重和所有专家都常驻。 |
| `weight_hbm_experts` | 指定所有层中常驻 HBM 的路由专家编号；与层覆盖取并集，不包含共享专家。 |
| `kv_hbm_layers` | 指定 KV 固定留在 HBM 的层；其余层使用 HBF 后备和 KV 缓存。 |
| `cache_bytes_per_device` / `slot_bytes` | HBF 权重启用时必填；slot 按 4 KiB 对齐，缓存容量为 slot 的倍数，需放得下最大的后备权重对象。 |
| `kv_cache_bytes_per_device` | KV HBF 启用时必填，至少容纳一个 KV 页，为页大小的整数倍。 |
| `kv_capacity_bytes_per_device` | KV HBF 启用时必填，为页大小的整数倍，且不小于 KV 缓存。 |
| `hbf_capacity_bytes` | KV HBF 启用时必填，4 KiB 对齐；权重 HBF 单独使用时可省略，由小场景自动预留。 |
| `kv_cache_policy` | `lru`、`hot` 或 `write_aware`，默认 `lru`。 |
| `kv_hot_blocks` | `hot` 策略下每个请求、每层最近多少个块优先保留，默认 1。0 表示不保护。 |
| `prefetch` | 仅控制权重预取；KV 当前按需回迁。 |
| `writeback_layer` | 兼容原有服务结束后的权重写回验证，不是 KV offload 开关。 |

混合放置示例：

```yaml
storage:
  weights: hbf
  weight_hbm_layers: [0]
  weight_hbm_experts: [0]
  kv: hbf
  kv_hbm_layers: [0]
  cache_bytes_per_device: 8192
  slot_bytes: 8192
  prefetch: false
  kv_cache_bytes_per_device: 4096
  kv_capacity_bytes_per_device: 65536
  hbf_capacity_bytes: 1048576
  kv_cache_policy: hot
  kv_hot_blocks: 1
```

这个配置把第 0 层全部权重、各层专家 0，以及第 0 层的 KV 常驻 HBM。MoE 的
Attention、门控与共享专家属于基础权重对象；可通过层规则固定其位置。
覆盖列表要求唯一、合法的编号，只能在相应默认层级为 HBF 时使用。

旧 `storage.hbf` 仍表示权重 HBF，KV 保持 HBM；若同时填写 `weights`，二者必须一致。

## KV 分页、回写和热度

一个逻辑块由 `serving.kv_block_tokens` 个 token、一个请求和一层组成。TP rank
分别持有其 KV 分片；当前 offload 要求单 token KV 字节数能被 TP 整除。

每卡物理 KV 缓存页大小为：

```text
align_up(kv_block_tokens × 2 × kv_heads × head_dim × dtype_bytes / TP, 4096)
```

- 新块在 HBM 创建；每轮只新增本轮 token 的 KV，已有 token 的版本不重复写入。
- 读取历史块时，缓存命中直接读取 HBM；未命中则等待 HBF → HBM 完成。
- 淘汰干净块可直接释放缓存槽；脏块先经 HBM 读出，再按 4 KiB DLU 写入新的 HBF
  地址。后备地址和版本在成功完成后更新，然后才释放旧槽。
- 脏状态按有效 token 前缀跟踪；回写使用整页，因此物理写入量可大于新增 KV 字节。
  回迁检查后备版本与逻辑有效 token 一致，避免读到尚未完成的增量更新。
- 请求结束后，其 KV 生命周期结束，剩余脏块直接丢弃并单独计数，无需为了回写而
  延长请求延迟；已经写入 HBF 的旧版本仍占本轮物理日志空间。
- `lru` 淘汰最久未使用的块。`hot` 优先淘汰不在近期窗口中的块，同组内使用 LRU；
  如果缓存全是热块，仍允许淘汰热块，保证单槽场景能继续执行。这里的热度是最近
  token 的窗口代理，不是学习得到的 Attention 权重或访问频率预测。
- `write_aware` 依次选择干净块、已填满的脏块、仍在追加的未满脏块作为淘汰对象，
  同类内使用 LRU。这样尽量让未满块在 HBM 合并多个 token 的更新，并避免尾块
  反复写新版本。偏好是软约束；缓存全部为未满脏块时仍会回写，保证进度。

例如下面配置用块大小调节有效数据量，用淘汰策略控制重复写入：

```yaml
serving:
  kv_block_tokens: 64
storage:
  weights: hbm
  kv: hbf
  kv_cache_bytes_per_device: 8192
  kv_capacity_bytes_per_device: 65536
  hbf_capacity_bytes: 1048576
  kv_cache_policy: write_aware
```

在当前小模型中每 token KV 为 32 字节，4/16/64-token 块的有效载荷为
128/512/2048 字节，均映射到一个 4 KiB 页。增大块可减少填充浪费和请求次数，
也可能增加部分有效页的读取量、改变准入粒度；实际模型应按每卡 KV 字节和物理
DLU 对齐关系选择块大小，不能固定套用 64。已填满表示逻辑块封口，并不一定填满
整个物理页。

当前策略仍是脏淘汰时整页同步写回。尚无跨块打包的写合并缓冲区、按 dirty delta
记录的日志、后台批量刷写与超时触发。后续实现这些能力必须同时计入 HBM 缓冲
容量、未持久化版本的读取转发、压力刷写以及请求完成时取消写入，避免把缓存
里尚未写入 HBF 的数据视为已经持久化。完整 Attention 仍逐 token 读取历史 KV，
因此减少写回不保证减少 HBF 读取或 TTFT/TPOT。

逻辑准入会为请求完整生成长度预留 KV 容量；容量不足的请求等待已有请求完成。
单个请求超过配置的逻辑容量时明确失败。HBM 固定 KV 层沿用容量预算机制。

## 计时与容量边界

显式填写 `weights` 或 `kv` 时，四种组合使用统一的执行顺序：权重/激活访问 →
QKV 投影 → KV 增量写和逐块读取 → Attention 与其余矩阵计算。NPU 还计入相应
SRAM/DMA 服务。Attention 计算在块流量完成后汇总收费，没有模拟块内计算与搬运
流水重叠，因此属于保守模型。HBM 常驻层和 HBF 层都使用该顺序，方便比较放置策略。

未使用新字段的旧场景保留原有算子聚合路径；报告的 `storage_model.execution_mode`
区分 `block_stream` 和 `legacy_aggregate`。比较存储策略时应统一采用显式配置。

HBM 容量检查包含固定权重、权重缓存预留、KV 缓存预留、固定 KV 层和临时空间。
每卡预留使用跨 PP stage/EP lane 的保守最大值，因此可能高于具体某张卡的用量。
缓存预留不会因为当前没有活跃请求而消失。HBF 权重只为实际所属 PP/EP 设备建立后备。

HBF 使用一个共享后端；KV 搬运经过一个串行提交队列，并与权重 Fill 竞争同一后端。
**当前 HBF 为有限的追加日志，没有 GC/擦除回收模型**。旧版本不会被覆盖，也不会
免费腾出物理空间；空间耗尽时返回 `HBF append log capacity exhausted`。扩大
`hbf_capacity_bytes` 可评估更长的有界运行；长期稳态磨损和 GC 对性能的影响尚未覆盖。

详细模式沿用原有小模型限制，新增：每种缓存预留最多 1 MiB，HBF 物理容量最多
8 MiB，KV 页最多 64 KiB，保守估计的总 KV 块访问最多 8,192 次。超限提示使用 fast。
快速模式报告权重后备及单份 KV 的容量检查，不将其标为已考虑脏版本的物理容量保证。

## 报告字段

`storage_metrics.weights`：固定 HBM 权重、权重缓存预留、HBF 后备容量、逻辑 HBM
权重读取量、HBF Fill 字节、权重访问服务时间之和。

`storage_metrics.kv`：

- `hbm_append_logical_bytes` / `hbm_read_logical_bytes`：按计算需求统计的新增/读取字节。
- `offloaded_append_bytes`：使用 HBF 后备的层新增的 KV 字节。
- `hbf_fill_bytes` / `hbf_writeback_bytes`：包含 4 KiB 对齐的实际 KV 搬运量。
- `dirty_logical_bytes_written` / `discarded_dirty_bytes_at_completion`：已持久化新增字节
  与请求结束时丢弃的新增字节；二者之和应等于 `offloaded_append_bytes`。
- `cache_hits` / `cache_misses`、`evictions` / `dirty_evictions` / `reloads`：块级事件数。
- `hot_protections`：由于近期块优先策略，改为淘汰较新的冷块的次数。
- `write_aware_protections`：因写入状态优先级而改变 LRU 淘汰对象的次数。
- `partial_block_writebacks` / `sealed_block_writebacks`：未满/已填满逻辑块的写回次数；
  两者之和等于成功的脏块淘汰数。
- `rewrite_writebacks`：已有后备版本的块再次写回次数。
- `write_amplification`：实际 KV HBF 写字节 / 新持久化的 dirty 字节；无写回时为 null，
  不包含 NAND 内部 GC、磨损和额外元数据。
- `migration_wait_ps_sum`：块获取过程中等待淘汰、回写、回迁的时间之和，跨设备可重叠。
- `hbm_cache_peak_bytes_per_device` / `logical_peak_bytes_per_device`：缓存实际峰值和
  存活逻辑页峰值；`live_bytes_at_end` 应为 0。

`storage_metrics.hbf_log`：配置容量、已分配物理空间，以及 KV HBF 读写字节与队列等待。
`kv_write_dlu_ops` 是 KV 后端 4 KiB 写提交的成功次数，不含权重初始化，
也不等同于 NAND 内部 program 次数。
顶层原有 `hbf_fill_bytes`、`hbf_writeback_bytes` 保持权重指标语义，KV 使用上述独立字段。
时间线增加 `hbm_kv_append/read/fill/evict`、`hbf_kv_fill/writeback/queue`，仍可导出
Chrome/Perfetto，并计入请求关键路径。

这些指标来自性能事件和版本元数据，不模拟 KV 数值内容，也不构成真实 GPU/NPU
性能校准结果。
