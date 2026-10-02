# 执行模型与计时边界

## 模型工作量

令隐藏维度为 H，单头维度为 D = head_dim，查询维度 Q = heads × D，
KV 维度 V = kv_heads × D，FFN 维度 F。省略 head_dim 时默认 D = H / heads；
显式指定时 Q 可以不等于 H，覆盖部分 Qwen3 模型。
每层注意力投影参数 A = H(Q + 2V) + QH，SwiGLU 专家参数 E = 3HF。
Dense 每层参数为 A + E。

MoE 每层有 N 个路由专家、S 个共享专家、top-k = K：

- 总参数：A + E(N + S) + HN。
- 每 token 激活参数：A + E(K + S) + HN。
- 投影/FFN FLOPs：2 × 激活参数 × token 数。
- 注意力 FLOPs：4Q × attention_positions。
- 每 token KV：2 × layers × V × dtype_bytes。

词嵌入、词表输出头、归一化参数、投影 bias 未计入参数量；归一化、激活函数、RoPE、
采样未单独计时。该工作量对应抽象
Transformer 层；对照真实框架时必须控制模型结构差异。GEMM 和 Attention 服务
可分别拟合；Grouped GEMM 把同一设备本轮活跃专家合为一组，权重只读活跃专家。
共享专家放入基础层计算，复制在各 EP lane，按该 lane 的 token 数计费。

## 请求与并行

逻辑设备顺序为 `(((dp * PP + pp) * EP + ep) * TP + tp)`，再通过
`parallel.placement` 映射到物理设备。TP 分摊权重与算子工作量；PP 连续切分层；
DP 使用独立请求队列。请求 i 属于 `dp=i%DP`、`ep=(i/DP)%EP`。
同一个请求的 KV 留在其 EP lane；批内各 lane 分别执行注意力。
TP 采用理想权重/KV 分片，未模拟真实框架在 TP 大于 KV head 数时的 KV head 复制；
这类布局的每卡 KV 容量可能被低估。模型预设与统计范围见 [MODEL_CONFIG.md](MODEL_CONFIG.md)。

MoE 执行顺序：基础层与共享专家 → TP AllReduce → EP dispatch → 每卡 Grouped
GEMM → 专家输出 TP AllReduce → EP combine。每轮有全组依赖屏障。
EP 通信按来源 lane/目的 lane 聚合，TP rank 内传递相应分片；本地路由不占网络。
源 token 可以路由到 K 个不同专家；专家之间没有容量丢弃。

路由使用 `(seed, request_id, token_position, layer, expert)` 的固定 64 位混合值，
以加权无放回抽样选 K 个专家。同一平台/构建可重复，改变批处理不会改变路由。
`expert_weights` 控制长期偏好；`expert_tokens` 计数不重复计算 TP 副本，
`device_assignments` 按 TP 副本计数。`straggler_ps_sum` 是各专家阶段最晚与最早
完成卡之间的时间差之和，空闲卡也参与屏障，因此能反映闲忙不均。

Prefill 分块，Decode 每轮生成 1 个 token。当前约定 **每个输出 token 都有一次
显式 Decode，包括第一个**，不把 Prefill 的最后位置直接作为首 token。
TTFT 包含排队、Prefill 和第一次 Decode；TPOT 是后续实际 token 间隔的分布。
完成预算在准入时预留，实际 KV 页面仍按已消费 token 分块增长，防止多个未完成
请求相互占满 KV 后无法前进。单请求也装不下时明确报错。

## 计算、HBM、HBF、片上缓冲

以下聚合计时描述适用于旧场景。显式配置 `storage.weights` 或 `storage.kv` 的
场景使用统一的 QKV 投影 → 块级 KV 访问 → 剩余计算路径，四种存储组合都采用
相同粒度，见 [STORAGE_CONFIG.md](STORAGE_CONFIG.md)。KV offload 的脏块回写、
回迁及版本状态已接入，权重和 KV 仍共享每卡 HBM 控制器。

GPU 默认每个算子阶段同时启动计算服务和内存服务，完成时间取二者较晚。
计算时间只包括计算服务，不允许直接填入已含 HBM 等待的实测 kernel 总时间。
HBM 采用真实 HBMSim 队列/时序，访问最小 64 B 且向上对齐；Fill 和 KV 通过同一
HBM 控制器，拒绝的访问按容量释放重试，不用固定轮询间隔。

选择 `hbm2_2000`、`hbm3_6400` 或 `hbm4_8000` 会切换已有 HBMSim 标准。
用户配置的可用 HBM 容量由上层管理，HBM 地址为合成流量地址，和真实张量地址
映射、缓存行复用、TLB、缓存命中率不等价。

MoE 的 HBF 对象为每层一个基础权重对象及每专家一个对象；TP 分片向 4 KiB
对齐，必须放得进 slot。多专家读取逐个取得和释放租约，使一个 slot 也能工作；
多对象组合的计算在权重流完成后开始。基础单对象计算在 Fill 完成后开始，可与
驻留权重读重叠。预取只预测同一对象的下一层，不推测尚未观察的路由。
初始化逐个写入小量虚拟数据，绕开后端在途上限，初始化耗时不计入请求延迟。

NPU 默认路径采用保守的串行分析：HBM 输入服务 → 分 tile 的片上 DMA → 计算
服务 → 输出 DMA → HBM 写。DMA 时间为 bytes / 带宽 + tile_count × 启动开销；
每卡 DMA 独占服务，tile 必须适合 SRAM，工作缓冲占用不超过一个 tile。
这是汇总后的分块代价模型，未模拟逐 tile 的载入/消费/复用过程或 DMA/计算流水，
不应把它解释为指令级 SRAM 驻留轨迹。GPU 也可显式打开同样的 DMA 模型。

KV 容量以拥有专家最多的 lane 保守计费；MoE 临时空间按最坏的 dispatch/combine
token 数预留。未模拟 SRAM 权重缓存、专家动态迁移或过载丢 token。
固定 KV 层占用 HBM；offload 层使用独立的有界 KV 缓存及逻辑页容量预算。
HBF 后备按新地址追加版本，当前不模拟 GC 回收，容量耗尽会显式失败。

## 物理网络

`device_nodes[physical_device]` 指定节点；`node_racks[node]` 指定机柜。
同节点使用 NVSwitch 出/入端口或有向 NVLink mesh 链路。
跨节点路径经过源 NIC、机柜交换、可选跨机柜上联、目的 NIC；每一跳有自己的
带宽、延迟、共享资源和等待。采用确定性的 FCFS 消息预留与存储转发模型。

分层 AllReduce：节点内归约 → 机柜内节点代表归约 → 机柜代表 Ring AllReduce
→ 机柜广播 → 节点广播。按每个集合的实际物理成员构造分组。
归约采用消息级数据量模型，未单独收费归约 ALU 工作。

`dispatch_payload_bytes`/`combine_payload_bytes` 是端到端消息负载；
`network_tiers.*.bytes` 与 `fabric_bytes` 是 **物理逐跳字节数**，同一负载经过多个
NIC/交换层会重复计费。`busy_ps`、`queue_wait_ps` 为跨资源累加，不能除以全局时长
后直接当成单链路利用率。DP/PP 并发会竞争同一 NIC 或上联。
不同卡间的网络预留可与其他计算和 HBM 事件重叠。

## 关键路径与快速模式

详细报告输出算子依赖 DAG，网络链路、计算和 DMA 资源依赖显式入图；HBMSim
内部命令/队列作为不透明内存区间。关键路径从最后完成的请求回溯，每个区间只
计一次，其总和必须等于 makespan。报告分别归因计算、内存、通信、DMA、排队与
到达等待；它是耗时解释，不能直接当成删除某资源后加速多少的预测。

快速模式适合大尺寸参数和容量筛选。计算下界假定所有设备理想利用，忽略启动与
通信；MoE 内存估算保守读取全部专家且无批次复用。它不执行路由、HBM 事件或链路
竞争，也不产生 TTFT/TPOT。大规模详细仿真/trace 抽样是以后可扩展的性能工程，
当前版本主动限制本机运行规模。
