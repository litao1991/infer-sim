# 阿里百炼 trace 驱动的推理仿真评估

## 数据与复现

数据来自 [Qwen-Bailian 使用轨迹公开仓库](https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon)，固定在提交
`5f7439c51ec248a0c585f7d90a41a6f57773b912`。下载器按文件核对大小和 SHA-256；
原始数据存放在 `data/bailian/`，不会进入 Git。四份文件共 312,520,831 字节，269,681 行。

```sh
python3 tools/download_bailian.py
python3 tools/download_bailian.py --verify-only
python3 tools/bailian_eval.py
```

主结果在 `reports/bailian/`：`manifest.json` 记录源文件指纹和统计量，`samples/` 保存样本行号、原始长度和时间戳，`cases/` 保存每组完整场景和仿真报告，`summary.csv/json/md` 用于汇总。所有输出都可从固定版本的四份原始 JSONL 重建。

| 文件 | 行数 | 输入 token p50 / p95 | 输出 token p50 / p95 | 时长 |
| --- | ---: | ---: | ---: | ---: |
| trace A | 43,058 | 1,046 / 8,808 | 376 / 1,032 | 7,199 秒 |
| trace B | 172,800 | 574 / 2,491 | 39 / 328 | 7,200 秒 |
| coder | 43,011 | 4,540 / 13,407 | 469 / 2,699 | 7,199 秒 |
| thinking | 10,812 | 3,680 / 16,221 | 1,666 / 12,670 | 7,192 秒 |

全量数据只进行流式扫描和负载统计，详细事件仿真选择每个文件的两个确定性窗口：最密集的连续 12 个有效请求，以及全段时间中点之后的 12 个请求。每个窗口保留原始到达间隔，同时增加一组到达间隔压缩 1000 倍的压力场景。非零输入/输出 token 数分别映射为 `min(24, ceil(input/64))` 与 `min(8, ceil(output/64))`。每个场景比较四种放置：权重/KV 为 HBM/HBM、HBF/HBM、HBM/HBF、HBF/HBF，共 64 组。权重缓存为每设备 8 KiB；HBF KV 缓存默认为每设备 8 KiB，采用 LRU。小模型见 [bailian-small.json](../examples/bailian-small.json)。

模板使用 2 层、隐藏维度 16 的 Dense 模型与单设备合成 GPU；标称算力 0.5 TFLOPS、HBM 容量 1 MiB。它刻意足够小，以在本机生成完整事件报告，并不对应百炼实际使用的模型与设备。

可替换模板，或使用 `--architecture` 和 `--model` 直接覆盖硬件与模型，进一步评估 GPU/NPU、卡数、并行方式或 YAML 配置。配置必须符合详细模式的本地预算；程序会将采样请求及存储放置写入组合后的配置，再交给同一仿真核心执行。下面的 NPU 示例和单卡 GPU 使用相同的小型 Dense 模型，但硬件和 TP 放置不同，不能仅把延迟差异归因于设备种类。

```sh
python3 tools/bailian_eval.py --template examples/bailian-small.json \
  --kv-cache-bytes 8192 --arrival-speeds 1 1000

.venv/bin/python tools/bailian_eval.py \
  --architecture configs/hardware/npu-2card.yaml \
  --sources qwen_traceB_blksz_16.jsonl --arrival-speeds 1000 \
  --output reports/bailian/npu-dense-demo

# 单独做容量敏感性实验；需要相应的模板 HBM 容量。
python3 tools/bailian_eval.py --kv-cache-bytes 32768 \
  --arrival-speeds 1000 --output reports/bailian/cache-32k
python3 tools/bailian_eval.py --kv-cache-bytes 524288 \
  --arrival-speeds 1000 --output reports/bailian/cache-512k
```

## 如何解读结果

每组报告有平均和 p95 TTFT/TPOT、排队、吞吐、HBM 峰值、权重 Fill、KV Fill/回写和缓存缺失等指标。比较时固定同一文件、同一窗口、同一到达速率，以 HBM/HBM 为基线。`x1` 窗口的相邻请求往往隔得远，适于观察单请求路径；`x1000` 用于观察并发和缓存争用。`summary.md` 给出全部逐组结果，`summary.csv` 便于二次分析。

本机验证中主评估 64 组、两个容量对照各 32 组，合计 128 组全部成功。下表只汇总 x1000 的 8 个窗口，给出各窗口平均 TTFT 相对于同窗口 HBM/HBM 的**中位比值**；读/写量是 8 个窗口的总和。缓存增大时只有 HBF KV 的缓存字段改变，模型、请求和硬件配置均保持相同。

| 每卡 KV 缓存 | 权重/KV | TTFT 中位比值 | KV HBF 读 / 写合计 |
| --- | --- | ---: | ---: |
| 8 KiB | HBF/HBM | 7.25× | 0 / 0 MiB |
| 8 KiB | HBM/HBF | 625.55× | 19.58 / 6.09 MiB |
| 8 KiB | HBF/HBF | 638.88× | 19.58 / 6.10 MiB |
| 32 KiB | HBM/HBF | 599.26× | 19.28 / 5.91 MiB |
| 512 KiB | HBM/HBF | 0.99× | 0 / 0 MiB |
| 512 KiB | HBF/HBF | 7.25× | 0.30 / 0.30 MiB |

8 KiB 情况下频繁发生 KV 回迁；32 KiB 仍不足以容纳这个样本的工作集。512 KiB 时 HBM/HBF 组合没有发生 HBF 读写，所以接近 HBM 基线，**不能据此认为 HBF 与 HBM 等速**。这些数字说明当前模型对缓存容量极敏感，研究 HBF 收益前必须明确真实工作集、驻留策略和设备测量。

另以 Trace B 验证了两卡 NPU Dense、两卡 NPU MoE、八卡 NVSwitch 和八卡跨机柜配置，各 8 组均成功；报告分别在 `reports/bailian/npu-dense-demo/`、`npu-moe-demo/`、`gpu-nvswitch-demo/` 和 `gpu-crossrack-demo/`。八卡详细模式达到本地 KV 事件预算，因此这两组改用每窗口 4 个请求、输入/输出分别封顶 8/4 token。相同四请求突发窗口与 HBM/HBM 放置下，NVSwitch 配置平均 TTFT 为 0.0030 ms；跨机柜配置为 0.0070 ms，跨机柜链路记录 4,224 字节。这验证了跨机柜路径被实际计入事件图，不构成真实硬件的速度比较。

这些结果是**缩小模型和 token 数后的结构性压力测试**。本机没有实测 GPU/NPU/HBF 校准数据，示例算力、带宽、延迟和容量都不是百炼线上硬件规格。因此绝对毫秒数、放大倍数不能当作线上性能预测，也不能宣称复现论文结果。当前仿真器尚未消费 `hash_ids` 前缀块或父子会话链，结束请求会释放 KV；对话共享、前缀缓存命中、跨轮 KV 生命周期和图像/文件等多模态代价均未重放。窗口只抽样 24 个请求/文件，不代表全量延迟分布。全量统计的 token 长度仍保留原始值。

进一步提高真实性应先引入 session/prefix 复用与真实权重/KV 工作集，再采集硬件微基准和线上 TTFT/TPOT 作独立校准。容量敏感性结果可用来检查较小 HBM 缓存造成的回迁，不能单凭单个容量点判断 HBF 固有性能。

## KV 写回策略与块大小对照

`write_aware` 在淘汰时依次选择干净块、已满脏块、仍在追加的未满脏块，同类内按
LRU 排序；缓存全是未满脏块时允许回写，不会无限阻塞。它将多个 token 的更新
尽可能合并在 HBM 中，当前写回仍为完整对齐页。配置、计数器和版本保证见
[存储说明](STORAGE_CONFIG.md)。

```sh
# 在普通评估里切换策略和 KV 块大小。
python3 tools/bailian_eval.py --kv-cache-policy write_aware --kv-block-tokens 16 \
  --output reports/bailian/write-aware

# 相同请求，固定 8 KiB 缓存：3 种策略 × 3 种块大小 × 4 份 trace × 2 个窗口。
python3 tools/bailian_kv_sweep.py --output reports/bailian/kv-policies
```

这次写回对照使用每窗口 2 个请求，长度缩放为 `ceil(original/32)`，输入/输出封顶
64/16，到达间隔压缩 1000 倍；保持 HBM 权重、HBF KV、小模型与 8 KiB KV 缓存相同。
这些请求与前面容量扫描的 12-request 窗口不同，不能直接比较两份实验的绝对延迟。
72 组全部成功。每块大小下的三种策略使用完全相同的请求，改变块大小也保持
逻辑 KV 追加/读取字节及模型 FLOPs 守恒。

| 块 token 数 | 淘汰策略 | KV HBF 写入合计 | 写提交次数 | 写入放大 |
| ---: | --- | ---: | ---: | ---: |
| 4 | LRU | 2,612 KiB | 653 | 56.51× |
| 4 | write_aware | 2,500 KiB | 625 | 54.53× |
| 16 | write_aware | 1,952 KiB | 488 | 42.55× |
| 64 | write_aware | 1,620 KiB | 405 | 40.00× |

单请求控制实验中，1 层、8-token prompt、12-token output、两页缓存，将 LRU 换成
write_aware 后写回从 52 KiB 降到 16 KiB，未满块写回和重复写均归零。四份 trace
的多请求、两层场景中，同块大小的总写量只改善约 2%–4%；两页缓存下仍会被迫
淘汰正在追加的块。较大块降低了总请求数，但反复写整页导致写入放大仍高。
因此这项策略只能作为基线，不能声称已经解决小有效载荷写入。

下一层需要单独建模有界写合并缓冲区：淘汰时暂存 dirty delta，按 DLU/页对齐打包，
以容量水位或超时触发刷写。未刷出的版本必须能从 HBM 转发读取；请求结束可取消
无继续复用需求的缓冲数据。评估必须计入额外 HBM 预留、合并等待、压力下的部分
刷写、HBF 读取/恢复多个 delta 的开销、元数据和写入放大。仅把多个写请求放进一个
队列，而后仍逐块完整写页，不会减少实际写字节。当前未实现该缓冲区、增量后备
日志和后台刷写，报告明确记录这一边界。
