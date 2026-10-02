# v1.0 验证与示例结果

## 自动验证

Release 与 ASan/UBSan 均通过 16 组测试。测试覆盖原有 v0.1–v0.6 行为及新增能力。

- [Release 日志](release-tests.txt)
- [ASan/UBSan 日志](asan-tests.txt)
- [8 组参数扫描](sweep/summary.md)
- [6 组仿真微基准](microbench/manifest.json)

## 小模型详细仿真

以下时间是仿真结果，非硬件实测。网络字节数按物理逐跳累计。

| 场景 | 完成时间 µs | TTFT P95 µs | TPOT P95 µs | 网络字节 | 专家拖尾合计 µs | DMA tile 数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [dense-small](dense-small.json) | 21.625 | 13.004 | 4.315 | 0 | 0.000 | 0 |
| [moe-small](moe-small.json) | 12.505 | 7.926 | 2.586 | 1408 | 0.774 | 0 |
| [moe-hbf-small](moe-hbf-small.json) | 188.736 | 128.349 | 41.402 | 1408 | 32.954 | 0 |
| [moe-balanced-small](moe-balanced-small.json) | 18.879 | 11.607 | 2.661 | 5440 | 3.274 | 0 |
| [moe-skew-small](moe-skew-small.json) | 15.741 | 9.412 | 2.020 | 5120 | 6.146 | 0 |
| [cross-rack-ring-small](cross-rack-ring-small.json) | 29.908 | 18.209 | 5.910 | 58240 | 0.000 | 0 |
| [cross-rack-hierarchical-small](cross-rack-hierarchical-small.json) | 9.941 | 6.269 | 1.905 | 49920 | 0.000 | 0 |
| [placement-local-small](placement-local-small.json) | 12.422 | 7.447 | 2.451 | 7488 | 0.000 | 0 |
| [placement-cross-small](placement-cross-small.json) | 15.554 | 9.538 | 3.138 | 17472 | 0.000 | 0 |
| [npu-moe-small](npu-moe-small.json) | 14.664 | 9.398 | 2.966 | 1408 | 0.858 | 152 |
| [npu-small-sram](npu-small-sram.json) | 16.474 | 10.548 | 3.346 | 1408 | 0.958 | 786 |

## 结果解释

- 路由和 FLOPs 守恒；改变 Prefill 分块不改变同一 token 的专家选择。
- 热门专家提高屏障拖尾；当前小模型访存占比较高，少读专家权重可能让总时间缩短，因此拖尾与总耗时分别报告。
- 分层集合通信减少该拓扑上的跨机柜流量；改变物理放置可以增加 TP 的跨机柜代价。
- 较小 SRAM 使 DMA tile 数和启动开销增加，FLOPs、路由和 DMA 字节数保持相同。
- MoE+HBF 用例包含超过后端单批初始化请求上限的对象数；有界初始化与单 slot 流式读取均通过。
- 关键路径长度等于 makespan；网络逐层字节/等待之和与总计一致。

## 本地资源

检测设备：Apple M5，内存 32 GB。
2 层小模型、16 个逻辑设备的 Release 仿真约 0.075 秒。
整个顺序示例生成期间，子进程最大 RSS 15.28 MiB；
Python 驱动最大 RSS 34.27 MiB。
这是各进程的高水位值，不是系统总占用；实际采样见 [local-resources.json](local-resources.json)。

## 校准状态

[校准拟合](synthetic-profile.json)、[外部适配器参考](reference-synthetic.json) 与
[误差计算](compared-synthetic.json) 均使用明确标注的合成样例，验证数据链路。
本机未采集真实 GPU/NPU 的独立校准数据；未比较的报告保持 `prediction_error.status=unmeasured`。
真实框架需按 [校准协议](../docs/CALIBRATION.md) 提供测量，才能给出目标硬件预测误差。

[执行时间线](npu-trace.json) 可导入 Chrome Trace/Perfetto。
