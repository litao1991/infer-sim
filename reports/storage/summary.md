# 权重/KV 分层存储验证结果

7 个小模型 YAML 场景，使用显式存储配置和统一的块级 KV 执行顺序。
所有请求均完成，KV 逻辑字节守恒，结束时没有遗留 KV 页。

| 场景 | 仿真完成时间 / µs | 每卡 HBM 峰值 / KiB | 权重 Fill / KiB | KV Fill / KiB | KV 写回 / KiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| [storage-hbf-hbf.yaml](storage-hbf-hbf.json) | 585.033 | 12.12 | 64 | 40 | 40 |
| [storage-hbf-hbm.yaml](storage-hbf-hbm.json) | 52.091 | 8.88 | 64 | 0 | 0 |
| [storage-hbm-hbf.yaml](storage-hbm-hbf.json) | 538.848 | 13.12 | 0 | 40 | 40 |
| [storage-hbm-hbm.yaml](storage-hbm-hbm.json) | 6.003 | 9.88 | 0 | 0 | 0 |
| [storage-hot-kv.yaml](storage-hot-kv.json) | 1075.762 | 21.12 | 0 | 152 | 76 |
| [storage-mixed-moe.yaml](storage-mixed-moe.json) | 349.170 | 26.62 | 120 | 24 | 20 |
| [storage-npu-moe.yaml](storage-npu-moe.json) | 911.501 | 12.75 | 264 | 72 | 56 |

前四个 HBM/HBF 组合场景使用同一个 Dense 模型和负载；混合 MoE、NPU 与 hot 场景使用不同负载，不能直接横向比较耗时。

本机顺序运行总耗时 0.979 秒；仿真器子进程最大 RSS 5.47 MiB，Python 驱动最大 RSS 19.27 MiB。

这些是事件级仿真结果，未加载真实权重/KV 张量，未进行真实 GPU/NPU 精度校准。
HBF 使用有限追加日志，旧版本不回收；没有模拟 GC 和长期稳态行为。

配置和统计范围见 [STORAGE_CONFIG.md](../../docs/STORAGE_CONFIG.md)。
完整测试日志见 [storage-tests.txt](../storage-tests.txt) 和 [storage-asan-tests.txt](../storage-asan-tests.txt)。
