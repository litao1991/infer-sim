# 主流模型预设运行结果

2026-10-03：10 个 YAML 预设均完成快速估算。全部使用 `examples/model-evaluation.yaml`：
假设的 8 × 80 GiB GPU、TP=8，一条 512-token 输入 / 128-token 输出请求。

**统计范围为 Transformer 层矩阵。容量通过只覆盖这些权重、KV 和临时空间，
不能据此保证完整 checkpoint 可部署；TP 的 KV 采用理想分片，尚未加入框架复制规则。**

| 模型 | 层矩阵参数 / B | 每 token 激活矩阵 / B | 每 token 全模型 KV / KiB | 每卡层权重 / GiB | 已建模容量检查 |
| --- | ---: | ---: | ---: | ---: | --- |
| [meta-llama/Llama-3.1-70B-Instruct](llama-3.1-70b.json) | 68.451 | 68.451 | 320 | 15.938 | 通过 |
| [meta-llama/Llama-3.1-8B-Instruct](llama-3.1-8b.json) | 6.979 | 6.979 | 128 | 1.625 | 通过 |
| [mistralai/Mixtral-8x22B-Instruct-v0.1](mixtral-8x22b.json) | 140.227 | 38.758 | 224 | 32.649 | 通过 |
| [mistralai/Mixtral-8x7B-Instruct-v0.1](mixtral-8x7b.json) | 46.440 | 12.618 | 128 | 10.813 | 通过 |
| [Qwen/Qwen2.5-32B-Instruct](qwen2.5-32b.json) | 31.206 | 31.206 | 256 | 7.266 | 通过 |
| [Qwen/Qwen2.5-7B-Instruct](qwen2.5-7b.json) | 6.525 | 6.525 | 56 | 1.519 | 通过 |
| [Qwen/Qwen3-235B-A22B](qwen3-235b-a22b.json) | 233.848 | 20.945 | 188 | 54.447 | 通过 |
| [Qwen/Qwen3-30B-A3B](qwen3-30b-a3b.json) | 29.910 | 2.730 | 96 | 6.964 | 通过 |
| [Qwen/Qwen3-32B](qwen3-32b.json) | 31.206 | 31.206 | 256 | 7.266 | 通过 |
| [Qwen/Qwen3-8B](qwen3-8b.json) | 6.946 | 6.946 | 144 | 1.617 | 通过 |

本机串行执行总耗时 0.046 秒；Python 进程峰值驻留内存 16.42 MiB。未加载权重张量。

快速模式不运行 HBMSim、路由或网络竞争，不输出 TTFT/TPOT；预测误差状态为 `unmeasured`。
来源、独立 head_dim、导入方法及支持范围见 [模型配置说明](../../docs/MODEL_CONFIG.md)。

复现一个模型：

```sh
.venv/bin/python run.py examples/model-evaluation.yaml \
  --model configs/models/qwen3-30b-a3b.yaml --mode fast \
  --output reports/models/qwen3-30b-a3b.json
```
