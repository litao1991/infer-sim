# 校准、真实框架对照与扫描

## 1. 计算服务校准

校准输入是 JSON 数组，每条记录包含以下字段：

```json
{
  "op": "gemm",
  "flops": 8192,
  "latency_ps": 17384,
  "split": "test",
  "hardware_kind": "npu",
  "hardware_id": "synthetic-npu-v1",
  "source": "synthetic linear fixture; not hardware measurements",
  "evidence": "synthetic",
  "measurement_scope": "compute_only"
}
```

`op` 可为 `gemm`、`attention`、`grouped_gemm`。每类至少两个不同 FLOP 数的
训练点，可提供独立 `test` 点。拟合 `launch_ps + flops × ps_per_flop`，负斜率或
负截距被拒绝。保留训练误差、留出集 MAPE、来源和原始数据哈希。

`compute_only` 必须是隔离/分解访存之后的计算服务数据。GPU/NPU kernel 总时延
通常已包含访存，不能不经分解直接填写这个范围。工具拒绝 `kernel_total` 和
`end_to_end` 作为计算校准输入，防止与 HBMSim 重复收费。

`hardware_kind` 和具体 `hardware_id` 均需匹配。可只校准有实测数据的算子，
其余使用场景效率。报告显示每类算子调用数、训练 FLOPs 范围及超范围调用数；
不能把 FLOPs 范围覆盖解释为所有矩阵形状、布局和精度均已覆盖。应按硬件、精度
和形状族分别建立 profile，必要时增加分段模型。

`examples/calibration-synthetic.json` 专门测试拟合过程。其零拟合误差不代表任何
GPU/NPU 的真实误差。`status=synthetic` 会一路保留到最终报告。

## 2. 外部推理框架适配

先生成预测报告，并在场景中设置实际 `hardware_id`。运行：

```sh
python3 tools/reference_runner.py reports/prediction.json --output reports/reference.json \
  --adapter python3 /path/to/adapter.py --framework-config /path/to/config.json
```

适配器收到额外参数 `--scenario /tmp/.../scenario.json`，读取解析后的同一工作负载，
执行模型、等待设备同步，用单调时钟采集 token 时刻。stdout 输出一个 JSON 对象；
日志输出到 stderr。适配器自行声明测量框架、版本、设备、模型结构与精度，保存在
`source` 字符串中；推荐将原始 trace 一同归档。接口不安装任何推理框架或下载权重。

```json
{
  "hardware_id": "the-exact-hardware-id-in-scenario",
  "source": "framework + revision + device + precision + benchmark command",
  "evidence": "measured",
  "token_convention": "explicit_first_decode",
  "requests": [
    {"request_id": 0, "arrival_ps": 0, "token_times_ps": [10000000, 13000000, 16000000]}
  ]
}
```

时间戳以此次基准开始时刻为零，单位 ps。每条请求必须完整返回，token 数需匹配；
不能把网络流式输出的一次 chunk 到达时刻当成多个 token 的真实 Decode 时刻。
当前模拟器对第一个输出 token 也执行显式 Decode；常见框架直接从 Prefill 产生
首 token，因此适配器必须对齐这一测量约定，不能只改字段标签。其他框架结构
差异、未建模的词表头/归一化等，也会体现在残差中。

`reference_runner` 校验请求身份、token 数、时序和硬件身份，并计算平均
TTFT/TPOT/请求时延与吞吐。`tests/fixtures/reference_adapter.py` 是测试该接口的
合成适配器，**不是实际推理实现**。本机没有安装 PyTorch/真实 GPU/NPU 推理栈；
本次验证没有宣称真实模型对照已达成任何误差指标。

## 3. 对照误差

```sh
python3 tools/experiments.py compare reports/prediction.json reports/reference.json \
  --output reports/compared.json
```

也可从现有基准工具直接导入简化参考 JSON：

```json
{
  "scenario_sha256": "copy-from-prediction-report",
  "hardware_id": "same-as-prediction",
  "source": "benchmark source/version",
  "evidence": "measured",
  "measurement_scope": "end_to_end",
  "metrics": {"mean_ttft_ps": 10000000, "mean_tpot_ps": 3000000}
}
```

逐指标报告 `(预测 - 实测) / 实测 × 100%`，同时报告绝对百分比误差均值。
场景哈希不匹配、硬件不匹配、无来源、非正值/非有限数据均拒绝。
没有参考数据时 `prediction_error.status=unmeasured`，不会填入虚构误差。

## 4. 方案扫描

grid 是字段路径到候选值数组的映射，例如：

```json
{
  "model.top_k": [1, 2],
  "model.dtype_bytes": [1, 2],
  "hardware.collective": ["ring", "hierarchical"]
}
```

产生笛卡尔积，逐个运行，保存每份完整报告、JSON/CSV 汇总和 Markdown 对比。
二维/三维模型和拓扑参数有关联时，需成套修改设备数、placement 和 device_nodes。
无效组合显式记录为失败，扫描退出状态为非零。支持 `--profile` 和 `--mode fast`。

dtype 字节数减少本身不意味着算力翻倍；要研究量化，需同时设置对应精度的
`peak_tflops`、效率或校准 profile。快速模式只适合大模型容量/理想下界筛选。

## 5. 可重复微基准

`microbench` 固化 prompt 长度 1/4/16、batch 1/2、输出 3 token，共六种小场景，
保存完整配置、每个算子 trace 和场景哈希。输出标注 `evidence=simulated`；可将
这些配置交给真实框架适配器采集测量，再导入对照。它不把仿真时间当作实测数据。
