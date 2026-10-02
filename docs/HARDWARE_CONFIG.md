# 硬件架构配置

可以在完整场景中直接写 `hardware`、`parallel`，也可以通过 `--architecture`
复用独立 JSON/YAML 文件。后一种方式适合在同一个 Dense/MoE 模型上比较不同架构。

## 可配置范围

| 资源 | 配置 |
| --- | --- |
| 计算 | `kind`、每卡 `peak_tflops`、通用/算子效率、启动开销、校准系数 |
| HBM | 每卡容量；HBM2/3/4 时序 profile；快速估算带宽；请求队列容量 |
| 片上存储 | SRAM 容量、DMA tile 大小、DMA 带宽和启动时间 |
| 卡间互联 | NVSwitch 或有向 mesh；带宽、延迟 |
| 集群 | 设备所属节点、节点所属机柜、NIC/机柜内/跨柜带宽和延迟 |
| 并行放置 | TP、PP、DP、EP、逻辑→物理设备排列、专家放置 |
| HBF | 在场景的 `storage` 段独立配置权重/KV 放置、容量、slot、预取和近期块策略，见 [存储配置](STORAGE_CONFIG.md) |

`peak_tflops`、HBM 容量与带宽均按单卡定义。时间使用 ps，容量使用 B；
网络 `*_gbps` 使用 Gbit/s，`dma_GBps` 使用 GB/s。

## 跨机柜示例

以下是 `configs/hardware/gpu-8card-crossrack.yaml` 的拓扑部分。
完整文件还包含算力、HBM 与启动开销，可直接运行。

```yaml
hardware:
  kind: nvidia_gpu
  devices: 8
  fabric_topology: nvswitch
  collective: hierarchical
  device_nodes: [0, 0, 1, 1, 2, 2, 3, 3]
  node_racks: [0, 0, 1, 1]
  nvlink_gbps: 100
  nvlink_latency_ps: 10000
  nic_gbps: 100
  nic_latency_ps: 20000
  rack_gbps: 200
  rack_latency_ps: 20000
  cross_rack_gbps: 25
  cross_rack_latency_ps: 100000
parallel:
  tp: 8
  pp: 1
  dp: 1
  ep: 1
```

设备 0/1 属于节点 0，2/3 属于节点 1；节点 0/1 在机柜 0，节点 2/3 在机柜 1。
以上是合成验证参数。可独立修改跨柜带宽或把 `collective` 改为 `ring`，比较
`network_tiers.cross_rack`、TTFT/TPOT 和关键路径的变化。

## 使用规则

- `--architecture` 的文件只包含 `hardware` 和可选的 `parallel`，以完整段替换
  场景对应内容。未提供 `parallel` 时保留场景的并行设置，并重新检查设备数。
- `tp * pp * dp * ep` 必须等于 `devices`；PP 整除层数；当前 Dense 要求 EP=1。
- `parallel.placement` 指定物理设备排列，不写时自动生成顺序排列。
- `expert_placement` 不写时按专家编号轮流分配到 EP lane。
- `hardware.hbm_profile` 控制详细 HBM 时序；`hbm_bandwidth_GBps` 用于快速估算。
- NPU 需要填写 SRAM/DMA 字段；`nvlink_*` 在当前接口中也是 NPU 本地互联的通用
  带宽/延迟参数名，不表示 NPU 实际使用 NVIDIA NVLink 协议。
- `hardware_id` 是用户定义的配置/硬件身份，用于匹配校准数据，不会自动下载硬件规格。

YAML 使用 PyYAML 的 SafeLoader 派生加载器，不接受 Python 对象标签；
重复键、非有限数值、日期对象和循环别名均拒绝。JSON 同样拒绝重复键。
YAML 合并锚点若产生重名字段也会报错；推荐使用独立架构文件切换完整配置。
布尔值使用 `true`/`false`；若填写 `config_version`，使用字符串 `"1.0"`。
格式不同但解析后相同的场景，会产生相同的配置哈希与仿真结果。

加载器依据 [PyYAML 官方接口](https://pyyaml.org/wiki/PyYAMLDocumentation) 实现，
可选依赖固定在 `requirements-yaml.txt`。未安装 PyYAML 的环境仍可运行 JSON 场景。

## 当前模型粒度

支持切换同构 GPU/NPU 系统及设备/节点/机柜布局；每张卡采用不同型号的异构配置、
任意交换机图和包级路由尚未实现。HBM 详细时序从三个内置标准中选择，不会仅凭
一个带宽数值推导任意内存标准。具体计算和 DMA 抽象见 [MODEL.md](MODEL.md)。

新机器运行包含 YAML 的测试时，可在 CMake 配置中指定已安装依赖的解释器：

```sh
cmake -S . -B build -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
ctest --test-dir build --output-on-failure
```
