# 验证说明

测试使用 2 层、隐藏维度 16/32 的模型和少量请求，不分配真实模型张量。

## 重现

```sh
ctest --test-dir build --output-on-failure
cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug \
  -DHBFSIM_SOURCE_DIR=/path/to/hbfsim -DHBMSIM_SOURCE_DIR=/path/to/hbmsim \
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined"
cmake --build build-asan --target infer_sim -j 2
ctest --test-dir build-asan --output-on-failure
python3 tools/demo_report.py
```

`INFER_SIM_BINARY` 可指定其他构建的可执行文件。每次模拟有 180 秒实际运行上限。
示例生成脚本写入 `reports/`；两份测试日志由 CTest 的 `Testing/Temporary/LastTest.log`
复制而来。该脚本不会用新示例运行结果替代测试日志。

## 验证矩阵

| 范围 | 检查内容 |
| --- | --- |
| Dense | 参数量/FLOPs 与手算一致、请求完成、HBM 请求字节守恒。 |
| 确定性和背压 | 完整报告重复一致；HBM 队列容量 1 时可重试并完成。 |
| HBF | 预取减少固定场景耗时；Fill 字节一致；单 slot 淘汰、显式回写不改变请求延迟。 |
| TP/PP/DP | FLOPs/输出 token 守恒；流水线实际重叠；独立副本改善固定场景等待。 |
| MoE | token × layer × top-k 路由守恒，无重复专家；改变 Prefill 分块不改变路由。 |
| 专家倾斜/放置 | 热门专家计数、每卡工作迁移、屏障拖尾；不强制把权重复用收益误判为性能回退。 |
| MoE + HBF | 多对象有界初始化、单缓存 slot 流式读取、Fill/淘汰正常完成。 |
| 跨机柜 | 逐跳字节/等待总量守恒；分层集合通信减少特定拓扑的跨机柜流量。 |
| 物理放置 | 同等计算工作量下，TP 跨机柜放置增加链路流量和关键路径时间。 |
| NPU | 共用路由/FLOPs；DMA 依赖、SRAM 容量约束；tile 更小时启动开销增加。 |
| 并行组合 | EP 与 TP/PP/DP 组合，最大 16 卡；多个副本共享 NIC 时发生排队。 |
| 报告 | TTFT/TPOT 样本数正确；DAG 不倒流；关键路径覆盖 makespan；时间线可导出。 |
| 校准 | 训练/留出集拟合、profile 改变计算服务、拒绝含内存的 kernel 总时延与错误硬件。 |
| 对照适配 | 外部进程协议、token 数与时序检查、合成来源保留、逐指标误差计算。 |
| 扫描/微基准 | 合法组合产出完整报告，无效组合保留错误；6 组固定小场景。 |
| 容量/输入 | KV 完成预算准入避免多请求死锁；非法维度、NaN、路由、拓扑被拒绝。 |
| HBM 配置 | HBMSim HBM2/3/4 配置改变访存服务，计算工作量不变。 |

## 本次结果（2026-10-02）

- Release：16 组 Python 场景测试通过，CTest 时间 **8.93 秒**。
- Debug + AddressSanitizer/UBSan：同一套 16 组测试通过，CTest 时间 **225.27 秒**。
- 自动生成 11 个示例详细报告、8 组方案扫描、6 组仿真微基准。
- 外部测量适配器、计算服务拟合和误差计算已用明确标注的合成数据跑通。
- 上一轮 v0.1–v0.6 依赖接口调整后，HBFSim/HBMSim 集成套件已通过 46/46；
  本轮未再修改该依赖接口。当前新增路径通过本项目的 Release 与 sanitizer 测试。

硬件资源实测见 [本次结果报告](../reports/VALIDATION_REPORT.md)。本机为 Apple M5、
32 GB 内存；测试及示例均按该本地规模控制。仿真核心没有加载模型参数张量。

## 已修复的问题

1. MoE 增加 HBF 对象数后，批量初始化超出 HBF 在途请求上限：改为有界逐对象页
   初始化，提交时间使用后端明确的纳秒接口，统一执行与报告仍用皮秒。
2. 多个请求逐步扩展 KV 时可能相互占满内存：增加完成预算准入，实际 KV 页面仍
   随消费量分配，等待者在请求完成释放预算后继续。
3. 增加整数时间溢出、非有限数值、无效拓扑、重复设备和校准范围/硬件身份检查。

## 准确性结论

这些验证证明本版本的执行依赖、资源约束和统计在列出的模型粒度下可重复且一致，
不能据此声称特定真实硬件的 TTFT/TPOT 预测误差已低于某个阈值。
真实校准仍需目标 GPU/NPU 的测量数据；相关导入、拟合、对照流程已经实现。

## YAML 与硬件预设验证（2026-10-02）

配置读取层新增 6 组检查，加上原有测试共 **22 组通过**，Release CTest 用时
**9.81 秒**。使用项目 `.venv` 中的 PyYAML 6.0.3，所有 YAML 测试均实际执行。

- 等价 JSON/YAML 的完整报告（含配置哈希）一致。
- 四种架构预设分别运行 Dense 和 MoE，计算工作量与输出 token 守恒。
- GPU/NPU 切换以完整段替换，设备放置重建；跨机柜产生对应通信，NPU 产生 DMA。
- YAML 场景、架构、扫描表、校准数据与 profile 的命令行入口跑通。
- 重复键、对象标签、循环别名、非数据类型被拒绝；未安装 YAML 依赖仍能读 JSON。

本轮日志：[yaml-tests.txt](../reports/yaml-tests.txt)。可运行配置和说明见
[HARDWARE_CONFIG.md](HARDWARE_CONFIG.md)。上面的 16 组 sanitizer 记录为 v1.0 核心验证。

## 主流模型配置验证（2026-10-02–03）

新增 6 组模型测试后，完整 Release 套件 **28 组通过**，CTest 用时 **11.52 秒**；
本轮修改后的 C++ 核心也通过新增 6 组测试的 AddressSanitizer/UBSan 检查，
用时 **57.753 秒**。此轮 sanitizer 运行范围为 `test_models.py`，完整核心套件的历史
结果见上文。Release 与本轮 sanitizer 均使用项目 `.venv`，YAML 测试全部实际执行。

- 10 个预设完成快速估算，检查来源、上游快照 SHA-256 与导入维度一致。
- 独立 `head_dim` 的参数量、KV 容量及 FLOPs 与手算一致；小模型在单卡 GPU、
  TP=2 NPU 的 Dense/MoE 路径均保持计算量守恒。
- Qwen3-30B-A3B 统计的层矩阵参数为 29,909,581,824，每 token 激活层矩阵参数
  为 2,730,491,904，2 字节精度的全模型每 token KV 为 98,304 B。
- 模型/硬件组合替换、来源覆盖标记、本地 HF 配置导入、命令行运行及扫描跑通。
- 拒绝未经建模的 MLA、异构层、滑动窗口与量化 checkpoint 导入；真实尺寸模型
  进入详细模式时主动提示使用 fast，避免本机展开大量 HBM 事件。

10 个真实尺寸模型的解析估算总耗时 **0.046 秒**，Python 进程峰值驻留内存
**16.42 MiB**，没有加载权重。计算仍采用明确假设的硬件参数，没有真实硬件精度校准。
所有完整报告及统计范围见 [模型运行汇总](../reports/models/summary.md)。

测试日志：[完整 Release](../reports/model-tests.txt)、
[新增模型 sanitizer](../reports/model-asan-tests.txt)。可重现新增 sanitizer 测试：

```sh
INFER_SIM_BINARY="$PWD/build-asan/infer_sim" .venv/bin/python \
  -m unittest discover -s tests -p test_models.py -v
```

## v1.1 权重/KV 独立放置（2026-10-03）

完整 Release 套件 **37 组通过**，CTest 总耗时 **14.52 秒**。新增存储功能的
**9 组 AddressSanitizer/UBSan 测试通过**，用时 **29.915 秒**。本轮 sanitizer
只运行 `test_storage.py`，覆盖新回调与缓存生命周期；历史全套结果见上文。

验证包含：

- 权重/KV 的四种 HBM/HBF 组合，使用统一显式配置路径，计算量与逻辑读写字节守恒。
- 单槽缓存的脏淘汰、部分块增量追加、反复回迁；大缓存无需写回时减少等待。
- 按层/专家常驻和按层 KV 放置，容量与解析计算一致；近期块策略实际影响淘汰选择。
- TP/PP/DP/EP、跨机柜 GPU、NPU DMA、HBM 队列容量 1 和共享 HBF 排队。
- 请求完整 KV 预算准入与完成释放，HBM 恰好装下缓存的边界，HBF 物理日志耗尽失败。
- PP 后级请求完成与前级缓存写回重叠时，延后释放仍在使用的页元数据。
- YAML 命令行、配置非法值、快速容量估算，以及本地规模限制。

逐次校验：`offloaded_append_bytes = dirty_logical_bytes_written +
discarded_dirty_bytes_at_completion`；KV 结束时存活页为 0；缓存与逻辑页峰值不超过
预算；HBF 回迁/写回字节与物理日志读写字节一致；请求 DAG 覆盖完整 makespan。

7 个可运行场景顺序执行 **0.979 秒**；仿真器子进程最大 RSS **5.47 MiB**，
Python 驱动最大 RSS **19.27 MiB**。没有加载实际模型或 KV 张量。

- [场景结果与资源记录](../reports/storage/summary.md)
- [完整 Release 日志](../reports/storage-tests.txt)
- [新增存储 sanitizer 日志](../reports/storage-asan-tests.txt)
- [配置与已实现边界](STORAGE_CONFIG.md)

```sh
ctest --test-dir build --output-on-failure
INFER_SIM_BINARY="$PWD/build-asan/infer_sim" .venv/bin/python \
  -m unittest discover -s tests -p test_storage.py -v
```

HBF 使用有限的追加空间，本轮不实现 GC/物理擦除回收；近期块策略使用 token 窗口
代理热度。小模型自测不代表真实 GPU/NPU 的预测误差已校准。


## GitHub 独立构建复验（2026-10-03）

从 GitHub 下载 `dependencies.lock.json` 固定的两个后端到全新的 `.deps/`，应用
仓库内补丁，并使用默认 CMake 路径重新构建。完整 **37 组测试通过**，CTest
耗时 **14.97 秒**；不依赖开发机器原有的 `/Users/.../Code/` 后端目录。
初始化脚本再次执行成功，已应用的补丁不会重复修改。

日志：[github-build-tests.txt](../reports/github-build-tests.txt)。GitHub Actions 将在
每次推送和 PR 时，从固定依赖开始构建并运行同一套测试。
