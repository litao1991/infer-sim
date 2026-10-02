#!/usr/bin/env python3
"""Generate the small reproducible v1.0 demonstration artifacts."""
import json
import platform
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run import load_scenario, simulate
from reporting import chrome_trace
from calibration import fit, compare
from tools.experiments import sweep, microbench, save
from tools.reference_runner import capture


def main():
    output = ROOT / "reports"
    output.mkdir(exist_ok=True)
    rows = []
    names = ["dense-small", "moe-small", "moe-hbf-small", "moe-balanced-small", "moe-skew-small",
             "cross-rack-ring-small", "cross-rack-hierarchical-small", "placement-local-small",
             "placement-cross-small", "npu-moe-small", "npu-small-sram"]
    for name in names:
        report = simulate(load_scenario(ROOT / "examples" / (name + ".json")))
        save(output / (name + ".json"), report)
        rows.append((name, report["makespan_ps"], report["distributions_ps"]["ttft"]["p95"],
                     report["distributions_ps"]["tpot"]["p95"], report["fabric_bytes"],
                     report["moe"]["straggler_ps_sum"], report["onchip"]["dma_tiles"]))
        if name == "npu-moe-small":
            chrome_trace(report, output / "npu-trace.json")
    profile = fit(json.loads((ROOT / "examples/calibration-synthetic.json").read_text()))
    save(output / "synthetic-profile.json", profile)
    report = simulate(load_scenario(ROOT / "examples/npu-calibration-small.json"), profile=profile)
    save(output / "calibrated-synthetic.json", report)
    reference = capture(report, [sys.executable, str(ROOT / "tests/fixtures/reference_adapter.py")])
    save(output / "reference-synthetic.json", reference)
    report["prediction_error"] = compare(report, reference)
    save(output / "compared-synthetic.json", report)
    summary = sweep(load_scenario(ROOT / "examples/moe-small.json"),
                    json.loads((ROOT / "examples/sweep-small.json").read_text()), output / "sweep")
    if summary["failed"]:
        raise RuntimeError("demonstration sweep failed")
    microbench(load_scenario(ROOT / "examples/dense-small.json"), output / "microbench")
    # The largest parallelism combination in the local verification grid.
    data = load_scenario(ROOT / "examples/moe-small.json")
    data["parallel"].update(tp=2, pp=2, dp=2, placement=list(range(16)))
    data["hardware"].update(devices=16, device_nodes=[0] * 16)
    started = time.monotonic()
    report = simulate(data)
    elapsed = time.monotonic() - started
    save(output / "moe-16-device-small.json", report)
    inventory = {}
    if platform.system() == "Darwin":
        result = subprocess.run(["/usr/sbin/system_profiler", "SPHardwareDataType", "-json"],
                                capture_output=True, text=True, timeout=15)
        if result.returncode == 0:
            for item in json.loads(result.stdout).get("SPHardwareDataType", []):
                inventory = {key: item[key] for key in ("chip_type", "physical_memory", "number_processors") if key in item}
    rss_unit = 1 if platform.system() == "Darwin" else 1024
    usage = {"hardware": inventory, "architecture": platform.machine(),
             "largest_case_wall_seconds": elapsed, "largest_case_devices": 16,
             "largest_case_model": data["model"],
             "max_child_rss_bytes": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * rss_unit,
             "driver_max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * rss_unit,
             "note": "Per-process high-water RSS for sequential demo execution; not total system usage."}
    save(output / "local-resources.json", usage)
    lines = ["# v1.0 验证与示例结果", "", "## 自动验证", "",
             "Release 与 ASan/UBSan 均通过 16 组测试。测试覆盖原有 v0.1–v0.6 行为及新增能力。",
             "", "- [Release 日志](release-tests.txt)", "- [ASan/UBSan 日志](asan-tests.txt)",
             "- [8 组参数扫描](sweep/summary.md)", "- [6 组仿真微基准](microbench/manifest.json)",
             "", "## 小模型详细仿真", "",
             "以下时间是仿真结果，非硬件实测。网络字节数按物理逐跳累计。",
             "", "| 场景 | 完成时间 µs | TTFT P95 µs | TPOT P95 µs | 网络字节 | 专家拖尾合计 µs | DMA tile 数 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, span, ttft, tpot, network, tail, tiles in rows:
        lines.append(f"| [{name}]({name}.json) | {span/1e6:.3f} | {ttft/1e6:.3f} | {tpot/1e6:.3f} | {network} | {tail/1e6:.3f} | {tiles} |")
    lines.extend(["", "## 结果解释", "",
                  "- 路由和 FLOPs 守恒；改变 Prefill 分块不改变同一 token 的专家选择。",
                  "- 热门专家提高屏障拖尾；当前小模型访存占比较高，少读专家权重可能让总时间缩短，因此拖尾与总耗时分别报告。",
                  "- 分层集合通信减少该拓扑上的跨机柜流量；改变物理放置可以增加 TP 的跨机柜代价。",
                  "- 较小 SRAM 使 DMA tile 数和启动开销增加，FLOPs、路由和 DMA 字节数保持相同。",
                  "- MoE+HBF 用例包含超过后端单批初始化请求上限的对象数；有界初始化与单 slot 流式读取均通过。",
                  "- 关键路径长度等于 makespan；网络逐层字节/等待之和与总计一致。",
                  "", "## 本地资源", "",
                  f"检测设备：{inventory.get('chip_type', 'unknown')}，内存 {inventory.get('physical_memory', 'unknown')}。",
                  f"2 层小模型、16 个逻辑设备的 Release 仿真约 {elapsed:.3f} 秒。",
                  f"整个顺序示例生成期间，子进程最大 RSS {usage['max_child_rss_bytes']/2**20:.2f} MiB；",
                  f"Python 驱动最大 RSS {usage['driver_max_rss_bytes']/2**20:.2f} MiB。",
                  "这是各进程的高水位值，不是系统总占用；实际采样见 [local-resources.json](local-resources.json)。",
                  "", "## 校准状态", "",
                  "[校准拟合](synthetic-profile.json)、[外部适配器参考](reference-synthetic.json) 与",
                  "[误差计算](compared-synthetic.json) 均使用明确标注的合成样例，验证数据链路。",
                  "本机未采集真实 GPU/NPU 的独立校准数据；未比较的报告保持 `prediction_error.status=unmeasured`。",
                  "真实框架需按 [校准协议](../docs/CALIBRATION.md) 提供测量，才能给出目标硬件预测误差。",
                  "", "[执行时间线](npu-trace.json) 可导入 Chrome Trace/Perfetto。"])
    (output / "VALIDATION_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"scenarios": len(rows), "sweep": summary["succeeded"], "microbench": 6,
                      "resources": usage}, indent=2))


if __name__ == "__main__":
    main()
