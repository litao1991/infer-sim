"""Latency distributions and a causal critical path from the operation DAG."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

VERSION = "1.1.0"


def distribution(values):
    values = sorted(values)
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "p99": None, "max": None}
    def quantile(q):
        index = (len(values) - 1) * q
        lo = int(index)
        hi = min(lo + 1, len(values) - 1)
        return values[lo] + (values[hi] - values[lo]) * (index - lo)
    return {"count": len(values), "mean": sum(values) / len(values),
            "p50": quantile(.5), "p95": quantile(.95), "p99": quantile(.99), "max": values[-1]}


def critical_path(report):
    nodes = report["trace"]
    for index, node in enumerate(nodes):
        if node["id"] != index or node["end_ps"] < node["start_ps"]:
            raise ValueError("invalid trace node")
        for parent in node["parents"]:
            if parent >= index or nodes[parent]["end_ps"] > node["start_ps"]:
                raise ValueError("invalid trace dependency")
    endpoint = max((r["completion_node"] for r in report["requests_detail"]),
                   key=lambda n: nodes[n]["end_ps"])
    cursor = endpoint
    path = []
    totals = defaultdict(int)
    while True:
        node = nodes[cursor]
        duration = node["end_ps"] - node["start_ps"]
        if duration:
            path.append(cursor)
            totals[node["kind"]] += duration
        if not node["parents"]:
            break
        parent = max(node["parents"], key=lambda n: (nodes[n]["end_ps"], -n))
        gap = node["start_ps"] - nodes[parent]["end_ps"]
        if gap:
            totals["unattributed_wait"] += gap
        cursor = parent
    duration = sum(totals.values())
    if duration != report["makespan_ps"]:
        raise ValueError("critical path does not cover service makespan")
    return {"scope": "operation dependencies; HBM internal scheduling is an opaque timed span",
            "duration_ps": duration, "node_ids": list(reversed(path)),
            "time_by_kind_ps": dict(sorted(totals.items(), key=lambda p: -p[1]))}


def enrich(report, data, fingerprint, profile=None):
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(list(root.glob("*.py")) + list((root / "tools").glob("*.py")) +
                       list((root / "src").glob("*")) + [root / "CMakeLists.txt"]):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    report.update({"simulator_version": VERSION, "config_version": data["config_version"],
                   "simulator_source_sha256": digest.hexdigest(),
                   "scenario_sha256": fingerprint, "scenario": data,
                   "calibration": profile or {"status": "uncalibrated", "source": "scenario assumptions"},
                   "prediction_error": {"status": "unmeasured", "metrics": None}})
    metadata = data.get("model_metadata", {})
    report["storage_model"] = {
        "placement_scope": "HBM residence or HBF backing with bounded HBM staging; no direct compute on HBF",
        "kv_offload_timing": "QKV projection, block append/read stream, then aggregate attention and remaining GEMM; no stream/compute overlap",
        "execution_mode": "block_stream" if any(k in data.get("storage", {}) for k in ("weights", "kv")) else "legacy_aggregate",
        "kv_hotness": "recent token blocks per request and layer; soft eviction preference",
        "kv_eviction_policy": data.get("storage", {}).get("kv_cache_policy", "lru"),
        "kv_writeback": "dirty eviction writes a complete aligned KV page; write_aware prefers clean then sealed dirty victims; no packing buffer",
        "write_amplification_scope": "KV physical log write bytes / newly persisted dirty logical bytes; excludes NAND internal GC and wear",
        "hbf_log": "shared append-only physical backing; old versions are not garbage collected in this run",
        "capacity_scope": "HBM maxima use conservative per-stage/lane bounds; KV cache and weight cache are separate reservations",
        "legacy_metrics": "top-level hbf_fill/writeback fields refer to weights; storage_metrics.kv and hbf_log report KV transfers"}
    report["model_accounting"] = {
        "scope": "transformer_block_matrices",
        "model_id": metadata.get("model_id"),
        "omitted": ["token embeddings", "LM head", "normalization weights and operations",
                    "projection biases", "activation functions", "RoPE", "sampling"],
        "capacity_scope": "HBM feasibility covers modeled block weights, KV and scratch; not the full checkpoint",
        "parallelism_assumption": "ideal TP sharding of weights and KV; framework-specific KV-head replication is not modeled",
        "source": metadata,
        "overrides_from_preset": {key: {"preset": value, "resolved": data["model"].get(key)}
                                  for key, value in metadata.get("reference_dimensions", {}).items()
                                  if data["model"].get(key) != value}}
    if report["mode"] == "detailed":
        details = report["requests_detail"]
        report["distributions_ps"] = {
            "ttft": distribution([r["ttft_ps"] for r in details]),
            "tpot": distribution([v for r in details for v in r["token_gaps_ps"]]),
            "request_latency": distribution([r["latency_ps"] for r in details]),
            "queue_wait": distribution([r["queue_wait_ps"] for r in details])}
        report["critical_path"] = critical_path(report)
        breakdown = report["critical_path"]["time_by_kind_ps"]
        components = defaultdict(int)
        for name, value in breakdown.items():
            kind = ("compute" if name in ("compute", "grouped_gemm") else
                    "memory" if name.startswith(("hbm", "hbf")) else
                    "communication" if name in ("card_interconnect", "nic", "intra_rack", "cross_rack") else
                    "onchip_dma" if name == "dma" else "queue_or_arrival")
            components[kind] += value
        report["bottleneck"] = {"critical_path_time_ps": dict(components),
                                "dominant_component": max(components, key=components.get),
                                "interpretation": "time attribution, not a counterfactual speedup prediction"}
        if profile:
            coverage = {}
            for op in ("gemm", "attention", "grouped_gemm"):
                values = [node["attention_flops"] if op == "attention" else node["gemm_flops"]
                          for node in report["trace"]
                          if node["kind"] == ("grouped_gemm" if op == "grouped_gemm" else "compute")]
                values = [v for v in values if v]
                operator = profile.get("operators", {}).get(op)
                limits = operator["training_flops_range"] if operator else None
                coverage[op] = {"invocations": len(values), "training_flops_range": limits,
                                "outside_training_range": sum(not limits or not limits[0] <= v <= limits[1]
                                                              for v in values)}
            report["calibration_coverage"] = coverage
    else:
        report["distributions_ps"] = None
        report["critical_path"] = None
        report["bottleneck"] = {"interpretation": "analytic bounds have no execution timeline"}
    return report


def chrome_trace(report, path):
    events = []
    for node in report.get("trace", []):
        if node["end_ps"] == node["start_ps"]:
            continue
        events.append({"name": node["kind"], "ph": "X", "pid": 0,
                       "tid": node["device"] if node["device"] >= 0 else "scheduler/network",
                       "ts": node["start_ps"] / 1e6,
                       "dur": (node["end_ps"] - node["start_ps"]) / 1e6,
                       "args": {"node": node["id"], "parents": node["parents"]}})
    Path(path).write_text(json.dumps({"traceEvents": events, "displayTimeUnit": "ns"}, indent=2))
