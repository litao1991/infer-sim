#!/usr/bin/env python3
"""Stream Qwen-Bailian traces and run bounded, explicitly scaled simulations.

Source JSONL stays in data/ and is never embedded in the generated scenarios.
This is a structural stress test, not a full-size production performance replay.
"""

import argparse
from collections import Counter, deque
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run import load_scenario, simulate

SOURCE_REPOSITORY = "https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon"
SOURCE_REVISION = "5f7439c51ec248a0c585f7d90a41a6f57773b912"
FILES = (
    "qwen_traceA_blksz_16.jsonl",
    "qwen_traceB_blksz_16.jsonl",
    "qwen_coder_blksz_16.jsonl",
    "qwen_thinking_blksz_16.jsonl",
)
POLICIES = {
    "hbm-hbm": ("hbm", "hbm"),
    "hbf-hbm": ("hbf", "hbm"),
    "hbm-hbf": ("hbm", "hbf"),
    "hbf-hbf": ("hbf", "hbf"),
}


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    return ordered[low] + (ordered[min(low + 1, len(ordered) - 1)] - ordered[low]) * (position - low)


def record(line, line_number):
    try:
        return record_from_data(json.loads(line), line_number)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid trace record at line {line_number}: {error}") from error


def inspect(path, sample_size):
    """One sequential pass; stores only scalar distributions and a short window."""
    digest = hashlib.sha256()
    prompts, outputs = [], []
    kinds = Counter()
    window = deque(maxlen=sample_size)
    burst = None
    burst_span = math.inf
    total = eligible = zero_output = parent_rows = hash_blocks = 0
    first_time = last_time = None
    with Path(path).open("rb") as stream:
        for total, line in enumerate(stream, 1):
            digest.update(line)
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid trace record at line {total}: {error}") from error
            item = record_from_data(raw, total)
            now = item["timestamp_s"]
            if last_time is not None and now < last_time:
                raise ValueError(f"timestamps out of order in {path} at line {total}")
            first_time = now if first_time is None else first_time
            last_time = now
            kinds[str(raw.get("type", "unknown"))] += 1
            parent_rows += item["parent_chat_id"] not in (-1, None)
            hash_blocks += item["hash_blocks"]
            prompts.append(item["input_length"])
            outputs.append(item["output_length"])
            if item["input_length"] == 0 or item["output_length"] == 0:
                zero_output += 1
                continue
            eligible += 1
            window.append(item)
            if len(window) == sample_size:
                span = window[-1]["timestamp_s"] - window[0]["timestamp_s"]
                if span < burst_span:
                    burst_span = span
                    burst = list(window)
    if burst is None:
        raise ValueError(f"{path} has fewer than {sample_size} positive-length requests")
    return {
        "source_url": f"{SOURCE_REPOSITORY}/blob/{SOURCE_REVISION}/{Path(path).name}",
        "source_revision": SOURCE_REVISION,
        "sha256": digest.hexdigest(),
        "bytes": Path(path).stat().st_size,
        "rows": total,
        "eligible_rows": eligible,
        "zero_length_rows": zero_output,
        "duration_s": last_time - first_time,
        "parent_rows": parent_rows,
        "hash_blocks": hash_blocks,
        "type_counts": dict(kinds),
        "prompt_tokens": {"mean": sum(prompts) / total, "p50": percentile(prompts, .5),
                          "p95": percentile(prompts, .95), "max": max(prompts)},
        "output_tokens": {"mean": sum(outputs) / total, "p50": percentile(outputs, .5),
                          "p95": percentile(outputs, .95), "max": max(outputs)},
        "burst_span_s": burst_span,
        "burst": burst,
        "midpoint_s": (first_time + last_time) / 2,
    }


def record_from_data(data, line_number):
    try:
        timestamp = float(data["timestamp"])
        prompt = int(data["input_length"])
        output = int(data["output_length"])
        hashes = data["hash_ids"]
        if (not math.isfinite(timestamp) or timestamp < 0 or prompt < 0 or output < 0
                or not isinstance(hashes, list)):
            raise ValueError("invalid length, timestamp, or hash_ids")
    except (ValueError, TypeError, KeyError) as error:
        raise ValueError(f"invalid trace record at line {line_number}: {error}") from error
    return {
        "line": line_number, "timestamp_s": timestamp,
        "input_length": prompt, "output_length": output,
        "chat_id": data.get("chat_id"), "parent_chat_id": data.get("parent_chat_id"),
        "hash_blocks": len(hashes),
    }


def midpoint_window(path, midpoint, sample_size):
    selected = []
    with Path(path).open() as stream:
        for line_number, line in enumerate(stream, 1):
            item = record(line, line_number)
            if (item["timestamp_s"] >= midpoint and item["input_length"] > 0
                    and item["output_length"] > 0):
                selected.append(item)
                if len(selected) == sample_size:
                    return selected
    raise ValueError(f"{path} has fewer than {sample_size} eligible requests after midpoint")


def scaled_tokens(length, divisor, cap):
    return min(cap, max(1, (length + divisor - 1) // divisor))


def milliseconds(value):
    return value / 1e9 if value is not None else None


def formatted_ms(value):
    return f"{value:.4f}" if value is not None else "—"


def scenario(base, requests, policy, speed, divisor, prompt_cap, output_cap,
             kv_cache_bytes):
    data = copy.deepcopy(base)
    data["requests"] = [{
        "arrival_ps": int(round((r["timestamp_s"] - requests[0]["timestamp_s"]) * 1e12 / speed)),
        "prompt_tokens": scaled_tokens(r["input_length"], divisor, prompt_cap),
        "output_tokens": scaled_tokens(r["output_length"], divisor, output_cap),
    } for r in requests]
    weights, kv = POLICIES[policy]
    storage = {"weights": weights, "kv": kv}
    if weights == "hbf":
        storage.update(cache_bytes_per_device=8192, slot_bytes=8192, prefetch=False)
    if kv == "hbf":
        storage.update(kv_cache_bytes_per_device=kv_cache_bytes,
                       kv_capacity_bytes_per_device=1048576,
                       hbf_capacity_bytes=8388608, kv_cache_policy="lru")
    data["storage"] = storage
    return data


def compact(report, label, source, window, speed, policy):
    kv = report["storage_metrics"]["kv"]
    weights = report["storage_metrics"]["weights"]
    return {
        "case": label, "source": source, "window": window, "arrival_speed": speed,
        "policy": policy, "status": "ok", "scenario_sha256": report["scenario_sha256"],
        "requests": len(report["requests_detail"]),
        "input_tokens": sum(r["prompt_tokens"] for r in report["scenario"]["requests"]),
        "output_tokens": report["output_tokens"],
        "mean_ttft_ms": milliseconds(report["mean_ttft_ps"]),
        "p95_ttft_ms": milliseconds(report["distributions_ps"]["ttft"]["p95"]),
        "mean_tpot_ms": milliseconds(report["mean_tpot_ps"]),
        "p95_tpot_ms": milliseconds(report["distributions_ps"]["tpot"]["p95"]),
        "mean_queue_ms": milliseconds(report["distributions_ps"]["queue_wait"]["mean"]),
        "makespan_ms": milliseconds(report["makespan_ps"]),
        "tokens_per_second": report["tokens_per_second"],
        "weight_hbf_fill_bytes": weights["hbf_fill_bytes"],
        "kv_hbf_fill_bytes": kv["hbf_fill_bytes"],
        "kv_hbf_writeback_bytes": kv["hbf_writeback_bytes"],
        "kv_cache_misses": kv["cache_misses"],
        "peak_hbm_bytes_per_device": report["peak_hbm_bytes_per_device"],
        "bottleneck": report["bottleneck"]["dominant_component"],
        "report": f"cases/{label}/report.json",
    }


def run(data_dir, output, binary, sample_size=12, divisor=64, prompt_cap=24,
        output_cap=8, speeds=(1, 1000), sources=FILES, template=None,
        kv_cache_bytes=8192, architecture=None, model=None):
    if (sample_size < 2 or sample_size > 16 or divisor < 1 or prompt_cap < 1
            or output_cap < 1 or sample_size * (prompt_cap + output_cap) > 1024):
        raise ValueError("sample/token settings exceed the local detailed-mode budget")
    if not speeds or any(speed < 1 or speed > 1000000 for speed in speeds):
        raise ValueError("arrival speeds must be in [1, 1000000]")
    if kv_cache_bytes < 4096 or kv_cache_bytes > 1048576 or kv_cache_bytes % 4096:
        raise ValueError("KV cache size must be a 4096-byte multiple in [4096, 1048576]")
    data_dir, output = Path(data_dir), Path(output)
    template = Path(template) if template else ROOT / "examples/bailian-small.json"
    base = load_scenario(template, architecture, model)
    manifest = {"repository": SOURCE_REPOSITORY, "revision": SOURCE_REVISION,
                "method": {"sample_size": sample_size, "token_divisor": divisor,
                           "prompt_cap": prompt_cap, "output_cap": output_cap,
                           "template": str(template),
                           "architecture": str(architecture) if architecture else None,
                           "model": str(model) if model else None,
                           "kv_cache_bytes_per_device": kv_cache_bytes,
                           "arrival_speeds": list(speeds),
                           "windows": ["burst_minimum_span", "midpoint_next_requests"],
                           "raw_prefix_hashes_used_by_simulator": False,
                           "actual_device_calibration": False}, "sources": {}}
    rows = []
    for name in sources:
        if name not in FILES:
            raise ValueError(f"unknown source file: {name}")
        path = data_dir / name
        stats = inspect(path, sample_size)
        windows = {"burst": stats.pop("burst"),
                   "midpoint": midpoint_window(path, stats.pop("midpoint_s"), sample_size)}
        manifest["sources"][name] = stats
        for window_name, requests in windows.items():
            save(output / "samples" / f"{Path(name).stem}-{window_name}.json",
                 {"source": name, "window": window_name, "requests": requests})
            for speed in speeds:
                for policy in POLICIES:
                    label = f"{Path(name).stem}-{window_name}-x{speed}-{policy}"
                    case_dir = output / "cases" / label
                    data = scenario(base, requests, policy, speed, divisor, prompt_cap,
                                    output_cap, kv_cache_bytes)
                    save(case_dir / "scenario.json", data)
                    try:
                        report = simulate(data, binary=str(binary))
                        save(case_dir / "report.json", report)
                        rows.append(compact(report, label, name, window_name, speed, policy))
                    except (OSError, ValueError) as error:
                        rows.append({"case": label, "source": name, "window": window_name,
                                     "arrival_speed": speed, "policy": policy, "status": "error",
                                     "error": str(error)})
    save(output / "manifest.json", manifest)
    baseline = {(r["source"], r["window"], r["arrival_speed"]): r for r in rows
                if r["status"] == "ok" and r["policy"] == "hbm-hbm"}
    for row in rows:
        if row["status"] == "ok":
            reference = baseline.get((row["source"], row["window"], row["arrival_speed"]))
            row["ttft_ratio_to_hbm"] = (row["mean_ttft_ms"] / reference["mean_ttft_ms"]
                                         if reference and reference["mean_ttft_ms"] else None)
            row["tpot_ratio_to_hbm"] = (row["mean_tpot_ms"] / reference["mean_tpot_ms"]
                                         if reference and row["mean_tpot_ms"] is not None
                                         and reference["mean_tpot_ms"] else None)
    save(output / "summary.json", {"method": manifest["method"], "runs": rows,
                                   "succeeded": sum(r["status"] == "ok" for r in rows),
                                   "failed": sum(r["status"] != "ok" for r in rows)})
    fields = sorted({key for row in rows for key in row})
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Qwen-Bailian trace-driven small-model evaluation", "",
             f"All {len(sources)} supplied JSONL files were streamed for workload statistics. Detailed simulation",
             f"uses two {sample_size}-request windows per file, scaled token counts, and synthetic hardware.",
             "These are structural stress tests, not paper-result reproductions or calibrated",
             "predictions for a production GPU or HBF device.", "",
             "| Source | Rows | Duration (s) | Prompt p50/p95 | Output p50/p95 | Parent rows |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, stats in manifest["sources"].items():
        lines.append(f"| {name} | {stats['rows']} | {stats['duration_s']:.1f} | "
                     f"{stats['prompt_tokens']['p50']:.0f}/{stats['prompt_tokens']['p95']:.0f} | "
                     f"{stats['output_tokens']['p50']:.0f}/{stats['output_tokens']['p95']:.0f} | "
                     f"{stats['parent_rows']} |")
    lines += ["", "Arrival speed xN compresses interarrival times by N; x1 preserves them.",
              f"Prompt/output tokens are ceil(original/{divisor}), capped at {prompt_cap}/{output_cap} respectively.",
              "The current simulator does not use session links or prefix hashes; requests",
              "release KV on completion. See manifest.json for exact samples and SHA-256.", "",
              "| Source | Window | Arrival | Placement | TTFT mean/p95 (ms) | TPOT mean/p95 (ms) | TTFT vs HBM | HBF weight/KV read/KV write (KiB) |",
              "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |"]
    for row in rows:
        if row["status"] != "ok":
            lines.append(f"| {row['source']} | {row['window']} | x{row['arrival_speed']} | "
                         f"{row['policy']} | ERROR: {row['error']} | — | — | — |")
            continue
        ratio = (f"{row['ttft_ratio_to_hbm']:.2f}×"
                 if row["ttft_ratio_to_hbm"] is not None else "—")
        lines.append(f"| {row['source']} | {row['window']} | x{row['arrival_speed']} | "
                     f"[{row['policy']}]({row['report']}) | "
                     f"{formatted_ms(row['mean_ttft_ms'])}/{formatted_ms(row['p95_ttft_ms'])} | "
                     f"{formatted_ms(row['mean_tpot_ms'])}/{formatted_ms(row['p95_tpot_ms'])} | "
                     f"{ratio} | "
                     f"{row['weight_hbf_fill_bytes']/1024:.1f}/"
                     f"{row['kv_hbf_fill_bytes']/1024:.1f}/"
                     f"{row['kv_hbf_writeback_bytes']/1024:.1f} |")
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(ROOT / "data/bailian"))
    parser.add_argument("--output", default=str(ROOT / "reports/bailian"))
    parser.add_argument("--binary", default=str(ROOT / "build/infer_sim"))
    parser.add_argument("--template", default=str(ROOT / "examples/bailian-small.json"))
    parser.add_argument("--architecture", help="JSON/YAML hardware and parallel preset")
    parser.add_argument("--model", help="JSON/YAML model preset within detailed-mode limits")
    parser.add_argument("--kv-cache-bytes", type=int, default=8192)
    parser.add_argument("--sources", nargs="+", default=list(FILES), choices=FILES)
    parser.add_argument("--sample-size", type=int, default=12)
    parser.add_argument("--token-divisor", type=int, default=64)
    parser.add_argument("--prompt-cap", type=int, default=24)
    parser.add_argument("--output-cap", type=int, default=8)
    parser.add_argument("--arrival-speeds", type=int, nargs="+", default=[1, 1000])
    args = parser.parse_args()
    try:
        rows = run(args.data_dir, args.output, args.binary, args.sample_size,
                   args.token_divisor, args.prompt_cap, args.output_cap,
                   tuple(args.arrival_speeds), tuple(args.sources), args.template,
                   args.kv_cache_bytes, args.architecture, args.model)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, str(error) + "\n")
    print(json.dumps({"runs": len(rows), "failed": sum(r["status"] != "ok" for r in rows),
                      "summary": str(Path(args.output) / "summary.md")}))
    if any(r["status"] != "ok" for r in rows):
        parser.exit(1)


if __name__ == "__main__":
    main()
