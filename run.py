#!/usr/bin/env python3
"""Validated scenario frontend for the inference discrete-event simulator."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile

from schema import validate
from calibration import validate_profile
from reporting import enrich, chrome_trace
from config_io import load_document, scenario_document
from storage import weights_on_hbf, layout


def load_scenario(path, architecture=None, model=None):
    return validate(scenario_document(path, architecture, model))


def model_numbers(data):
    m = data["model"]
    h = m["hidden"]
    head_dim = m.get("head_dim", h // m["heads"])
    query, kv = m["heads"] * head_dim, m["kv_heads"] * head_dim
    experts = m.get("experts", 1) if m.get("kind") == "moe" else 1
    shared = m.get("shared_experts", 0) if m.get("kind") == "moe" else 0
    gate = h * experts if m.get("kind") == "moe" else 0
    params_layer = h * (query + 2 * kv) + query * h + 3 * h * m["ffn"] * (experts + shared) + gate
    weight_bytes_layer = params_layer * m["dtype_bytes"]
    kv_bytes_token = 2 * m["layers"] * kv * m["dtype_bytes"]
    return params_layer, weight_bytes_layer, kv_bytes_token


def fast_report(data):
    m, h, p = data["model"], data["hardware"], data["parallel"]
    st = data.get("storage", {})
    placement = layout(data)
    params_layer, weight_bytes_layer, kv_bytes_token = model_numbers(data)
    total_prompt = sum(r["prompt_tokens"] for r in data["requests"])
    total_output = sum(r["output_tokens"] for r in data["requests"])
    token_work = total_prompt + total_output
    active_params = params_layer
    if m.get("kind") == "moe":
        active_params -= 3 * m["hidden"] * m["ffn"] * (m["experts"] - m["top_k"])
    flops = 2 * active_params * m["layers"] * token_work
    query_dim = m["heads"] * m.get("head_dim", m["hidden"] // m["heads"])
    flops += 4 * query_dim * m["layers"] * sum(
        r["prompt_tokens"] * (r["prompt_tokens"] + 1) // 2
        + r["output_tokens"] * (r["prompt_tokens"] + (r["output_tokens"] + 1) / 2)
        for r in data["requests"])
    attention_work = flops - 2 * active_params * m["layers"] * token_work
    grouped_work = (2 * 3 * m["hidden"] * m["ffn"] * m["top_k"] * m["layers"] * token_work
                    if m.get("kind") == "moe" else 0)
    gemm_work = flops - attention_work - grouped_work
    compute_bound = sum(work * h.get(op + "_ps_per_flop", 1 / (h["peak_tflops"] *
                        h.get(op + "_efficiency", h["compute_efficiency"])))
                        for op, work in (("gemm", gemm_work), ("attention", attention_work),
                                         ("grouped_gemm", grouped_work))) / h["devices"]
    weight_per_device = math.ceil(weight_bytes_layer * m["layers"] / (p["tp"] * p["pp"]))
    if m.get("kind") == "moe":
        expert_params = 3 * m["hidden"] * m["ffn"]
        base_params = params_layer - expert_params * m["experts"]
        most_experts = max(p["expert_placement"].count(lane) for lane in range(p["ep"]))
        weight_per_device = math.ceil((base_params + most_experts * expert_params) *
                                      m["dtype_bytes"] * m["layers"] / (p["tp"] * p["pp"]))
    block = data["serving"]["kv_block_tokens"]
    request_kv = [placement["kv_hbm_bytes_per_token_device"] * math.ceil(
        (r["prompt_tokens"] + r["output_tokens"]) / block) * block for r in data["requests"]]
    kv_per_request_device = max(request_kv)
    groups = p["dp"] * p.get("ep", 1)
    kv_per_replica = [sum(request_kv[i] for i in range(group, len(request_kv), groups))
                      for group in range(groups)]
    max_kv_replica = max(kv_per_replica)
    scratch_per_device = math.ceil(data["serving"]["max_batch_tokens"] * max(m["hidden"], query_dim)
                                   * m["dtype_bytes"] * (1 + m["top_k"] if m.get("kind") == "moe" else 1) / p["tp"])
    prefill_chunks = sum(math.ceil(r["prompt_tokens"] /
                                   data["serving"]["prefill_chunk_tokens"])
                         for r in data["requests"])
    weight_reads = weight_bytes_layer * m["layers"] * (prefill_chunks + total_output)
    attention_positions = sum(
        r["prompt_tokens"] * (r["prompt_tokens"] + 1) // 2
        + r["output_tokens"] * r["prompt_tokens"]
        + r["output_tokens"] * (r["output_tokens"] + 1) // 2
        for r in data["requests"])
    kv_reads = kv_bytes_token * attention_positions
    kv_writes = kv_bytes_token * token_work
    activation_bytes = token_work * m["hidden"] * m["dtype_bytes"] * m["layers"]
    hbm_bytes = weight_reads + kv_reads + kv_writes + activation_bytes
    resident_bytes = placement["resident_weight_bytes_per_device"]
    resident_bytes += st.get("cache_bytes_per_device", 0) if weights_on_hbf(st) else 0
    resident_bytes += st.get("kv_cache_bytes_per_device", 0) if st.get("kv") == "hbf" else 0
    hbf_slot_feasible = not weights_on_hbf(st) or placement["max_weight_object_bytes"] <= st["slot_bytes"]
    hbf_requests = [math.ceil((r["prompt_tokens"] + r["output_tokens"]) / block) *
                    placement["kv_page_bytes"] * placement["kv_hbf_layers_per_device"] for r in data["requests"]]
    hbf_minimum = placement["hbf_weight_bytes_total"] + sum(
        math.ceil((r["prompt_tokens"] + r["output_tokens"]) / block) * placement["kv_page_bytes"] *
        (m["layers"] - len(st.get("kv_hbm_layers", []))) * p["tp"]
        for r in data["requests"]) if st.get("kv") == "hbf" else placement["hbf_weight_bytes_total"]
    return {
        "active_parameters_per_token": active_params * m["layers"],
        "memory_traffic_assumption": "all experts read per layer invocation, no batch reuse; conservative MoE estimate",
        "mode": "fast_lower_bound", "model_parameters": params_layer * m["layers"],
        "weight_bytes_total": weight_bytes_layer * m["layers"],
        "weight_bytes_per_device": weight_per_device,
        "kv_bytes_per_token_total": kv_bytes_token,
        "max_kv_bytes_per_request_device": kv_per_request_device,
        "max_kv_bytes_per_replica": max_kv_replica,
        "scratch_bytes_per_device": scratch_per_device,
        "hbf_layer_slot_feasible": hbf_slot_feasible,
        "storage_capacity": dict(placement,
            kv_hbf_max_request_bytes_per_device=max(hbf_requests),
            kv_hbf_single_request_feasible=(max(hbf_requests) <= st.get("kv_capacity_bytes_per_device", 0)),
            hbf_single_copy_bytes_all_requests=hbf_minimum,
            hbf_single_copy_capacity_feasible=(hbf_minimum <= st["hbf_capacity_bytes"] if "hbf_capacity_bytes" in st else True),
            hbf_log_traffic="not predicted in fast mode; dirty versions consume additional append-log capacity"),
        "estimated_flops": int(flops),
        "estimated_hbm_bytes_no_batch_reuse": hbm_bytes,
        "hbm_time_no_batch_reuse_ps": math.ceil(hbm_bytes * 1000 /
                                                 h["hbm_bandwidth_GBps"] / h["devices"]),
        "compute_lower_bound_ps": math.ceil(compute_bound),
        "hbm_capacity_feasible": resident_bytes + max_kv_replica + scratch_per_device
                                 <= h["hbm_capacity_bytes"],
    }


def flattened(data):
    lines = []
    for section in ("model", "hardware", "parallel", "serving", "storage"):
        for key, value in data.get(section, {}).items():
            if key == "hardware_id":
                continue
            if isinstance(value, list):
                value = ",".join(str(v) for v in value)
            if isinstance(value, bool):
                value = int(value)
            lines.append(f"{section}.{key}={value}")
    for req in data["requests"]:
        lines.append("request=" + ",".join(str(req[key]) for key in
                                         ("arrival_ps", "prompt_tokens", "output_tokens")))
    return "\n".join(lines) + "\n"


def source_state(binary):
    cache = Path(binary).resolve().parent / "CMakeCache.txt"
    if not cache.exists():
        return {}
    paths = {}
    for line in cache.read_text().splitlines():
        for name in ("HBFSIM_SOURCE_DIR", "HBMSIM_SOURCE_DIR"):
            prefix = name + ":PATH="
            if line.startswith(prefix):
                paths[name] = line[len(prefix):]
    state = {}
    for name, path in paths.items():
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=path,
                              text=True, capture_output=True)
        changes = subprocess.run(["git", "status", "--porcelain"], cwd=path,
                                 text=True, capture_output=True)
        if head.returncode == 0:
            state[name] = {"revision": head.stdout.strip(),
                           "dirty": bool(changes.stdout.strip())}
            diff = subprocess.run(["git", "diff", "HEAD"], cwd=path, text=True, capture_output=True)
            state[name]["tracked_diff_sha256"] = hashlib.sha256(diff.stdout.encode()).hexdigest()
    return state


def simulate(data, mode="detailed", binary=None, profile=None):
    data = validate(data)
    if profile:
        validate_profile(profile, data["hardware"])
        data["hardware"].update(profile["hardware_overrides"])
        data = validate(data)
    binary = binary or str(Path(__file__).parent / "build" / "infer_sim")
    fingerprint = hashlib.sha256(json.dumps(data, sort_keys=True,
                                            separators=(",", ":")).encode()).hexdigest()
    if mode == "fast":
        return enrich(fast_report(data), data, fingerprint, profile)
    _, weight_bytes_layer, _ = model_numbers(data)
    if (weight_bytes_layer > 128 * 1024 or len(data["requests"]) > 128
            or data["model"]["layers"] > 16 or data["hardware"]["devices"] > 16
            or sum(r["prompt_tokens"] + r["output_tokens"] for r in data["requests"]) > 1024):
        raise ValueError("detailed mode exceeds local scale limits; use --mode fast")
    storage = data.get("storage", {})
    shape = layout(data)
    if max(storage.get("cache_bytes_per_device", 0), storage.get("kv_cache_bytes_per_device", 0)) > 1024 * 1024:
        raise ValueError("detailed cache reservation exceeds 1 MiB local limit; use --mode fast")
    if storage.get("hbf_capacity_bytes", 0) > 8 * 1024 * 1024:
        raise ValueError("detailed HBF capacity exceeds 8 MiB local limit; use --mode fast")
    if weights_on_hbf(storage):
        seed_bytes = shape["hbf_weight_bytes_total"]
        if seed_bytes > 1024 * 1024:
            raise ValueError("detailed HBF seed exceeds 1 MiB; use --mode fast")
    if "weights" in storage or "kv" in storage:
        block = data["serving"]["kv_block_tokens"]
        # Upper bound: one block walk per token per layer and TP rank.
        visits = sum((r["prompt_tokens"] + r["output_tokens"]) * math.ceil(
            (r["prompt_tokens"] + r["output_tokens"]) / block) for r in data["requests"])
        visits *= data["model"]["layers"] * data["parallel"]["tp"]
        if visits > 8192 or shape["kv_page_bytes"] > 65536:
            raise ValueError("detailed KV block stream exceeds local scale limits; use --mode fast")
    with tempfile.TemporaryDirectory() as directory:
        config = Path(directory) / "scenario.txt"
        config.write_text(flattened(data))
        result = subprocess.run([binary, str(config)], text=True, capture_output=True, timeout=180)
        if result.returncode:
            raise ValueError(result.stderr.strip() or f"simulator exited {result.returncode}")
        report = json.loads(result.stdout)
    report["backend_revisions"] = source_state(binary)
    report["binary_sha256"] = hashlib.sha256(Path(binary).read_bytes()).hexdigest()
    return enrich(report, data, fingerprint, profile)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", help="scenario JSON or YAML")
    parser.add_argument("--model", help="replace model using a JSON or YAML preset")
    parser.add_argument("--architecture", help="replace hardware/parallel using a JSON or YAML preset")
    parser.add_argument("--mode", choices=("fast", "detailed"), default="detailed")
    parser.add_argument("--binary", default=str(Path(__file__).parent / "build" / "infer_sim"))
    parser.add_argument("--profile", help="compute-only calibration profile JSON or YAML")
    parser.add_argument("--output", help="write report JSON to a file")
    parser.add_argument("--trace-out", help="write Chrome/Perfetto timeline JSON")
    args = parser.parse_args()
    try:
        profile = load_document(args.profile) if args.profile else None
        report = simulate(load_scenario(args.scenario, args.architecture, args.model), args.mode, args.binary, profile)
        if args.trace_out:
            chrome_trace(report, args.trace_out)
        content = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.output:
            Path(args.output).write_text(content)
        else:
            print(content, end="")
    except (ValueError, KeyError, OSError, subprocess.TimeoutExpired) as error:
        parser.exit(2, str(error) + "\n")


if __name__ == "__main__":
    main()
