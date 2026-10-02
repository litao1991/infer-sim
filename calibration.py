"""Fit compute-only service profiles; keep measured end-to-end latency separate."""
import hashlib
import json
import math

OPS = ("gemm", "attention", "grouped_gemm")
OVERRIDES = {op + suffix for op in OPS for suffix in ("_ps_per_flop", "_launch_ps")}


def errors(pairs):
    if not pairs:
        return {"count": 0, "mape_percent": None, "max_abs_percent": None}
    values = [abs(predicted - measured) / measured * 100 for predicted, measured in pairs]
    return {"count": len(values), "mape_percent": sum(values) / len(values),
            "max_abs_percent": max(values)}


def fit(records):
    if not records:
        raise ValueError("no calibration records")
    kinds = {r["hardware_kind"] for r in records}
    targets = {r["hardware_id"] for r in records}
    if len(kinds) != 1 or len(targets) != 1:
        raise ValueError("calibrate one hardware target at a time")
    if not kinds <= {"npu", "nvidia_gpu"}:
        raise ValueError("unsupported hardware_kind")
    for r in records:
        if r["measurement_scope"] != "compute_only":
            raise ValueError("kernel/end-to-end timings contain memory; use compare, not compute fitting")
        if r["op"] not in OPS or r.get("split") not in ("train", "test"):
            raise ValueError("each calibration row needs op and train/test split")
        if not r.get("source") or not r.get("hardware_id") or r.get("evidence") not in ("measured", "synthetic"):
            raise ValueError("calibration provenance is required")
        for key in ("flops", "latency_ps"):
            if (type(r[key]) not in (int, float) or not math.isfinite(r[key]) or r[key] <= 0):
                raise ValueError("calibration numbers must be positive and finite")
    result = {"profile_version": "1.0", "hardware_kind": next(iter(kinds)),
              "hardware_id": next(iter(targets)), "measurement_scope": "compute_only",
              "status": "measured" if all(r["evidence"] == "measured" for r in records) else "synthetic",
              "source": sorted({r["source"] for r in records}),
              "records_sha256": hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest(),
              "hardware_overrides": {}, "operators": {}}
    for op in OPS:
        rows = [r for r in records if r["op"] == op]
        if not rows:
            continue
        train = [r for r in rows if r["split"] == "train"]
        if len(train) < 2 or len({r["flops"] for r in train}) < 2:
            raise ValueError(f"{op} needs two or more distinct training FLOP counts")
        mean_x = sum(r["flops"] for r in train) / len(train)
        mean_y = sum(r["latency_ps"] for r in train) / len(train)
        slope = sum((r["flops"] - mean_x) * (r["latency_ps"] - mean_y) for r in train) / sum(
            (r["flops"] - mean_x) ** 2 for r in train)
        intercept = mean_y - slope * mean_x
        if slope <= 0 or intercept < -.001:
            raise ValueError(f"{op} linear fit is nonphysical; use a narrower workload range")
        intercept = max(0, round(intercept))
        result["hardware_overrides"].update({op + "_ps_per_flop": slope, op + "_launch_ps": intercept})
        result["operators"][op] = {
            "model": "launch_ps + flops * ps_per_flop",
            "training_flops_range": [min(r["flops"] for r in train), max(r["flops"] for r in train)],
            "fit_error": errors([(intercept + slope * r["flops"], r["latency_ps"]) for r in train]),
            "held_out_error": errors([(intercept + slope * r["flops"], r["latency_ps"])
                                       for r in rows if r["split"] == "test"])}
    return result


def validate_profile(profile, hardware):
    if profile.get("profile_version") != "1.0":
        raise ValueError("unsupported profile_version")
    if profile["hardware_kind"] != hardware["kind"]:
        raise ValueError("calibration hardware_kind does not match scenario")
    if hardware.get("hardware_id") != profile["hardware_id"]:
        raise ValueError("calibration hardware_id does not match scenario")
    if profile.get("measurement_scope") != "compute_only":
        raise ValueError("only compute_only profiles can be applied")
    if not profile.get("hardware_overrides") or set(profile["hardware_overrides"]) - OVERRIDES:
        raise ValueError("profile may only override compute service coefficients")
    if profile.get("status") not in ("measured", "synthetic") or not profile.get("source"):
        raise ValueError("profile provenance is required")


def compare(report, reference):
    if reference["scenario_sha256"] != report["scenario_sha256"]:
        raise ValueError("reference scenario hash does not match prediction")
    if (not reference.get("source") or reference.get("evidence") not in ("measured", "synthetic")
            or reference.get("measurement_scope") != "end_to_end"):
        raise ValueError("reference needs source, evidence and end_to_end scope")
    if reference.get("hardware_id") != report["scenario"]["hardware"].get("hardware_id"):
        raise ValueError("reference hardware_id does not match prediction")
    metrics = {}
    for key, measured in reference["metrics"].items():
        if key not in ("mean_ttft_ps", "mean_tpot_ps", "mean_request_latency_ps", "tokens_per_second"):
            raise ValueError("unsupported comparison metric: " + key)
        if type(measured) not in (int, float) or not math.isfinite(measured) or measured <= 0:
            raise ValueError("reference metric must be positive and finite")
        predicted = report[key]
        metrics[key] = {"predicted": predicted, "measured": measured,
                        "signed_error_percent": (predicted - measured) / measured * 100}
    if not metrics:
        raise ValueError("reference has no metrics")
    return {"status": reference["evidence"], "source": reference["source"],
            "hardware_id": reference.get("hardware_id"), "metrics": metrics,
            "aggregate": errors([(v["predicted"], v["measured"]) for v in metrics.values()])}
