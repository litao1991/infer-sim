#!/usr/bin/env python3
"""Capture an external inference adapter's token timestamps as a comparison reference.

Adapter invocation: ADAPTER [ARGS...] --scenario RESOLVED_SCENARIO.json
Adapter stdout is JSON; diagnostics belong on stderr. No shell is involved.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from schema import validate


def normalize(prediction, observed):
    scenario = prediction["scenario"]
    if observed.get("hardware_id") != scenario["hardware"].get("hardware_id"):
        raise ValueError("adapter hardware_id does not match prediction")
    if not observed.get("source") or observed.get("evidence") not in ("measured", "synthetic"):
        raise ValueError("adapter must declare source and measured/synthetic evidence")
    if observed.get("token_convention") != "explicit_first_decode":
        raise ValueError("adapter must use the simulator's explicit_first_decode convention")
    details = observed["requests"]
    if (len(details) != len(scenario["requests"]) or
            sorted(r["request_id"] for r in details) != list(range(len(details)))):
        raise ValueError("adapter must return each request exactly once")
    ttft, latency, gaps, done = [], [], [], []
    tokens = 0
    for row in details:
        request = scenario["requests"][row["request_id"]]
        times = row["token_times_ps"]
        if len(times) != request["output_tokens"]:
            raise ValueError("adapter token count does not match workload")
        arrival = row["arrival_ps"]
        if type(arrival) is not int or arrival < 0 or any(type(t) is not int for t in times):
            raise ValueError("adapter timestamps must be integer picoseconds")
        if times != sorted(times) or times[0] <= arrival:
            raise ValueError("adapter timestamps violate request causality")
        ttft.append(times[0] - arrival)
        latency.append(times[-1] - arrival)
        gaps.extend(b - a for a, b in zip(times, times[1:]))
        done.append(times[-1])
        tokens += len(times)
    metrics = {"mean_ttft_ps": sum(ttft) / len(ttft),
               "mean_request_latency_ps": sum(latency) / len(latency),
               "tokens_per_second": tokens * 1e12 / max(done)}
    if gaps and sum(gaps) > 0:
        metrics["mean_tpot_ps"] = sum(gaps) / len(gaps)
    return {"scenario_sha256": prediction["scenario_sha256"],
            "hardware_id": observed["hardware_id"], "source": observed["source"],
            "evidence": observed["evidence"], "measurement_scope": "end_to_end",
            "token_convention": observed["token_convention"], "metrics": metrics,
            "requests": details,
            "observations_sha256": hashlib.sha256(json.dumps(observed, sort_keys=True).encode()).hexdigest()}


def capture(prediction, adapter, timeout=180):
    validate(prediction["scenario"])
    if not prediction["scenario"]["hardware"].get("hardware_id"):
        raise ValueError("real-reference collection requires an explicit hardware_id")
    if not adapter:
        raise ValueError("an inference framework adapter command is required")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "scenario.json"
        path.write_text(json.dumps(prediction["scenario"], indent=2))
        result = subprocess.run(adapter + ["--scenario", str(path)], capture_output=True,
                                text=True, timeout=timeout)
        if result.returncode:
            raise ValueError("inference adapter failed: " + result.stderr.strip())
        return normalize(prediction, json.loads(result.stdout))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction", help="detailed report to compare against")
    parser.add_argument("--output", required=True)
    parser.add_argument("--adapter", nargs=argparse.REMAINDER, required=True)
    args = parser.parse_args()
    try:
        reference = capture(json.loads(Path(args.prediction).read_text()), args.adapter)
        Path(args.output).write_text(json.dumps(reference, indent=2) + "\n")
    except (ValueError, KeyError, subprocess.TimeoutExpired) as error:
        parser.exit(2, str(error) + "\n")


if __name__ == "__main__":
    main()
