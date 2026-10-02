#!/usr/bin/env python3
"""Small sequential sweeps, compute calibration, and measured-reference comparisons."""
import argparse
import copy
import csv
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from calibration import fit, compare
from run import load_scenario, simulate
from config_io import load_document


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def sweep(scenario, grid, output, binary=None, mode="detailed", profile=None):
    """One subprocess at a time; errors remain visible rather than disappearing."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    keys = list(grid)
    if not keys or any(not isinstance(v, list) or not v for v in grid.values()):
        raise ValueError("grid must map scenario field paths to nonempty lists")
    count = 1
    for options in grid.values():
        count *= len(options)
    if count > 256:
        raise ValueError("local sweep limit is 256 runs")
    rows = []
    for index, values in enumerate(itertools.product(*(grid[k] for k in keys))):
        data = copy.deepcopy(scenario)
        changes = dict(zip(keys, values))
        row = {"run": index, "changes": changes}
        try:
            for key, value in changes.items():
                section, field = key.split(".", 1)
                if section not in ("model", "hardware", "parallel", "serving", "storage"):
                    raise ValueError("unsupported sweep field: " + key)
                data[section][field] = value
            report = simulate(data, mode, binary, profile)
            name = f"run-{index:03d}.json"
            save(output / name, report)
            row.update(status="ok", report=name, scenario_sha256=report["scenario_sha256"])
            for key in ("makespan_ps", "mean_ttft_ps", "mean_tpot_ps", "tokens_per_second",
                        "compute_flops", "fabric_bytes", "peak_hbm_bytes_per_device", "estimated_flops"):
                if key in report:
                    row[key] = report[key]
            row["bottleneck"] = report["bottleneck"].get("dominant_component")
        except (ValueError, KeyError) as error:
            row.update(status="error", error=str(error))
        rows.append(row)
    summary = {"config_version": "1.0", "runs": rows,
               "succeeded": sum(r["status"] == "ok" for r in rows),
               "failed": sum(r["status"] == "error" for r in rows)}
    save(output / "summary.json", summary)
    fields = sorted({key for row in rows for key in row})
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row, changes=json.dumps(row["changes"], sort_keys=True)))
    lines = ["# Simulation sweep", "", f"Completed: {summary['succeeded']}; failed: {summary['failed']}.",
             "", "| Run | Changes | TTFT (µs) | TPOT (µs) | Bottleneck |", "| --- | --- | ---: | ---: | --- |"]
    for row in rows:
        if row["status"] == "ok":
            lines.append(f"| [{row['run']}]({row['report']}) | `{json.dumps(row['changes'])}` | "
                         f"{row.get('mean_ttft_ps', 0)/1e6:.3f} | {row.get('mean_tpot_ps', 0)/1e6:.3f} | "
                         f"{row.get('bottleneck') or 'analytic'} |")
        else:
            lines.append(f"| {row['run']} | `{json.dumps(row['changes'])}` | — | — | ERROR: {row['error']} |")
    lines.extend(["", "Values are simulator predictions. Measurement provenance and calibration errors are in each JSON report."])
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    return summary


def microbench(scenario, output, binary=None):
    """Fixed one-request probes. These are simulated workloads, never fake measurements."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cases = []
    for prompt in (1, 4, 16):
        for batch in (1, 2):
            data = copy.deepcopy(scenario)
            data["requests"] = [{"arrival_ps": 0, "prompt_tokens": prompt, "output_tokens": 3}
                                for _ in range(batch)]
            data["serving"].update(max_seqs=batch, max_batch_tokens=prompt * batch,
                                    prefill_chunk_tokens=prompt)
            report = simulate(data, binary=binary)
            name = f"prompt-{prompt}-batch-{batch}.json"
            save(output / name, report)
            cases.append({"prompt_tokens": prompt, "batch": batch, "report": name,
                          "scenario_sha256": report["scenario_sha256"],
                          "mean_ttft_ps": report["mean_ttft_ps"], "mean_tpot_ps": report["mean_tpot_ps"]})
    save(output / "manifest.json", {"evidence": "simulated", "cases": cases})
    return {"cases": len(cases), "evidence": "simulated"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for name in ("sweep", "microbench"):
        command = commands.add_parser(name)
        command.add_argument("scenario")
        command.add_argument("--model", help="model JSON or YAML preset")
        command.add_argument("--architecture", help="hardware/parallel JSON or YAML preset")
        command.add_argument("--output", required=True)
        command.add_argument("--binary")
        if name == "sweep":
            command.add_argument("--grid", required=True)
            command.add_argument("--mode", default="detailed", choices=("fast", "detailed"))
            command.add_argument("--profile")
    command = commands.add_parser("calibrate")
    command.add_argument("records")
    command.add_argument("--output", required=True)
    command = commands.add_parser("compare")
    command.add_argument("prediction")
    command.add_argument("reference")
    command.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        if args.action == "sweep":
            result = sweep(load_scenario(args.scenario, args.architecture, args.model), load_document(args.grid),
                           args.output, args.binary, args.mode,
                           load_document(args.profile) if args.profile else None)
            print(json.dumps({k: result[k] for k in ("succeeded", "failed")}))
            if result["failed"]:
                parser.exit(1)
        elif args.action == "microbench":
            print(json.dumps(microbench(load_scenario(args.scenario, args.architecture, args.model), args.output, args.binary)))
        elif args.action == "calibrate":
            save(args.output, fit(load_document(args.records)))
        else:
            report = load_document(args.prediction)
            report["prediction_error"] = compare(report, load_document(args.reference))
            save(args.output, report)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(2, str(error) + "\n")


if __name__ == "__main__":
    main()
