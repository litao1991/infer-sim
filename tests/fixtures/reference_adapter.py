"""Synthetic adapter fixture, deliberately not a hardware benchmark."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--scenario", required=True)
args = parser.parse_args()
scenario = json.loads(Path(args.scenario).read_text())
print(json.dumps({"source": "synthetic adapter integration fixture", "evidence": "synthetic",
                  "hardware_id": scenario["hardware"]["hardware_id"],
                  "token_convention": "explicit_first_decode",
                  "requests": [{"request_id": i, "arrival_ps": row["arrival_ps"],
                                "token_times_ps": [row["arrival_ps"] + 10000000 + 3000000 * token
                                                   for token in range(row["output_tokens"])]}
                               for i, row in enumerate(scenario["requests"])]}))
