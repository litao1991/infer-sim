import hashlib
import importlib.util
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from bailian_eval import inspect, midpoint_window, run, scaled_tokens
from download_bailian import verify
from bailian_kv_sweep import sweep


class BailianEvalTests(unittest.TestCase):
    binary = os.environ.get("INFER_SIM_BINARY", str(ROOT / "build/infer_sim"))

    def trace(self, directory):
        records = [
            (0, 64, 8), (0.1, 128, 16), (0.11, 130, 17),
            (2, 32, 0), (5, 512, 64), (5.1, 640, 128),
        ]
        path = Path(directory) / "qwen_traceA_blksz_16.jsonl"
        path.write_text("".join(json.dumps({
            "timestamp": timestamp, "input_length": prompt,
            "output_length": output, "hash_ids": ["abc"],
            "chat_id": i, "parent_chat_id": -1, "type": "text",
        }) + "\n" for i, (timestamp, prompt, output) in enumerate(records)))
        return path

    def test_streaming_stats_and_window_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.trace(directory)
            stats = inspect(path, 2)
            self.assertEqual(stats["rows"], 6)
            self.assertEqual(stats["eligible_rows"], 5)
            self.assertEqual([r["line"] for r in stats["burst"]], [2, 3])
            self.assertEqual([r["line"] for r in midpoint_window(path, stats["midpoint_s"], 2)],
                             [5, 6])
            self.assertEqual(stats["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(scaled_tokens(130, 64, 24), 3)
            self.assertEqual(scaled_tokens(100000, 64, 8), 8)
            with self.assertRaisesRegex(ValueError, "checksum or size mismatch"):
                verify(path, path.name)

    def test_eight_detailed_placement_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.trace(directory)
            output = Path(directory) / "reports"
            rows = run(directory, output, self.binary, sample_size=2, speeds=(1000,),
                       sources=(path.name,))
            self.assertEqual(len(rows), 8)
            self.assertTrue(all(row["status"] == "ok" for row in rows))
            self.assertEqual({row["policy"] for row in rows},
                             {"hbm-hbm", "hbf-hbm", "hbm-hbf", "hbf-hbf"})
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual((summary["succeeded"], summary["failed"]), (8, 0))
            self.assertEqual(summary["method"]["token_divisor"], 64)
            self.assertIn("two 2-request windows", (output / "summary.md").read_text())
            scenario = json.loads((output / "cases" /
                                   "qwen_traceA_blksz_16-burst-x1000-hbm-hbm" /
                                   "scenario.json").read_text())
            self.assertEqual([r["prompt_tokens"] for r in scenario["requests"]], [2, 3])
            self.assertEqual([r["output_tokens"] for r in scenario["requests"]], [1, 1])
            self.assertEqual([r["arrival_ps"] for r in scenario["requests"]], [0, 10000000])
            self.assertGreater(next(r["mean_ttft_ms"] for r in rows
                                    if r["policy"] == "hbm-hbf"), 0)
            other = Path(directory) / "reports-again"
            run(directory, other, self.binary, sample_size=2, speeds=(1000,),
                sources=(path.name,))
            self.assertEqual((output / "summary.json").read_bytes(),
                             (other / "summary.json").read_bytes())

    def test_rejects_unsorted_and_insufficient_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.trace(directory)
            rows = path.read_text().splitlines()
            path.write_text("\n".join([rows[1], rows[0]] + rows[2:]) + "\n")
            with self.assertRaisesRegex(ValueError, "timestamps out of order"):
                inspect(path, 2)
            with self.assertRaisesRegex(ValueError, "local detailed-mode budget"):
                run(directory, Path(directory) / "out", self.binary, sample_size=17,
                    sources=(path.name,))

    def test_selected_placement_write_policy_and_block_override(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.trace(directory)
            output = Path(directory) / "policy"
            rows = run(directory, output, self.binary, sample_size=2, speeds=(1000,),
                       sources=(path.name,), cache_policy="write_aware", block_tokens=8,
                       placements=("hbm-hbf",))
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(r['status'] == 'ok' for r in rows))
            self.assertIsNone(rows[0]['ttft_ratio_to_hbm'])
            report = json.loads((output / rows[0]['report']).read_text())
            self.assertEqual(report['scenario']['storage']['kv_cache_policy'], 'write_aware')
            self.assertEqual(report['scenario']['serving']['kv_block_tokens'], 8)

    def test_write_policy_sweep_on_identical_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.trace(directory)
            output = Path(directory) / 'sweep'
            with contextlib.redirect_stdout(io.StringIO()):
                summary = sweep(directory, output, self.binary, blocks=(4,), sources=(path.name,),
                                sample_size=2, divisor=64, prompt_cap=24, output_cap=8)
            self.assertEqual((summary['succeeded'], summary['failed']), (6, 0))
            self.assertEqual(len(summary['aggregates']), 3)
            for row in summary['aggregates']:
                self.assertEqual(row['write_dlu_ops'] * 4096, row['hbf_write_bytes'])
            # Eviction choice cannot change the workload being compared.
            counts = {(r['input_tokens'], r['output_tokens']) for r in summary['runs']
                      if r['window'] == 'burst'}
            self.assertEqual(len(counts), 1)

    @unittest.skipUnless(importlib.util.find_spec("yaml"), "PyYAML is not installed")
    def test_yaml_npu_architecture(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.trace(directory)
            output = Path(directory) / "npu"
            rows = run(directory, output, self.binary, sample_size=2, speeds=(1000,),
                       sources=(path.name,),
                       architecture=ROOT / "configs/hardware/npu-2card.yaml")
            self.assertEqual(sum(r["status"] == "ok" for r in rows), 8)
            report = json.loads((output / rows[0]["report"]).read_text())
            self.assertEqual(report["hardware_kind"], "npu")
            self.assertEqual(report["scenario"]["parallel"]["tp"], 2)


if __name__ == "__main__":
    unittest.main()
