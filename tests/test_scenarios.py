import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run  # noqa: E402


class ScenarioTests(unittest.TestCase):
    def report(self, scenario):
        result = subprocess.run(
            [sys.executable, str(ROOT / "run.py"), str(ROOT / "examples" / scenario),
             "--mode", "detailed", "--binary",
             os.environ.get("INFER_SIM_BINARY", str(ROOT / "build" / "infer_sim"))],
            capture_output=True, text=True, check=True)
        return json.loads(result.stdout)

    def test_fast_formula_and_detailed_conservation(self):
        data = run.load_scenario(ROOT / "examples/dense-small.json")
        fast = run.fast_report(data)
        self.assertEqual(fast["model_parameters"], 18432)
        self.assertEqual(fast["kv_bytes_per_token_total"], 128)
        self.assertEqual(fast["estimated_flops"], 492288)
        detailed = self.report("dense-small.json")
        self.assertEqual(detailed["compute_flops"], fast["estimated_flops"])
        self.assertEqual(detailed["direct_hbm_requested_bytes"],
                         detailed["hbm_read_bytes"] + detailed["hbm_write_bytes"])
        self.assertLessEqual(detailed["peak_hbm_bytes_per_device"],
                             data["hardware"]["hbm_capacity_bytes"])
        self.assertEqual(detailed["output_tokens"], 5)
        self.assertTrue(all(r["ttft_ps"] > 0 and r["latency_ps"] >= r["ttft_ps"]
                            for r in detailed["requests_detail"]))

    def test_repeatability_and_bounded_backpressure(self):
        self.assertEqual(self.report("dense-small.json"), self.report("dense-small.json"))
        limited = self.report("backpressure-small.json")
        self.assertGreater(limited["hbm_backpressure"], 0)
        self.assertEqual(limited["output_tokens"], 5)
        self.assertEqual(limited["direct_hbm_requested_bytes"],
                         limited["hbm_read_bytes"] + limited["hbm_write_bytes"])

    def test_hbf_prefetch_eviction_and_writeback(self):
        prefetch = self.report("hbf-small.json")
        baseline = self.report("hbf-no-prefetch-small.json")
        evicted = self.report("hbf-eviction-small.json")
        flushed = self.report("hbf-writeback-small.json")
        self.assertGreater(prefetch["hbf_fill_bytes"], 0)
        self.assertGreater(prefetch["hbf_prefetches"], 0)
        self.assertLess(prefetch["makespan_ps"], baseline["makespan_ps"])
        self.assertEqual(prefetch["hbf_fill_bytes"], baseline["hbf_fill_bytes"])
        self.assertGreater(evicted["hbf_evictions"], 0)
        self.assertGreater(evicted["hbf_fill_bytes"], prefetch["hbf_fill_bytes"])
        self.assertEqual(flushed["hbf_writeback_bytes"], 8192)
        self.assertEqual(flushed["requests_detail"], prefetch["requests_detail"])
        self.assertEqual(flushed["service_hbm_read_bytes"], prefetch["hbm_read_bytes"])
        self.assertGreater(flushed["hbm_read_bytes"], flushed["service_hbm_read_bytes"])

    def test_tensor_pipeline_and_data_parallel(self):
        single = self.report("dense-small.json")
        tp = self.report("tp-small.json")
        dp = self.report("dp-small.json")
        single_pipeline = self.report("single-pipeline-check.json")
        pipelined = self.report("pipeline-pipeline-check.json")
        self.assertEqual(tp["compute_flops"], single["compute_flops"])
        self.assertGreater(tp["fabric_bytes"], 0)
        self.assertEqual(dp["fabric_bytes"], 0)
        self.assertLess(dp["mean_request_latency_ps"], single["mean_request_latency_ps"])
        self.assertEqual(pipelined["output_tokens"], single_pipeline["output_tokens"])
        self.assertEqual(pipelined["compute_flops"], single_pipeline["compute_flops"])
        self.assertGreater(pipelined["fabric_bytes"], 0)
        self.assertLess(pipelined["makespan_ps"], single_pipeline["makespan_ps"])

    def test_capacity_rejection_and_input_validation(self):
        data = run.load_scenario(ROOT / "examples/dense-small.json")
        bad = copy.deepcopy(data)
        bad["hardware"]["hbm_capacity_bytes"] = 37887
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(bad))
            result = subprocess.run([sys.executable, str(ROOT / "run.py"), str(path),
                                     "--binary", os.environ.get("INFER_SIM_BINARY",
                                     str(ROOT / "build" / "infer_sim"))],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("KV capacity deadlock", result.stderr)
            bad["parallel"]["tp"] = 2
            path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, "tp \* pp \* dp"):
                run.load_scenario(path)


if __name__ == "__main__":
    unittest.main()
