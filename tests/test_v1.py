import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run
from calibration import fit, compare
from reporting import chrome_trace
from tools.experiments import sweep, microbench
from tools.reference_runner import capture, normalize


class V1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.binary = os.environ.get("INFER_SIM_BINARY", str(ROOT / "build/infer_sim"))

    def data(self, name="moe-small"):
        return run.load_scenario(ROOT / "examples" / (name + ".json"))

    def simulate(self, data, profile=None):
        return run.simulate(self.data(data) if isinstance(data, str) else data,
                            binary=self.binary, profile=profile)

    def test_moe_conservation_and_batch_independent_routing(self):
        data = self.data()
        report = self.simulate(data)
        visits = sum(r["prompt_tokens"] + r["output_tokens"] for r in data["requests"]) * data["model"]["layers"]
        self.assertEqual(report["moe"]["token_layer_visits"], visits)
        self.assertEqual(report["moe"]["routed_assignments"], visits * data["model"]["top_k"])
        self.assertEqual(sum(report["moe"]["expert_tokens"]), report["moe"]["routed_assignments"])
        self.assertEqual(report["moe"]["dispatch_payload_bytes"], report["moe"]["combine_payload_bytes"])
        self.assertEqual(report["compute_flops"], run.fast_report(data)["estimated_flops"])
        data["serving"]["prefill_chunk_tokens"] = 1
        other = self.simulate(data)
        self.assertEqual(report["moe"]["expert_tokens"], other["moe"]["expert_tokens"])
        self.assertEqual(report["compute_flops"], other["compute_flops"])
        self.assertEqual(report, self.simulate("moe-small"))

    def test_skew_and_expert_placement(self):
        balanced = self.simulate("moe-balanced-small")
        skew = self.simulate("moe-skew-small")
        self.assertEqual(skew["compute_flops"], balanced["compute_flops"])
        self.assertGreater(skew["moe"]["straggler_ps_sum"], balanced["moe"]["straggler_ps_sum"])
        self.assertEqual(skew["moe"]["expert_tokens"][0], skew["moe"]["routed_assignments"])
        data = self.data("moe-skew-small")
        data["parallel"]["expert_placement"] = [1, 0, 1, 0]
        moved = self.simulate(data)
        self.assertEqual(moved["moe"]["expert_tokens"], skew["moe"]["expert_tokens"])
        self.assertEqual(moved["moe"]["device_assignments"], list(reversed(skew["moe"]["device_assignments"])))

    def test_hbf_moe_and_one_slot_streaming(self):
        report = self.simulate("moe-hbf-small")
        gpu = self.simulate("moe-small")
        self.assertEqual(report["compute_flops"], gpu["compute_flops"])
        self.assertEqual(report["moe"]["expert_tokens"], gpu["moe"]["expert_tokens"])
        self.assertGreater(report["hbf_fill_bytes"], 0)
        data = self.data("moe-hbf-small")
        data["storage"]["cache_bytes_per_device"] = data["storage"]["slot_bytes"]
        streamed = self.simulate(data)
        self.assertGreater(streamed["hbf_evictions"], 0)
        self.assertEqual(streamed["output_tokens"], gpu["output_tokens"])

    def test_cross_rack_hierarchy_and_placement(self):
        ring = self.simulate("cross-rack-ring-small")
        hierarchy = self.simulate("cross-rack-hierarchical-small")
        for report in (ring, hierarchy):
            self.assertEqual(sum(v["bytes"] for v in report["network_tiers"].values()), report["fabric_bytes"])
            self.assertEqual(sum(v["queue_wait_ps"] for v in report["network_tiers"].values()),
                             report["fabric_queue_wait_ps_sum"])
            self.assertTrue(all(v["bytes"] > 0 for v in report["network_tiers"].values()))
        self.assertLess(hierarchy["network_tiers"]["cross_rack"]["bytes"],
                        ring["network_tiers"]["cross_rack"]["bytes"])
        self.assertLess(hierarchy["makespan_ps"], ring["makespan_ps"])
        local, cross = self.simulate("placement-local-small"), self.simulate("placement-cross-small")
        self.assertEqual(local["compute_flops"], cross["compute_flops"])
        self.assertGreater(cross["network_tiers"]["cross_rack"]["bytes"],
                           local["network_tiers"]["cross_rack"]["bytes"])
        self.assertGreater(cross["makespan_ps"], local["makespan_ps"])

    def test_npu_dma_capacity_and_shared_graph(self):
        gpu = self.simulate("moe-small")
        npu = self.simulate("npu-moe-small")
        small = self.simulate("npu-small-sram")
        self.assertEqual(gpu["compute_flops"], npu["compute_flops"])
        self.assertEqual(gpu["moe"]["expert_tokens"], npu["moe"]["expert_tokens"])
        self.assertEqual(small["onchip"]["dma_bytes"], npu["onchip"]["dma_bytes"])
        self.assertLessEqual(small["onchip"]["peak_sram_bytes"], 256)
        self.assertLessEqual(npu["onchip"]["peak_sram_bytes"], 2048)
        self.assertGreater(small["onchip"]["dma_tiles"], npu["onchip"]["dma_tiles"])
        self.assertGreater(small["makespan_ps"], npu["makespan_ps"])
        for node in npu["trace"]:
            if node["kind"] in ("compute", "grouped_gemm"):
                parents = npu["trace"][node["parents"][0]]["parents"]
                self.assertEqual(npu["trace"][parents[0]]["kind"], "dma")

    def test_ep_tp_pp_dp_and_shared_nic_contention(self):
        for tp, pp, dp in ((2, 1, 1), (1, 2, 1), (1, 1, 2), (2, 2, 2)):
            with self.subTest(tp=tp, pp=pp, dp=dp):
                data = self.data()
                devices = 2 * tp * pp * dp
                data["parallel"].update(tp=tp, pp=pp, dp=dp, placement=list(range(devices)))
                data["hardware"].update(devices=devices, device_nodes=[0] * devices)
                report = self.simulate(data)
                self.assertEqual(report["compute_flops"], run.fast_report(data)["estimated_flops"])
                self.assertLessEqual(report["peak_hbm_bytes_per_device"], data["hardware"]["hbm_capacity_bytes"])
                self.assertEqual(sum(report["moe"]["device_assignments"]), report["moe"]["routed_assignments"] * tp)
        data = self.data("cross-rack-ring-small")
        data["parallel"].update(tp=2, dp=4, placement=list(range(8)))
        data["hardware"].update(device_nodes=[0, 1] * 4, node_racks=[0, 1])
        data["requests"] = [dict(arrival_ps=0, prompt_tokens=4, output_tokens=2) for _ in range(4)]
        report = self.simulate(data)
        self.assertGreater(report["network_tiers"]["nic"]["queue_wait_ps"], 0)

    def test_timeline_distribution_and_trace_export(self):
        report = self.simulate("npu-moe-small")
        self.assertEqual(report["critical_path"]["duration_ps"], report["makespan_ps"])
        self.assertNotIn("unattributed_wait", report["critical_path"]["time_by_kind_ps"])
        self.assertEqual(report["distributions_ps"]["ttft"]["count"], 2)
        self.assertEqual(report["distributions_ps"]["tpot"]["count"], 3)
        self.assertEqual(report["prediction_error"]["status"], "unmeasured")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            chrome_trace(report, path)
            self.assertTrue(json.loads(path.read_text())["traceEvents"])

    def test_calibration_fit_held_out_and_reference_errors(self):
        records = json.loads((ROOT / "examples/calibration-synthetic.json").read_text())
        profile = fit(records)
        self.assertEqual(profile["hardware_overrides"]["gemm_ps_per_flop"], 2)
        self.assertEqual(profile["status"], "synthetic")
        self.assertEqual(profile["operators"]["gemm"]["held_out_error"]["mape_percent"], 0)
        data = self.data("npu-calibration-small")
        report = self.simulate(data, profile)
        self.assertNotEqual(report["compute_service_ps_sum"], self.simulate(data)["compute_service_ps_sum"])
        self.assertIn("calibration_coverage", report)
        reference = {"scenario_sha256": report["scenario_sha256"], "hardware_id": "synthetic-npu-v1",
                     "source": "test fixture, not hardware", "evidence": "synthetic",
                     "measurement_scope": "end_to_end", "metrics": {"mean_ttft_ps": report["mean_ttft_ps"] * 2}}
        result = compare(report, reference)
        self.assertEqual(result["aggregate"]["mape_percent"], 50)
        reference["scenario_sha256"] = "mismatch"
        with self.assertRaisesRegex(ValueError, "hash"):
            compare(report, reference)
        records[0]["measurement_scope"] = "kernel_total"
        with self.assertRaisesRegex(ValueError, "contain memory"):
            fit(records)
        data["hardware"]["hardware_id"] = "wrong-target"
        with self.assertRaisesRegex(ValueError, "hardware_id"):
            self.simulate(data, profile)

    def test_sweeps_report_failures_and_microbench(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = sweep(self.data(), {"model.top_k": [1, 2, 10]}, directory, self.binary)
            self.assertEqual(summary["succeeded"], 2)
            self.assertEqual(summary["failed"], 1)
            self.assertTrue((Path(directory) / "summary.csv").exists())
            result = microbench(self.data("dense-small"), Path(directory) / "micro", self.binary)
            self.assertEqual(result, {"cases": 6, "evidence": "simulated"})

    def test_kv_admission_progress_and_invalid_inputs(self):
        data = self.data("dense-small")
        data["hardware"]["hbm_capacity_bytes"] = 38500
        report = self.simulate(data)
        self.assertEqual(report["output_tokens"], 5)
        self.assertLessEqual(report["peak_hbm_bytes_per_device"], 38500)
        cases = [("model", "top_k", 8), ("model", "expert_weights", [0, 0, 0, 0]),
                 ("hardware", "peak_tflops", float("nan")), ("parallel", "placement", [0, 0]),
                 ("hardware", "device_nodes", [0, 2]), ("model", "layers", True)]
        for section, field, value in cases:
            with self.subTest(field=field):
                data = self.data()
                data[section][field] = value
                with self.assertRaises(ValueError):
                    run.validate(data)

    def test_reference_adapter_and_memory_profiles(self):
        report = self.simulate("npu-calibration-small")
        reference = capture(report, [sys.executable, str(ROOT / "tests/fixtures/reference_adapter.py")])
        self.assertEqual(reference["evidence"], "synthetic")
        self.assertEqual(reference["metrics"]["mean_tpot_ps"], 3000000)
        self.assertEqual(compare(report, reference)["aggregate"]["count"], 4)
        bad = dict(reference, token_convention="prefill_emits_first_token")
        with self.assertRaisesRegex(ValueError, "convention"):
            normalize(report, bad)
        times = []
        for profile in ("hbm2_2000", "hbm3_6400", "hbm4_8000"):
            data = self.data("dense-small")
            data["hardware"]["hbm_profile"] = profile
            result = self.simulate(data)
            self.assertEqual(result["compute_flops"], 492288)
            times.append(result["makespan_ps"])
        self.assertGreater(len(set(times)), 1)


if __name__ == "__main__":
    unittest.main()
