import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config_io import load_document, scenario_document
from run import load_scenario, simulate


class JsonConfigurationTests(unittest.TestCase):
    def test_missing_yaml_dependency_preserves_json(self):
        with patch.dict(sys.modules, {"yaml": None}):
            self.assertEqual(load_scenario(ROOT / "examples/moe-small.json")["model"]["kind"], "moe")
            with self.assertRaisesRegex(ValueError, "requires PyYAML"):
                load_document(ROOT / "examples/moe-small.yaml")

    def test_duplicate_json_key_and_invalid_architecture(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text('{"hardware": {}, "hardware": {}}')
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_document(path)
            path.write_text(json.dumps({"hardware": {}, "model": {}}))
            with self.assertRaisesRegex(ValueError, "architecture must contain"):
                scenario_document(ROOT / "examples/moe-small.json", path)


@unittest.skipUnless(importlib.util.find_spec("yaml"), "optional YAML dependency is not installed")
class YamlConfigurationTests(unittest.TestCase):
    binary = os.environ.get("INFER_SIM_BINARY", str(ROOT / "build/infer_sim"))

    def test_yaml_json_full_report_equivalence(self):
        json_data = load_scenario(ROOT / "examples/moe-small.json")
        yaml_data = load_scenario(ROOT / "examples/moe-small.yaml")
        self.assertEqual(json_data, yaml_data)
        self.assertEqual(simulate(json_data, binary=self.binary), simulate(yaml_data, binary=self.binary))

    def test_architectures_on_dense_and_moe(self):
        for model in ("dense-small.json", "moe-small.yaml"):
            baseline = simulate(load_scenario(ROOT / "examples" / model), binary=self.binary)
            for name in ("gpu-single", "gpu-8card-nvswitch", "gpu-8card-crossrack", "npu-2card"):
                with self.subTest(model=model, architecture=name):
                    data = load_scenario(ROOT / "examples" / model,
                                         ROOT / "configs/hardware" / (name + ".yaml"))
                    report = simulate(data, binary=self.binary)
                    self.assertEqual(report["compute_flops"], baseline["compute_flops"])
                    self.assertEqual(report["output_tokens"], baseline["output_tokens"])
                    self.assertEqual(report["hardware_kind"], data["hardware"]["kind"])
                    if name == "gpu-8card-crossrack":
                        self.assertGreater(report["network_tiers"]["cross_rack"]["bytes"], 0)
                    if name == "npu-2card":
                        self.assertGreater(report["onchip"]["dma_bytes"], 0)
        # Whole-section replacement drops NPU DMA settings and stale 2-card placement.
        data = load_scenario(ROOT / "examples/npu-moe-small.json", ROOT / "configs/hardware/gpu-single.yaml")
        self.assertNotIn("sram_bytes", data["hardware"])
        self.assertEqual(data["parallel"]["placement"], [0])

    def test_yaml_rejects_ambiguous_or_nondata_values(self):
        contents = ["hardware: {}\nhardware: {}\n", "hardware: {devices: 1, devices: 2}",
                    "!!python/name:builtins.len ''", "value: 2026-10-02", "value: .nan",
                    "loop: &loop [*loop]", "[1, 2, 3]", "hardware: [invalid"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yml"
            for content in contents:
                with self.subTest(content=content):
                    path.write_text(content)
                    with self.assertRaises((ValueError, TypeError)):
                        data = load_scenario(path)
                        self.fail(f"invalid YAML scenario accepted: {data}")

    def test_cli_yaml_architecture_grid_and_calibration(self):
        import yaml
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            report_path = output / "report.json"
            subprocess.run([sys.executable, str(ROOT / "run.py"), str(ROOT / "examples/moe-small.yaml"),
                            "--architecture", str(ROOT / "configs/hardware/npu-2card.yaml"),
                            "--binary", self.binary, "--output", str(report_path)], check=True,
                           capture_output=True, text=True)
            self.assertEqual(json.loads(report_path.read_text())["hardware_kind"], "npu")
            result = subprocess.run([sys.executable, str(ROOT / "tools/experiments.py"), "sweep",
                                     str(ROOT / "examples/moe-small.yaml"), "--grid",
                                     str(ROOT / "examples/sweep-small.yaml"), "--architecture",
                                     str(ROOT / "configs/hardware/gpu-single.yaml"), "--binary", self.binary,
                                     "--mode", "fast", "--output", str(output / "sweep")],
                                    check=True, capture_output=True, text=True)
            self.assertEqual(json.loads(result.stdout), {"succeeded": 8, "failed": 0})
            rows = json.loads((ROOT / "examples/calibration-synthetic.json").read_text())
            (output / "calibration.yml").write_text(yaml.safe_dump(rows))
            subprocess.run([sys.executable, str(ROOT / "tools/experiments.py"), "calibrate",
                            str(output / "calibration.yml"), "--output", str(output / "profile.json")],
                           check=True, capture_output=True, text=True)
            profile = json.loads((output / "profile.json").read_text())
            (output / "profile.yaml").write_text(yaml.safe_dump(profile))
            subprocess.run([sys.executable, str(ROOT / "run.py"),
                            str(ROOT / "examples/npu-calibration-small.json"), "--profile",
                            str(output / "profile.yaml"), "--binary", self.binary, "--output", str(report_path)],
                           check=True, capture_output=True, text=True)
            self.assertEqual(json.loads(report_path.read_text())["calibration"]["status"], "synthetic")


if __name__ == "__main__":
    unittest.main()
