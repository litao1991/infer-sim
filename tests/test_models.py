import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config_io import load_document
from run import load_scenario, simulate, fast_report, validate
from tools.import_model import convert


class ModelDimensionTests(unittest.TestCase):
    binary = os.environ.get("INFER_SIM_BINARY", str(ROOT / "build/infer_sim"))

    def test_independent_head_dim_matches_hand_calculation(self):
        data = load_scenario(ROOT / "examples/dense-small.json")
        data["model"]["head_dim"] = 16  # H=32, query width=64, KV width=32.
        expected = fast_report(data)
        self.assertEqual(expected["model_parameters"], 24576)
        self.assertEqual(expected["kv_bytes_per_token_total"], 256)
        self.assertEqual(expected["estimated_flops"], 665088)
        for topology in ("dense-small.json", "dense-npu-small.json"):
            with self.subTest(topology=topology):
                configured = load_document(ROOT / "examples" / topology)
                configured["model"] = copy.deepcopy(data["model"])
                if topology == "dense-npu-small.json":
                    configured["hardware"]["devices"] = 2
                    configured["parallel"]["tp"] = 2
                configured = validate(configured)
                report = simulate(configured, binary=self.binary)
                self.assertEqual(report["compute_flops"], 665088)
                self.assertEqual(report["output_tokens"], 5)
                self.assertLessEqual(report["peak_hbm_bytes_per_device"], configured["hardware"]["hbm_capacity_bytes"])
                configured["model"].update(kind="moe", experts=4, top_k=2)
                configured = validate(configured)
                moe_report = simulate(configured, binary=self.binary)
                self.assertEqual(moe_report["compute_flops"], 991232)
                self.assertEqual(fast_report(configured)["estimated_flops"], 991232)

    def test_explicit_head_dim_supports_nondivisible_hidden(self):
        data = load_scenario(ROOT / "examples/dense-small.json")
        data["model"].update(hidden=18, head_dim=8)
        report = simulate(data, binary=self.binary)
        self.assertEqual(report["compute_flops"], fast_report(data)["estimated_flops"])
        del data["model"]["head_dim"]
        with self.assertRaisesRegex(ValueError, "head_dim"):
            validate(data)

    def test_importer_preserves_qwen_moe_shape_and_rejects_unsupported(self):
        root = ROOT / "configs/models/upstream"
        raw = load_document(root / "qwen3-30b-a3b.json")
        model = convert(raw, "Qwen/Qwen3-30B-A3B")["model"]
        self.assertEqual((model["hidden"], model["heads"], model["head_dim"]), (2048, 32, 128))
        self.assertEqual((model["ffn"], model["experts"], model["top_k"]), (768, 128, 8))
        with self.assertRaisesRegex(ValueError, "MLA"):
            convert(load_document(root / "deepseek-v3.json"), "deepseek-ai/DeepSeek-V3")
        for field, value in (("quantization_config", {"quant_method": "fp8"}),
                             ("decoder_sparse_step", 2), ("mlp_only_layers", [0]),
                             ("use_sliding_window", True), ("hidden_act", "gelu")):
            with self.subTest(field=field):
                bad = dict(raw, **{field: value})
                with self.assertRaises(ValueError):
                    convert(bad, "unsupported-variant")


@unittest.skipUnless(importlib.util.find_spec("yaml"), "optional YAML dependency is not installed")
class ModelPresetTests(unittest.TestCase):
    def test_all_ten_presets_are_fast_runnable_with_provenance(self):
        files = sorted((ROOT / "configs/models").glob("*.yaml"))
        self.assertEqual(len(files), 10)
        for path in files:
            with self.subTest(preset=path.name):
                data = load_scenario(ROOT / "examples/model-evaluation.yaml", model=path)
                report = simulate(data, mode="fast")
                self.assertGreater(report["model_parameters"], 1e9)
                self.assertTrue(report["model_accounting"]["source"]["source_url"].startswith("https://"))
                self.assertEqual(report["model_accounting"]["scope"], "transformer_block_matrices")
                self.assertFalse(report["model_accounting"]["overrides_from_preset"])
                self.assertIsNone(report["critical_path"])
                self.assertEqual(report["prediction_error"]["status"], "unmeasured")
                upstream = ROOT / "configs/models/upstream" / (path.stem + ".json")
                if upstream.exists():
                    metadata = data["model_metadata"]
                    self.assertEqual(metadata["source_sha256"], hashlib.sha256(upstream.read_bytes()).hexdigest())
                    self.assertEqual(metadata["reference_dimensions"],
                                     convert(load_document(upstream), metadata["model_id"])["model"])
        data = load_scenario(ROOT / "examples/model-evaluation.yaml", model=ROOT / "configs/models/qwen3-30b-a3b.yaml")
        report = simulate(data, mode="fast")
        self.assertEqual(report["model_parameters"], 29909581824)
        self.assertEqual(report["active_parameters_per_token"], 2730491904)
        self.assertEqual(report["kv_bytes_per_token_total"], 98304)
        with self.assertRaisesRegex(ValueError, "local scale limits"):
            simulate(data)

    def test_model_switch_replaces_old_model_and_records_overrides(self):
        data = load_scenario(ROOT / "examples/moe-small.yaml", ROOT / "configs/hardware/gpu-single.yaml",
                             ROOT / "configs/models/llama-3.1-8b.yaml")
        self.assertEqual(data["model"]["kind"], "dense")
        self.assertNotIn("experts", data["model"])
        data["model"]["dtype_bytes"] = 1
        report = simulate(data, mode="fast")
        self.assertEqual(report["model_accounting"]["overrides_from_preset"]["dtype_bytes"],
                         {"preset": 2, "resolved": 1})

    def test_model_import_and_selection_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            output = folder / "model.yaml"
            subprocess.run([sys.executable, str(ROOT / "tools/import_model.py"),
                            str(ROOT / "configs/models/upstream/qwen3-30b-a3b.json"),
                            "--model-id", "Qwen/Qwen3-30B-A3B", "--output", str(output)],
                           check=True, capture_output=True, text=True)
            subprocess.run([sys.executable, str(ROOT / "run.py"),
                            str(ROOT / "examples/model-evaluation.yaml"), "--model", str(output),
                            "--mode", "fast", "--output", str(folder / "report.json")],
                           check=True, capture_output=True, text=True)
            report = json.loads((folder / "report.json").read_text())
            self.assertEqual(report["scenario"]["model"]["ffn"], 768)
            (folder / "grid.yaml").write_text("model.dtype_bytes: [1, 2]\n")
            result = subprocess.run([sys.executable, str(ROOT / "tools/experiments.py"), "sweep",
                                     str(ROOT / "examples/model-evaluation.yaml"), "--model", str(output),
                                     "--mode", "fast", "--grid", str(folder / "grid.yaml"),
                                     "--output", str(folder / "sweep")], check=True, capture_output=True, text=True)
            self.assertEqual(json.loads(result.stdout), {"succeeded": 2, "failed": 0})


if __name__ == "__main__":
    unittest.main()
