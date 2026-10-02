#!/usr/bin/env python3
"""Convert a local Hugging Face config.json into a simulator model preset.

Reads configuration only. Never downloads weights or executes remote model code.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config_io import load_document
from schema import number

SUPPORTED = {"llama", "qwen2", "qwen3", "qwen3_moe", "mixtral"}
DTYPE_BYTES = {"bfloat16": 2, "float16": 2, "float32": 4}


def convert(config, model_id, source_url=None, revision=None, source_sha256=None):
    if not isinstance(config, dict):
        raise ValueError("Hugging Face config must be a mapping")
    if config.get("kv_lora_rank") or config.get("model_type", "").startswith("deepseek"):
        raise ValueError("MLA / DeepSeek mixed dense-MoE layers are not implemented; cannot map them to GQA")
    family = config.get("model_type")
    if family not in SUPPORTED:
        raise ValueError(f"unsupported model_type {family!r}; supported: {', '.join(sorted(SUPPORTED))}")
    if config.get("quantization_config"):
        raise ValueError("quantized checkpoint metadata cannot be reduced to one dtype_bytes value")
    if config.get("hidden_act", "silu") != "silu":
        raise ValueError("current FFN model requires SwiGLU/silu")
    if config.get("use_sliding_window") or (family == "mixtral" and config.get("sliding_window")):
        raise ValueError("sliding-window attention is not implemented")
    if config.get("layer_types") or config.get("mlp_only_layers") or config.get("decoder_sparse_step", 1) != 1:
        raise ValueError("heterogeneous attention/MLP layer schedules are not implemented")
    dtype = config.get("dtype", config.get("torch_dtype"))
    if dtype not in DTYPE_BYTES:
        raise ValueError("config must specify bfloat16, float16, or float32 dtype")
    moe = family in ("mixtral", "qwen3_moe")
    heads, hidden = config["num_attention_heads"], config["hidden_size"]
    for key in ("num_attention_heads", "hidden_size", "num_hidden_layers"):
        number(key, config[key], integer=True)
    if "head_dim" not in config and hidden % heads:
        raise ValueError("non-divisible hidden/heads requires explicit head_dim")
    model = {"kind": "moe" if moe else "dense", "layers": config["num_hidden_layers"],
             "hidden": hidden, "heads": heads, "kv_heads": config.get("num_key_value_heads", heads),
             "head_dim": config.get("head_dim", hidden // heads),
             "ffn": config["moe_intermediate_size"] if family == "qwen3_moe" else config["intermediate_size"],
             "dtype_bytes": DTYPE_BYTES[dtype]}
    if moe:
        model.update(experts=config["num_experts"] if family == "qwen3_moe" else config["num_local_experts"],
                     top_k=config["num_experts_per_tok"], shared_experts=0, routing_seed=1)
    for key, value in model.items():
        if key != "kind":
            number(key, value, integer=True, strict=key not in ("shared_experts", "routing_seed"))
    if model["heads"] % model["kv_heads"] or (moe and model["top_k"] > model["experts"]):
        raise ValueError("invalid attention or expert dimensions")
    limitations = ["models Transformer block matrices; excludes embeddings, LM head, norms, biases, RoPE and sampling",
                   "real-size presets require fast mode on this local machine",
                   "routing is deterministic synthetic routing, not learned checkpoint routing"]
    if family in ("qwen3", "qwen3_moe"):
        limitations.append("Q/K normalization is not separately timed")
    metadata = {"model_id": model_id, "model_type": family,
                "source_url": source_url or "local config.json", "revision": revision,
                "source_sha256": source_sha256 or hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                "source_dtype": dtype, "vocab_size": config.get("vocab_size"),
                "max_position_embeddings": config.get("max_position_embeddings"),
                "reference_dimensions": dict(model), "scope": "transformer_block_matrices",
                "limitations": limitations}
    return {"model": model, "model_metadata": metadata}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="local Hugging Face config.json")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--source-url")
    parser.add_argument("--revision")
    parser.add_argument("--output", required=True, help=".json, .yaml, or .yml")
    args = parser.parse_args()
    try:
        preset = convert(load_document(args.config), args.model_id, args.source_url, args.revision,
                         hashlib.sha256(Path(args.config).read_bytes()).hexdigest())
        path = Path(args.output)
        if path.suffix.lower() == ".json":
            content = json.dumps(preset, indent=2) + "\n"
        elif path.suffix.lower() in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError as error:
                raise ValueError("YAML output requires requirements-yaml.txt") from error
            content = yaml.safe_dump(preset, sort_keys=False, allow_unicode=True)
        else:
            raise ValueError("output must be .json, .yaml, or .yml")
        path.write_text(content)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(2, str(error) + "\n")


if __name__ == "__main__":
    main()
