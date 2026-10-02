"""One JSON/YAML entry point for scenarios, architecture presets, and experiments."""
import json
from pathlib import Path


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if not isinstance(key, str):
            raise ValueError("configuration mapping keys must be strings")
        if key in result:
            raise ValueError(f"duplicate configuration key: {key}")
        result[key] = value
    return result


def load_document(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in (".json", ".yaml", ".yml"):
        raise ValueError(f"{path}: expected .json, .yaml, or .yml")
    content = path.read_text(encoding="utf-8-sig")
    try:
        if suffix == ".json":
            data = json.loads(content, object_pairs_hook=_unique_pairs)
        else:
            try:
                import yaml
            except ImportError as error:
                raise ValueError("YAML support requires PyYAML; install requirements-yaml.txt "
                                 "with the Python environment used to run this command") from error

            class UniqueSafeLoader(yaml.SafeLoader):
                pass

            def mapping(loader, node):
                loader.flatten_mapping(node)
                return _unique_pairs((loader.construct_object(key, deep=True),
                                      loader.construct_object(value, deep=True))
                                     for key, value in node.value)

            UniqueSafeLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
            try:
                data = yaml.load(content, Loader=UniqueSafeLoader)
            except yaml.YAMLError as error:
                raise ValueError(str(error)) from error
        # Keep both formats restricted to the same finite JSON-compatible values.
        # This also rejects YAML date/binary/set values and recursive aliases.
        json.dumps(data, allow_nan=False)
        return data
    except (ValueError, TypeError, RecursionError) as error:
        raise ValueError(f"{path}: {error}") from error


def scenario_document(path, architecture=None, model=None):
    data = load_document(path)
    if not isinstance(data, dict):
        raise ValueError("scenario must be a mapping")
    if model:
        preset = load_document(model)
        if (not isinstance(preset, dict) or not isinstance(preset.get("model"), dict)
                or set(preset) - {"model", "model_metadata"}):
            raise ValueError("model preset must contain model and optional model_metadata sections only")
        data["model"] = preset["model"]
        data.pop("model_metadata", None)
        if "model_metadata" in preset:
            data["model_metadata"] = preset["model_metadata"]
    if architecture:
        preset = load_document(architecture)
        if (not isinstance(preset, dict) or "hardware" not in preset
                or set(preset) - {"hardware", "parallel"}):
            raise ValueError("architecture must contain hardware and optional parallel sections only")
        for section, values in preset.items():
            if not isinstance(values, dict):
                raise ValueError(f"architecture.{section} must be a mapping")
            # Replace complete sections to prevent stale GPU/NPU fields or
            # device placements leaking across architecture changes.
            data[section] = values
    return data
