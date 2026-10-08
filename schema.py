"""Versioned, dependency-free scenario validation shared by runs and sweeps."""
import copy
import math
from storage import weights_on_hbf, layout

VERSION = "1.0"
FIELDS = {
    "model": set("layers hidden heads kv_heads head_dim ffn dtype_bytes kind experts top_k shared_experts routing_seed expert_weights".split()),
    "hardware": set("devices hbm_profile peak_tflops compute_efficiency hbm_capacity_bytes hbm_bandwidth_GBps kernel_launch_ps nvlink_gbps nvlink_latency_ps kind hardware_id fabric_topology collective hbm_queue_capacity gemm_efficiency attention_efficiency grouped_gemm_efficiency gemm_ps_per_flop attention_ps_per_flop grouped_gemm_ps_per_flop gemm_launch_ps attention_launch_ps grouped_gemm_launch_ps device_nodes node_racks nic_gbps rack_gbps cross_rack_gbps nic_latency_ps rack_latency_ps cross_rack_latency_ps sram_bytes dma_tile_bytes dma_latency_ps dma_GBps".split()),
    "parallel": set("tp pp dp ep expert_placement placement".split()),
    "serving": set("max_batch_tokens max_seqs prefill_chunk_tokens kv_block_tokens".split()),
    "storage": set("hbf prefetch cache_bytes_per_device slot_bytes writeback_layer weights kv weight_hbm_layers weight_hbm_experts kv_hbm_layers kv_cache_bytes_per_device kv_cache_policy kv_hot_blocks kv_capacity_bytes_per_device hbf_capacity_bytes".split())}


def number(name, value, minimum=0, integer=False, strict=True):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or (integer and not isinstance(value, int))
            or (value <= minimum if strict else value < minimum)):
        raise ValueError(f"{name} has an invalid numeric value")
    if value > 2 ** 53:
        raise ValueError(f"{name} exceeds exact supported numeric range")
    return value


def validate(data):
    data = copy.deepcopy(data)
    if data.get("config_version", VERSION) != VERSION:
        raise ValueError("unsupported config_version")
    data["config_version"] = VERSION
    if set(data) - set(FIELDS) - {"config_version", "requests", "model_metadata"}:
        raise ValueError("unknown scenario section")
    if "model_metadata" in data:
        if not isinstance(data["model_metadata"], dict):
            raise ValueError("model_metadata must be a mapping")
        if not isinstance(data["model_metadata"].get("reference_dimensions", {}), dict):
            raise ValueError("model_metadata.reference_dimensions must be a mapping")
    for section, fields in FIELDS.items():
        if set(data.get(section, {})) - fields:
            raise ValueError("unknown scenario field in " + section)
    m, h, p, s = (data[k] for k in ("model", "hardware", "parallel", "serving"))
    st = data.setdefault("storage", {})
    for section in ("model", "serving", "parallel"):
        for key, value in data[section].items():
            if type(value) is int and value >= 2 ** 32:
                raise ValueError(section + "." + key + " exceeds supported dimension")
    for k in ("layers", "hidden", "heads", "kv_heads", "ffn", "dtype_bytes"):
        number("model." + k, m[k], integer=True)
    if "head_dim" not in m:
        if m["hidden"] % m["heads"]:
            raise ValueError("hidden/heads must divide evenly when head_dim is omitted")
        m["head_dim"] = m["hidden"] // m["heads"]
    number("model.head_dim", m["head_dim"], integer=True)
    if m["heads"] % m["kv_heads"]:
        raise ValueError("heads/kv_heads must divide evenly")
    m.setdefault("kind", "dense")
    if m["kind"] not in ("dense", "moe"):
        raise ValueError("model.kind must be dense or moe")
    for k in ("tp", "pp", "dp"):
        number("parallel." + k, p[k], integer=True)
    p.setdefault("ep", 1)
    number("parallel.ep", p["ep"], integer=True)
    number("hardware.devices", h["devices"], integer=True)
    if h["devices"] > 4096:
        raise ValueError("local scenario device limit is 4096")
    if math.prod(p[k] for k in ("tp", "pp", "dp", "ep")) != h["devices"]:
        raise ValueError("tp * pp * dp * ep must equal devices")
    if m["layers"] % p["pp"]:
        raise ValueError("layers must be divisible by pp")
    if m["hidden"] % p["tp"] or m["ffn"] % p["tp"]:
        raise ValueError("hidden and ffn must be divisible by tp")
    if m["kind"] == "dense" and p["ep"] != 1:
        raise ValueError("ep > 1 requires a moe model")
    if m["kind"] == "moe":
        for k in ("experts", "top_k"):
            number("model." + k, m[k], integer=True)
        if m["experts"] > 4096:
            raise ValueError("local scenario expert limit is 4096")
        m.setdefault("shared_experts", 0)
        m.setdefault("routing_seed", 1)
        number("shared_experts", m["shared_experts"], integer=True, strict=False)
        number("routing_seed", m["routing_seed"], integer=True, strict=False)
        if m["top_k"] > m["experts"] or p["ep"] > m["experts"]:
            raise ValueError("top_k and ep must not exceed experts")
        m.setdefault("expert_weights", [1] * m["experts"])
        if len(m["expert_weights"]) != m["experts"]:
            raise ValueError("expert_weights length must equal experts")
        for v in m["expert_weights"]:
            number("expert_weights", v, strict=False)
        if sum(v > 0 for v in m["expert_weights"]) < m["top_k"]:
            raise ValueError("at least top_k experts need positive routing weights")
        p.setdefault("expert_placement", [e % p["ep"] for e in range(m["experts"])])
        if len(p["expert_placement"]) != m["experts"]:
            raise ValueError("expert_placement length must equal experts")
        for v in p["expert_placement"]:
            number("expert_placement", v, integer=True, strict=False)
            if v >= p["ep"]:
                raise ValueError("expert_placement must name an EP lane")
    p.setdefault("placement", list(range(h["devices"])))
    if (any(type(v) is not int for v in p["placement"])
            or sorted(p["placement"]) != list(range(h["devices"]))):
        raise ValueError("placement must be a permutation of physical device IDs")
    for k in ("peak_tflops", "compute_efficiency", "hbm_bandwidth_GBps", "nvlink_gbps"):
        number("hardware." + k, h[k])
    for k in ("hbm_capacity_bytes", "kernel_launch_ps", "nvlink_latency_ps"):
        number("hardware." + k, h[k], integer=True)
    h.setdefault("kind", "nvidia_gpu")
    h.setdefault("hbm_profile", "hbm3_6400")
    if h["hbm_profile"] not in ("hbm2_2000", "hbm3_6400", "hbm4_8000"):
        raise ValueError("unsupported HBMSim hbm_profile")
    h.setdefault("fabric_topology", "nvswitch")
    h.setdefault("collective", "ring")
    if h["kind"] not in ("nvidia_gpu", "npu"):
        raise ValueError("hardware.kind must be nvidia_gpu or npu")
    if h["fabric_topology"] not in ("nvswitch", "nvlink_mesh"):
        raise ValueError("unknown fabric_topology")
    if h["collective"] not in ("ring", "hierarchical"):
        raise ValueError("collective must be ring or hierarchical")
    for k in ("compute_efficiency", "gemm_efficiency", "attention_efficiency", "grouped_gemm_efficiency"):
        if k in h:
            number(k, h[k])
            if h[k] > 1:
                raise ValueError(k + " must be <= 1")
    for k in ("gemm_ps_per_flop", "attention_ps_per_flop", "grouped_gemm_ps_per_flop"):
        if k in h:
            number(k, h[k])
    for k in ("hbm_queue_capacity", "gemm_launch_ps", "attention_launch_ps", "grouped_gemm_launch_ps"):
        if k in h:
            number(k, h[k], integer=True, strict=False)
    h.setdefault("device_nodes", [0] * h["devices"])
    h.setdefault("node_racks", [0])
    if len(h["device_nodes"]) != h["devices"] or not h["node_racks"]:
        raise ValueError("device_nodes/node_racks dimensions are invalid")
    for v in h["device_nodes"] + h["node_racks"]:
        number("topology ID", v, integer=True, strict=False)
    if max(h["device_nodes"]) >= len(h["node_racks"]):
        raise ValueError("device_nodes references an undefined node")
    if len(set(h["device_nodes"])) > 1:
        for k in ("nic_gbps", "rack_gbps", "cross_rack_gbps"):
            number(k, h[k])
        for k in ("nic_latency_ps", "rack_latency_ps", "cross_rack_latency_ps"):
            number(k, h[k], integer=True)
    if h["kind"] == "npu" or h.get("dma_GBps", 0):
        for k in ("sram_bytes", "dma_tile_bytes", "dma_latency_ps"):
            number(k, h[k], integer=True)
        number("dma_GBps", h["dma_GBps"])
        if h["dma_tile_bytes"] > h["sram_bytes"]:
            raise ValueError("DMA tile must fit in SRAM")
    for k in ("max_batch_tokens", "max_seqs", "prefill_chunk_tokens", "kv_block_tokens"):
        number("serving." + k, s[k], integer=True)
    if s["prefill_chunk_tokens"] > s["max_batch_tokens"]:
        raise ValueError("prefill_chunk_tokens exceeds max_batch_tokens")
    for k in ("hbf", "prefetch"):
        if k in st and type(st[k]) is not bool:
            raise ValueError(k + " must be boolean")
    for key in ("weights", "kv"):
        if key in st and st[key] not in ("hbm", "hbf"):
            raise ValueError("storage." + key + " must be hbm or hbf")
    if "weights" in st and "hbf" in st and (st["weights"] == "hbf") != st["hbf"]:
        raise ValueError("storage.weights conflicts with legacy storage.hbf")
    for key, limit in (("weight_hbm_layers", m["layers"]), ("kv_hbm_layers", m["layers"]),
                       ("weight_hbm_experts", m.get("experts", 0) if m["kind"] == "moe" else 0)):
        if key in st:
            values = st[key]
            if not isinstance(values, list) or any(type(v) is not int or not 0 <= v < limit for v in values) or len(set(values)) != len(values):
                raise ValueError("storage." + key + " must contain unique valid IDs")
            if values and (st.get("kv", "hbm") != "hbf" if key == "kv_hbm_layers" else not weights_on_hbf(st)):
                raise ValueError(key + " overrides require hbf as the default tier")
    if weights_on_hbf(st):
        for k in ("slot_bytes", "cache_bytes_per_device"):
            number(k, st[k], integer=True)
        if st["slot_bytes"] % 4096 or st["cache_bytes_per_device"] % st["slot_bytes"]:
            raise ValueError("HBF slots must be DLU aligned and divide cache capacity")
        if "writeback_layer" in st:
            number("writeback_layer", st["writeback_layer"], integer=True, strict=False)
            if st["writeback_layer"] >= m["layers"]:
                raise ValueError("writeback_layer must name an existing layer")
            if st["writeback_layer"] in st.get("weight_hbm_layers", []):
                raise ValueError("writeback_layer must use HBF backing")
    elif "writeback_layer" in st:
        raise ValueError("writeback_layer requires HBF weights")
    if st.get("kv", "hbm") == "hbf":
        for key in ("kv_cache_bytes_per_device", "kv_capacity_bytes_per_device", "hbf_capacity_bytes"):
            number("storage." + key, st.get(key), integer=True)
        shape = layout(data)
        if 2 * m['kv_heads'] * m['head_dim'] * m['dtype_bytes'] % p['tp']:
            raise ValueError("KV offload requires per-token KV bytes divisible by tp")
        if not shape["kv_hbf_layers_per_device"]:
            raise ValueError("KV HBF placement requires at least one offloaded layer")
        for key in ("kv_cache_bytes_per_device", "kv_capacity_bytes_per_device"):
            if st[key] % shape["kv_page_bytes"]:
                raise ValueError(key + " must be a multiple of the aligned KV page bytes")
        if st["kv_cache_bytes_per_device"] > st["kv_capacity_bytes_per_device"]:
            raise ValueError("KV cache must not exceed logical KV capacity")
        if st["hbf_capacity_bytes"] % 4096:
            raise ValueError("hbf_capacity_bytes must be 4 KiB aligned")
        if st.get("kv_cache_policy", "lru") not in ("lru", "hot", "write_aware"):
            raise ValueError("kv_cache_policy must be lru, hot or write_aware")
        number("kv_hot_blocks", st.get("kv_hot_blocks", 1), integer=True, strict=False)
    elif any(k in st for k in ("kv_cache_bytes_per_device", "kv_capacity_bytes_per_device", "kv_cache_policy", "kv_hot_blocks")):
        raise ValueError("KV cache policy/capacity fields require kv: hbf")
    if "hbf_capacity_bytes" in st:
        number("hbf_capacity_bytes", st["hbf_capacity_bytes"], integer=True)
        if st["hbf_capacity_bytes"] % 4096:
            raise ValueError("hbf_capacity_bytes must be 4 KiB aligned")
    if not data["requests"]:
        raise ValueError("at least one request is required")
    last = 0
    for r in data["requests"]:
        if set(r) != {"arrival_ps", "prompt_tokens", "output_tokens"}:
            raise ValueError("unknown or missing request field")
        number("arrival_ps", r["arrival_ps"], integer=True, strict=False)
        if r["arrival_ps"] < last:
            raise ValueError("arrival times must be sorted")
        last = r["arrival_ps"]
        for k in ("prompt_tokens", "output_tokens"):
            number(k, r[k], integer=True)
            if r[k] >= 2 ** 32:
                raise ValueError(k + " exceeds supported dimension")
    return data
