"""Storage placement and conservative capacity accounting, without tensor allocation."""
import math


def weights_on_hbf(storage):
    return storage.get('weights', 'hbf' if storage.get('hbf', False) else 'hbm') == 'hbf'


def weight_on_hbf(storage, layer, expert=None):
    return (weights_on_hbf(storage) and layer not in storage.get('weight_hbm_layers', [])
            and (expert is None or expert not in storage.get('weight_hbm_experts', [])))


def layout(data):
    m, p, s = data['model'], data['parallel'], data.get('storage', {})
    d = m.get('head_dim', m['hidden'] // m['heads'])
    q, v = m['heads'] * d, m['kv_heads'] * d
    expert = 3 * m['hidden'] * m['ffn']
    moe = m.get('kind') == 'moe'
    base = m['hidden'] * (q + 2 * v) + q * m['hidden']
    base += expert * m.get('shared_experts', 0) + m['hidden'] * m['experts'] if moe else expert
    base_bytes, expert_bytes = (math.ceil(x * m['dtype_bytes'] / p['tp']) for x in (base, expert))
    layers_per_stage = m['layers'] // p['pp']
    resident, backing, hbm_layers, hbf_layers = [], [], [], []
    offload = s.get('kv', 'hbm') == 'hbf'
    max_object = 0
    experts_by_lane = [[] for _ in range(p.get('ep', 1))]
    if moe:
        for e, lane in enumerate(p['expert_placement']):
            experts_by_lane[lane].append(e)
    for stage in range(p['pp']):
        start, stop = stage * layers_per_stage, (stage + 1) * layers_per_stage
        pinned = (sum(start <= layer < stop for layer in s.get('kv_hbm_layers', []))
                  if offload else layers_per_stage)
        hbm_layers.append(pinned)
        hbf_layers.append(layers_per_stage - pinned)
        weight_pinned = (sum(start <= layer < stop for layer in s.get('weight_hbm_layers', []))
                         if weights_on_hbf(s) else layers_per_stage)
        offloaded_layers = layers_per_stage - weight_pinned
        for experts in experts_by_lane:
            pinned_experts = sum(e in s.get('weight_hbm_experts', []) for e in experts)
            hbm = weight_pinned * (base_bytes + len(experts) * expert_bytes)
            hbm += offloaded_layers * pinned_experts * expert_bytes
            hbf = offloaded_layers * (math.ceil(base_bytes / 4096) * 4096 +
                    (len(experts) - pinned_experts) * math.ceil(expert_bytes / 4096) * 4096)
            if offloaded_layers:
                max_object = max(max_object, math.ceil(base_bytes / 4096) * 4096)
                if len(experts) > pinned_experts:
                    max_object = max(max_object, math.ceil(expert_bytes / 4096) * 4096)
            resident.append(hbm)
            backing.append(hbf)
    page_bytes = math.ceil(math.ceil(data['serving']['kv_block_tokens'] * 2 * v * m['dtype_bytes'] / p['tp']) / 4096) * 4096
    return dict(resident_weight_bytes_per_device=max(resident),
                hbf_weight_bytes_total=sum(backing) * p['tp'] * p['dp'],
                max_weight_object_bytes=max_object,
                kv_hbm_layers_per_device=max(hbm_layers),
                kv_hbf_layers_per_device=max(hbf_layers),
                kv_page_bytes=page_bytes,
                kv_hbm_bytes_per_token_device=math.ceil(2 * v * m['dtype_bytes'] * max(hbm_layers) / p['tp']))
