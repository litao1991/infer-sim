#!/usr/bin/env python3
"""Compare KV eviction policies and block sizes on identical Bailian windows."""

import argparse
import csv
import json
from pathlib import Path

if __package__:
    from .bailian_eval import FILES, ROOT, run, save
else:
    from bailian_eval import FILES, ROOT, run, save

POLICIES = ("lru", "hot", "write_aware")


def sweep(data_dir, output, binary, blocks=(4, 16, 64), sources=FILES,
          sample_size=2, divisor=32, prompt_cap=64, output_cap=16,
          speed=1000, cache_bytes=8192, template=None, architecture=None):
    output = Path(output)
    if not blocks or len(set(blocks)) != len(blocks) or any(b < 1 for b in blocks):
        raise ValueError("blocks must contain unique positive token counts")
    if len(blocks) * len(POLICIES) * len(sources) * 2 > 256:
        raise ValueError("local KV sweep limit is 256 cases")
    rows = []
    for block in blocks:
        for policy in POLICIES:
            folder = f"block-{block}-{policy}"
            cases = run(data_dir, output / folder, binary, sample_size, divisor,
                        prompt_cap, output_cap, (speed,), sources, template,
                        cache_bytes, architecture, cache_policy=policy,
                        block_tokens=block, placements=("hbm-hbf",))
            for row in cases:
                row.update(kv_block_tokens=block, kv_cache_policy=policy)
                if row['status'] == 'ok':
                    row['report'] = f"{folder}/{row['report']}"
            rows.extend(cases)
            print(json.dumps({"block_tokens": block, "policy": policy,
                              "succeeded": sum(r['status'] == 'ok' for r in cases)}), flush=True)
    aggregates = []
    for block in blocks:
        for policy in POLICIES:
            cases = [r for r in rows if r['kv_block_tokens'] == block
                     and r['kv_cache_policy'] == policy and r['status'] == 'ok']
            physical = sum(r['kv_hbf_writeback_bytes'] for r in cases)
            dirty = sum(r['kv_dirty_logical_bytes_written'] for r in cases)
            aggregates.append({
                "kv_block_tokens": block, "kv_cache_policy": policy, "cases": len(cases),
                "hbf_write_bytes": physical, "new_dirty_bytes_persisted": dirty,
                "write_amplification": physical / dirty if dirty else None,
                "write_dlu_ops": sum(r['kv_write_dlu_ops'] for r in cases),
                "partial_writebacks": sum(r['kv_partial_writebacks'] for r in cases),
                "sealed_writebacks": sum(r['kv_sealed_writebacks'] for r in cases),
                "rewrites": sum(r['kv_rewrite_writebacks'] for r in cases),
                "hbf_read_bytes": sum(r['kv_hbf_fill_bytes'] for r in cases),
                "mean_ttft_ms": sum(r['mean_ttft_ms'] for r in cases) / len(cases) if cases else None,
            })
    for row in aggregates:
        baseline = next(r for r in aggregates if r['kv_block_tokens'] == row['kv_block_tokens']
                        and r['kv_cache_policy'] == 'lru')
        row['write_bytes_ratio_to_lru'] = (row['hbf_write_bytes'] / baseline['hbf_write_bytes']
                                          if baseline['hbf_write_bytes'] else None)
    summary = {"evidence": "scaled_small_model_uncalibrated",
               "method": {"sample_size": sample_size, "token_divisor": divisor,
                          "prompt_cap": prompt_cap, "output_cap": output_cap,
                          "arrival_speed": speed, "cache_bytes": cache_bytes,
                          "placements": ["hbm-hbf"], "block_tokens": list(blocks),
                          "policies": list(POLICIES), "sources": list(sources)},
               "succeeded": sum(r['status'] == 'ok' for r in rows),
               "failed": sum(r['status'] != 'ok' for r in rows),
               "aggregates": aggregates, "runs": rows}
    save(output / 'summary.json', summary)
    with (output / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, list(aggregates[0]))
        writer.writeheader()
        writer.writerows(aggregates)
    lines = ["# Bailian KV write policy sweep", "",
             f"Succeeded: {summary['succeeded']}; failed: {summary['failed']}.", "",
             "Scaled small-model stress test; all policies use the same source windows,",
             "token lengths, arrival times, modeled hardware and HBM cache capacity.",
             "Write amplification = physical HBF write bytes / newly persisted dirty bytes.",
             "No packing buffer, background writeback, session persistence or NAND GC is modeled.", "",
             "| Block tokens | Policy | HBF write (KiB) | 4 KiB write submissions | Partial / sealed / rewrites | Write amplification | Writes vs LRU | HBF read (KiB) |",
             "| ---: | --- | ---: | ---: | --- | ---: | ---: | ---: |"]
    for row in aggregates:
        wa = f"{row['write_amplification']:.2f}×" if row['write_amplification'] is not None else '—'
        ratio = f"{row['write_bytes_ratio_to_lru']:.2f}×" if row['write_bytes_ratio_to_lru'] is not None else '—'
        lines.append(f"| {row['kv_block_tokens']} | {row['kv_cache_policy']} | "
                     f"{row['hbf_write_bytes']/1024:.1f} | {row['write_dlu_ops']} | "
                     f"{row['partial_writebacks']} / {row['sealed_writebacks']} / {row['rewrites']} | "
                     f"{wa} | {ratio} | {row['hbf_read_bytes']/1024:.1f} |")
    lines += ["", "A partial victim is still evictable when all slots are under pressure.",
              "Larger blocks may also increase reload traffic and change page admission;",
              "a lower write count alone does not establish a latency improvement.",
              "Detailed per-case reports and source fingerprints are in the subdirectories."]
    (output / 'summary.md').write_text('\n'.join(lines) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default=str(ROOT / 'data/bailian'))
    parser.add_argument('--output', default=str(ROOT / 'reports/bailian/kv-policies'))
    parser.add_argument('--binary', default=str(ROOT / 'build/infer_sim'))
    parser.add_argument('--template')
    parser.add_argument('--architecture')
    parser.add_argument('--blocks', nargs='+', type=int, default=[4, 16, 64])
    parser.add_argument('--sources', nargs='+', choices=FILES, default=list(FILES))
    parser.add_argument('--sample-size', type=int, default=2)
    parser.add_argument('--token-divisor', type=int, default=32)
    parser.add_argument('--prompt-cap', type=int, default=64)
    parser.add_argument('--output-cap', type=int, default=16)
    parser.add_argument('--arrival-speed', type=int, default=1000)
    parser.add_argument('--kv-cache-bytes', type=int, default=8192)
    args = parser.parse_args()
    try:
        summary = sweep(args.data_dir, args.output, args.binary, tuple(args.blocks),
                        tuple(args.sources), args.sample_size, args.token_divisor,
                        args.prompt_cap, args.output_cap, args.arrival_speed,
                        args.kv_cache_bytes, args.template, args.architecture)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, str(error) + '\n')
    print(json.dumps({"succeeded": summary['succeeded'], "failed": summary['failed'],
                      "summary": str(Path(args.output) / 'summary.md')}))
    if summary['failed']:
        parser.exit(1)


if __name__ == '__main__':
    main()
