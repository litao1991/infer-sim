import copy
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
from run import load_scenario, simulate, fast_report, validate
from storage import layout


class StorageTests(unittest.TestCase):
    binary = os.environ.get('INFER_SIM_BINARY', str(ROOT / 'build/infer_sim'))

    def config(self, weights='hbm', kv='hbf', example='hbf-small.json'):
        data = load_scenario(ROOT / 'examples' / example)
        data['storage'] = {'weights': weights, 'kv': kv}
        if weights == 'hbf':
            data['storage'].update(cache_bytes_per_device=8192, slot_bytes=8192, prefetch=False)
        if kv == 'hbf':
            data['storage'].update(kv_cache_bytes_per_device=4096, kv_capacity_bytes_per_device=65536,
                                   hbf_capacity_bytes=1048576)
        return data

    def run_case(self, data):
        report = simulate(data, binary=self.binary)
        self.assertEqual(report['compute_flops'], fast_report(validate(data))['estimated_flops'])
        self.assertEqual(report['output_tokens'], sum(r['output_tokens'] for r in data['requests']))
        self.assertLessEqual(report['peak_hbm_bytes_per_device'], data['hardware']['hbm_capacity_bytes'])
        kv = report['storage_metrics']['kv']
        self.assertEqual(kv['offloaded_append_bytes'], kv['dirty_logical_bytes_written'] + kv['discarded_dirty_bytes_at_completion'])
        self.assertEqual(kv['live_bytes_at_end'], 0)
        self.assertEqual(kv['hbf_fill_bytes'], report['storage_metrics']['hbf_log']['kv_read_bytes'])
        self.assertEqual(kv['hbf_writeback_bytes'], report['storage_metrics']['hbf_log']['kv_write_bytes'])
        if data['storage'].get('kv') == 'hbf':
            self.assertLessEqual(kv['hbm_cache_peak_bytes_per_device'], data['storage']['kv_cache_bytes_per_device'])
            self.assertLessEqual(kv['logical_peak_bytes_per_device'], data['storage']['kv_capacity_bytes_per_device'])
            self.assertLessEqual(report['storage_metrics']['hbf_log']['allocated_bytes'], data['storage']['hbf_capacity_bytes'])
        return report

    def test_four_storage_combinations_and_logical_bytes(self):
        for weights in ('hbm', 'hbf'):
            for kv_tier in ('hbm', 'hbf'):
                with self.subTest(weights=weights, kv=kv_tier):
                    report = self.run_case(self.config(weights, kv_tier))
                    kv, w = (report['storage_metrics'][k] for k in ('kv', 'weights'))
                    self.assertEqual(kv['hbm_append_logical_bytes'], 640)
                    self.assertEqual(kv['hbm_read_logical_bytes'], 1984)
                    self.assertEqual(w['hbf_fill_bytes'] > 0, weights == 'hbf')
                    self.assertEqual(kv['reloads'] > 0, kv_tier == 'hbf')

    def test_dirty_versions_reload_and_large_cache(self):
        small = self.config()
        report = self.run_case(small)
        self.assertEqual(report, self.run_case(small))
        stats = report['storage_metrics']['kv']
        self.assertGreater(stats['dirty_evictions'], 0)
        self.assertGreater(stats['reloads'], 0)
        self.assertGreater(stats['hbf_writeback_bytes'], stats['dirty_logical_bytes_written'])
        large = copy.deepcopy(small)
        large['storage']['kv_cache_bytes_per_device'] = 32768
        cached = self.run_case(large)
        self.assertEqual(cached['storage_metrics']['kv']['reloads'], 0)
        self.assertEqual(cached['storage_metrics']['kv']['hbf_writeback_bytes'], 0)
        self.assertLess(cached['makespan_ps'], report['makespan_ps'])
        # Every fill has HBF completion before HBM fill, and request DAG remains causal.
        self.assertEqual(report['critical_path']['duration_ps'], report['makespan_ps'])
        self.assertTrue(any(n['kind'] == 'hbf_kv_fill' for n in report['trace']))

    def test_layer_and_expert_residency(self):
        data = self.config('hbf', 'hbf', 'moe-small.json')
        data['storage'].update(weight_hbm_layers=[0], weight_hbm_experts=[0], kv_hbm_layers=[0])
        report = self.run_case(data)
        expected = layout(validate(data))
        weights, kv = (report['storage_metrics'][k] for k in ('weights', 'kv'))
        self.assertEqual(weights['hbm_resident_bytes_per_device'], 13952)
        self.assertEqual(weights['hbf_backing_bytes'], 28672)
        self.assertEqual(weights['hbf_backing_bytes'], expected['hbf_weight_bytes_total'])
        self.assertGreater(kv['hbm_fixed_layers_peak_bytes_per_device'], 0)
        self.assertEqual(kv['hbm_append_logical_bytes'], 832)
        self.assertEqual(kv['offloaded_append_bytes'], 416)
        fast = fast_report(validate(data))
        self.assertEqual(fast['storage_capacity']['resident_weight_bytes_per_device'], 13952)

    def test_parallel_gpu_npu_and_hbm_backpressure(self):
        for example, tp, pp, dp, ep in (('moe-small.json', 2, 2, 1, 2),
                                       ('npu-moe-small.json', 1, 1, 2, 2)):
            with self.subTest(example=example):
                data = self.config('hbf', 'hbf', example)
                n = tp * pp * dp * ep
                data['hardware'].update(devices=n, device_nodes=[i // 2 for i in range(n)],
                    node_racks=[i // 2 for i in range(n // 2)], nic_gbps=100, rack_gbps=100,
                    cross_rack_gbps=50, nic_latency_ps=10000, rack_latency_ps=10000,
                    cross_rack_latency_ps=20000, hbm_queue_capacity=1)
                data['parallel'] = dict(tp=tp, pp=pp, dp=dp, ep=ep)
                data['requests'] *= 2
                for r in data['requests']: r['arrival_ps'] = 0
                report = self.run_case(data)
                # The NPU DMA path serializes services and may not fill the queue.
                if 'npu' not in example: self.assertGreater(report['hbm_backpressure'], 0)
                self.assertGreater(report['storage_metrics']['hbf_log']['kv_queue_wait_ps_sum'], 0)
                if 'npu' in example: self.assertGreater(report['onchip']['dma_bytes'], 0)
                else: self.assertGreater(report['network_tiers']['cross_rack']['bytes'], 0)

    def test_logical_capacity_admission_and_reclamation(self):
        data = self.config()
        data['hardware']['hbm_capacity_bytes'] = 13440  # weights + cache + scratch exactly fit
        exact = self.run_case(data)
        self.assertEqual(exact['peak_hbm_bytes_per_device'], 13440)
        self.assertTrue(fast_report(validate(data))['hbm_capacity_feasible'])
        data['model']['layers'] = 1
        data['requests'] = [dict(arrival_ps=0, prompt_tokens=5, output_tokens=1) for _ in range(2)]
        data['storage']['kv_capacity_bytes_per_device'] = 8192
        report = self.run_case(data)
        self.assertGreater(report['requests_detail'][1]['queue_wait_ps'], 0)
        self.assertLessEqual(report['storage_metrics']['kv']['logical_peak_bytes_per_device'], 8192)
        data['storage']['kv_capacity_bytes_per_device'] = 4096
        self.assertFalse(fast_report(validate(data))['storage_capacity']['kv_hbf_single_request_feasible'])
        with self.assertRaisesRegex(ValueError, 'KV capacity deadlock'):
            simulate(data, binary=self.binary)

    def test_pp_completion_while_another_request_uses_cache(self):
        data = self.config()
        data['hardware'].update(devices=2, device_nodes=[0, 0])
        data['parallel'] = dict(tp=1, pp=2, dp=1)
        data['serving'].update(max_seqs=1)
        data['requests'] = [dict(arrival_ps=0, prompt_tokens=1, output_tokens=1),
                            dict(arrival_ps=0, prompt_tokens=8, output_tokens=4),
                            dict(arrival_ps=0, prompt_tokens=2, output_tokens=2)]
        report = self.run_case(data)
        self.assertGreater(report['storage_metrics']['kv']['dirty_evictions'], 0)
        self.assertEqual(report['storage_metrics']['kv']['live_bytes_at_end'], 0)

    def test_flash_exhaustion_and_invalid_configuration(self):
        data = self.config()
        data['storage']['hbf_capacity_bytes'] = 4096
        with self.assertRaisesRegex(ValueError, 'append log capacity exhausted'):
            simulate(data, binary=self.binary)
        invalid = [dict(kv_cache_bytes_per_device=1), dict(kv_hbm_layers=[2]),
                   dict(kv_hbm_layers=[0, 0]), dict(kv_cache_policy='random'),
                   dict(weights='other'), dict(hbf=True), dict(hbf_capacity_bytes=123),
                   dict(weight_hbm_experts=[0]), dict(kv_hot_blocks=-1)]
        for overrides in invalid:
            with self.subTest(overrides=overrides):
                bad = self.config()
                bad['storage'].update(overrides)
                with self.assertRaises(ValueError): validate(bad)
        data = self.config()
        data['storage']['hbf_capacity_bytes'] = 16 * 1024 * 1024
        with self.assertRaisesRegex(ValueError, 'local limit'):
            simulate(data, binary=self.binary)

    def test_hot_recent_blocks_policy(self):
        data = self.config()
        data['hardware']['hbm_capacity_bytes'] = 1048576
        data['storage'].update(kv_cache_bytes_per_device=12288, kv_cache_policy='hot', kv_hot_blocks=1)
        data['requests'] = [dict(arrival_ps=0, prompt_tokens=8, output_tokens=4),
                            dict(arrival_ps=0, prompt_tokens=6, output_tokens=3)]
        hot = self.run_case(data)
        self.assertGreater(hot['storage_metrics']['kv']['hot_protections'], 0)
        data['storage']['kv_cache_policy'] = 'lru'
        lru = self.run_case(data)
        self.assertEqual(lru['storage_metrics']['kv']['hot_protections'], 0)
        self.assertEqual(hot['storage_metrics']['kv']['hbm_append_logical_bytes'],
                         lru['storage_metrics']['kv']['hbm_append_logical_bytes'])

    @unittest.skipUnless(importlib.util.find_spec('yaml'), 'optional YAML dependency is not installed')
    def test_yaml_cli_and_fast_capacity(self):
        import yaml
        data = self.config('hbf', 'hbf')
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp) / 'scenario.yaml', Path(temp) / 'report.json'
            source.write_text(yaml.safe_dump(data))
            subprocess.run([sys.executable, str(ROOT / 'run.py'), str(source), '--binary', self.binary,
                            '--output', str(output)], check=True, capture_output=True, text=True)
            report = json.loads(output.read_text())
            self.assertGreater(report['storage_metrics']['kv']['hbf_writeback_bytes'], 0)
            large = copy.deepcopy(data)
            large['model'].update(layers=48, hidden=2048, heads=32, kv_heads=4, head_dim=128, ffn=768)
            # Full-size KV block cache capacity must use its aligned block size.
            large['storage'].update(kv_cache_bytes_per_device=65536, kv_capacity_bytes_per_device=1048576)
            fast = simulate(large, mode='fast')
            self.assertIsNone(fast['critical_path'])
            self.assertIn('not predicted', fast['storage_capacity']['hbf_log_traffic'])


if __name__ == '__main__':
    unittest.main()
