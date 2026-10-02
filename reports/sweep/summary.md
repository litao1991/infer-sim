# Simulation sweep

Completed: 8; failed: 0.

| Run | Changes | TTFT (µs) | TPOT (µs) | Bottleneck |
| --- | --- | ---: | ---: | --- |
| [0](run-000.json) | `{"model.top_k": 1, "model.dtype_bytes": 1, "hardware.collective": "ring"}` | 3.930 | 1.162 | memory |
| [1](run-001.json) | `{"model.top_k": 1, "model.dtype_bytes": 1, "hardware.collective": "hierarchical"}` | 3.930 | 1.162 | memory |
| [2](run-002.json) | `{"model.top_k": 1, "model.dtype_bytes": 2, "hardware.collective": "ring"}` | 7.505 | 2.140 | memory |
| [3](run-003.json) | `{"model.top_k": 1, "model.dtype_bytes": 2, "hardware.collective": "hierarchical"}` | 7.505 | 2.140 | memory |
| [4](run-004.json) | `{"model.top_k": 2, "model.dtype_bytes": 1, "hardware.collective": "ring"}` | 4.070 | 1.274 | memory |
| [5](run-005.json) | `{"model.top_k": 2, "model.dtype_bytes": 1, "hardware.collective": "hierarchical"}` | 4.070 | 1.274 | memory |
| [6](run-006.json) | `{"model.top_k": 2, "model.dtype_bytes": 2, "hardware.collective": "ring"}` | 7.881 | 2.387 | memory |
| [7](run-007.json) | `{"model.top_k": 2, "model.dtype_bytes": 2, "hardware.collective": "hierarchical"}` | 7.881 | 2.387 | memory |

Values are simulator predictions. Measurement provenance and calibration errors are in each JSON report.
