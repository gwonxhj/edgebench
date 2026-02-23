# Benchmarking Methodology

EdgeBench는 ONNX Runtime 기반 CPU 벤치마크를 다음 원칙으로 수행합니다.

## Measurement

- Timer: `time.perf_counter()`
- Warmup: 기본 10회 (캐시/그래프 초기화 영향 감소)
- Runs: 기본 100회 (통계 안정화)
- Metrics:
  - mean, std
  - p50, p90, p99
  - min, max

## Reproducibility

벤치마크 분산을 줄이기 위해 ONNX Runtime 스레드 수를 고정할 수 있습니다.

- `--intra-threads` → `intra_op_num_threads`
- `--inter-threads` → `inter_op_num_threads`

예:
```bash
edgebench profile model.onnx --intra-threads 1 --inter-threads 1
```