# Benchmarking Methodology

이 문서는 EdgeBench의 벤치마크 측정 방법론을 정의합니다.  
본 문서는 재현 가능성(Reproducibility)을 최우선 목표로 작성되었습니다.

---

## 1. 목표

EdgeBench는 다음 두 지표를 통합 분석합니다:

1. **정적 지표 (Static Metrics)**
   - Parameter count
   - FLOPs (Conv / Linear 중심 근사)

2. **동적 지표 (Runtime Metrics)**
   - Mean latency
   - P50 / P90 / P99 latency
   - Std / Min / Max

정적 연산량과 실제 지연시간 간의 스케일링 관계를 분석하는 것이 목적입니다.

---

## 2. 측정 환경

### Hardware / System

- OS: Linux (Codespaces x86_64)
- Python: 3.12.x
- Machine: x86_64
- Engine: ONNX Runtime CPU

### Runtime Configuration

- intra_op_num_threads: 1
- inter_op_num_threads: 1
- warmup: 10
- runs: 300 (ToyNet 기준)
- batch size: 1

멀티스레드 환경에서는 latency 분산이 커질 수 있으므로,  
기본 벤치마크는 single-thread 환경에서 수행합니다.

---

## 3. 입력 텐서 생성 규칙

ONNX 모델 입력이 dynamic shape인 경우:

- Batch dimension: `--batch` 옵션으로 override
- Height / Width: `--height`, `--width` 옵션으로 override
- dtype: FLOAT32
- 데이터: numpy random normal

예시:

```bash
edgebench profile model.onnx \
  --height 640 --width 640 \
  --batch 1 \
  --runs 300 \
  --intra-threads 1 \
  --inter-threads 1
```