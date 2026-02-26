# EdgeBench

> Edge AI Inference Profiling Framework  
> ONNX 모델의 구조 분석과 실제 추론 성능을 정량화하는 개발자용 벤치마크 도구

---

## 📌 프로젝트 개요

EdgeBench는 엣지 환경에서 AI 모델을 배포하기 전에  
모델의 구조적 특성과 실제 추론 성능을 분석하기 위한 CLI 기반 도구입니다.

정확도(Accuracy)만으로는 모델의 배포 가능성을 판단할 수 없습니다.

EdgeBench는 다음을 제공합니다:

- 모델 파라미터 수 계산
- 모델 파일 크기 확인
- FLOPs 추정
- CPU 기반 실제 추론 latency 측정
- JSON 형태의 정량 리포트 출력

---

## 🎯 왜 필요한가?

Jetson, RK3588, CPU-only 환경과 같은 엣지 디바이스에서는  
모델의 정확도보다 다음 요소가 더 중요합니다:

- 실시간 처리 가능 여부
- 연산량
- 메모리 요구량
- 실제 추론 지연 시간

EdgeBench는 이러한 정보를 하나의 CLI 인터페이스에서 통합 제공합니다.

---

## 🧠 아키텍처

CLI 기반 구조:

- Analyzer: 정적 모델 분석
- Profiler: 동적 추론 성능 측정
- Engine Interface: 추론 엔진 추상화 계층

현재 지원:
- ONNX Runtime CPU

향후 확장 예정:
- TensorRT
- RKNN
- Jetson CUDA Backend
- C++ 추론 엔진

---

## 🛠 예정 기능 (MVP)

- ONNX 모델 로드
- 파라미터 수 계산
- FLOPs 추정
- CPU latency 벤치마크
- JSON 리포트 출력

---

## 🗺 개발 로드맵

자세한 계획은 Roadmap.md 참고

---

## 📈 Benchmarks

EdgeBench는 정적 지표(FLOPs, Parameters)와 동적 지표(Latency)를 하나의 리포트 스키마로 통합 제공합니다.

> 환경: GitHub Codespaces (Linux x86_64), ONNX Runtime CPU  
> 설정: warmup=10, intra_threads=1, inter_threads=1

---

### 🔄 Auto-Generated Benchmark Results
> 아래 표는 'make demo' 또는 CI 실행 시 자동 갱신됩니다.

<!-- EDGE_BENCH:START -->

## Latest (deduplicated)

| Model | Engine | Device | Batch | Input(HxW) | FLOPs | Mean (ms) | P99 (ms) | Timestamp (UTC) |
|---|---|---:|---:|---:|---:|---:|---:|---|
| toy.onnx | onnxruntime | cpu | 1 | 224x224 | 126,444,160 | 0.546 | 1.027 | 2026-02-25T09:22:26Z |
| toy.onnx | onnxruntime | cpu | 1 | 320x320 | 258,048,640 | 1.073 | 1.470 | 2026-02-25T09:22:34Z |
| toy.onnx | onnxruntime | cpu | 1 | 640x640 | 1,032,192,640 | 4.424 | 6.771 | 2026-02-25T09:22:41Z |
| toy224.onnx | onnxruntime | cpu | 1 | 224x224 | 126,444,160 | 0.518 | 0.740 | 2026-02-26T11:25:05Z |
| toy320.onnx | onnxruntime | cpu | 1 | 320x320 | 258,048,640 | 1.062 | 1.447 | 2026-02-26T11:25:06Z |
| toy640.onnx | onnxruntime | cpu | 1 | 640x640 | 1,032,192,640 | 4.477 | 13.694 | 2026-02-26T11:25:09Z |

## Recent stats (last 5 runs per key)

| Model | Engine | Device | Batch | Input(HxW) | N | Mean(avg) | P99(avg) | P99(max) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| toy.onnx | onnxruntime | cpu | 1 | 224x224 | 1 | 0.546 | 1.027 | 1.027 |
| toy.onnx | onnxruntime | cpu | 1 | 320x320 | 1 | 1.073 | 1.470 | 1.470 |
| toy.onnx | onnxruntime | cpu | 1 | 640x640 | 1 | 4.424 | 6.771 | 6.771 |
| toy224.onnx | onnxruntime | cpu | 1 | 224x224 | 5 | 0.563 | 1.352 | 3.886 |
| toy320.onnx | onnxruntime | cpu | 1 | 320x320 | 5 | 1.109 | 1.840 | 3.470 |
| toy640.onnx | onnxruntime | cpu | 1 | 640x640 | 5 | 4.887 | 12.695 | 17.021 |

> Full history: `BENCHMARKS.md`

<!-- EDGE_BENCH:END -->

> 전체 히스토리(raw)는 BENCHMARKS.md 참고

---

## 📜 License

MIT License

---


