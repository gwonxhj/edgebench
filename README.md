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

## 📈 Benchmarks (예시 결과)

> 환경: GitHub Codespaces (Linux x86_64), ONNX Runtime CPU  
> 설정: warmup=10, intra_threads=1, inter_threads=1

### YOLOv8n (640×640, batch=1)

- Parameters: 3,193,923
- Latency (ms):
  - mean: 120.22
  - p50: 115.67
  - p90: 125.57
  - p99: 166.38
  - std: 11.84
  - min/max: 113.42 / 172.68

리포트 JSON: `reports/yolov8n__onnxruntime_cpu__b1__r50__*.json`

### ToyNet (dynamic H/W, batch=1)

> 환경: GitHub Codespaces (Linux x86_64), ONNX Runtime CPU  
> 설정: warmup=10, runs=300, intra_threads=1, inter_threads=1

| Input | Mean (ms) | P50 (ms) | P90 (ms) | P99 (ms) |
|---|---:|---:|---:|---:|
| 224×224 | 0.538 | 0.501 | 0.723 | 0.764 |
| 320×320 | 1.131 | 1.014 | 1.424 | 1.455 |
| 640×640 | 4.589 | 4.083 | 5.834 | 9.302 |

> 입력 해상도 증가에 따라 latency가 면적에 비례해 증가하는 경향을 확인할 수 있습니다.

### 실행 명령

```bash
edgebench profile models/yolov8n.onnx \
  --warmup 10 --runs 50 --batch 1 \
  --intra-threads 1 --inter-threads 1
```

리포트 JSON: `reports/yolov8n__onnxruntime_cpu__b1__r50__*.json`

벤치마크 측정 방법론: `docs/benchmarking.md`

## 📜 License

MIT License

---


