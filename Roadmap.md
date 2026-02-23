# EdgeBench Roadmap

## 🚀 Phase 1 – MVP (macOS, CPU 기반)

목표:  
CLI 환경에서 ONNX 모델의 구조 분석과 CPU 기반 추론 성능을 정량적으로 측정할 수 있는 최소 기능 구현

### 구현 항목

- ONNX 모델 구조 파싱
- 파라미터 수 계산
- 모델 파일 크기 분석
- FLOPs 추정 (이론적 연산량 계산)
- CPU 기반 추론 latency 벤치마크
- JSON 형식의 성능 리포트 출력

---

## 🧠 Phase 2 – 엔진 추상화 계층 설계

목표:  
추론 엔진을 교체 가능하도록 구조를 일반화

### 구현 항목

- Engine Base Interface 정의
- ONNX Runtime CPU 백엔드 구현
- TensorRT 백엔드 추가 (선택적)
- RKNN 백엔드 지원

---

## 📊 Phase 3 – 성능 분석 확장

목표:  
단일 실행 결과가 아닌 통계 기반 성능 분석 제공

### 구현 항목

- Warmup 반복 횟수 제어
- Batch size별 벤치마크 기능
- 다중 실행 평균 및 표준편차 계산
- 메모리 사용량 측정

---

## 🔥 Phase 4 – 고급 엣지 디바이스 지원

목표:  
실제 엣지 환경에서의 성능 비교 및 최적화 분석

### 구현 항목

- Jetson GPU 가속 지원
- RK3588 NPU 벤치마크 기능
- 양자화 모델(INT8 등) 성능 비교
- FP32 vs INT8 성능 차이 분석

---

## 🧩 Phase 5 – 개발자 경험(Developer Experience) 강화

목표:  
도구의 사용성 향상 및 브랜딩 확장

### 구현 항목

- Rich 기반 CLI UI 개선
- HTML 성능 리포트 생성 기능
- Web Dashboard 모드
- CI 기반 자동 벤치마크 시스템
