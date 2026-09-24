# Decisions

## 2026-09-24 — 프로젝트 전용 venv + NumPy(Accelerate) 사용

### Context
시스템 Python(Homebrew 3.13.5)에 NumPy 없음. 이후 PyTorch, onnx, onnxruntime이 필요.

### Options
1. 시스템 Python에 전역 설치
2. repo 내 `.venv` (python -m venv)
3. conda/uv

### Decision
2 — `.venv` + `requirements.txt`.

### Reason
추가 도구 설치 없이 바로 가능, 실험 재현성을 repo 단위로 고정. pip wheel NumPy는 macOS에서 Accelerate BLAS를 사용 (측정 해석 시 기억해 둘 것).

### Revisit When
onnxruntime/torch 버전 충돌이 생기거나 여러 Python 버전 비교가 필요할 때.

## 2026-09-24 — 벤치마크 통계는 p50 기준, warmup 분리

### Context
단일 실행 시간은 cache/thermal/스케줄러 영향으로 흔들림.

### Decision
warmup 3회 제외, 최소 10회·0.3초 반복, p50을 대표값으로 하고 p95/min을 함께 저장.

### Reason
Phase 5 cost model의 profile 포맷(`p50_us`, `p95_us`)과 맞춘다.

### Revisit When
tail latency(p99)나 queue delay를 봐야 하는 Phase 7 이후.

## 2026-09-24 — GPT-2 실습은 transformers 없이 NumPy + tiktoken + 자체 safetensors 파서

### Context
LLM 내부 계산을 직접 보고 싶음. transformers/PyTorch는 내부를 가려서 학습 목적에 불리하고 설치도 무거움.

### Options
1. transformers로 로드 후 hook
2. PyTorch로 직접 구현
3. NumPy로 직접 구현, weight는 safetensors를 직접 파싱, tokenizer만 tiktoken

### Decision
3.

### Reason
모든 연산이 눈에 보이고, 01 실험(NumPy MatMul)과 같은 환경이라 수치 비교가 가능. safetensors 형식이 단순해서 파서가 20줄.

### Revisit When
GPU(Metal/MPS) 실험, ONNX export(Phase 3)가 필요해질 때 → PyTorch 도입.
