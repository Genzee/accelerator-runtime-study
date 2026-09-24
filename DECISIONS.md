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

## 2026-09-25 — [후보, 미확정] 하드웨어 추상화: Compute / Memory / Link 토폴로지

> 상태: 방향 초안. 다음 세션에서 워크로드 쪽 IO 분석과 함께 다듬는다.

### Context
하드웨어 형태가 계속 바뀐다 (분리형 GPU+HBM, unified SoC, 칩렛, 3D 적층, wafer-scale SRAM). 스케줄러가 특정 칩 구조에 묶이면 안 된다.

### Options
1. 칩 종류별로 전용 로직 (GPU용, NPU용 …)
2. 공통 부품의 연결(토폴로지)로 추상화하고 숫자는 profile로 채움

### Decision (잠정)
2. 어떤 하드웨어든 세 부품의 그래프로 표현:
- **Compute**: 지원 연산/dtype/layout(capability), 연산별 비용(profile)
- **Memory**: 용량, 어떤 Compute에 붙어 있나
- **Link**: Memory↔Memory 파이프. 대역폭, 고정 지연, **공유 여부**(경합)

예:
- H100 서버: [GPU] ─ [HBM 80GB] ═PCIe═ [CPU DRAM] ─ [CPU]
- M4: [CPU][GPU][NPU] ─ 공유 Link ~90 GB/s(실측) ─ [LPDDR 24GB]
- Cerebras류: [Compute 다수] ─ 인접 [SRAM] 분산

스케줄러가 묻는 것:
- 이 연산을 이 Compute가 할 수 있나 / 몇 µs인가
- 이 데이터(weight/activation/KV)는 어느 Memory에 있고, 옮기면 어떤 Link를 몇 bytes 지나나
- 그 Link를 다른 작업과 나눠 쓰나

### Reason
- 새 하드웨어는 부품·연결만 기술하면 스케줄러 코드 재사용 가능.
- 숫자는 **스펙이 아니라 실측** (M4 공칭 120 vs 실측 90 GB/s, 정렬·dtype으로 12~15배 차이 경험).
- 선례: K8s DRA/device plugin(속성), Linux NUMA/hwloc(거리), ONNX Runtime EP(capability), OpenXLA PJRT(플러그인).

### 다음 세션에서 다듬을 것
- 워크로드 쪽 추상화(연산 그래프 + 데이터 3종 + bytes)와의 접점 정의
- Link 공유(경합)를 cost model에 어떻게 넣을지
- 칩 **안**의 계층(SRAM/L2)은 모델링할지, kernel에 맡기고 생략할지
- 기존 표현(hwloc, DRA ResourceSlice 등)을 재사용할 수 있는지 조사

### Revisit When
Phase 5(cost model) 설계 시작 시, 또는 M4 CPU+GPU 경합 측정 결과가 나올 때.
