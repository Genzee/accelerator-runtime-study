# Progress

## Current Phase
Phase 0 — Tensor / Operator / Graph

## Current Task
- [ ] Task 3: Transformer block을 operator graph로 그리기 (Phase 1 진입 전, Phase 0 마무리 점검 포함)

## Completed
- [x] 2026-09-24 — repo 골격, `.venv` (NumPy 2.5.3 / Accelerate BLAS) 구성
- [x] 2026-09-24 — Task 1: Tensor shape / dtype / byte 계산 (+ stride/contiguous 실험)
- [x] 2026-09-24 — Task 2: MatMul shape와 FLOP 계산 (+ 실측 latency, elementwise Add 비교, 작은 DAG + topo sort)
- Phase 0 완료 기준
  - [x] `[B,S,H]` tensor가 몇 byte인지 바로 계산할 수 있다
  - [x] MatMul 입력 shape를 보고 output shape를 설명할 수 있다
  - [x] graph edge가 실제로는 tensor dependency라는 것을 설명할 수 있다

## Experiments
| Date | Experiment | Result | Path |
|---|---|---|---|
| 2026-09-24 | NumPy MatMul shape/dtype sweep (M4 CPU) | fp32 큰 MatMul ~1.8 TFLOP/s에서 plateau. M=1(decode-like)은 34 GFLOP/s, 유효 69 GB/s → memory-bound. fp16은 BLAS 경로 없어 ~1,000배 느림 | `experiments/01-matmul/` |
| 2026-09-24 | elementwise Add (fp32, 48/96 MiB) | 유효 대역폭 93–108 GB/s, AI=0.083 | `experiments/01-matmul/` |
| 2026-09-24 | 5-node DAG + topo sort | 뒤섞은 선언을 의존성만으로 복원. activation 14 KiB vs weight 129 KiB | `experiments/01-matmul/` |

## What I Understand
- tensor = (data ptr, dtype, shape, strides, device). transpose는 stride만 바꾸고 복사 안 함; contiguous 강제 시 copy 발생.
- MatMul `[M,K]·[K,N]→[M,N]`, FLOPs = `2MNK`. Transformer의 `[B,S,H]·[H,H']`는 `M=B·S`인 2D MatMul.
- graph edge = tensor dependency. device를 바꾸면 edge 하나가 memcpy/DMA 하나가 된다.
- FLOP이 같아도 latency는 shape·dtype·메모리 접근 패턴에 따라 수백 배 다를 수 있다.

## What Is Still Unclear
- M4 CPU에서 fp32 1.8 TFLOP/s가 어디서 나오는가 (AMX/SME 추측) → QUESTIONS.md
- fp64가 fp32의 2배가 아니라 4배 느린 이유 → QUESTIONS.md

## Next Task
- Task 3: 작은 Transformer block(PyTorch)을 만들고 operator 단위 graph와 각 edge tensor shape/bytes를 그린다 (`notes/02-transformer-execution.md`, `experiments/02-transformer-block/`). PyTorch 설치 필요.
