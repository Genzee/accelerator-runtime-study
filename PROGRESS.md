# Progress

## Current Phase
Phase 2 — CPU / GPU / NPU 실행구조

## Current Task
- [ ] 작업(연산)별 IO 요구 정리 → 배치 설계 (다음 세션 시작점)

## Completed
- [x] 2026-09-24 — repo 골격, `.venv` (NumPy 2.5.3 / Accelerate BLAS) 구성
- [x] 2026-09-24 — Task 1: Tensor shape / dtype / byte 계산 (+ stride/contiguous 실험)
- [x] 2026-09-24 — Task 2: MatMul shape와 FLOP 계산 (+ 실측 latency, elementwise Add 비교, 작은 DAG + topo sort)
- Phase 0 완료 기준
  - [x] `[B,S,H]` tensor가 몇 byte인지 바로 계산할 수 있다
  - [x] MatMul 입력 shape를 보고 output shape를 설명할 수 있다
  - [x] graph edge가 실제로는 tensor dependency라는 것을 설명할 수 있다

- [x] 2026-09-24 — LLM 동작 개요 정리 (루프, shape 흐름, 벡터 의미, block 구조, prefill/decode/KV cache) → notes/02
- [x] 2026-09-24 — 워밍업: attention 손계산 (토큰 3개 × 2차원)
- [x] 2026-09-24 — Task 3·4: GPT-2 small NumPy 구현 (shape trace, 생성, KV cache on/off, prefill 길이 sweep, op별 시간 비중)
- Phase 1 완료 기준
  - [x] Transformer block을 operator 단위로 그릴 수 있다
  - [x] "모델을 쪼갠다"가 layer/subgraph/tensor 중 무엇인지 구분할 수 있다
  - [x] prefill과 decode의 실행 특성이 왜 다른지 설명할 수 있다

## Experiments
| Date | Experiment | Result | Path |
|---|---|---|---|
| 2026-09-24 | NumPy MatMul shape/dtype sweep (M4 CPU) | fp32 큰 MatMul ~1.8 TFLOP/s에서 plateau. M=1(decode-like)은 34 GFLOP/s, 유효 69 GB/s → memory-bound. fp16은 BLAS 경로 없어 ~1,000배 느림 | `experiments/01-matmul/` |
| 2026-09-24 | elementwise Add (fp32, 48/96 MiB) | 유효 대역폭 93–108 GB/s, AI=0.083 | `experiments/01-matmul/` |
| 2026-09-24 | attention 손계산 (3토큰×2차원) | Wq만 바꿔도 '먹었다'의 참고 비율이 밥 0.248→0.496, 나 0.248→0.007로 바뀜. score 표는 S² (8K 토큰 = layer당 4 GiB) | `experiments/02-transformer-block/` |
| 2026-09-24 | 메모리 대역폭 경합 (프로세스 1–8개) | 합계 ~90 GB/s에서 고정, 코어 1개가 거의 포화 | `experiments/07-bandwidth-contention/` |
| 2026-09-24 | GPT-2 파일 header 조회 | tensor 160개, block당 31 MiB, 파일은 이름→숫자 사전 | `experiments/02-transformer-block/inspect_model_file.py` |
| 2026-09-24 | GPT-2 small NumPy 실행 | decode 8.44 ms/토큰(119 tok/s), 대역폭 하한 5.5 ms의 62%. prefill 토큰당 7.85→0.45 ms(1→256토큰). MatMul 85%, lm_head 26%. misaligned 12배·float64 승격 15배 느려짐 | `experiments/02-transformer-block/gpt2_numpy.py` |
| 2026-09-24 | 5-node DAG + topo sort | 뒤섞은 선언을 의존성만으로 복원. activation 14 KiB vs weight 129 KiB | `experiments/01-matmul/` |

## What I Understand
- tensor = (data ptr, dtype, shape, strides, device). transpose는 stride만 바꾸고 복사 안 함; contiguous 강제 시 copy 발생.
- MatMul `[M,K]·[K,N]→[M,N]`, FLOPs = `2MNK`. Transformer의 `[B,S,H]·[H,H']`는 `M=B·S`인 2D MatMul.
- graph edge = tensor dependency. device를 바꾸면 edge 하나가 memcpy/DMA 하나가 된다.
- 벡터 = 토큰별 의미 좌표(상태). block마다 attention으로 문맥을 흡수해 갱신된다.
- attention = Q·Kᵀ로 관련도 표 → softmax 비율 → V 가중합. weight가 '누구를 참고할지'를 결정.
- causal mask 때문에 앞 토큰의 K/V는 불변 → KV cache가 성립.
- LLM 추론 흐름 전체(토크나이징→임베딩→[attention+FFN]×N→출구→반복)를 개념으로 설명 가능. 요약은 notes/02 §10.
- 메모리 계층(레지스터→SRAM→L2→HBM/VRAM→PCIe→DRAM→SSD)과 층마다 파이프 한계. 데이터 3종(weight/activation/KV cache)의 배치 원칙. notes/03.
- 모델 = 코드(아키텍처) + {이름: 숫자} weight 사전. block들은 같은 코드에 다른 숫자.
- prefill은 weight를 여러 토큰이 재사용해서 토큰당 싸고, decode는 토큰마다 weight 전체를 읽어서 비싸다 (실측 17배).
- layout(정렬)·dtype 하나로 같은 계산이 10배 이상 느려질 수 있다.
- FLOP이 같아도 latency는 shape·dtype·메모리 접근 패턴에 따라 수백 배 다를 수 있다.

## What Is Still Unclear
- M4 CPU에서 fp32 1.8 TFLOP/s가 어디서 나오는가 (AMX/SME 추측) → QUESTIONS.md
- fp64가 fp32의 2배가 아니라 4배 느린 이유 → QUESTIONS.md

## Next Task
- **다음 세션 (사용자 제안)**: "각 작업이 어떤 IO가 필요하니 어떻게 디자인해야 하나"
  1. 작업 목록: prefill, decode, attention(KV cache 읽기), FFN, lm_head, KV cache 쓰기, 칩 간 activation 전달, 도구 대기
  2. 각 작업마다: 읽는 데이터(weight/activation/KV) · bytes · 어느 메모리 층 · 어느 파이프를 지나나 · compute-bound/memory-bound
  3. 그 표를 바탕으로 배치 설계 원칙 도출 (무엇을 어디에 상주시키고 어디를 자를지)
  4. 이어서 Roofline(Task 5–6)으로 M4에서 수치 확인
  5. DECISIONS.md의 "하드웨어 추상화 후보(Compute/Memory/Link)"를 워크로드 IO 표와 맞춰 다듬기
