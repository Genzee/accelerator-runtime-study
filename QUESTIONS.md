# Questions

## Open
- [ ] M4 CPU에서 NumPy fp32 MatMul이 ~1.8 TFLOP/s가 나온다. P-core NEON 이론 peak로 설명 가능한가, 아니면 Accelerate가 AMX/SME 행렬 unit을 쓰는가?
  - 검증 아이디어: 스레드 수(`VECLIB_MAXIMUM_THREADS=1`) 바꿔 scaling 보기, M4 P/E core NEON 이론치 계산, Apple 문서/공개 자료 확인.
  - 왜 중요: "CPU backend" 안에 이미 matrix engine이 있다면 CPU vs NPU 비교 해석이 달라진다.
  - 관련 실험: `experiments/01-matmul/`
- [ ] fp64 MatMul이 fp32 대비 2배가 아니라 ~4배 느린 이유는? (행렬 unit의 fp64 처리량?)
- [ ] elementwise Add 48 MiB(108 GB/s)가 96 MiB(93 GB/s)보다 대역폭이 높은 이유 — SLC 캐시 효과인가? 크기 sweep 필요 (Phase 2).
- [ ] M=1(decode-like) MatMul 유효 대역폭 69 GB/s가 Add(93–108 GB/s)보다 낮은 이유 — 단일 스레드 GEMV 경로인가?

- [ ] M4 unified memory에서 CPU와 GPU가 memory-bound 작업을 동시에 돌리면 대역폭을 나눠 먹는가? (contention 측정)
- [ ] unified memory에서도 CoreML/ANE로 넘길 때 layout 변환 copy가 생기는가? (Phase 4)
- [ ] llama.cpp `--n-gpu-layers`를 0→전체로 바꾸며 CPU/GPU 분할 지점별 속도 측정 (한 모델을 두 칩에 나눠 싣는 가장 싼 실험 후보)

## Answered
- [x] transpose는 데이터를 복사하는가?
  - 결론: 아니다. stride만 바뀌고 같은 버퍼를 공유. contiguous로 만들 때 copy 발생.
  - 근거: `np.shares_memory` True/False, strides (48,16,4) → (48,4,16)
  - 관련 실험: `experiments/01-matmul/`
- [x] 같은 FLOP이면 latency도 비슷한가?
  - 결론: 아니다. decode-like(0.034 GFLOP) 975 µs vs 128³(0.004 GFLOP) 4.5 µs — FLOP비 8배, latency비 ~216배. memory-bound 여부가 지배.
  - 관련 실험: `experiments/01-matmul/`
