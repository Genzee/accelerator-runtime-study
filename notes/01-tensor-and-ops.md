# 01 — Tensor, Operator, Computational Graph (시스템 프로그래머 관점)

> Phase 0 / Task 1–2. 실험 코드: `experiments/01-matmul/matmul_basics.py`
> 결과 원본: `experiments/01-matmul/results.{csv,json}`
> 측정 환경: Apple M4 (macOS 26.5.2, 24 GB unified memory), Python 3.13.5, NumPy 2.5.3 (BLAS = Apple Accelerate)

표기 규칙: **[계산]** = 공식으로 얻은 값, **[측정]** = 실제로 돌려서 얻은 값, **[추측]** = 아직 검증 안 된 해석.

---

## 1. Tensor = "타입이 붙은 포인터 + 인덱스→주소 변환 규칙"

C 프로그래머 관점에서 tensor는 다음 struct로 생각하면 된다.

```c
struct Tensor {
    void*    data;      // 연속된 메모리 버퍼 (어느 device의 메모리인지가 중요!)
    dtype_t  dtype;     // 원소 하나의 타입 → itemsize (fp32=4, fp16/bf16=2, int8=1)
    int      ndim;
    int64_t  shape[ndim];    // 각 축의 길이
    int64_t  strides[ndim];  // 각 축으로 1칸 이동할 때 몇 byte 건너뛰는가
    device_t device;    // cpu / gpu:0 / npu:0 ...
};
// 원소 주소: data + Σ index[i] * strides[i]
```

- **shape**: 논리적 모양. `[B, S, H]` = batch × sequence length × hidden.
- **dtype**: 원소 크기와 연산 unit을 결정한다. 같은 shape라도 dtype이 바뀌면 bytes와 **쓸 수 있는 kernel**이 바뀐다 (→ 아래 fp16 결과).
- **stride/layout**: 같은 bytes를 어떤 순서로 해석하는가. transpose는 데이터를 복사하지 않고 stride만 바꾼다.
- **contiguous**: stride가 "마지막 축부터 촘촘히"인 상태. 대부분의 kernel(특히 accelerator)은 contiguous 입력을 원하므로, non-contiguous tensor는 실행 전에 **숨은 copy**를 유발할 수 있다.

### [측정] stride 실험

| tensor | shape | strides (bytes) | C-contiguous | x와 메모리 공유 |
|---|---|---|---|---|
| `x` (fp32) | (2,3,4) | (48,16,4) | True | — |
| `x.transpose(0,2,1)` | (2,4,3) | (48,4,16) | **False** | **True** (복사 없음) |
| `np.ascontiguousarray(...)` | (2,4,3) | (48,12,4) | True | **False** (새 버퍼로 복사) |

→ "shape이 같다"와 "메모리 배치가 같다"는 다른 말이다. device 경계에서 tensor를 넘길 때 layout 변환이 필요하면 copy 비용이 추가된다.

### [계산] `[B,S,H]` 크기 암산법

`bytes = B × S × H × itemsize`

| shape | fp32 | fp16/bf16 |
|---|---|---|
| [1, 128, 768] (BERT-base 1문장) | 384 KiB | 192 KiB |
| [1, 2048, 4096] (7B급 hidden, 2K 토큰) | 32 MiB | 16 MiB |
| [8, 2048, 4096] | 256 MiB | 128 MiB |
| [32, 4096, 8192] | 4 GiB | 2 GiB |

암산 팁: `2048 × 4096 = 8M` 원소 → fp16이면 16 MiB. 2의 거듭제곱으로 쪼개서 계산하면 빠르다.

---

## 2. Operator = "tensor를 받아 tensor를 내는 순수 함수(kernel)"

| op | 입력 → 출력 | FLOPs | 성격 |
|---|---|---|---|
| MatMul `A[M,K]·B[K,N]` | → `C[M,N]` | `2·M·N·K` | 데이터 재사용이 많음 → compute-bound가 될 수 있음 |
| Gemm | `α·A·B + β·C` | ≈ MatMul | MatMul + bias를 한 kernel로 (op fusion의 원형) |
| Add / Mul (elementwise) | shape 동일(broadcast) | `numel` | 원소당 1 FLOP, 3번 메모리 접근 → memory-bound |
| Relu / GELU / SiLU | 동일 shape | `numel × 소수` | memory-bound |
| Softmax | 동일 shape | `~5·numel` | reduce(max, sum) + elementwise |
| LayerNorm / RMSNorm | 동일 shape | `~5–8·numel` | reduce + elementwise |
| Gather / Scatter | index로 원소를 모음/뿌림 | ≈ 0 | 불규칙 메모리 접근 (embedding lookup) |

### MatMul shape 규칙

```
A: [M, K]   B: [K, N]   →   C: [M, N]
          ^^^ 안쪽 차원(K)이 같아야 하고, 사라진다(reduce된다)
```
- 배치가 붙으면 `[B, M, K] · [K, N] → [B, M, N]` — 앞쪽 축은 그대로 따라간다(broadcast).
- Transformer에서는 `x[B,S,H] · W[H,H'] → [B,S,H']` 이고, 실제 kernel은 이를 `M = B·S` 인 2D MatMul로 본다.

### 왜 `2·M·N·K` 인가
출력 원소 `M·N`개 각각이 길이 `K`의 내적 = 곱셈 K번 + 덧셈 K번(FMA 1회 = 2 FLOP으로 셈).

### Arithmetic Intensity (AI) — 미리 맛보기 (Phase 2에서 본격)
`AI = FLOPs / 이동 bytes`. 이상적으로 A, B, C를 DRAM에서 한 번씩만 읽고 쓴다고 가정하면

- MatMul(정사각 n): `2n³ / (3n²·itemsize)` → n에 비례해 커진다 → 크면 compute-bound
- MatMul(M=1, decode): `≈ 2·K·N / (K·N·itemsize)` → fp32에서 0.5 → **weight를 읽는 속도가 전부** (memory-bound)
- Add: `n / (3n·4)` = 0.083 → 항상 memory-bound

---

## 3. Computational Graph = "tensor가 흐르는 operator DAG"

- **node** = operator 호출 1회
- **edge** = tensor. "A → B"는 *B가 A의 출력 tensor를 입력으로 쓴다*는 **데이터 의존성**이다. 제어 흐름(call)이 아니라 **데이터가 준비되면 실행 가능**이라는 뜻.
- **graph input** = 요청마다 바뀌는 activation, **initializer** = 고정된 weight.
- **실행 순서** = topological order. 의존성만 지키면 여러 순서가 가능하고, 독립 node는 병렬 실행 가능.

시스템 관점 비유:
- graph = Makefile / 빌드 DAG. target=tensor, rule=operator.
- runtime = make 실행기. "어떤 rule을 어느 머신(device)에서 돌리고, 결과 파일(tensor)을 어디로 복사할지" 결정.
- **device를 바꾸는 순간 edge 하나 = memcpy/DMA 하나**가 된다. 이게 이 스터디 전체의 핵심 비용.

### [측정] 작은 DAG: `y = softmax(relu(x·W1 + b1)·W2)` (B=4, H=64, F=256, fp32)

노드를 **일부러 뒤섞어 선언**했는데 topo sort가 의존성만으로 순서를 복원했다: `n1 → n2 → n3a → n3b → n4`.

| producer → consumer | tensor | shape | bytes | 종류 |
|---|---|---|---|---|
| graph_input → n1 | x | (4,64) | 1,024 | activation |
| initializer → n1 | W1 | (64,256) | 65,536 | weight |
| n1 → n2 | t1 | (4,256) | 4,096 | activation |
| initializer → n2 | b1 | (256,) | 1,024 | weight |
| n2 → n3a | t2 | (4,256) | 4,096 | activation |
| n3a → n3b | t2r | (4,256) | 4,096 | activation |
| initializer → n3b | W2 | (256,64) | 65,536 | weight |
| n3b → n4 | t3 | (4,64) | 1,024 | activation |

- activation 합 14 KiB vs weight 합 129 KiB — **weight가 훨씬 크다** (batch가 작을 때).
- `n3a | n3b` 사이를 device 경계로 자르면 이동 대상은 `t2r` 4 KiB 하나. **단, W2가 그 device에 상주하지 않았다면 W2(64 KiB)도 옮겨야 한다.** → "weight는 device에 상주" 원칙의 이유.
- 참고: `n2(Add)`, `n3a(Relu)`는 `n1` 결과에 붙은 elementwise op → 실제 runtime은 보통 `MatMul+Add+Relu`를 하나로 **fuse**한다 (ONNX Runtime의 graph optimization, Phase 4에서 확인).

---

## 4. [측정] MatMul: 계산값 vs 실측

p50 latency, warmup 3회 후 최소 10회·0.3초 반복. `GFLOP/s = 2MNK / p50`.

| shape (M,K,N) | dtype | in | weight | out | GFLOP | AI [계산] | p50 [측정] | GFLOP/s [측정] |
|---|---|---|---|---|---|---|---|---|
| decode-like (1,4096,4096) | fp32 | 16 KiB | 64 MiB | 16 KiB | 0.034 | 0.5 | 975 µs | 34 |
| decode-like | fp64 | 32 KiB | 128 MiB | 32 KiB | 0.034 | 0.25 | 2,551 µs | 13 |
| decode-like | fp16 | 8 KiB | 32 MiB | 8 KiB | 0.034 | 1.0 | **41,165 µs** | **0.8** |
| small (128³) | fp32 | 64 KiB | 64 KiB | 64 KiB | 0.004 | 21 | 4.5 µs | 941 |
| mid (512³) | fp32 | 1 MiB | 1 MiB | 1 MiB | 0.268 | 85 | 151 µs | 1,776 |
| mid (512³) | fp16 | | | | 0.268 | 171 | **155,774 µs** | **1.7** |
| big (2048³) | fp32 | 16 MiB | 16 MiB | 16 MiB | 17.2 | 341 | 9,451 µs | 1,818 |
| big (2048³) | fp64 | | | | 17.2 | 171 | 37,719 µs | 456 |
| prefill-like (2048,4096,4096) | fp32 | 32 MiB | 64 MiB | 32 MiB | 68.7 | 512 | 40,517 µs | 1,696 |
| skinny K (2048,64,2048) | fp32 | 512 KiB | 512 KiB | 16 MiB | 0.54 | 30 | 516 µs | 1,040 |

비교용 elementwise Add (fp32):

| n | bytes [계산] | AI | p50 [측정] | 유효 대역폭 [측정] |
|---|---|---|---|---|
| 4M | 48 MiB | 0.083 | 467 µs | 108 GB/s |
| 8M | 96 MiB | 0.083 | 1,078 µs | 93 GB/s |

### 관찰 (측정에서 바로 읽을 수 있는 것)

1. **FLOP이 같아도 latency는 shape에 따라 크게 다르다.** decode-like(0.034 GFLOP)는 975 µs, small square(0.004 GFLOP)는 4.5 µs. FLOP 비는 8배인데 latency 비는 ~216배.
2. **decode-like(M=1)은 memory-bound.** 64 MiB weight를 975 µs에 읽음 → 유효 69 GB/s. Add의 93–108 GB/s와 같은 자릿수. FLOP/s는 34로 peak(~1,800)의 2%. → **LLM decode가 왜 "weight 읽기 속도" 싸움인지**의 가장 작은 재현.
3. **큰 MatMul은 compute-bound로 plateau.** 512³ 이상에서 fp32 ~1.7–1.8 TFLOP/s로 수렴. AI가 85 → 512로 늘어도 GFLOP/s는 거의 안 늘어남 = compute 천장에 닿음.
4. **dtype이 bytes를 절반으로 줄여도 kernel이 없으면 끝.** fp16은 NumPy에 BLAS 경로가 없어 **fp32 대비 ~1,000배 느림**. → accelerator의 "supported dtype/operator 제약"이 CPU에서도 똑같이 존재한다. "지원 안 하는 op/dtype는 fallback 되고, fallback이 치명적일 수 있다"의 예고편.
5. **fp64는 fp32의 2배가 아니라 ~4배 느림** (1,818 vs 456 GFLOP/s).
6. **skinny K(K=64)는 AI가 30인데도 peak의 57%.** 출력(16 MiB)이 입력(1 MiB)보다 훨씬 커서 쓰기 비중이 크다.

### [추측] — 검증 필요 (QUESTIONS.md에 등록)

- fp32 1.8 TFLOP/s는 M4 P-core NEON만으로 설명하기엔 높다. Accelerate가 CPU 옆의 행렬 unit(AMX/SME 계열)을 쓰는 것으로 **추측**. 그렇다면 "CPU" backend 안에 이미 matrix engine이 숨어 있다는 뜻 → Phase 2/4에서 CPU vs NPU 비교할 때 반드시 고려.
- fp64 4배 차이도 위 matrix unit의 fp64 처리량이 낮기 때문으로 **추측**.
- Add 4M(48 MiB)이 8M보다 대역폭이 높은 건 일부가 SLC(system level cache)에 걸렸기 때문으로 **추측**. 크기 sweep 필요.
- 측정은 싱글 프로세스·p50 기준이며 thermal/전원 상태는 통제하지 않았다.

---

## 5. 완료 기준 체크

- [x] `[B,S,H]` tensor가 몇 byte인지 바로 계산할 수 있다 → `B·S·H·itemsize`, 표 §1
- [x] MatMul 입력 shape를 보고 output shape를 설명할 수 있다 → `[M,K]·[K,N]→[M,N]`, K가 reduce됨, 앞 축은 broadcast
- [x] graph edge가 실제로는 tensor dependency라는 것을 설명할 수 있다 → §3 DAG 실험 (뒤섞인 선언 → topo sort 복원, edge별 bytes)

## 6. 이 노트에서 얻은 설계 힌트 (나중 Phase용)

- device 경계 비용 = 경계를 가로지르는 **activation bytes + 상주하지 않은 weight bytes + layout 변환 copy**.
- cost model 키에 `dtype`은 필수. 같은 op·shape라도 dtype에 따라 1,000배 차이.
- 이론 FLOP/s로 latency를 예측하면 memory-bound op(M=1, elementwise)에서 크게 틀린다 → profile 우선 원칙의 근거.
