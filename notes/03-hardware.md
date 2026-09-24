# 03 — CPU / GPU / NPU 실행구조

> Phase 2. 아직 본격 시작 전. 먼저 나온 질문부터 기록.

## Unified memory vs 분리형(discrete) 메모리 — 트레이드오프

| | 분리형 (PCIe GPU/NPU 카드) | Unified (Apple M 시리즈, 모바일 SoC) |
|---|---|---|
| 구조 | 칩마다 전용 메모리(HBM/GDDR), 칩 사이는 PCIe | CPU·GPU·NPU가 같은 DRAM(LPDDR)을 공유 |
| 대역폭 (공칭) | H100 HBM3 ~3.35 TB/s, RTX 4090 ~1 TB/s | M4 ~120 GB/s, M4 Max ~546 GB/s |
| 칩 간 이동 | PCIe 4.0 x16 ~32 GB/s — **로컬 대역폭의 1/30~1/100** | 복사 불필요(원칙상) |
| 용량 | 카드당 24–80 GB 수준, 모자라면 카드 추가 | 시스템 전체 용량을 전부 사용 가능 |
| 경합 | 칩마다 자기 메모리 → 서로 간섭 없음 | **모든 칩이 같은 대역폭을 나눠 씀** |
| 확장/비용 | 카드 추가로 확장, HBM은 비쌈 | 구매 시 고정, 확장 불가 |

LLM decode 관점 [계산, 공칭값 기준]:
- decode 속도 상한 ≈ 대역폭 / 모델 bytes. 8B bf16(16 GB): H100 ≈ 200 tok/s, M4 ≈ 7 tok/s.
- 70B bf16(140 GB): H100 1장(80 GB)에 안 들어감 → 여러 장 + 분할 필요. 큰 unified memory 머신은 한 곳에 올라가지만 느림.
- → **분리형은 속도(대역폭), unified는 용량과 이동 비용에서 유리.**

전제 수정: "NPU는 보통 unified를 안 쓴다"는 반만 맞다.
- 엣지/모바일 NPU(스마트폰, 카메라 SoC, Apple ANE): 대부분 **시스템 DRAM 공유**.
- 데이터센터 NPU/GPU 카드: **전용 HBM + PCIe**.
- 중간 형태: AMD MI300A(CPU+GPU가 HBM 공유), NVIDIA Grace Hopper(메모리는 분리돼 있지만 NVLink-C2C로 coherent하게 연결).

스케줄러 cost model에 주는 의미:
- 분리형: `transfer_cost`가 지배적 → device 경계 최소화가 핵심.
- Unified: `transfer_cost ≈ 0`이지만 **contention**이 생김 → memory-bound op 두 개를 CPU와 GPU에서 동시에 돌려도 2배 빨라지지 않을 수 있음 [추측, 검증 필요].
- 즉 **같은 planner라도 메모리 구조에 따라 cost model이 달라져야 한다.**

[추측] unified라도 zero-copy가 항상 되는 건 아닐 수 있다: 칩마다 요구하는 layout/format(예: ANE 전용 형식)이 다르면 변환 copy가 생긴다. → Phase 4에서 확인.

## [측정] 대역폭 경합: memory-bound 작업을 프로세스 N개로 동시에

실험: `experiments/07-bandwidth-contention/cpu_procs.py` — 프로세스마다 64 MiB fp32 배열 Add를 2초간 반복 (M4: P-core 4, E-core 6).

| 프로세스 수 | 합계 GB/s | 프로세스당 GB/s |
|---|---|---|
| 1 | 87.7 | 87.7 |
| 2 | 91.5 | 45.8 |
| 4 | 90.5 | 22.6 |
| 8 | 89.8 | 11.2 |

- **합계가 ~90 GB/s에서 고정**. 프로세스를 늘려도 전체는 안 늘고, 각자 몫만 줄어든다.
- M4에서는 **코어 1개가 이미 DRAM 통로를 거의 다 채운다.**
- 경합 대상은 용량(공간)이 아니라 **대역폭(통로)**. 64 MiB × 3 × 8개 = 1.5 GiB로 용량은 넉넉했다.
- 스케줄러 의미: unified memory에서 GPU/NPU도 **이 같은 통로**를 쓴다면, memory-bound 작업(LLM decode)을 CPU→GPU로 옮기거나 둘에 나눠도 통로가 넓어지지 않는다. 반면 compute-bound 작업은 칩마다 연산기가 따로 있어서 나누면 이득 가능.
  - → cost model에서 **compute는 칩별 자원, 대역폭은 공유 자원**으로 모델링해야 한다.
- [남은 검증] CPU+GPU(Metal) 동시 실행 시에도 같은 천장을 공유하는지는 아직 미측정.

## 칩 ↔ 자기 메모리 대역폭 비교 (공칭 스펙, 기억 기반 — 인용 전 재확인 필요)

| 분류 | 칩 | 메모리 종류 | 대역폭 | 용량 |
|---|---|---|---|---|
| Unified (LPDDR) | Apple M4 | LPDDR5X 128-bit | 120 GB/s (실측 ~90) | ~32 GB |
| | Apple M4 Pro / Max | LPDDR5X 256 / 512-bit | 273 / 546 GB/s | ~64 / 128 GB |
| | Apple M3 Ultra | LPDDR5 1024-bit | 819 GB/s | ~512 GB |
| | NVIDIA DGX Spark (GB10) | LPDDR5X | 273 GB/s | 128 GB |
| Unified (HBM) | AMD MI300A | HBM3 | 5.3 TB/s | 128 GB |
| 소비자 GPU (GDDR) | RTX 4060 | GDDR6 | 272 GB/s | 8 GB |
| | RTX 4090 / 5090 | GDDR6X / GDDR7 | 1.0 / 1.8 TB/s | 24 / 32 GB |
| 워크스테이션 GPU | RTX A6000 | GDDR6 | 768 GB/s | 48 GB |
| 데이터센터 GPU (HBM) | A100 / H100 / H200 / B200 | HBM2e / 3 / 3e / 3e | 2.0 / 3.35 / 4.8 / 8 TB/s | 80 / 80 / 141 / 192 GB |
| | AMD MI300X | HBM3 | 5.3 TB/s | 192 GB |
| NPU | FuriosaAI RNGD | HBM3 | 1.5 TB/s | 48 GB |
| | Google TPU v5e / v5p | HBM2e | 0.8 / 2.8 TB/s | 16 / 95 GB |
| 칩 사이 | PCIe 4.0 / 5.0 x16 | — | ~32 / ~64 GB/s (단방향) | — |

- 대역폭을 결정하는 건 "unified냐 아니냐"가 아니라 **메모리 종류(LPDDR < GDDR < HBM)와 버스 폭**. MI300A는 unified인데 HBM이라 5 TB/s.
- 트레이드오프: HBM은 빠르지만 비싸고 용량이 작음, LPDDR은 느리지만 싸고 용량이 큼.
- 분리형 GPU의 핵심 불균형: 자기 메모리 1–8 TB/s vs 호스트와 잇는 PCIe 32–64 GB/s → **30–100배 차이**. 칩 경계를 넘는 게 비싼 이유.
- 공칭값은 이론 최대. 실측은 보통 70–90% (M4: 90/120 ≈ 75%).
