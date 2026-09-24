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
