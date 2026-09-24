# Glossary

| 용어 | 한 줄 정의 | 처음 나온 곳 |
|---|---|---|
| Tensor | (data 포인터, dtype, shape, strides, device)로 해석되는 다차원 배열 | notes/01 |
| shape | 각 축의 길이. 예: `[B,S,H]` | notes/01 |
| dtype | 원소 타입. itemsize와 사용 가능한 kernel을 결정 (fp32=4B, fp16/bf16=2B, int8=1B) | notes/01 |
| stride | 한 축으로 1칸 이동 시 건너뛰는 byte 수. layout을 결정 | notes/01 |
| contiguous | stride가 마지막 축부터 촘촘한 상태. 많은 kernel의 입력 요구조건 | notes/01 |
| operator (op) | tensor를 받아 tensor를 내는 함수. 실제 실행 단위는 kernel | notes/01 |
| kernel | 특정 device·dtype·layout용으로 구현된 op의 실제 코드 | notes/01 |
| MatMul | `[M,K]·[K,N]→[M,N]`, FLOPs = `2MNK` | notes/01 |
| Gemm | `α·A·B + β·C`. MatMul + bias 결합 형태 | notes/01 |
| elementwise | 원소별 독립 연산 (Add, Relu 등). 대개 memory-bound | notes/01 |
| FLOP / FLOPS | 부동소수 연산 횟수 / 초당 연산 횟수 | notes/01 |
| arithmetic intensity (AI) | FLOPs / 이동 bytes. 높을수록 compute-bound 쪽 | notes/01 |
| compute-bound | 연산 유닛이 병목 | notes/01 |
| memory-bound | 메모리 대역폭이 병목 | notes/01 |
| computational graph | node=op, edge=tensor dependency 인 DAG | notes/01 |
| DAG | 사이클 없는 방향 그래프 | notes/01 |
| topological order | 모든 edge가 앞→뒤를 향하는 node 나열. 유효한 실행 순서 | notes/01 |
| initializer | graph에 고정 저장된 tensor (주로 weight) | notes/01 |
| activation | 요청마다 새로 계산되어 node 사이를 흐르는 tensor | notes/01 |
| op fusion | 여러 op를 하나의 kernel로 합쳐 중간 tensor 메모리 왕복 제거 | notes/01 |
| latency / throughput | 요청 1건 소요 시간 / 단위시간당 처리량 | notes/01 |
| p50 / p95 | 측정 분포의 중앙값 / 95번째 백분위 | notes/01 |
| warmup | 캐시·JIT·할당 효과를 제거하기 위해 버리는 초기 실행 | notes/01 |
