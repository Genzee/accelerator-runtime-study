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
| embedding | 토큰을 의미 공간의 좌표(벡터)로 바꾸는 표 lookup. 좌표는 학습으로 정해짐 | notes/02 |
| Q / K / V | 같은 벡터를 weight로 변환한 '찾는 것 / 이름표 / 넘겨줄 내용' | notes/02 |
| attention | Q·Kᵀ 관련도 → softmax 비율 → V 가중합. 토큰 간 정보 교환 | notes/02 |
| causal mask | 뒤(미래) 토큰을 못 보게 score를 -inf로 가림 | notes/02 |
| residual | block 출력에 입력을 더해 원래 정보를 보존 | notes/02 |
| prefill / decode | 프롬프트 일괄 처리(compute-bound) / 토큰 1개씩 생성(memory-bound) | notes/02 |
| KV cache | 이전 토큰의 K/V를 저장해 decode 때 재계산을 피함 | notes/02 |
| logits | 단어장 전체에 대한 점수. softmax 후 확률 | notes/02 |
| bandwidth (대역폭) | 초당 옮길 수 있는 데이터 양 (GB/s). 통로의 폭. 버스 폭 × 전송 속도로 결정 | notes/03 |
| memory latency | 첫 바이트가 도착하기까지 걸리는 시간 (DRAM ~100 ns). 대역폭과 별개 | notes/03 |
| capacity (용량) | 메모리에 담을 수 있는 총량 (GB). 대역폭과 별개 | notes/03 |
| unified memory | CPU·GPU·NPU가 같은 물리 DRAM을 공유하는 구조. 복사는 없지만 대역폭을 나눠 씀 | notes/03 |
| contention (경합) | 여러 주체가 같은 공유 자원(여기선 메모리 대역폭)을 나눠 쓰며 각자 느려지는 현상 | notes/03 |
| decode (단계) | 답을 토큰 1개씩 만드는 바퀴들. 토크나이징과 무관 | notes/02 |
| lap/바퀴 | 토큰 하나를 만들기 위해 전체 층을 한 번 통과하는 것 | notes/02 |
| early exit | 쉬운 토큰은 중간 층에서 끝내자는 연구 아이디어. 일반 서빙에선 안 씀 | notes/02 |
| prefix caching | 이전 요청의 KV cache를 보관했다가 앞부분이 같으면 재사용 | notes/02 |
| tool call | 모델이 도구 호출문을 토큰으로 생성, 실행은 앱/에이전트 층 | notes/02 |
| RAG | 관련 문서를 벡터 검색해 문맥에 붙여 주는 방식 | notes/02 |
| logit lens | 중간 층 벡터를 출구에 바로 넣어 "그 시점의 다음 단어"를 엿보는 관찰 기법 | notes/02 |
| VRAM / HBM / GDDR | GPU 전용 메모리(역할) / 그 기술 종류(HBM: 적층·초광대역, GDDR: 게이밍용) | notes/03 |
| SRAM | 칩 안의 작고 빠른 메모리(L1/shared, 캐시) | notes/03 |
| op fusion | 여러 연산을 한 kernel로 합쳐 중간 결과를 SRAM에 둔 채 처리 | notes/01, 03 |
| systolic array | 곱셈기 격자에 데이터를 흘려보내며 재사용하는 NPU 구조 | notes/03 |
