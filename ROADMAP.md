# Heterogeneous Accelerator Runtime Study Plan

## 목표

- AI 모델이 실제 하드웨어에서 어떻게 실행되는지 이해한다.
- ONNX 그래프를 읽고 operator / tensor / shape / dtype / dependency를 설명할 수 있다.
- CPU/GPU/NPU의 차이를 compute, memory, transfer 관점에서 설명할 수 있다.
- ONNX Runtime의 Execution Provider와 graph partitioning을 이해한다.
- 간단한 cost model을 만들고, "어떤 subgraph를 어느 device에서 실행할 것인가"를 실험한다.
- 최종적으로 heterogeneous accelerator node runtime / scheduler 아이디어의 타당성을 검증한다.

## 중요 원칙

- 처음부터 MLIR/IREE 소스코드를 파지 않는다.
- 수학 증명보다 "실행 관점"을 우선한다.
- 매 단계마다 반드시 작은 코드/실험을 남긴다.
- 추측과 실측을 분리한다.
- op 단위의 미세 분할보다 device boundary와 transfer cost를 먼저 본다.
- 모델마다 최적 실행계획이 달라질 수 있음을 전제로 한다.
- 학습 노트는 repo 안에 계속 누적한다.

---

## Phase 0 — 용어와 실행 모델

**목표**: 모델을 "거대한 함수"가 아니라 "tensor가 operator DAG를 흐르는 프로그램"으로 본다.

**공부**
- Tensor: shape, dtype, stride/layout, contiguous/non-contiguous
- Operator: MatMul/Gemm, Add, Softmax, LayerNorm, activation, gather/scatter
- Computational graph: node, edge, dependency, DAG
- FLOP / FLOPS, latency / throughput, batch, sequence length, static shape / dynamic shape

**실습**
- NumPy로 MatMul을 직접 실행한다.
- matrix shape를 바꾸며 입력/weight/output byte 수를 계산한다.
- `2*M*N*K`로 MatMul FLOP을 계산한다.
- 작은 operator DAG를 Python 코드로 표현한다.

**완료 기준**
- `[B,S,H]` tensor가 몇 byte인지 바로 계산할 수 있다.
- MatMul 입력 shape를 보고 output shape를 설명할 수 있다.
- graph edge가 실제로는 tensor dependency라는 것을 설명할 수 있다.

**산출물**: `notes/01-tensor-and-ops.md`, `experiments/01-matmul/`

## Phase 1 — Transformer를 "실행 관점"에서 이해

**목표**: Transformer를 수학 논문이 아니라 실행 graph로 본다.

**공부**: embedding, Transformer block, Q/K/V projection, attention, softmax, output projection, residual connection, LayerNorm/RMSNorm, FFN/MLP, logits, sampling

**반드시 이해**
```
input tensor
  ↓
Q/K/V MatMul
  ↓
attention
  ↓
projection
  ↓
residual/norm
  ↓
FFN MatMul
  ↓
activation
  ↓
MatMul
  ↓
output tensor
```

**LLM inference 추가**: prefill, decode, KV cache, batch, continuous batching의 개념, sequence length가 compute/memory에 주는 영향

**병렬화 개념**: pipeline parallelism, tensor parallelism, sequence parallelism, expert parallelism(MoE)

**주의**
- 이 단계에서는 distributed training 깊게 들어가지 않는다.
- attention 수학 증명보다 tensor shape 변화와 memory movement를 우선한다.

**실습**
- PyTorch로 아주 작은 Transformer block 하나를 만든다.
- forward hook 또는 profiler로 주요 tensor shape를 출력한다.
- batch/sequence length를 바꿔 tensor 크기를 비교한다.

**완료 기준**
- Transformer block을 operator 단위로 그릴 수 있다.
- "모델을 쪼갠다"는 말이 layer/subgraph/tensor 중 무엇을 의미하는지 구분할 수 있다.
- prefill과 decode의 실행 특성이 왜 다른지 설명할 수 있다.

**산출물**: `notes/02-transformer-execution.md`, `experiments/02-transformer-block/`

## Phase 2 — CPU / GPU / NPU 실행구조

**목표**: 왜 같은 operator라도 chip마다 비용이 다른지 이해한다.

- **CPU**: core, SIMD/vector, cache hierarchy, DRAM
- **GPU**: SM/CU, warp/wavefront, thread block, register/shared memory, VRAM/HBM, kernel launch, occupancy
- **NPU**: MAC array / matrix engine, systolic-array 개념, local SRAM, DMA, supported dtype, supported operator/shape 제약
- **공통**: compute throughput, memory bandwidth, memory capacity, host/device memory, DMA, PCIe, unified/shared memory, synchronization
- **핵심 개념**: arithmetic intensity, compute-bound, memory-bound, Roofline model

**실습**
- 큰 MatMul과 elementwise Add를 각각 benchmark한다.
- shape를 바꿔 latency scaling을 측정한다.
- logical bytes와 실제 latency가 단순 비례하지 않는 이유를 기록한다.

**완료 기준**
- FLOPs / compute throughput만으로 latency를 정확히 예측할 수 없는 이유를 설명한다.
- transfer cost가 작은 op의 accelerator 이득을 없앨 수 있는 이유를 설명한다.
- "NPU가 3배 빠르다"와 "NPU로 보내는 것이 3배 빠르다"가 다른 문장임을 설명한다.

**산출물**: `notes/03-hardware.md`, benchmark CSV/JSON

## Phase 3 — ONNX와 graph IR

**목표**: 모델을 실제 graph 데이터 구조로 읽는다.

**공부**: ONNX Model, Graph, Node, Value/Tensor, initializer(weight), input/output, op_type, attributes, shape inference, opset

**실습**
- 작은 PyTorch 모델을 ONNX로 export.
- Python onnx package로 graph를 순회.
- 각 node에 대해 op type / input names / output names / inferred shape / dtype 출력.
- 각 edge tensor의 예상 byte 크기를 계산.
- graph를 topological order로 출력.
- 추가: 작은 Transformer block을 ONNX로 export해서 graph가 얼마나 많은 operator로 풀리는지 관찰.

**완료 기준**
- ONNX graph 하나를 보고 dependency를 따라갈 수 있다.
- weight와 activation을 구분할 수 있다.
- subgraph boundary에서 어떤 tensor가 이동해야 하는지 계산할 수 있다.

**산출물**: `notes/04-onnx-graph.md`, `src/graph_analyzer/`, `experiments/03-onnx-inspect/`

## Phase 4 — ONNX Runtime / Execution Provider

**목표**: "누가 graph를 어떤 hardware에 넘기는가"를 이해한다.

**공부**: ONNX Runtime session, graph optimization, Execution Provider(EP), EP priority, capability, graph partitioning, fallback, compiled subgraph, I/O Binding, profiling

**중점**
- `GetCapability()`가 어떤 의미인지 코드/문서 기준으로 이해한다.
- EP가 지원 가능한 node/subgraph를 어떻게 선언하는지 본다.
- CPU EP fallback 구조를 이해한다.
- 왜 여러 EP가 있다고 자동으로 globally optimal placement가 되는 것은 아닌지 설명한다.

**실습**
- CPUExecutionProvider로 ONNX 모델 실행.
- ORT profiling 활성화, node별 latency 수집.
- graph optimization 전후 비교.
- 가능하면 CPU + CoreMLExecutionProvider 비교.

**M4 실험**
- macOS ONNX Runtime의 CoreML EP 사용 가능 여부 확인, `ort.get_available_providers()` 기록.
- CoreML `MLComputeUnits` 옵션별 테스트: `CPUOnly`, `CPUAndGPU`, `CPUAndNeuralEngine`, `ALL`
- 같은 model/input shape로 latency 비교. warm-up과 steady-state latency를 분리.

**주의**
- Apple CoreML이 내부적으로 CPU/GPU/ANE 배치를 결정할 수 있으므로 이것을 곧바로 "우리가 만든 scheduler"라고 해석하지 않는다.
- 우선 "가속기 backend와 CPU backend의 실제 비용 차이를 관찰하는 실험 환경"으로 사용한다.

**완료 기준**
- ONNX와 ONNX Runtime의 역할을 구분할 수 있다.
- EP와 device driver/library의 관계를 설명할 수 있다.
- graph partitioning과 execution scheduling을 구분할 수 있다.

**산출물**: `notes/05-onnx-runtime.md`, `experiments/04-ort-profile/`, `experiments/05-coreml-vs-cpu/`

**공식 참고**
- ONNX Runtime Execution Providers — https://onnxruntime.ai/docs/execution-providers/
- CoreML Execution Provider — https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html
- Add a new Execution Provider — https://onnxruntime.ai/docs/execution-providers/add-execution-provider.html

## Phase 5 — Cost Model 기초

**목표**: "어디서 실행할까?"를 숫자로 표현한다.

**첫 모델**
```
compute_cost(op, device)
transfer_cost(tensor, src, dst)
```

**정적 추정**
```
compute_time  ≈ FLOPs / effective_compute_throughput
transfer_time ≈ fixed_overhead + bytes / effective_bandwidth
```
하지만 실제 plan에서는 profiling 값을 우선한다.

**Profile key**: `(device, op_type, input_shapes, dtype, batch, sequence_length)`

```json
{ "device": "cpu", "op": "MatMul", "shape": [[1, 2048, 4096], [4096, 4096]],
  "dtype": "bf16", "p50_us": 1200, "p95_us": 1350 }
```
```json
{ "src": "deviceA", "dst": "deviceB", "bytes": 16777216, "p50_us": 700 }
```

**공부**: average/p50/p95, warmup, cache effects, dynamic shapes, bucketing, interpolation, contention, queue delay

**실습**
- MatMul shape별 latency table 생성.
- tensor size별 copy latency curve 생성(가능하면).
- 단순 linear model과 실제 측정값 오차 비교, prediction error 기록.

**완료 기준**
- logical bytes와 physical transfer를 구분한다.
- latency를 이론값으로만 고정하면 안 되는 이유를 설명한다.
- shape bucket을 쓰는 이유를 설명한다.

**산출물**: `notes/06-cost-model.md`, `src/profiler/`, `src/cost_model/`, `experiments/06-cost-model/`

## Phase 6 — Graph Partitioning

**목표**: "쪼갤 수 있다"와 "쪼개는 것이 유리하다"를 분리한다.

**먼저 capability partition**
```
Node A -> GPU/NPU 가능
Node B -> GPU만 가능
Node C -> CPU/GPU 가능
```

**다음 cost-aware partition**
```
subgraph compute saving
- boundary transfer cost
- synchronization cost
- queue delay
```

**중요 원칙**
- 작은 op마다 device를 바꾸지 않는다.
- 큰 contiguous subgraph를 우선한다.
- device boundary 수를 penalty로 둔다.
- weight는 가능하면 device에 상주시킨다.
- activation transfer를 계산한다.

**첫 planner objective**
```
minimize estimated end-to-end latency
subject to:
- device capability
- device memory capacity
- DAG dependency
- valid tensor layout/dtype
추가 penalty:
+ transfer_cost
+ synchronization_cost
+ boundary_penalty
```

**실습**
- 가짜 device 2개를 만든다. 각각 op latency profile을 다르게 설정한다.
- graph에 대해 최저 예상 비용 partition을 찾는다.
- greedy 방식부터 시작. 이후 shortest-path / dynamic programming / ILP 가능성을 조사.

**완료 기준**
- fastest-op placement가 global optimum이 아닌 예제를 직접 만든다.
- device boundary가 늘면 성능이 나빠지는 예제를 만든다.
- request-level routing과 graph-level partitioning의 차이를 설명한다.

**산출물**: `notes/07-partitioning.md`, `src/planner/`

## Phase 7 — Request-level routing 먼저 실험

**목표**: fine-grained graph partition보다 쉬운 현실적 문제부터 검증한다.

```
Request
  ↓
Router
  ├─ Backend A
  └─ Backend B
```

- 입력 feature: batch, sequence length, model, precision, current queue depth
- 결정: `backend = argmin(predicted latency)`

**실습**
- 동일 모델을 CPU/CoreML 두 backend로 실행.
- seq_len 또는 input shape별 latency profile 생성.
- 간단한 router 작성. static priority와 profile-aware routing 비교.

**왜 먼저 하는가**: tensor cross-device transfer가 없다 / planner·cost-model·observability를 먼저 검증할 수 있다 / 실제 scheduler 설계의 기본 피드백 루프를 만들 수 있다.

**완료 기준**
- 고정 backend보다 profile-aware routing이 유리한 workload를 재현한다.
- 잘못된 prediction이 어떤 성능 악화를 만드는지 기록한다.

**산출물**: `src/planner/request_router.py`, benchmark report

## Phase 8 — Coarse Subgraph Prototype

**목표**: 실제 heterogeneous partition의 최소 형태를 검증한다. 처음부터 Transformer 전체를 하지 않는다.

```
MatMul → Relu → MatMul → Softmax → MatMul
```

**실험**
- graph를 두 큰 subgraph로 나눈다. A만 backend 1, B만 backend 2에서 실행.
- 중간 tensor serialization/copy 비용 측정.
- single backend 대비 end-to-end latency 비교.

**반드시 기록**: compute saved, transfer added, synchronization added, total latency, boundary tensor size

**성공 기준**
- hybrid execution이 이기는 경우와 지는 경우를 둘 다 재현.
- "어떤 조건에서 split할 가치가 있는가"를 문장과 수치로 설명.

**산출물**: `notes/08-prototype.md`, `src/planner/subgraph_planner.py`, end-to-end benchmark report

## Phase 9 — 그 다음에 볼 것

- **Compiler/IR**: MLIR, StableHLO, IREE
- **Runtime/plugin abstraction**: OpenXLA PJRT, ONNX Runtime Plugin EP
- **Scheduling**: list scheduling, critical path, HEFT, dynamic programming, ILP, online scheduling, queue-aware scheduling
- **Memory**: zero-copy, shared/unified memory, pinned memory, peer-to-peer DMA, NUMA, CXL, coherent interconnect
- **LLM serving**: vLLM, paged KV cache, continuous batching, prefill/decode disaggregation, speculative decoding, MoE routing
- **관측**: per-op latency, transfer latency, queue time, device utilization, memory pressure, cache hit/miss 가능성, p50/p95/p99

## 지금은 하지 않을 것

자체 ML framework / 자체 ONNX 대체 IR / 자체 compiler backend / CUDA kernel 최적화 깊게 / distributed training 구현 / 여러 서버를 넘는 heterogeneous partition / RDMA 기반 tensor transport / custom NPU driver / 완성형 dynamic scheduler / 전체 LLM을 처음부터 여러 chip에 자동 partition
→ Phase 0~8 이후 다시 판단한다.

## 최종 프로젝트 가설

> 서로 다른 accelerator가 한 node 안에 있을 때, 모델 graph와 runtime profile을 이용해 device capability, compute cost, transfer cost, memory와 queue 상태를 고려한 execution plan을 만들면 단순한 고정 device priority보다 더 나은 배치를 만들 수 있다.

- **비교군**: CPU only / Accelerator only / static EP priority / request-level profile-aware routing / coarse cost-aware partition
- **평가**: end-to-end latency, throughput, p95/p99, transfer overhead, device utilization, memory usage, plan stability

## 첫 10개 실제 Task

1. Tensor shape / dtype / byte 계산
2. MatMul shape와 FLOP 계산
3. Transformer block을 operator graph로 그리기
4. prefill / decode / KV cache 이해
5. CPU/GPU/NPU memory hierarchy 비교
6. Arithmetic Intensity / Roofline 이해
7. 작은 PyTorch 모델을 ONNX로 export
8. ONNX graph node/edge/shape 출력기 작성
9. ONNX Runtime CPU profiling
10. M4에서 CoreML EP와 CPU EP latency 비교

**첫 10개가 끝나기 전에는 custom scheduler 구현을 시작하지 않는다.**
