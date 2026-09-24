# 02 — Transformer / LLM을 실행 관점에서 이해

> Phase 1. 완료 기준 충족 (2026-09-24).
> 실험: `experiments/02-transformer-block/attention_by_hand.py`
> 예시 수치는 Llama-3-8B 기준 (H=4096, layer 32, head 32, KV head 8, FFN 14336, vocab 128256).

---

## 1. LLM = "다음 토큰 맞추기 함수"를 반복 호출하는 루프

```python
tokens = tokenize("오늘 날씨가")
while not done:
    logits = model(tokens)          # graph 1회 실행
    next_id = sample(logits[-1])    # 확률분포에서 토큰 하나 뽑기
    tokens.append(next_id)
```

모델은 "지금까지 토큰 → 다음 토큰 확률"을 내는 순수 함수. 문장 생성은 이 함수를 루프로 부르는 것.

## 2. 함수 1회 안의 shape 흐름

```
token ids   [S]
  ↓ embedding (Gather: 표[128256, 4096]에서 행 꺼내기)
x           [S, 4096]
  ↓ Transformer block × 32   (shape 불변, 내용만 갱신)
x           [S, 4096]
  ↓ lm_head MatMul [4096, 128256]
logits      [S, 128256] → 마지막 행 softmax → sampling
```

block을 통과해도 `[S, H]`가 유지된다 → block 단위로 잘라 device에 나눠 싣기 쉽다 (pipeline parallelism).

## 3. 벡터(embedding)는 무엇인가

- 토큰 하나 = **의미 공간 속 좌표** (숫자 H개). 비슷한 의미 = 가까운 좌표. 좌표는 **학습**이 정한다.
- 실제 모델에서 축 하나하나는 사람이 해석할 수 없고, 좌표 전체의 방향/위치가 의미를 가진다.
- embedding 직후 벡터는 **단어 자체의 뜻**만 담는다 ("배" = 과일/선박/신체가 섞인 애매한 좌표).
- block을 지날 때마다 attention으로 **문맥을 흡수**해 좌표가 이동한다 ("배를 타고 바다로" → 선박 쪽).
- 시스템 비유: token id = 배열 인덱스, 벡터 = **토큰별 상태(state)**, block 32개 = 그 상태를 차례로 갱신하는 함수들.

## 3.5 [측정] 모델 파일 안에는 무엇이 있나 (GPT-2 small, safetensors header만 조회)

실험: `experiments/02-transformer-block/inspect_model_file.py` (파일 전체 523 MiB 중 header 14 KB만 HTTP Range로 받음)

- 파일 = `{이름: (dtype, shape, byte 위치)}` 목차 + **숫자 덩어리**. tensor 160개.
- 이름이 곧 구조: `wte`(토큰 embedding 표 [50257,768], 147 MiB), `wpe`(위치 embedding), `h.0`~`h.11`(block 12개), `ln_f`(마지막 norm).
- block 하나 = 13개 tensor, **31.04 MiB**. `h.0`과 `h.11`은 **이름 규칙·shape가 완전히 같고 숫자만 다르다.**
  - `attn.c_attn.weight [768, 2304]` = Wq|Wk|Wv를 가로로 붙인 행렬 (768×3 = 2304)
  - `attn.c_proj.weight [768, 768]` = Wo
  - `mlp.c_fc.weight [768, 3072]`, `mlp.c_proj.weight [3072, 768]` = FFN 두 MatMul
  - `ln_*` = LayerNorm의 scale/shift (768개짜리, 무시할 만큼 작음)
  - `attn.bias [1,1,1024,1024]` = causal mask 삼각형 표. 학습값이 아니라 고정 버퍼 (block마다 4 MiB씩 중복 저장돼 있음)
- 파일 안 byte 위치는 **뒤죽박죽**이다 → 파일은 그냥 사전이고, **실행 순서는 모델 코드(아키텍처 정의)가 안다.**

**쪼개서 올린다 = 이 사전에서 이름으로 골라 각 칩 메모리에 복사하는 것.**
- 예: `h.0.*`~`h.5.*` + `wte`/`wpe` → 칩 A, `h.6.*`~`h.11.*` + `ln_f` → 칩 B.
- 각 칩은 **같은 block 코드**를 돌리되 **자기가 가진 숫자**로 계산한다. 쪼개진 조각끼리 다른 건 숫자 값뿐.
- [주의] 칩 A는 `wte`로 시작하고 칩 B는 마지막에 lm_head로 `wte`를 다시 씀(weight tying) → 양쪽에 사본이 필요할 수 있다.

## 4. Block 하나 = "토큰끼리 대화(attention)" + "토큰별 혼자 생각(FFN)"

```
x [S,4096]
 ├─ RMSNorm
 ├─ Attention   ← 토큰 간 정보 교환이 일어나는 유일한 곳
 │    Q = x·Wq [S,4096], K = x·Wk [S,1024], V = x·Wv [S,1024]   (GQA: KV head 8개)
 │    score = Q·Kᵀ [S,S] → ÷√d → causal mask → softmax → ·V → Wo
 ├─ + x (residual)
 ├─ RMSNorm
 ├─ FFN         ← 토큰별 독립
 │    [S,4096] → [S,14336] → SiLU(gate) → [S,4096]
 └─ + x (residual)
```

- Q = "내가 찾는 정보", K = "내가 가진 정보의 이름표", V = "선택되면 넘겨줄 내용". 같은 벡터를 weight 3개로 역할별 변환한 것.
- [계산] 파라미터: block당 attention ≈ 42M, FFN ≈ 176M → ×32 ≈ 7B + embedding/lm_head ≈ 1B = **8B**. bf16이면 **16 GB**의 read-only weight.
- 연산 종류는 적다: MatMul, Gather, Norm, Softmax, SiLU, Add.

## 5. [측정] 워밍업: attention 손계산 (토큰 3개 × 2차원)

축에 이해용 의미를 붙임: `[사람, 음식]`. `나=[1,0]`, `밥=[0,1]`, `먹었다=[1,1]`.

### A. 변환 없음 (Wq=Wk=Wv=I)

| 단계 | 먹었다 행 (나, 밥, 먹었다) |
|---|---|
| score = Q·Kᵀ | 1, 1, 2 |
| ÷√2 | 0.707, 0.707, 1.414 |
| softmax 비율 | **0.248, 0.248, 0.503** |
| 출력 = 비율·V | `[0.752, 0.752]` |

→ 자기와 비슷한 것(자기 자신)을 가장 많이 참고. "비슷한 것끼리 섞임"만 일어남.

### B. Wq가 "음식을 찾는 질문"으로 학습됐다고 가정 (`Wq=[[0,3],[0,3]]`)

| 단계 | 먹었다 행 (나, 밥, 먹었다) |
|---|---|
| Q | `[0, 6]` ("음식 어디 있어?") |
| score | 0, 6, 6 |
| softmax 비율 | **0.007, 0.496, 0.496** |
| 출력 | `[0.504, 0.993]` |

→ **weight(Wq)가 바뀌니 같은 입력에서도 "누구를 참고할지"가 바뀌었다.** "나"는 거의 무시(0.007)하고 음식을 가진 토큰을 참고 → 출력의 음식 축이 0.75 → 0.99로 커짐 = "먹었다"가 "음식을 먹은"이라는 문맥을 흡수.
→ 학습이란 결국 이런 Wq/Wk/Wv 값을 찾는 과정이다.

### causal mask
- "나"는 자기만, "밥"은 나·밥까지, "먹었다"는 전부 볼 수 있다 (score 표의 위쪽 삼각형 = -inf → 비율 0).
- 미래를 못 보게 하기 때문에, **이미 계산한 앞 토큰의 K/V는 뒤 토큰이 추가돼도 변하지 않는다** → 저장해서 재사용 가능 = **KV cache가 성립하는 이유**.

### [계산] 스케줄러 관점: score 표는 S²으로 커진다

Llama-3-8B (32 heads, bf16), layer 하나 기준, 표를 통째로 만든다고 가정:

| S | score 표 크기 |
|---|---|
| 3 | 0.6 KiB |
| 2,048 | 256 MiB |
| 8,192 | 4 GiB |
| 32,768 | 64 GiB |

→ 긴 입력은 attention이 메모리를 폭발시킨다. FlashAttention 같은 kernel은 표를 통째로 만들지 않고 조각내서 계산해 이를 피한다. **"이 op을 어느 device에서 돌릴 수 있는가"는 op 종류뿐 아니라 kernel 구현과 입력 길이에 따라 달라진다.**

## 6. prefill vs decode, KV cache

| | prefill | decode |
|---|---|---|
| 입력 | 프롬프트 S개 한꺼번에 | 새 토큰 1개 |
| MatMul | M=S | **M=1** |
| 병목 | compute-bound | **memory-bound** (weight 읽기) |
| 01 실험 대응 | prefill-like 1.7 TFLOP/s | decode-like 34 GFLOP/s |

- [계산] KV cache: 토큰당 2(K,V) × 32 layer × 1024 × 2B = **128 KiB**. 8K 토큰 = 1 GiB, 동시 사용자 수만큼 곱해짐.
- 시스템 비유: weight = 공유 read-only 세그먼트, KV cache = 세션마다 커지는 힙.
- [계산] decode 상한: 토큰 1개당 weight 16 GB를 1번 읽음. M4 ~100 GB/s(01 실험 Add 93–108 GB/s) → **≈6 tok/s**. 그래서 LLM 서빙은 "대역폭 대비 모델 크기" 싸움 → 양자화·batching이 효과적.

## 6.5 요청 하나의 일생 (일반적인 LLM 서빙 기준)

1. **입력 조립**: 시스템 지시 + 이전 대화 전체 + 새 메시지를 chat template(역할 표시 토큰 포함)으로 이어 붙여 **긴 텍스트 하나**로 만든다. 모델은 기억이 없으므로 매번 대화 전체를 다시 넣는다.
2. **tokenize**: 텍스트 → token id 배열.
3. **대기열 + 스케줄러**: 서버가 다른 사용자 요청들과 함께 batch를 구성 (continuous batching: 매 decode step마다 끝난 요청은 빼고 새 요청을 넣음).
4. **prefill**: 입력 토큰 전부를 한 번에 모든 layer에 통과 → 모든 토큰의 K/V를 KV cache에 저장 + 첫 출력 토큰. prefix caching이 있으면 이전 턴까지의 KV cache를 재사용해 새 부분만 계산.
   - 이 시간이 **TTFT (Time To First Token)** = 첫 글자가 나오기까지 기다리는 시간.
5. **decode 루프**: 토큰 1개씩. 매 step마다 모든 layer 통과(= weight 전체 읽기), KV cache에 1칸 추가, 토큰을 **바로 스트리밍** (글자가 조금씩 나타나는 이유).
   - 이 속도가 **TPOT / tokens per second**.
6. **종료**: 끝 토큰(EOS)이나 길이 제한 → 종료, KV cache 해제.

### 추론에서 "파이프라인"의 두 가지 뜻

- **토큰 사이에는 파이프라인이 불가능**: 토큰 t+1은 토큰 t가 나와야 계산 가능 (의존성). 한 요청의 decode는 본질적으로 순차.
- **칩 사이 pipeline parallelism**: block 0~5는 칩 A, 6~11은 칩 B일 때, 요청 하나만 있으면 한 칩이 일할 동안 다른 칩은 논다 (**bubble**). 여러 요청을 엇갈려 넣어야 둘 다 바쁘다:

```
시간 →     t1      t2      t3      t4
칩 A     [요청1]  [요청2]  [요청1]  [요청2]
칩 B              [요청1]  [요청2]  [요청1]
```

  - → pipeline 분할은 **단일 요청 latency는 줄이지 못하고**(오히려 전송만큼 늘어남), **여러 요청의 throughput**을 올리는 수단이다. 스케줄러 평가 지표를 latency와 throughput으로 나눠야 하는 이유.

## 7. "칩에 나눠준다"의 세 가지 의미

| | 무엇을 나누나 | 예 | 칩 사이 데이터 이동 | 이 스터디 |
|---|---|---|---|---|
| **A. 요청 나누기** | 같은 모델 **복사본**을 칩마다 두고 요청을 배분 | 짧은 질문 → NPU, 긴 문서 → GPU | 거의 없음 (요청 입출력만) | Phase 7 |
| **B. 모델 하나를 안에서 쪼개기** | 한 모델의 연산을 여러 칩이 나눠 실행 | block 1–16 GPU / 17–32 NPU, prefill GPU / decode NPU | **경계마다 activation 이동** | Phase 6, 8 (본체) |
| **C. 여러 모델을 배치하기** | 서로 다른 모델을 어느 칩에 올릴지 | 카메라: 검출 모델 NPU, 추적 모델 CPU / 음성인식→LLM→TTS | 모델 사이 결과만 | A의 일반화 |

B 안에서도 자르는 단위가 다시 나뉜다:
- **layer/block 단위** (pipeline parallelism): block 경계에서 `[S, H]` activation 하나만 넘기면 됨 → 가장 깔끔
- **subgraph/phase 단위**: prefill과 decode를 다른 칩에 → KV cache를 넘겨야 함 (토큰당 128 KiB)
- **tensor 단위** (tensor parallelism): MatMul 하나의 weight를 칩들이 쪼개 들고 동시에 계산 → 매 layer마다 결과를 모으는 통신 필요. 칩 사이 연결이 매우 빨라야 함 (보통 같은 종류 GPU끼리)

### 나눠 싣는다 = 물리적으로 어디에 무엇이 있나

- **분리형**: 각 칩의 전용 메모리에 **자기 몫의 weight만 복사**해 둔다 (DRAM 공유 아님). block 1–16 weight는 GPU VRAM, 17–32는 NPU 메모리.
  - weight 이동은 **로딩 때 1번**. [계산] 8 GB를 PCIe 4.0(~32 GB/s)으로 ≈ 0.25 s.
  - 실행 중 칩 사이를 오가는 건 **경계의 activation뿐**. [계산] decode 1토큰 `[1,4096]` bf16 = 8 KiB, prefill 2048토큰 `[2048,4096]` = 16 MiB (PCIe로 ≈ 0.5 ms). 실제로는 여기에 고정 오버헤드(동기화, 드라이버 호출)가 더해짐.
  - A(요청 나누기)는 칩마다 **weight 전체 사본** → 메모리 사용량이 칩 수만큼 늘어남.
- **Unified**: weight는 DRAM에 **한 벌**만 두고, 여러 칩이 같은 주소를 읽는 것이 원칙. "나눈다"는 메모리 배치가 아니라 **어느 칩이 어느 부분을 계산할지**만 정하는 것.
  - [추측] 단, 칩이 전용 형식을 요구하면(ANE 등) 그 칩용으로 변환된 사본이 별도로 생길 수 있음 → Phase 4 확인.

### 나눠 실었을 때 입력은 어느 칩으로 가나 — "내용을 보고 고르나?"

| 방식 | 입력 내용을 보고 칩을 고르나? | 동작 |
|---|---|---|
| **pipeline (block 단위 분할)** | **아니오** | 모든 토큰이 항상 칩 A(앞 block) → 칩 B(뒤 block) 순서로 전부 통과. 조립 라인. 판단 과정 없음 |
| **MoE (Mixture of Experts)** | **예** | block 안의 FFN이 여러 개(expert). **router**(작은 MatMul + top-k)가 토큰 벡터를 보고 expert 몇 개만 고름. expert를 칩마다 나눠 두면 토큰이 내용에 따라 다른 칩으로 감 = expert parallelism |
| **embedding 표 행 분할** (vocab parallel) | 번호로만 | 단어 id 범위로 칩 결정 (0~25000은 A, 나머지는 B). 유사도 검색이 아니라 단순 산술 |

- dense 모델(GPT-2, Llama)의 forward에는 "가장 가까운 것 찾기" 과정이 없다. 문자→id는 사전 lookup, id→벡터는 표의 행 꺼내기. 벡터 간 "가까움"은 attention 내부(Q·K)에서 **참고 비율**을 정하는 데만 쓰인다.
- "내용 기반 라우팅"은 MoE에서 실제로 존재하고, 스케줄러 관점에서 가장 까다로운 경우다: 어느 expert(칩)가 바빠질지 **실행 전에 알 수 없고** 입력마다 달라짐 → 칩 간 부하 불균형, 토큰 all-to-all 통신.

난이도·위험: A < C < B(layer) < B(phase) < B(tensor). 그래서 A부터 검증하고 B로 간다.

## 9. [측정] GPT-2 small을 NumPy로 직접 실행

실험: `experiments/02-transformer-block/gpt2_numpy.py` (NumPy + tiktoken만 사용, weight는 `models/gpt2/`에 별도 다운로드)
환경: Apple M4 CPU, fp32, greedy decoding. 결과 원본: `results_gpt2.json`

### 9.1 tensor 흐름 (입력 "The weather today is", S=4)

| 단계 | shape | bytes |
|---|---|---|
| embedding (wte[ids] + wpe[pos]) | [4, 768] | 12 KiB |
| QKV projection | [4, 2304] | 36 KiB |
| Q / K (12 heads) | [12, 4, 64] | 12 KiB |
| FFN up + GELU | [4, 3072] | 48 KiB |
| block 출력 | [4, 768] | 12 KiB |
| **block 5→6 경계 (칩 분할 시 이동량)** | [4, 768] | **12 KiB** |
| logits (마지막 토큰만) | [1, 50257] | 196 KiB |

다음 토큰 top-5: ' very' 0.043, ' good' 0.032, ' pretty' 0.031 … → 생성 결과: "The weather today is **very good, and we're going to be able to get some good weather tomorrow.** …" (GPT-2 greedy 특유의 반복 포함)

### 9.2 KV cache 효과

| | 측정값 |
|---|---|
| decode (cache 사용) | 토큰당 **8.44 ms** (119 tok/s) |
| KV cache 크기 | 토큰당 72 KiB (12 layer × K,V × 768 × 4B) |
| cache 사용 vs 미사용 출력 | **동일** (계산 결과는 같고 재계산만 생략) |
| cache 미사용: 토큰 1개당 비용 | 문장 길이 4 → 9.9 ms, 43 → 26.9 ms (길이에 비례해 증가) |
| 40토큰 생성 총 시간 | cache 351 ms vs 미사용 736 ms |

### 9.3 prefill: 한 번에 넣는 토큰이 많을수록 토큰당 비용이 싸진다

| 입력 토큰 수 | prefill 시간 | 토큰당 |
|---|---|---|
| 1 | 7.85 ms | 7.85 ms |
| 4 | 9.85 ms | 2.46 ms |
| 16 | 15.9 ms | 0.99 ms |
| 64 | 36.2 ms | 0.57 ms |
| 256 | 114 ms | **0.45 ms** |
| 1000 | 667 ms | 0.67 ms |

- 1토큰 → 256토큰에서 토큰당 비용 **17배 감소**: weight(474 MiB)를 한 번 읽어 여러 토큰이 재사용 → memory-bound에서 compute-bound로 이동. = **prefill은 싸고 decode는 비싼 이유**, **batching이 효과적인 이유**.
- 1000토큰에서 다시 증가 [추측]: attention의 S² 비용이 커지기 시작.
- KV cache가 없으면 1000토큰 문맥에서 토큰 1개 생성마다 667 ms, cache가 있으면 ~8.4 ms → **~80배**.

### 9.4 decode 1 step 시간 비중 (문맥 ~40토큰)

| 연산 | ms/step | 비중 |
|---|---|---|
| MatMul lm_head (768→50257) | 2.28 | 25.9% |
| MatMul FFN down | 1.74 | 19.7% |
| MatMul FFN up | 1.70 | 19.4% |
| MatMul QKV | 1.31 | 14.8% |
| MatMul Wo | 0.46 | 5.2% |
| LayerNorm / Attention core / GELU / reshape / 기타 | 1.32 | 15.0% |

- **MatMul이 85%.** 그중 lm_head 하나가 26%: 단어장 50257 × 768 표(147 MiB)를 토큰마다 통째로 읽기 때문. 작은 모델일수록 lm_head 비중이 크다.
- attention core는 3.7%: 문맥이 짧으면 attention은 싸다. 문맥이 길어지면 커짐(9.3 참조).
- [계산] step당 읽는 weight 472 MiB ÷ 90 GB/s = **5.5 ms 하한** vs 측정 8.8 ms → 대역폭 한계의 약 62%. 나머지는 Python 오버헤드, 작은 op들 [추측].

### 9.5 실수에서 배운 것: 같은 계산인데 12~15배 느렸던 두 가지 원인

| 조건 | decode 토큰당 |
|---|---|
| 정상 (정렬된 메모리, fp32 유지) | **8.44 ms** |
| weight 메모리 주소 misaligned | 103 ms (12배) |
| `np.sqrt(64)`(float64 스칼라)로 나눠서 이후 전체가 float64로 승격 | 131 ms (15배) |

1. **정렬(alignment)**: safetensors header가 14,291 bytes라 파일을 그대로 참조하면 weight 시작 주소가 4의 배수가 아님 → NumPy가 BLAS를 못 쓰고 느린 루프로 처리. 로딩 시 복사해 정렬하면 해결.
2. **dtype 승격**: attention score 나눗셈 하나 때문에 activation이 float64가 되고, 이후 모든 MatMul에서 float32 weight(474 MiB)가 **매 토큰마다 float64로 변환**됨.
- 교훈: "같은 모델, 같은 연산"이라도 **layout/dtype이 kernel 경로를 바꾸면 10배 단위로 달라진다.** accelerator로 넘길 때도 똑같이 일어날 수 있는 일 (지원 dtype/layout이 아니면 변환 copy 또는 느린 fallback). cost model 키에 dtype·layout이 들어가야 하는 이유.

## 8. 한 줄 요약

> LLM = **거대한 read-only weight** + **토큰 수만큼 커지는 KV cache** 위에서, MatMul 위주의 **같은 block을 N번 반복하는 DAG**를 **토큰 하나 생성할 때마다 한 번** 실행하는 프로그램.

## 완료 기준 체크 (Phase 1)

- [x] Transformer block을 operator 단위로 그릴 수 있다 — §4 개념도 + §9 NumPy 실구현
- [x] "모델을 쪼갠다"가 layer/subgraph/tensor 중 무엇인지 구분할 수 있다 — §7
- [x] prefill과 decode의 실행 특성이 왜 다른지 설명할 수 있다 — §6 개념 + §9.2~9.3 실측
