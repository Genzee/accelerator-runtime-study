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

### 9.6 [측정] "연산만 하는데 어떻게 답이 나오나" — 다음 토큰 확률에 지식이 들어 있다

실험: `experiments/02-transformer-block/what_does_it_know.py` (GPT-2 small 124M)

| 입력 | 다음 토큰 top 후보 |
|---|---|
| Monday, Tuesday, Wednesday, | **Thursday 79%**, Friday 7% |
| 1, 2, 3, 4, | **5 87%** |
| The Eiffel Tower is located in the city of | **Paris 6.4%** (1위), London 4.6%, Amsterdam 3.4% |
| The capital of Japan is | the 9.4%, **Tokyo 6.7%** (2위) |
| The capital of France is | the 8.5%, now, a, France, **Paris 3.2%** (5위) |
| Water boils at a temperature of 100 degrees | **Fahrenheit 34.5%** (틀림), F, Celsius 14.4% |
| The CEO of Apple is | a, not, the … (모름) |

- 모델은 답을 "검색"하지 않는다. 다음 토큰 확률만 낸다. 그런데 확률을 잘 맞히려면 지식이 필요해서, 학습 과정에서 지식이 weight 숫자에 압축돼 들어간다.
- "the" 같은 토큰이 1위인 건 문법적으로 가능한 이어짐이 많아서 ("the city of Paris…").
- 124M짜리 작은 모델은 지식이 희미하고 틀리기도 한다. 모델이 클수록(파라미터 = 저장 공간) 더 많이, 더 정확히 담는다.
- 시스템 비유: CPU는 덧셈·곱셈·분기만 하지만 프로그램에 따라 무엇이든 한다. **MatMul이 명령어 실행이라면 weight가 프로그램**이고, 그 프로그램은 사람이 짠 게 아니라 학습이 데이터에서 찾아낸 것.

### 9.7 [측정] 텍스트 → 숫자 → layer별 변화 (logit lens)

실험: `experiments/02-transformer-block/text_to_numbers.py`, 입력 "The Eiffel Tower is located in the city of"

- 텍스트 42 bytes → 토큰 11개 (`'The'`=464, `' E'`=412, `'iff'`=733, `'el'`=417, `' Tower'`=8765 …). "Eiffel"은 사전에 없어서 3조각.
- 번호 → wte 행 + wpe 행 = x [11, 768] (33 KiB).
- 마지막 위치(' of')의 벡터를 layer마다 lm_head에 통과시켜 "이 시점에 떠올리는 다음 단어"를 본 결과:

| 지점 | 벡터 크기 | 떠올리는 다음 단어 top-3 |
|---|---|---|
| 입력 (embedding) | 4.7 | 무의미한 조각들 |
| block 0–3 | 54–65 | ' the' 66–82% (문법적으로 흔한 이어짐) |
| block 4–6 | 70–93 | ' the', ' La', ' England', ' East' (장소 느낌) |
| block 7–8 | 105–122 | ' San', ' Rome', ' La' (도시 이름) |
| block 9 | 150 | ' London' 24.8%, ' Paris' 19.4%, ' Amsterdam' 17.8% (유럽 도시) |
| block 10–11 | 265–444 | **' Paris' 1위** |

- 앞쪽 layer는 문법 수준("of 다음엔 the"), 중간은 범주("장소/도시"), 뒤쪽에서 구체적 사실("Paris")로 좁혀진다.
- 벡터 크기가 layer마다 커진다: residual로 각 block의 결과가 계속 **더해지기** 때문.
- 주의: logit lens는 마지막 layer용 출구를 중간에 쓰는 근사 관찰이다. "block 9가 유럽 도시를 생각한다"는 해석은 비유 수준.

## 10. 개념 정리 — 대화로 확정한 이해 (2026-09-24)

짧은 한 줄 버전. 자세한 근거는 위 섹션들.

- **임베딩** = 토큰(음절·단어·조각) 하나의 n차원 좌표를 표에서 찾는 것. 숫자 하나하나에 뜻은 없고, 모여서 위치를 나타냄. 비슷한 뜻은 가까이 있음(§9 실측: Monday↔Tuesday).
- **attention** = 앞 토큰들 좌표를 보고 관련 있는 쪽으로 내 좌표를 옮기는 것. Q=검색어, K=이름표, V=건네줄 내용. 좌표 수는 줄지 않음(6토큰 → 6좌표). 각 자리 좌표는 "처음부터 여기까지"의 뜻을 담게 됨. 순서대로 이어받는 게 아니라 **동시에** 계산.
- **FFN** = 각 자리가 혼자 "이 뜻이면 이런 게 떠오른다"를 더하는 것. 규칙 = (감지기 좌표, 밀어낼 방향) × 수천 개/층.
- **층** = attention + FFN 한 쌍. 모델에 정의된 층 수만큼 통과(GPT-2 12, Llama-8B 32). 층마다 행렬 숫자가 다름. 층 수는 고정, 중간 탈출(early exit)은 일반적으로 안 함.
- **출구** = 마지막 자리 좌표를 단어장 전체와 내적 → 점수 → 확률 → 토큰 하나. 임베딩의 역순(정확히 찾기가 아니라 가장 가까운 것 찾기).
- **바퀴** = 토큰 하나 만들기. 1바퀴(prefill)는 입력 전체, 이후(decode)는 새 토큰 1개만. 바퀴 수는 `<끝>` 토큰이 나올 때까지라 미리 모름(최대 토큰 수가 안전장치).
- **KV cache** = 앞 토큰들의 K, V를 층마다 저장한 메모. 한 요청(답변) 동안 바퀴를 넘어 유지, 끝나면 비움. 앞 토큰은 뒤를 못 보므로(causal mask) K, V가 안 바뀌어 재사용 가능.
- **학습 vs 추론** = 학습은 "틀린 만큼 정답 쪽으로 숫자를 조금씩 고치기"(경사하강법, 브루트포스 아님)로 행렬을 만드는 것 = 수치해석 방법을 만드는 것. 추론은 고정된 행렬로 정해진 층을 통과해 값을 추정하는 것.
- **모델 파일** = 학습으로 생긴 규칙(행렬 숫자)의 저장본. 계산 순서·방식(아키텍처)은 코드/설정에 따로 있음 → ONNX는 둘을 한 파일에 담음.
- **도구 호출** = `<tool_call>`도 단어장의 토큰. 부를지·인자는 모델이 토큰으로 생성(확률적), 실행은 모델 밖 앱/에이전트 층. 인자(위치 등)는 문맥에서 attention으로 베껴 씀 → 문맥에 없으면 되묻거나 지어낼 위험.
- **런타임 두 층** = ① 추론 엔진(PyTorch, ONNX Runtime, vLLM: 행렬 계산, KV cache, 칩 배치 — 이 스터디의 스케줄러 위치) / ② 앱·에이전트(Claude Code 등: 문서 조립, 도구 실행, 루프).
- **RAG** = 관련 문서를 벡터 검색해 문맥에 붙여 주는 것(도구 결과 붙이기와 같은 방식).
- **똑똑함의 차이** = 뼈대는 거의 같고, weight(크기·데이터·학습량·후처리)와 tokenizer가 가름.

## 8. 한 줄 요약

> LLM = **거대한 read-only weight** + **토큰 수만큼 커지는 KV cache** 위에서, MatMul 위주의 **같은 block을 N번 반복하는 DAG**를 **토큰 하나 생성할 때마다 한 번** 실행하는 프로그램.

## 완료 기준 체크 (Phase 1)

- [x] Transformer block을 operator 단위로 그릴 수 있다 — §4 개념도 + §9 NumPy 실구현
- [x] "모델을 쪼갠다"가 layer/subgraph/tensor 중 무엇인지 구분할 수 있다 — §7
- [x] prefill과 decode의 실행 특성이 왜 다른지 설명할 수 있다 — §6 개념 + §9.2~9.3 실측
