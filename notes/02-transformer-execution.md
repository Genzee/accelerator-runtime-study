# 02 — Transformer / LLM을 실행 관점에서 이해

> Phase 1. 진행 중.
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

난이도·위험: A < C < B(layer) < B(phase) < B(tensor). 그래서 A부터 검증하고 B로 간다.

## 8. 한 줄 요약

> LLM = **거대한 read-only weight** + **토큰 수만큼 커지는 KV cache** 위에서, MatMul 위주의 **같은 block을 N번 반복하는 DAG**를 **토큰 하나 생성할 때마다 한 번** 실행하는 프로그램.

## 완료 기준 체크 (Phase 1)

- [ ] Transformer block을 operator 단위로 그릴 수 있다 — 개념도는 §4. PyTorch/NumPy 실구현으로 확인 필요
- [x] "모델을 쪼갠다"가 layer/subgraph/tensor 중 무엇인지 구분할 수 있다 — §7
- [ ] prefill과 decode의 실행 특성이 왜 다른지 설명할 수 있다 — 개념은 §6, KV cache on/off 실측 필요
