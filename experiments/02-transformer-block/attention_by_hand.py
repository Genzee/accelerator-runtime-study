"""Phase 1 워밍업: 토큰 3개 × 2차원 벡터로 attention 한 번을 손계산 수준으로 따라간다.

실행:  .venv/bin/python experiments/02-transformer-block/attention_by_hand.py

벡터의 두 축에 이해용 의미를 붙인다: [사람 관련도, 음식 관련도]
(실제 모델에서는 축 하나하나에 사람이 읽을 수 있는 의미가 없다.)
"""

import numpy as np

np.set_printoptions(precision=3, suppress=True)

TOKENS = ["나", "밥", "먹었다"]
X = np.array([
    [1.0, 0.0],   # 나     : 사람
    [0.0, 1.0],   # 밥     : 음식
    [1.0, 1.0],   # 먹었다 : 사람이 하는 + 음식 관련 동작
])
D = X.shape[1]


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)  # 수치 안정용. 결과는 동일
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def show_table(title, M, rows=TOKENS, cols=TOKENS):
    print(f"  {title}")
    print("  " + " " * 8 + "".join(f"{c:>9s}" for c in cols))
    for r, row in zip(rows, M):
        print(f"  {r:>6s}  " + "".join(f"{v:9.3f}" for v in row))


def attention(Wq, Wk, Wv, label):
    print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")

    # 1) 같은 벡터를 세 가지 역할로 변환
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    print("1) Q = X·Wq (찾는 것),  K = X·Wk (이름표),  V = X·Wv (넘겨줄 내용)")
    for t, q, k, v in zip(TOKENS, Q, K, V):
        print(f"  {t:>6s}  Q={q}  K={k}  V={v}")

    # 2) 관련도 표: 모든 토큰 쌍의 내적  → [S, S]
    scores = Q @ K.T
    print("\n2) score = Q·Kᵀ   (행=질문하는 토큰, 열=참고 대상 토큰)")
    show_table("", scores)

    # 3) 크기 보정: 차원이 커지면 내적이 커지므로 √D로 나눔
    scaled = scores / np.sqrt(D)
    print(f"\n3) ÷ √D (=√{D}={np.sqrt(D):.3f})")
    show_table("", scaled)

    # 4) causal mask: 미래 토큰은 못 보게 -inf
    mask = np.triu(np.ones((3, 3), dtype=bool), k=1)
    masked = np.where(mask, -np.inf, scaled)
    print("\n4) causal mask (자기보다 뒤 토큰 = -inf → softmax 후 0)")
    show_table("", masked)

    # 5) softmax: 행마다 합이 1인 비율로
    weights = softmax(masked)
    print("\n5) softmax → 참고 비율 (각 행 합 = 1)")
    show_table("", weights)

    # 6) 비율대로 V를 섞음 → 문맥이 반영된 새 벡터
    out = weights @ V
    print("\n6) out = 비율·V   (문맥이 섞인 새 벡터)   [사람, 음식]")
    for t, x, o in zip(TOKENS, X, out):
        print(f"  {t:>6s}  원래 {x}  →  attention 출력 {o}")
    return weights, out


I = np.eye(D)

# A: 아무 변환 없음 (Q=K=V=X). "비슷한 것끼리 서로 참고" 만 일어남
wA, oA = attention(I, I, I, "A. Wq=Wk=Wv=I  (변환 없음)")

# B: Wq가 '음식' 축을 찾도록 학습됐다고 가정 → 모든 토큰의 질문이 "음식 어디 있어?"가 됨
# 사람축·음식축 입력 모두를 → 출력의 '음식' 칸으로 보냄
Wq_food = np.array([[0.0, 3.0],
                    [0.0, 3.0]])
wB, oB = attention(Wq_food, I, I, "B. Wq = '음식을 찾는 질문'으로 학습됐다고 가정")

print(f"\n{'=' * 60}\n비교: '먹었다'가 누구를 참고했나\n{'=' * 60}")
for name, w, o in [("A 변환 없음", wA, oA), ("B 음식 질문", wB, oB)]:
    print(f"  {name}:  나 {w[2,0]:.3f}  밥 {w[2,1]:.3f}  먹었다 {w[2,2]:.3f}   → 출력 {o[2]}")

print(f"\n{'=' * 60}\n스케줄러 관점: score 표는 [S, S] → 토큰 수의 제곱으로 커진다 [계산]\n{'=' * 60}")
heads, itemsize = 32, 2  # Llama-3-8B: 32 heads, bf16
for S in (3, 2048, 8192, 32768):
    b = S * S * heads * itemsize
    print(f"  S={S:>6d}  layer 하나 score 표 = {S}×{S}×{heads}heads×{itemsize}B = {b / 2**20:12.3f} MiB")
print("  (FlashAttention 같은 kernel은 이 표를 메모리에 통째로 만들지 않고 조각내서 계산한다)")
