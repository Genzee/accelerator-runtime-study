"""GPT-2 small(124M)을 NumPy만으로 실행한다. (PyTorch/transformers 없음)

준비:  models/gpt2/model.safetensors  (huggingface openai-community/gpt2)
실행:  .venv/bin/python experiments/02-transformer-block/gpt2_numpy.py [--unaligned]

하는 일
  1) trace   : 한 번의 forward에서 tensor shape/bytes가 어떻게 흐르는지 출력
  2) generate: KV cache 사용 — prefill 1번 + decode N번, 실제 문장 생성
  3) no-cache: KV cache 없이 매 토큰마다 전체 문장을 다시 계산 → 비교
  4) profile : decode 1 step에서 연산 종류별 시간 비중
결과는 results_gpt2.json에 저장.
"""

import json
import struct
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import tiktoken

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = ROOT / "models/gpt2/model.safetensors"
ALIGNED = "--unaligned" not in sys.argv
OUT = Path(__file__).parent / ("results_gpt2.json" if ALIGNED else "results_gpt2_unaligned.json")

N_LAYER, N_HEAD, D = 12, 12, 768
HD = D // N_HEAD  # head 하나의 차원 = 64


# ---------------------------------------------------------------------------
# 0. weight 로딩: safetensors = [8B 길이][JSON 목차][숫자 덩어리]  → 이름으로 꺼내기
# ---------------------------------------------------------------------------

def load_safetensors(path, aligned=True):
    """aligned=False면 파일 버퍼를 그대로 참조 → header 길이 때문에 주소가 4의 배수가 아님(misaligned)
    → NumPy가 BLAS를 못 쓰고 느린 경로로 감. 기본은 정렬된 메모리로 복사."""
    raw = path.read_bytes()
    (hlen,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8:8 + hlen])
    header.pop("__metadata__", None)
    base = 8 + hlen
    W = {}
    for name, info in header.items():
        if name.endswith(".attn.bias"):  # causal mask 버퍼 — 직접 만들어 쓰므로 불필요
            continue
        s, e = info["data_offsets"]
        assert info["dtype"] == "F32"
        W[name] = np.frombuffer(raw, dtype=np.float32, count=(e - s) // 4, offset=base + s).reshape(info["shape"])
        if aligned:
            W[name] = W[name].copy()  # 새로 할당된 버퍼는 정렬돼 있음
    return W


# ---------------------------------------------------------------------------
# 1. 연산들 (각각이 ONNX에서 보게 될 operator 하나 또는 몇 개)
# ---------------------------------------------------------------------------

PROF = defaultdict(float)  # 연산 종류별 누적 시간
PROFILING = False


def timed(kind):
    def deco(fn):
        def wrap(*a, **k):
            if not PROFILING:
                return fn(*a, **k)
            t0 = time.perf_counter()
            r = fn(*a, **k)
            PROF[kind] += time.perf_counter() - t0
            return r
        return wrap
    return deco


@timed("LayerNorm")
def layer_norm(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g + b


@timed("GELU")
def gelu(x):
    return 0.5 * x * (1 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))


@timed("MatMul: QKV projection")
def mm_qkv(x, w, b):
    return x @ w + b


@timed("MatMul: attn output (Wo)")
def mm_wo(x, w, b):
    return x @ w + b


@timed("MatMul: FFN up (768→3072)")
def mm_ffn_up(x, w, b):
    return x @ w + b


@timed("MatMul: FFN down (3072→768)")
def mm_ffn_down(x, w, b):
    return x @ w + b


@timed("MatMul: lm_head (768→50257)")
def mm_lm_head(x, wte):
    return x @ wte.T


@timed("Attention core (QKᵀ, softmax, ·V)")
def attention_core(q, K, V, past_len):
    # q: [H, T, 64], K/V: [H, P+T, 64]
    T, total = q.shape[1], K.shape[1]
    # 주의: np.sqrt(HD)는 float64 스칼라 → 나누면 이후 전부 float64로 승격돼 느려짐. Python float(HD**0.5)는 dtype 유지
    scores = q @ K.transpose(0, 2, 1) / HD ** 0.5                   # [H, T, P+T]
    mask = np.triu(np.ones((T, total), dtype=bool), k=past_len + 1)  # 미래 토큰 가리기
    scores = np.where(mask, -1e10, scores)
    scores = scores - scores.max(-1, keepdims=True)
    w = np.exp(scores)
    w /= w.sum(-1, keepdims=True)                                 # softmax
    return w @ V                                                  # [H, T, 64]


@timed("Reshape/concat (KV cache 등)")
def split_heads(x):
    T = x.shape[0]
    return x.reshape(T, N_HEAD, HD).transpose(1, 0, 2)  # [T,768] → [12, T, 64]


@timed("Reshape/concat (KV cache 등)")
def merge_heads(x):
    return x.transpose(1, 0, 2).reshape(x.shape[1], D)  # [12, T, 64] → [T, 768]


@timed("Reshape/concat (KV cache 등)")
def cache_append(cache, new):
    return new if cache is None else np.concatenate([cache, new], axis=1)


# ---------------------------------------------------------------------------
# 2. block 하나 / 모델 전체
# ---------------------------------------------------------------------------

def block(x, W, i, cache, trace=None):
    p = f"h.{i}."
    past_len = 0 if cache is None or cache["k"][i] is None else cache["k"][i].shape[1]

    h = layer_norm(x, W[p + "ln_1.weight"], W[p + "ln_1.bias"])
    qkv = mm_qkv(h, W[p + "attn.c_attn.weight"], W[p + "attn.c_attn.bias"])   # [T, 2304]
    q, k, v = np.split(qkv, 3, axis=-1)                                         # 각 [T, 768]
    q, k, v = split_heads(q), split_heads(k), split_heads(v)                    # 각 [12, T, 64]
    if cache is not None:
        cache["k"][i] = k = cache_append(cache["k"][i], k)                     # [12, P+T, 64]
        cache["v"][i] = v = cache_append(cache["v"][i], v)
    a = attention_core(q, k, v, past_len)
    a = mm_wo(merge_heads(a), W[p + "attn.c_proj.weight"], W[p + "attn.c_proj.bias"])
    x = x + a                                                                   # residual

    h = layer_norm(x, W[p + "ln_2.weight"], W[p + "ln_2.bias"])
    f = mm_ffn_up(h, W[p + "mlp.c_fc.weight"], W[p + "mlp.c_fc.bias"])          # [T, 3072]
    f_up = f = gelu(f)
    f = mm_ffn_down(f, W[p + "mlp.c_proj.weight"], W[p + "mlp.c_proj.bias"])   # [T, 768]
    x = x + f                                                                   # residual

    if trace is not None:
        trace += [("  ln_1 → c_attn (QKV)", qkv), ("  Q (12 heads)", q), ("  K (cache 포함)", k),
                  ("  attention 출력", a), ("  FFN up + GELU", f_up),
                  ("  FFN down", f), ("  block 출력 x", x)]
    return x


def new_cache():
    return {"k": [None] * N_LAYER, "v": [None] * N_LAYER}


def forward(ids, W, cache, start_pos=0, trace=None):
    """ids: 이번에 새로 넣는 토큰들. cache에 이전 토큰의 K/V가 있으면 이어서 계산."""
    pos = np.arange(start_pos, start_pos + len(ids))
    x = W["wte.weight"][ids] + W["wpe.weight"][pos]                  # embedding (Gather)
    if trace is not None:
        trace.append(("embedding (wte[ids] + wpe[pos])", x))
    for i in range(N_LAYER):
        x = block(x, W, i, cache, trace if (trace is not None and i == 0) else None)
        if trace is not None and i == 5:
            trace.append(("── block 5 → 6 경계 (칩을 나눈다면 여기서 이동) ──", x))
    assert x.dtype == np.float32, f"dtype 승격 발생: {x.dtype}"
    x = layer_norm(x[-1:], W["ln_f.weight"], W["ln_f.bias"])         # 다음 토큰 예측엔 마지막 위치만 필요
    logits = mm_lm_head(x, W["wte.weight"])                          # [1, 50257]
    if trace is not None:
        trace.append(("logits (마지막 토큰만)", logits))
    return logits[0]


# ---------------------------------------------------------------------------
# 3. 실험
# ---------------------------------------------------------------------------

def human(n):
    return f"{n/1024:.1f} KiB" if n < 2**20 else f"{n/2**20:.2f} MiB"


def exp_trace(W, enc, prompt):
    print("=" * 70 + "\n1) trace: 한 번의 forward에서 tensor가 흐르는 모습\n" + "=" * 70)
    ids = enc.encode(prompt)
    print(f"입력: {prompt!r}\ntoken ids: {ids}  → S={len(ids)}\n")
    trace = []
    logits = forward(ids, W, new_cache(), trace=trace)
    print("(block 0 내부만 자세히, block 1~11은 같은 구조 반복)")
    for name, t in trace:
        print(f"  {name:48s} shape={str(list(t.shape)):16s} {human(t.nbytes):>10s}")
    top = np.argsort(logits)[::-1][:5]
    p = np.exp(logits - logits.max()); p /= p.sum()
    print("\n다음 토큰 후보 top-5:")
    for t in top:
        print(f"  {enc.decode([t])!r:12s} p={p[t]:.3f}")
    return {"prompt": prompt, "ids": ids, "trace": [(n, list(t.shape), t.nbytes) for n, t in trace]}


def exp_generate(W, enc, prompt, n_new):
    print("\n" + "=" * 70 + f"\n2) 생성 (KV cache 사용): prefill 1번 + decode {n_new}번\n" + "=" * 70)
    ids = enc.encode(prompt)
    cache = new_cache()
    t0 = time.perf_counter()
    logits = forward(ids, W, cache)
    t_prefill = time.perf_counter() - t0
    out = [int(np.argmax(logits))]  # greedy: 가장 확률 높은 토큰
    step_times = []
    for _ in range(n_new - 1):
        t0 = time.perf_counter()
        logits = forward([out[-1]], W, cache, start_pos=len(ids) + len(out) - 1)
        step_times.append(time.perf_counter() - t0)
        out.append(int(np.argmax(logits)))
    text = enc.decode(out)
    kv_bytes = sum(k.nbytes + v.nbytes for k, v in zip(cache["k"], cache["v"]))
    n_tok = cache["k"][0].shape[1]
    dec_ms = np.median(step_times) * 1e3
    print(f"출력: {prompt}\033[1m{text}\033[0m\n")
    print(f"  prefill  : 입력 {len(ids)}토큰을 한 번에  → {t_prefill*1e3:7.2f} ms  (토큰당 {t_prefill*1e3/len(ids):.2f} ms)")
    print(f"  decode   : 1토큰씩 {len(step_times)}번  → 토큰당 p50 {dec_ms:7.2f} ms  (= {1000/dec_ms:.0f} tok/s)")
    print(f"  KV cache : 토큰 {n_tok}개 × 12 layer × (K+V) = {human(kv_bytes)}  (토큰당 {human(kv_bytes // n_tok)})")
    return {"prompt": prompt, "output": text, "prefill_tokens": len(ids), "prefill_ms": t_prefill * 1e3,
            "decode_p50_ms": dec_ms, "decode_steps_ms": [t * 1e3 for t in step_times],
            "kv_cache_bytes": kv_bytes, "kv_tokens": n_tok}


def exp_no_cache(W, enc, prompt, n_new):
    print("\n" + "=" * 70 + f"\n3) KV cache 없이: 매 토큰마다 지금까지의 문장 전체를 다시 계산\n" + "=" * 70)
    ids = enc.encode(prompt)
    seq = list(ids)
    step_times = []
    for _ in range(n_new):
        t0 = time.perf_counter()
        logits = forward(seq, W, cache=None)
        step_times.append(time.perf_counter() - t0)
        seq.append(int(np.argmax(logits)))
    text = enc.decode(seq[len(ids):])
    for i in (0, n_new // 4, n_new // 2, 3 * n_new // 4, n_new - 1):
        print(f"  {i+1:3d}번째 토큰: 문장 길이 {len(ids)+i:3d} 전체 재계산 → {step_times[i]*1e3:7.2f} ms")
    print(f"  합계 {sum(step_times)*1e3:.0f} ms   (출력 동일 여부: 아래에서 비교)")
    return {"output": text, "steps_ms": [t * 1e3 for t in step_times]}


def exp_profile(W, enc, prompt, n_steps=30):
    global PROFILING
    print("\n" + "=" * 70 + "\n4) decode 1 step에서 연산 종류별 시간 비중\n" + "=" * 70)
    ids = enc.encode(prompt)
    cache = new_cache()
    logits = forward(ids, W, cache)
    nxt = int(np.argmax(logits))
    PROF.clear()
    PROFILING = True
    t0 = time.perf_counter()
    for s in range(n_steps):
        logits = forward([nxt], W, cache, start_pos=len(ids) + s)
        nxt = int(np.argmax(logits))
    wall = time.perf_counter() - t0
    PROFILING = False
    measured = sum(PROF.values())
    rows = sorted(PROF.items(), key=lambda kv: -kv[1])
    for k, v in rows:
        print(f"  {k:36s} {v/n_steps*1e3:7.3f} ms/step  {v/wall*100:5.1f}%  " + "█" * int(v / wall * 50))
    other = wall - measured
    print(f"  {'(기타: Python 루프, embedding, 잔여)':36s} {other/n_steps*1e3:7.3f} ms/step  {other/wall*100:5.1f}%")

    # [계산] decode 1 step에 읽어야 하는 weight bytes → 대역폭 한계
    w_bytes = sum(a.nbytes for n, a in W.items() if n != "wpe.weight")
    bw = 90e9  # experiments/07에서 측정한 M4 실효 대역폭
    print(f"\n  [계산] decode 1 step이 읽는 weight ≈ {human(w_bytes)} (wte는 lm_head로 전체를 읽음)")
    print(f"         → 대역폭 90 GB/s 기준 하한 ≈ {w_bytes/bw*1e3:.2f} ms/step  (측정 {wall/n_steps*1e3:.2f} ms/step)")
    return {"per_step_ms": {k: v / n_steps * 1e3 for k, v in rows}, "other_ms": other / n_steps * 1e3,
            "wall_ms_per_step": wall / n_steps * 1e3, "weight_bytes": w_bytes}


def exp_prefill_sweep(W, enc):
    print("\n" + "=" * 70 + "\n5) 입력 길이별 prefill 시간 (= KV cache 없을 때 토큰 1개 만들 때마다 드는 비용)\n" + "=" * 70)
    base = enc.encode("The quick brown fox jumps over the lazy dog. " * 120)
    rows = []
    for L in (1, 4, 16, 64, 256, 1000):
        ids = base[:L]
        forward(ids, W, new_cache())  # warmup
        ts = []
        for _ in range(5):
            t0 = time.perf_counter()
            forward(ids, W, new_cache())
            ts.append(time.perf_counter() - t0)
        t = float(np.median(ts))
        rows.append({"tokens": L, "ms": t * 1e3, "ms_per_token": t * 1e3 / L})
        print(f"  입력 {L:5d}토큰 → {t*1e3:8.2f} ms   토큰당 {t*1e3/L:7.3f} ms")
    return rows


def main():
    if not WEIGHTS.exists():
        raise SystemExit(f"weight 파일 없음: {WEIGHTS}")
    t0 = time.perf_counter()
    W = load_safetensors(WEIGHTS, aligned=ALIGNED)
    print(f"[aligned={ALIGNED}] weight 로딩: {len(W)} tensors, {human(sum(a.nbytes for a in W.values()))}, {time.perf_counter()-t0:.2f}s\n")
    enc = tiktoken.get_encoding("gpt2")

    prompt = "The weather today is"
    res = {"trace": exp_trace(W, enc, prompt)}
    res["generate"] = exp_generate(W, enc, prompt, n_new=40)
    res["no_cache"] = exp_no_cache(W, enc, prompt, n_new=40)
    same = res["generate"]["output"] == res["no_cache"]["output"]
    print(f"\n  KV cache 사용/미사용 출력 동일: {same}")
    res["same_output"] = same
    c, n = res["generate"], res["no_cache"]
    total_cache = c["prefill_ms"] + sum(c["decode_steps_ms"])
    print(f"  총 시간: cache 사용 {total_cache:.0f} ms vs 미사용 {sum(n['steps_ms']):.0f} ms")

    long_prompt = ("In a small village near the mountains, there lived an old clockmaker who repaired "
                   "every clock in town. One winter morning, a stranger arrived carrying a broken watch that")
    res["profile"] = exp_profile(W, enc, long_prompt)
    res["prefill_sweep"] = exp_prefill_sweep(W, enc)
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\nsaved: {OUT}")


if __name__ == "__main__":
    main()
