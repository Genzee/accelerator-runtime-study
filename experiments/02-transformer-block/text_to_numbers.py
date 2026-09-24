"""텍스트가 숫자가 되어 layer를 하나씩 지나 다음 단어가 되는 과정을 실제 값으로 따라간다.

실행: cd experiments/02-transformer-block && ../../.venv/bin/python text_to_numbers.py

logit lens: 중간 layer의 벡터를 바로 lm_head에 넣어 "이 시점에 모델이 떠올리는 다음 단어"를 엿보는 기법.
(마지막 layer용으로 학습된 출구를 중간에 쓰는 것이라 근사적인 관찰이다.)
"""
import numpy as np
import tiktoken

from gpt2_numpy import N_LAYER, WEIGHTS, block, layer_norm, load_safetensors, new_cache

np.set_printoptions(precision=3, suppress=True)
W = load_safetensors(WEIGHTS)
enc = tiktoken.get_encoding("gpt2")
prompt = "The Eiffel Tower is located in the city of"


def top(vec, k=3):
    h = layer_norm(vec, W["ln_f.weight"], W["ln_f.bias"])
    logits = (h @ W["wte.weight"].T)[0]
    p = np.exp(logits - logits.max()); p /= p.sum()
    idx = np.argsort(p)[::-1][:k]
    return "  ".join(f"{enc.decode([int(i)])!r} {p[i]*100:4.1f}%" for i in idx)


print("① 텍스트 → 바이트 (컴퓨터에 저장된 그대로)")
print(f"   {prompt!r}\n   → {prompt.encode()[:24].hex(' ')} ...  ({len(prompt.encode())} bytes)")
print(f"   (한글이면: '날씨' → {'날씨'.encode().hex(' ')}  = 글자당 3 bytes)\n")

ids = enc.encode(prompt)
print("② 바이트 → 토큰 → 번호 (tokenizer: 자주 붙어 다니는 바이트 묶음을 사전에서 찾기)")
for t in ids:
    print(f"   {enc.decode([t])!r:12s} → id {t}")
print()

print("③ 번호 → 벡터 (embedding 표에서 행 꺼내기 + 위치 벡터 더하기)   [768개 중 앞 6개만 표시]")
x = W["wte.weight"][ids] + W["wpe.weight"][np.arange(len(ids))]
for i, t in enumerate(ids[:3]):
    print(f"   {enc.decode([t])!r:10s} wte[{t:5d}] = {W['wte.weight'][t][:6]}  + wpe[{i}] = {W['wpe.weight'][i][:6]}")
print("   ...")
print(f"   → x shape {list(x.shape)}  ({x.nbytes} bytes)\n")

print("④ layer를 하나씩 통과 — 마지막 위치(' of')의 벡터 변화와, 그 시점에 떠올리는 다음 단어(logit lens)")
print(f"   {'':9s} {'앞 4개 값':34s} {'크기':>7s}   떠올리는 다음 단어 top-3")
print(f"   {'입력':9s} {str(x[-1][:4]):34s} {np.linalg.norm(x[-1]):7.1f}   {top(x[-1:])}")
cache = new_cache()
for i in range(N_LAYER):
    x = block(x, W, i, cache)
    print(f"   block {i:2d}  {str(x[-1][:4]):34s} {np.linalg.norm(x[-1]):7.1f}   {top(x[-1:])}")

print("\n⑤ 마지막 벡터 → 단어 5만 개와 내적 → 확률 → 1등 선택 → 텍스트로")
print(f"   최종: {top(x[-1:], 5)}")
