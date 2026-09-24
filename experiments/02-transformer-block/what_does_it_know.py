"""'연산만 하는데 어떻게 답이 나오나?' — 지식이 필요한 문장을 넣고 다음 토큰 확률을 본다.

실행: .venv/bin/python experiments/02-transformer-block/what_does_it_know.py
"""
import numpy as np
import tiktoken

from gpt2_numpy import WEIGHTS, forward, load_safetensors, new_cache

W = load_safetensors(WEIGHTS)
enc = tiktoken.get_encoding("gpt2")

PROMPTS = [
    "The capital of France is",
    "The capital of Japan is",
    "The Eiffel Tower is located in the city of",
    "Water boils at a temperature of 100 degrees",
    "Monday, Tuesday, Wednesday,",
    "1, 2, 3, 4,",
    "The CEO of Apple is",
    "The capital of Australia is",
]

for prompt in PROMPTS:
    logits = forward(enc.encode(prompt), W, new_cache())
    p = np.exp(logits - logits.max())
    p /= p.sum()
    top = np.argsort(p)[::-1][:5]
    cands = "  ".join(f"{enc.decode([t])!r} {p[t]*100:4.1f}%" for t in top)
    print(f"{prompt!r:48s} → {cands}")
