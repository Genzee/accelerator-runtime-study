"""모델 파일(safetensors) 안에 무엇이 들어 있나 — 500MB 전체가 아니라 앞쪽 목차(header)만 받아서 본다.

safetensors 형식:  [8바이트: header 길이] [JSON header: 이름→dtype,shape,byte위치] [weight 숫자들이 그냥 이어 붙은 덩어리]

실행: .venv/bin/python experiments/02-transformer-block/inspect_model_file.py
"""
import json
import struct
import urllib.request
from collections import defaultdict

URL = "https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors"


def fetch(start, end):
    req = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req) as r:
        return r.read()


(hlen,) = struct.unpack("<Q", fetch(0, 7))
header = json.loads(fetch(8, 8 + hlen - 1))
header.pop("__metadata__", None)
tensors = sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0])
isz = {"F32": 4, "F16": 2, "BF16": 2}

print(f"header 크기: {hlen} bytes, tensor 개수: {len(tensors)}\n")
ROLE = {
    "ln_1.weight": "LayerNorm 1 (scale)", "ln_1.bias": "LayerNorm 1 (shift)",
    "attn.c_attn.weight": "Wq|Wk|Wv 를 한 행렬로 붙인 것", "attn.c_attn.bias": "Q/K/V bias",
    "attn.c_proj.weight": "Wo (attention 출력 projection)", "attn.c_proj.bias": "Wo bias",
    "attn.bias": "causal mask (학습값 아님, 고정 삼각형 표)",
    "ln_2.weight": "LayerNorm 2 (scale)", "ln_2.bias": "LayerNorm 2 (shift)",
    "mlp.c_fc.weight": "FFN 1: 768 → 3072", "mlp.c_fc.bias": "FFN 1 bias",
    "mlp.c_proj.weight": "FFN 2: 3072 → 768", "mlp.c_proj.bias": "FFN 2 bias",
}
ORDER = list(ROLE)
by_name = dict(tensors)


def line(name):
    info = by_name[name]
    s, e = info["data_offsets"]
    role = ROLE.get(name.split(".", 2)[-1], "") if name.startswith("h.") else ""
    print(f"  {name:26s} {str(info['shape']):18s} {(e-s)/2**20:6.2f} MiB   파일 위치 {8+hlen+s:>11,d}   {role}")


print("=== 1) 모델 입구 ===")
line("wte.weight"); print("       └ 토큰 embedding 표: 단어 50257개 × 768 (lm_head도 이 표를 재사용)")
line("wpe.weight"); print("       └ 위치 embedding 표: 최대 1024번째 위치 × 768")
for L in (0, 11):
    print(f"\n=== 2) block {L} (실행 순서대로) ===")
    for suffix in ORDER:
        line(f"h.{L}.{suffix}")
    if L == 0:
        print("\n  ... h.1 ~ h.10 : h.0과 이름 규칙·shape 완전히 동일, 숫자만 다름 ...")
print("\n=== 3) 모델 출구 ===")
line("ln_f.weight"); line("ln_f.bias")
print("\n※ '파일 위치'를 보면 순서가 뒤죽박죽이다 → 파일은 그냥 {이름: 숫자덩어리} 사전이고,")
print("   실행 순서는 파일이 아니라 모델 코드(아키텍처)가 알고 있다.")

groups = defaultdict(int)
for name, info in tensors:
    key = ".".join(name.split(".")[:2]) if name.startswith("h.") else name.split(".")[0]
    n = 1
    for d in info["shape"]:
        n *= d
    groups[key] += n * isz[info["dtype"]]
print("\n=== 묶음별 크기 = '쪼갤 수 있는 단위' ===")
total = sum(groups.values())
for k in sorted(groups, key=lambda k: (not k.startswith("w"), int(k.split(".")[1]) if k.startswith("h.") else 99)):
    print(f"  {k:6s} {groups[k]/2**20:7.2f} MiB")
print(f"  합계   {total/2**20:7.2f} MiB")
