"""Phase 0 / Task 1-2: tensor bytes, MatMul FLOPs, 실측 latency, 작은 operator DAG.

실행:  .venv/bin/python experiments/01-matmul/matmul_basics.py
결과:  experiments/01-matmul/results.csv, results.json  (콘솔에도 출력)

원칙: "계산값(estimate)"과 "측정값(measured)"을 컬럼 이름으로 분리한다.
"""

import csv
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# 1. Tensor = (dtype, shape, strides, data pointer)
# ---------------------------------------------------------------------------

def tensor_bytes(shape, dtype):
    """logical bytes = numel * itemsize. padding/정렬/메타데이터는 제외."""
    numel = 1
    for d in shape:
        numel *= d
    return numel * np.dtype(dtype).itemsize


def human(nbytes):
    for unit in ("B", "KiB", "MiB", "GiB"):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TiB"


def section_tensor():
    print("=== 1. [B,S,H] activation tensor bytes (계산값) ===")
    for B, S, H in [(1, 128, 768), (1, 2048, 4096), (8, 2048, 4096), (32, 4096, 8192)]:
        row = [f"[{B},{S},{H}]"]
        for dt in ("float32", "float16"):
            row.append(f"{dt}={human(tensor_bytes((B, S, H), dt))}")
        print("  ", "  ".join(row))

    print("\n=== 2. stride / contiguous (측정값: numpy 메타데이터) ===")
    x = np.zeros((2, 3, 4), dtype=np.float32)
    xt = x.transpose(0, 2, 1)  # 데이터 복사 없음, stride만 바뀜
    for name, t in [("x", x), ("x.transpose(0,2,1)", xt), ("ascontiguousarray", np.ascontiguousarray(xt))]:
        print(f"   {name:22s} shape={t.shape} strides(bytes)={t.strides} "
              f"C-contiguous={t.flags['C_CONTIGUOUS']} shares_memory_with_x={np.shares_memory(t, x)}")
    print()


# ---------------------------------------------------------------------------
# 2. MatMul: A[M,K] @ B[K,N] -> C[M,N]
# ---------------------------------------------------------------------------

def matmul_estimate(M, K, N, dtype):
    isz = np.dtype(dtype).itemsize
    a, b, c = M * K * isz, K * N * isz, M * N * isz
    flops = 2 * M * N * K  # 출력 원소 M*N개 × (곱 K + 덧셈 K)
    return {
        "input_bytes": a, "weight_bytes": b, "output_bytes": c,
        "total_bytes": a + b + c,
        "flops": flops,
        # arithmetic intensity (이상적: 각 tensor를 DRAM에서 정확히 1번씩만 읽고/씀)
        "ai_flop_per_byte": flops / (a + b + c),
    }


def bench(fn, warmup=3, min_repeat=10, min_time=0.3):
    for _ in range(warmup):
        fn()
    times = []
    t_start = time.perf_counter()
    while len(times) < min_repeat or (time.perf_counter() - t_start) < min_time:
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
        if len(times) >= 2000:
            break
    times.sort()
    return {
        "n": len(times),
        "p50_us": statistics.median(times) * 1e6,
        "p95_us": times[int(len(times) * 0.95) - 1] * 1e6 if len(times) >= 20 else times[-1] * 1e6,
        "min_us": times[0] * 1e6,
    }


SHAPES = [
    # (label, M, K, N)
    ("decode-like  (M=1)", 1, 4096, 4096),
    ("small square", 128, 128, 128),
    ("mid square", 512, 512, 512),
    ("big square", 2048, 2048, 2048),
    ("prefill-like (M=2048)", 2048, 4096, 4096),
    ("skinny K", 2048, 64, 2048),
]
DTYPES = ["float32", "float64", "float16"]


def section_matmul():
    rng = np.random.default_rng(0)
    rows = []
    print("=== 3. MatMul 계산값 vs 측정값 ===")
    hdr = f"{'shape':24s} {'dtype':8s} {'in':>9s} {'w':>9s} {'out':>9s} {'GFLOP':>8s} {'AI':>7s} {'p50 us':>11s} {'GFLOP/s':>9s}"
    print(hdr)
    for label, M, K, N in SHAPES:
        for dt in DTYPES:
            # float16 big shape는 numpy가 BLAS를 못 써서 매우 느림 → 큰 것은 생략
            if dt == "float16" and M * K * N > 512 ** 3:
                continue
            A = rng.standard_normal((M, K)).astype(dt)
            B = rng.standard_normal((K, N)).astype(dt)
            C = np.empty((M, N), dtype=dt)
            est = matmul_estimate(M, K, N, dt)
            meas = bench(lambda: np.matmul(A, B, out=C))
            gflops = est["flops"] / (meas["p50_us"] * 1e-6) / 1e9
            row = {"label": label.strip(), "M": M, "K": K, "N": N, "dtype": dt,
                   **{f"est_{k}": v for k, v in est.items()},
                   **{f"meas_{k}": v for k, v in meas.items()},
                   "meas_gflops_per_s": gflops}
            rows.append(row)
            print(f"{label:24s} {dt:8s} {human(est['input_bytes']):>9s} {human(est['weight_bytes']):>9s} "
                  f"{human(est['output_bytes']):>9s} {est['flops']/1e9:8.3f} {est['ai_flop_per_byte']:7.1f} "
                  f"{meas['p50_us']:11.1f} {gflops:9.1f}")
    print()
    return rows


def section_add_baseline():
    """비교용: elementwise Add. FLOP은 적고 bytes는 크다 → memory-bound 예고편."""
    rng = np.random.default_rng(1)
    rows = []
    print("=== 4. 비교: elementwise Add (같은 bytes, 훨씬 적은 FLOP) ===")
    for n in (2048 * 2048, 2048 * 4096):
        X = rng.standard_normal(n).astype(np.float32)
        Y = rng.standard_normal(n).astype(np.float32)
        Z = np.empty_like(X)
        nbytes = 3 * n * 4
        flops = n
        meas = bench(lambda: np.add(X, Y, out=Z))
        gbps = nbytes / (meas["p50_us"] * 1e-6) / 1e9
        rows.append({"label": f"add n={n}", "est_total_bytes": nbytes, "est_flops": flops,
                     "est_ai_flop_per_byte": flops / nbytes, **{f"meas_{k}": v for k, v in meas.items()},
                     "meas_effective_GBps": gbps})
        print(f"   add n={n:>9d} bytes={human(nbytes):>9s} AI={flops/nbytes:.3f} "
              f"p50={meas['p50_us']:9.1f} us  effective={gbps:6.1f} GB/s")
    print()
    return rows


# ---------------------------------------------------------------------------
# 3. 작은 operator DAG: node = op, edge = tensor dependency
# ---------------------------------------------------------------------------

class Node:
    def __init__(self, name, op, inputs, outputs, fn):
        self.name, self.op, self.inputs, self.outputs, self.fn = name, op, inputs, outputs, fn


def topo_sort(nodes, graph_inputs):
    """Kahn 알고리즘. 'input tensor가 모두 준비된 node'가 실행 가능."""
    ready = set(graph_inputs)
    order, pending = [], list(nodes)
    while pending:
        runnable = [n for n in pending if all(i in ready for i in n.inputs)]
        if not runnable:
            raise ValueError("cycle or missing input")
        for n in runnable:
            order.append(n)
            ready.update(n.outputs)
            pending.remove(n)
    return order


def section_dag():
    print("=== 5. 작은 DAG: y = softmax(relu(x @ W1 + b1) @ W2) ===")
    B, H, F = 4, 64, 256
    rng = np.random.default_rng(2)
    env = {  # graph inputs + initializers(weight)
        "x": rng.standard_normal((B, H)).astype(np.float32),
        "W1": rng.standard_normal((H, F)).astype(np.float32),
        "b1": rng.standard_normal((F,)).astype(np.float32),
        "W2": rng.standard_normal((F, H)).astype(np.float32),
    }
    initializers = {"W1", "b1", "W2"}

    def softmax(z):
        z = z - z.max(axis=-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=-1, keepdims=True)

    # 일부러 순서를 섞어서 선언 → topo sort가 dependency로 순서를 복원하는지 확인
    nodes = [
        Node("n4", "Softmax", ["t3"], ["y"], softmax),
        Node("n2", "Add", ["t1", "b1"], ["t2"], np.add),
        Node("n3a", "Relu", ["t2"], ["t2r"], lambda t: np.maximum(t, 0)),
        Node("n1", "MatMul", ["x", "W1"], ["t1"], np.matmul),
        Node("n3b", "MatMul", ["t2r", "W2"], ["t3"], np.matmul),
    ]
    order = topo_sort(nodes, env.keys())
    edges = []
    for n in order:
        out = n.fn(*[env[i] for i in n.inputs])
        env[n.outputs[0]] = out
        for i in n.inputs:
            producer = next((p.name for p in nodes if i in p.outputs), "graph_input" if i == "x" else "initializer")
            edges.append((producer, n.name, i, env[i].shape, env[i].nbytes, i in initializers))
        print(f"   {n.name:4s} {n.op:8s} {str(n.inputs):18s} -> {n.outputs[0]:4s} shape={out.shape}")
    print("\n   edges (producer -> consumer : tensor, shape, bytes, kind)")
    for src, dst, t, shp, nb, is_w in edges:
        kind = "weight" if is_w else "activation"
        print(f"   {src:12s} -> {dst:4s} : {t:4s} {str(shp):10s} {nb:7d} B  {kind}")
    act = sum(nb for *_, nb, w in edges if not w)
    wt = sum(nb for *_, nb, w in edges if w)
    print(f"\n   activation edge bytes 합={act} B, weight bytes 합={wt} B")
    print("   → 만약 n3a|n3b 사이에서 device를 바꾸면, 이동해야 하는 것은 't2r' 하나"
          f" ({env['t2r'].nbytes} B) + (W2가 상주하지 않았다면) W2")
    print()
    return {"order": [n.name for n in order],
            "edges": [dict(src=s, dst=d, tensor=t, shape=list(sh), bytes=nb, weight=w) for s, d, t, sh, nb, w in edges]}


def main():
    env_info = {"platform": platform.platform(), "machine": platform.machine(),
                "python": platform.python_version(), "numpy": np.__version__}
    print("env:", env_info, "\n")
    section_tensor()
    mm = section_matmul()
    add = section_add_baseline()
    dag = section_dag()

    with open(OUT_DIR / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(mm[0].keys()))
        w.writeheader()
        w.writerows(mm)
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump({"env": env_info, "matmul": mm, "add": add, "dag": dag}, f, indent=2)
    print(f"saved: {OUT_DIR/'results.csv'}, {OUT_DIR/'results.json'}")


if __name__ == "__main__":
    main()
