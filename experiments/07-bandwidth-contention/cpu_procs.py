"""메모리 대역폭 경합: memory-bound 작업(큰 배열 Add)을 프로세스 N개로 동시에 돌리면
합계 대역폭이 N배가 되는가, 아니면 천장(DRAM 대역폭)에서 멈추는가?

실행: .venv/bin/python experiments/07-bandwidth-contention/cpu_procs.py
"""
import multiprocessing as mp
import time

import numpy as np

N_ELEM = 16 * 1024 * 1024  # 64 MiB per array (fp32) → 캐시보다 훨씬 큼
DURATION = 2.0


def worker(start_evt, q):
    x = np.ones(N_ELEM, np.float32)
    y = np.ones(N_ELEM, np.float32)
    z = np.empty_like(x)
    for _ in range(3):
        np.add(x, y, out=z)  # warmup
    start_evt.wait()
    t0 = time.perf_counter()
    n = 0
    while time.perf_counter() - t0 < DURATION:
        np.add(x, y, out=z)
        n += 1
    q.put(n * 3 * x.nbytes / (time.perf_counter() - t0))  # 읽기 2 + 쓰기 1


def run(nproc):
    ctx = mp.get_context("spawn")
    evt, q = ctx.Event(), ctx.Queue()
    ps = [ctx.Process(target=worker, args=(evt, q)) for _ in range(nproc)]
    for p in ps:
        p.start()
    time.sleep(1.5)  # 모든 프로세스 준비 대기
    evt.set()
    bw = [q.get() for _ in ps]
    for p in ps:
        p.join()
    return bw


if __name__ == "__main__":
    print(f"{'procs':>5s} {'합계 GB/s':>10s} {'프로세스당 GB/s':>16s}")
    for n in (1, 2, 4, 8):
        bw = run(n)
        print(f"{n:5d} {sum(bw)/1e9:10.1f} {sum(bw)/n/1e9:16.1f}")
