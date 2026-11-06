#!/usr/bin/env python3
"""Cost of building the sorted token index list on device: top-k of [4 kv heads][ctx] scores -> sorted indices.
run: flock /tmp/gpu-engine.lock /media/fatman/data/workspace/qwen-warp/.venv/bin/python3 topk_cost.py
methods: A topk(sorted=False)+sort of the indices; B kthvalue threshold + (scores >= thr) mask + nonzero (indices come out sorted); C topk(sorted=True) only (lower bound of A)."""
import json
import torch

dev = "cuda"
out = []


def bench(fn, n=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(n):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / n * 1e3  # us per call (eager launches: includes python launch overhead only if GPU idle; many small kernels)


for ctx in (130816, 32512):
    sc = torch.randn(4, ctx, device=dev)
    for k in (1024, 2048, 4096, 8192, 16384):
        if k > ctx // 4:
            continue

        def A():
            v, i = torch.topk(sc, k, dim=1, sorted=False)
            return torch.sort(i, dim=1)[0]

        def B():
            thr = torch.kthvalue(sc, ctx - k + 1, dim=1, keepdim=True)[0]
            return torch.nonzero(sc >= thr)

        def C():
            return torch.topk(sc, k, dim=1, sorted=True)

        r = {"ctx": ctx, "k": k, "A_topk_unsorted_plus_sort_us": bench(A), "B_kthvalue_mask_nonzero_us": bench(B), "C_topk_sorted_only_us": bench(C)}
        print(r)
        out.append(r)
json.dump(out, open("/media/fatman/data/workspace/research/experiments/2026-11-06-sparsity-probes/results/kern/kern_topk_cost.json", "w"), indent=1)
