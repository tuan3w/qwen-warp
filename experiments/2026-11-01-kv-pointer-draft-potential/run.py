"""kv-pointer-draft-potential (E7): collect the model's own retrieval signals on E6's 32 greedy tasks.

  # private engine tree (git archive HEAD + probe patch), then with the GPU lease held:
  git archive HEAD engine oracle | tar -x -C ~/llm-lab/e7_engine
  python3 experiments/2026-11-01-kv-pointer-draft-potential/engine_patch/apply.py ~/llm-lab/e7_engine/engine && make -C ~/llm-lab/e7_engine/engine
  projects/llm-lab/bin/lab gpu --holder RunE7KvDraft acquire
  SMK_ROOT=~/llm-lab/e7_engine/engine ~/llm-lab/venv/bin/python experiments/2026-11-01-kv-pointer-draft-potential/run.py collect [--only a1,b2]
  ~/llm-lab/venv/bin/python experiments/2026-11-01-kv-pointer-draft-potential/run.py micro      # cosine-search / store micro-benchmarks (no engine)
  projects/llm-lab/bin/lab gpu --holder RunE7KvDraft release
  python3 experiments/2026-11-01-kv-pointer-draft-potential/analyze.py

`collect` teacher-forces prompt + the canonical greedy continuation of every E6 task (E6 trace.npz: F stream) through the private engine in T=4 steps (bf16 KV, head=2 = final norm only, no
logits).  Prompt steps run as CUDA graphs and only store hidden states; the decode region (rows P-1 .. P+N-2: row r = input token S[r], predicts S[r+1]; row P+i-1 is the one that produced
the anchor g_i of cycle i) runs as non-graph steps with the attention probe on (per layer / q head: top-4 key positions + softmax masses).  Hidden states (taps 16/32/48/60/65) of every
position are kept in bf16 on the GPU; R2 kNN is computed on the GPU per task (cosine on raw and prompt-mean-centred states, top-16 unconstrained and top-8 among positions whose next token
equals the anchor).  Outputs: runs/probe/<id>.npz (idx int16 [N,16,24,4], mass fp16 [N,16,24,6], kNN tables) and results/probe_check.json (probe vs real kernel).
"""
import argparse
import ctypes as C
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
E6 = ROOT / "experiments" / "2026-10-31-ngram-draft-potential"
ENGINE = Path(os.environ.get("SMK_ROOT", ROOT / "engine"))
OUT = Path(os.environ.get("E7_PROBE", HERE / "runs" / "probe"))
SLOTS = [16, 32, 48, 60, 65]  # debug taps: slot l+1 = residual stream after layer l, 65 = final norm out
KEY_AI = [7, 11, 15]  # attention layer indices (= layers 31, 47, 63) whose K cache serves as the key-like representation
H = 5120
NAT, NH = 16, 24


def load_tasks(only):
    sys.path.insert(0, str(E6))
    import analyze as A

    tasks = A.load()
    if only:
        tasks = [t for t in tasks if any(t["id"].startswith(o) for o in only)]
    return tasks


def setup_probe_api(lib):
    vp, ci = C.c_void_p, C.c_int
    lib.smk_probe_alloc.argtypes = [vp, ci]
    lib.smk_probe_cfg.argtypes = [vp, ci, ci, ci, ci]
    lib.smk_probe_read.argtypes = [vp, ci, vp, C.c_size_t]
    lib.smk_probe_store.argtypes = [vp, vp, ci, ci, ci, C.POINTER(ci), ci, vp]
    lib.smk_probe_copy_kv.argtypes = [vp, ci, ci, vp, vp]
    for n in ("smk_probe_alloc", "smk_probe_cfg", "smk_probe_read", "smk_probe_store", "smk_probe_copy_kv"):
        getattr(lib, n).restype = ci


def knn_tables(X, S1, P, N, mode):
    """X float32 [L, d] (positions 0..L-1, L = P+N-1), S1[j] = token at j+1 (int64 GPU).  Rows r = P-1..P+N-2 are the queries, anchor of row r = S[r+1] = S1[r].
    mode: 'raw' cosine of the raw states; 'c' centred on the prompt mean; 'z' centred and divided by the per-dimension std of the whole teacher-forced sequence (documented look-ahead of a
    global constant); 'n' every state L2-normalised first, then centred on the prompt mean."""
    import torch

    L = X.shape[0]
    if mode == "n":
        X = torch.nn.functional.normalize(X, dim=1)
    if mode in ("c", "z", "n"):
        X = X - X[:P].mean(0, keepdim=True)
    if mode == "z":
        X = X / (X.std(0, keepdim=True) + 1e-3 * X.std(0).mean())
    Xn = torch.nn.functional.normalize(X, dim=1).half()
    r = torch.arange(P - 1, P + N - 1, device=X.device)
    sims = (Xn[r] @ Xn.T).float()  # [N, L]
    j = torch.arange(L, device=X.device)[None, :]
    causal = j < r[:, None]  # j + 1 <= r: the neighbour's next token is a known token
    s_unc = sims.masked_fill(~causal, -2.0)
    vu, iu = s_unc.topk(16, dim=1)
    anchor = S1[r][:, None]
    ok = causal & (S1[None, :L] == anchor)
    s_msk = sims.masked_fill(~ok, -2.0)
    vm, im = s_msk.topk(8, dim=1)
    return dict(unc_idx=iu.short().cpu().numpy(), unc_sim=vu.half().cpu().numpy(), msk_idx=im.short().cpu().numpy(), msk_sim=vm.half().cpu().numpy())


def collect(only):
    import torch

    sys.path.insert(0, str(ENGINE))
    sys.path.insert(0, str(ENGINE / "tests"))
    from engine import Engine, smk

    tasks = load_tasks(only)
    cap = max(len(t["prompt"]) + len(t["g"]) for t in tasks) + 8
    ctx = cap + 64
    e = Engine(ctx_max=ctx, verbose=False, head="bf16")
    lib = e.lib
    setup_probe_api(lib)
    ROWS = max(len(t["g"]) for t in tasks) + 8
    assert lib.smk_probe_alloc(e.h, ROWS) == 0
    st = smk.stream()
    OUT.mkdir(parents=True, exist_ok=True)
    (HERE / "results").mkdir(exist_ok=True)
    dev = torch.device("cuda")
    Hs = torch.empty(len(SLOTS), cap, H, dtype=torch.bfloat16, device=dev)
    slots_c = (C.c_int * len(SLOTS))(*SLOTS)
    kv_buf = torch.empty(4, ctx, 256, dtype=torch.bfloat16, device=dev)
    kv_buf2 = torch.empty(4, ctx, 256, dtype=torch.bfloat16, device=dev)
    print(f"engine ready ctx_max {ctx}, free {torch.cuda.mem_get_info()[0] / 2**30:.2f} GiB", flush=True)
    chk = {}

    def store(pos, T):
        smk.check(lib.smk_probe_store(e.h, C.c_void_p(Hs.data_ptr()), pos, T, len(SLOTS), slots_c, cap, st), "store")

    def verify(ai, T, row_off, pos0):
        """torch recompute of the K2 output of attention layer ai from the engine's KV / q / gate vs the kernel's own output (snapshot), and of the probe's top-4 / masses."""
        torch.cuda.synchronize()
        q = torch.empty(NAT * 4 * NH * 256, dtype=torch.bfloat16)
        g = torch.empty_like(q)
        o = torch.empty(NAT * 4 * 6144, dtype=torch.bfloat16)
        smk.check(lib.smk_probe_read(e.h, 2, C.c_void_p(q.data_ptr()), q.numel() * 2), "r")
        smk.check(lib.smk_probe_read(e.h, 3, C.c_void_p(g.data_ptr()), g.numel() * 2), "r")
        smk.check(lib.smk_probe_read(e.h, 4, C.c_void_p(o.data_ptr()), o.numel() * 2), "r")
        q = q.view(NAT, 4, NH, 256)[ai, :T].to(dev).float()
        g = g.view(NAT, 4, NH, 256)[ai, :T].to(dev).float()
        o = o.view(NAT, 4, 6144)[ai, :T].to(dev).float()
        smk.check(lib.smk_probe_copy_kv(e.h, ai, 0, C.c_void_p(kv_buf.data_ptr()), st), "kv")
        smk.check(lib.smk_probe_copy_kv(e.h, ai, 1, C.c_void_p(kv_buf2.data_ptr()), st), "kv")
        torch.cuda.synchronize()
        n = pos0 + T
        K = kv_buf[:, :n].float()  # [4, n, 256]
        V = kv_buf2[:, :n].float()
        qh = q.view(T, 4, 6, 256)
        sc = torch.einsum("tgcd,gjd->tgcj", qh, K) / 16.0  # [T, 4, 6, n]
        jj = torch.arange(n, device=dev)
        for t in range(T):
            sc[t, :, :, jj > pos0 + t] = -float("inf")
        p = torch.softmax(sc, dim=-1)
        ref = torch.einsum("tgcj,gjd->tgcd", p, V).reshape(T, NH, 256) * torch.sigmoid(g)
        ref = ref.reshape(T, 6144)
        rel = float((ref - o).norm() / o.norm())
        rel_bf = float((ref.bfloat16().float() - o).norm() / o.norm())
        # noise floor: torch's own bf16 SDPA (flash/efficient kernel) vs the fp32 recompute, same inputs
        mask = (jj[None, :] <= (pos0 + torch.arange(T, device=dev))[:, None])
        sd = torch.nn.functional.scaled_dot_product_attention(q.bfloat16().permute(1, 0, 2)[None], kv_buf[None, :, :n], kv_buf2[None, :, :n], attn_mask=mask[None, None], enable_gqa=True)[0]  # [24, T, 256]
        sd = (sd.permute(1, 0, 2).float() * torch.sigmoid(g)).reshape(T, 6144)
        rel_sdpa = float((sd - ref).norm() / ref.norm())
        # probe top-4 / masses vs torch
        idx = torch.empty(ROWS * NAT * NH * 4, dtype=torch.int16)
        mas = torch.empty(ROWS * NAT * NH * 6, dtype=torch.float16)
        smk.check(lib.smk_probe_read(e.h, 0, C.c_void_p(idx.data_ptr()), idx.numel() * 2), "r")
        smk.check(lib.smk_probe_read(e.h, 1, C.c_void_p(mas.data_ptr()), mas.numel() * 2), "r")
        idx = idx.view(ROWS, NAT, NH, 4)[row_off : row_off + T, ai].to(dev).long()
        mas = mas.view(ROWS, NAT, NH, 6)[row_off : row_off + T, ai].to(dev).float()
        pf = p.reshape(T, NH, n)
        tv, ti = pf.topk(4, dim=-1)
        top1 = float((idx[..., 0] == ti[..., 0]).float().mean())
        top1_tie = float(((idx[..., 0] == ti[..., 0]) | ((pf.gather(-1, idx[..., :1].clamp(min=0))[..., 0] - tv[..., 0]).abs() <= 1e-4 * tv[..., 0])).float().mean())
        mass_err = float((mas[..., 0] - tv[..., 0]).abs().max())
        sink = float((mas[..., 4] - pf[..., 0]).abs().max())
        c = chk.setdefault(ai, dict(rel=[], rel_bf16=[], rel_torch_sdpa_bf16=[], top1=[], top1_tie=[], mass_err=[], sink_err=[], n=[]))
        c["rel"].append(rel); c["rel_bf16"].append(rel_bf); c["rel_torch_sdpa_bf16"].append(rel_sdpa); c["top1"].append(top1); c["top1_tie"].append(top1_tie); c["mass_err"].append(mass_err); c["sink_err"].append(sink); c["n"].append(n)

    for ti, t in enumerate(tasks):
        fn = OUT / f"{t['id']}.npz"
        if fn.exists():
            continue
        P, g = t["P"], np.asarray(t["g"]).astype(np.int64)
        N = len(g) - 1
        S = np.concatenate([np.asarray(t["prompt"]).astype(np.int64), g])
        assert len(S) == P + N + 1
        t0 = time.time()
        e.reset()
        pos = 0
        while pos < P - 1:  # prompt rows (graph steps), hidden states only
            T = min(4, P - 1 - pos)
            e._step(S[pos : pos + T].tolist(), 2)
            store(pos, T)
            pos += T
        torch.cuda.synchronize()
        t_pre = time.time() - t0
        t0 = time.time()
        row = 0
        while pos < P + N - 1:  # decode rows P-1 .. P+N-2: probe on, non-graph
            T = min(4, P + N - 1 - pos)
            smk.check(lib.smk_probe_cfg(e.h, 1, 0, pos, row), "cfg")
            e._step(S[pos : pos + T].tolist(), 2, graph=False)
            store(pos, T)
            pos += T
            row += T
        smk.check(lib.smk_probe_cfg(e.h, 0, 0, 0, 0), "cfg")
        torch.cuda.synchronize()
        t_dec = time.time() - t0
        # ---- probe results
        idx = torch.empty(ROWS * NAT * NH * 4, dtype=torch.int16)
        mas = torch.empty(ROWS * NAT * NH * 6, dtype=torch.float16)
        smk.check(lib.smk_probe_read(e.h, 0, C.c_void_p(idx.data_ptr()), idx.numel() * 2), "r")
        smk.check(lib.smk_probe_read(e.h, 1, C.c_void_p(mas.data_ptr()), mas.numel() * 2), "r")
        idx = idx.view(ROWS, NAT, NH, 4)[:N].numpy()
        mas = mas.view(ROWS, NAT, NH, 6)[:N].numpy()
        rec = dict(idx=idx, mass=mas)
        # ---- R2 kNN tables
        L = P + N - 1
        S1 = torch.from_numpy(S[1:]).to(dev)
        reps = {}
        for si, slot in enumerate(SLOTS):
            X = Hs[si, :L].float()
            if slot == 65:
                reps["h65raw"] = knn_tables(X, S1, P, N, "raw")
            reps[f"h{slot}"] = knn_tables(X, S1, P, N, "c")
            if slot in (32, 48, 60, 65):
                reps[f"h{slot}z"] = knn_tables(X, S1, P, N, "z")
                reps[f"h{slot}n"] = knn_tables(X, S1, P, N, "n")
            del X
        for ai in KEY_AI:
            smk.check(lib.smk_probe_copy_kv(e.h, ai, 0, C.c_void_p(kv_buf.data_ptr()), st), "kv")
            X = kv_buf[:, :L, 64:].permute(1, 0, 2).reshape(L, 768).float()
            reps[f"k{ai}"] = knn_tables(X, S1, P, N, "c")
            reps[f"k{ai}z"] = knn_tables(X, S1, P, N, "z")
        for rn, d in reps.items():
            for k, v in d.items():
                rec[f"{rn}/{k}"] = v
        np.savez(fn, **rec)
        print(f"{t['id']:6s} cls {t['cls']} P {P:5d} N {N:4d} prefill {t_pre:5.1f}s decode-probe {t_dec:5.1f}s ({N / t_dec:.0f} rows/s) -> {fn.name} {fn.stat().st_size / 1e6:.1f} MB", flush=True)
    # ---- probe validity vs the real kernel (fresh short replay on the first task, snapshots on)
    for t in [tasks[0]] + [x for x in tasks if x["cls"] in "bcde"][:3]:
        P, g = t["P"], np.asarray(t["g"]).astype(np.int64)
        S = np.concatenate([np.asarray(t["prompt"]).astype(np.int64), g])
        e.reset()
        pos = 0
        while pos < P - 1:
            T = min(4, P - 1 - pos)
            e._step(S[pos : pos + T].tolist(), 2)
            pos += T
        for k in range(3):  # three probed T=4 steps with snapshots; verify every attention layer after each
            T = 4
            smk.check(lib.smk_probe_cfg(e.h, 1, 1, pos, 0), "cfg")
            e._step(S[pos : pos + T].tolist(), 2, graph=False)
            for ai in range(NAT):
                verify(ai, T, 0, pos)
            pos += T
        smk.check(lib.smk_probe_cfg(e.h, 0, 0, 0, 0), "cfg")
    # ---- timing of the probe itself (per step, 16 layers) at the longest context
    t = max(tasks, key=lambda x: len(x["prompt"]))
    P, g = t["P"], np.asarray(t["g"]).astype(np.int64)
    S = np.concatenate([np.asarray(t["prompt"]).astype(np.int64), g])
    e.reset()
    pos = 0
    while pos < P - 1:
        T = min(4, P - 1 - pos)
        e._step(S[pos : pos + T].tolist(), 2)
        pos += T
    tim = {}
    for T in (1, 4):
        for on in (0, 1, 0, 1):
            ts = []
            for k in range(12):
                smk.check(lib.smk_probe_cfg(e.h, on, 0, pos, 0), "cfg")
                a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
                a.record()
                e._step(S[pos : pos + T].tolist(), 2, graph=False)
                b.record()
                torch.cuda.synchronize()
                ts.append(a.elapsed_time(b))
                pos += T
            tim.setdefault(f"T{T}_probe{on}", []).append(float(np.median(ts[2:])))
    smk.check(lib.smk_probe_cfg(e.h, 0, 0, 0, 0), "cfg")
    res = dict(per_layer={str(ai): {k: (float(np.mean(v)) if k != "n" else int(np.max(v))) for k, v in c.items()} for ai, c in chk.items()},
               worst_rel=max(max(c["rel"]) for c in chk.values()), worst_rel_bf16=max(max(c["rel_bf16"]) for c in chk.values()), mean_rel=float(np.mean([np.mean(c["rel"]) for c in chk.values()])), mean_rel_torch_sdpa_bf16=float(np.mean([np.mean(c["rel_torch_sdpa_bf16"]) for c in chk.values()])), min_top1_agree=min(min(c["top1_tie"]) for c in chk.values()),
               probe_step_ms=tim, probe_timing_task=t["id"], probe_timing_ctx=P)
    (HERE / "results" / "probe_check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1)[:3000], flush=True)


def micro():
    """cosine-search + store micro-benchmarks (single query vs L stored 5120-d states), independent of the engine."""
    import torch

    dev = torch.device("cuda")
    out = {}
    for L in (2048, 8192, 32768, 65536):
        X = torch.randn(L, H, device=dev).to(torch.bfloat16)
        q = torch.randn(H, device=dev, dtype=torch.bfloat16)
        tok = torch.randint(0, 248320, (L,), device=dev)
        anchor = tok[17]

        def f(X=X, q=q):
            return torch.mv(X, q)

        def f2(X=X, q=q, tok=tok, anchor=anchor):
            return torch.mv(X, q).masked_fill(tok != anchor, -1e4).argmax()

        for name, fn in (("mv", f), ("mv+mask+argmax", f2)):
            for _ in range(5):
                fn()
            torch.cuda.synchronize()
            a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
            a.record()
            for _ in range(200):
                fn()
            b.record()
            torch.cuda.synchronize()
            out[f"L{L}_bf16_{name}_ms"] = a.elapsed_time(b) / 200
        out[f"L{L}_bf16_MB"] = X.numel() * X.element_size() / 1e6
        del X
        torch.cuda.empty_cache()
    (HERE / "results" / "micro.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["collect", "micro"])
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    if a.cmd == "collect":
        collect([x for x in a.only.split(",") if x])
    else:
        micro()
