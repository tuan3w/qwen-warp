"""lm-head-quantisation-quality driver.  Lab venv (cu130 torch): ~/llm-lab/venv/bin/python experiments/2026-10-31-lm-head-quantisation-quality/run.py <stage>

Stages (each holds the GPU lease only while running; take it yourself: projects/llm-lab/bin/lab gpu acquire --holder <name>):
  corpus   CPU only: build + tokenise the natural-text docs, print counts                          (no GPU)
  collect  engine: greedy chat continuations + teacher-forced final-norm hidden states (T=4 rows for all docs, T=1 rows + engine logits for floor docs)
  analyze  torch: reference / quantised-head logits, metrics, noise floors, bootstrap -> results/summary.json, runs/per_position.npz
  report   CPU only: results/summary.json -> results/tables.md
Speed (separate): bench_head.cu (see README Method).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lmq  # noqa: E402
from lmq import HERE, RESULTS, RUNS, V, H  # noqa: E402

ENGINE = lmq.REPO / "engine"
GEN = 320
FLOOR_PER_CAT = 2  # floor docs (T=1 pass + engine logits) per category
LOGIT_EVERY = 10  # keep engine logits for positions with pos % 10 == 3 in the floor docs


def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(lmq.MODEL))


def stage_corpus():
    tok = tokenizer()
    docs = lmq.split_calib(lmq.build_docs(tok))
    from collections import Counter

    print(Counter((d["cat"], d["role"]) for d in docs), sum(len(d["ids"]) for d in docs), "tokens")
    for d in docs:
        print(d["cat"], d["role"], len(d["ids"]), d["name"])
    return docs


def stage_collect():
    sys.path.insert(0, str(ENGINE))
    from engine import Engine

    RUNS.mkdir(exist_ok=True)
    tok = tokenizer()
    docs = lmq.split_calib(lmq.build_docs(tok))
    for d in docs:
        d["eval_from"] = lmq.SKIP
    stops = {tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")} - {None}
    eng = Engine(ctx_max=2048)
    t0 = time.time()
    # greedy chat continuations (plain T=1 greedy decode graph) -> docs of category "greedy"
    for k, (kind, prompt) in enumerate(lmq.CHAT_PROMPTS):
        text = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False, enable_thinking=False)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        eng.reset()
        first = eng.prefill(ids)
        gen = [first] + eng.decode(GEN - 1)
        for j, t in enumerate(gen):
            if t in stops:
                gen = gen[: j + 1]
                break
        docs.append(dict(cat="greedy", sub=kind, name=f"chat:{prompt[:40]}", ids=ids + gen, eval_from=len(ids) - 1, role="calib" if k % 5 == 2 else "eval"))
    print(f"greedy docs done {time.time() - t0:.0f}s", flush=True)

    floor = []
    for cat in ("code", "prose", "math", "greedy"):
        c = [i for i, d in enumerate(docs) if d["cat"] == cat and d["role"] == "eval"]
        floor += c[: FLOOR_PER_CAT]
    for i in floor:
        docs[i]["floor"] = True

    def force(ids, T, want_logits_at=()):
        """teacher-force `ids` through the engine in T-row head steps; returns hidden [n,5120] bf16 and {pos: logits row}."""
        eng.reset()
        hid = torch.empty(len(ids), H, dtype=torch.bfloat16)
        lg = {}
        i = 0
        while i < len(ids):
            t = min(T, len(ids) - i)
            eng._step(ids[i : i + t], 1)
            hid[i : i + t] = eng._read(1, 66 * 4 * H, torch.bfloat16).view(66, 4, H)[65, :t]
            need = [p for p in want_logits_at if i <= p < i + t]
            if need:
                L = eng.logits(t)
                for p in need:
                    lg[p] = L[p - i].clone()
            i += t
        return hid, lg

    out_h4, out_h1, eng_l1, eng_l4, floor_pos = [], {}, {}, {}, {}
    for di, d in enumerate(docs):
        ids = d["ids"]
        want = [p for p in range(len(ids) - 1) if p % LOGIT_EVERY == 3 and p >= d["eval_from"]] if d.get("floor") else []
        h4, l4 = force(ids, 4, want)
        out_h4.append(h4)
        if d.get("floor"):
            h1, l1 = force(ids, 1, want)
            out_h1[di] = h1
            eng_l1[di], eng_l4[di], floor_pos[di] = l1, l4, want
        print(f"doc {di}/{len(docs)} {d['cat']} {len(ids)} tok  {time.time() - t0:.0f}s", flush=True)
    torch.save(dict(hidden_t4=out_h4, hidden_t1=out_h1), RUNS / "hidden.pt")
    torch.save(dict(l1={k: torch.stack([eng_l1[k][p] for p in floor_pos[k]]) for k in eng_l1 if floor_pos[k]},
                    l4={k: torch.stack([eng_l4[k][p] for p in floor_pos[k]]) for k in eng_l4 if floor_pos[k]}, pos=floor_pos), RUNS / "engine_logits.pt")
    json.dump([{k: d[k] for k in d if k != "ids"} | {"n": len(d["ids"]), "ids": d["ids"]} for d in docs], open(RUNS / "docs.json", "w"))
    print("collect done", time.time() - t0)


def _eval_positions(docs):
    """aligned arrays over evaluated positions of role=eval docs: (doc, pos_in_doc, tgt)."""
    D, P, T = [], [], []
    for di, d in enumerate(docs):
        if d["role"] != "eval":
            continue
        for p in range(d["eval_from"], d["n"] - 1):
            D.append(di)
            P.append(p)
            T.append(d["ids"][p + 1])
    return np.array(D), np.array(P), np.array(T)


def stage_analyze(chunk=384):
    sys.path.insert(0, str(ENGINE / "tests"))
    sys.path.insert(0, str(lmq.REPO / "oracle"))
    from weights import Checkpoint

    t0 = time.time()
    docs = json.load(open(RUNS / "docs.json"))
    hid = torch.load(RUNS / "hidden.pt")
    h4, h1 = hid["hidden_t4"], hid["hidden_t1"]
    D, P, T = _eval_positions(docs)
    print("eval positions", len(D), "docs", len(set(D.tolist())))
    X = torch.stack([h4[d][p] for d, p in zip(D, P)]).float().cuda()
    tgt = torch.tensor(T, device="cuda")
    cal = torch.cat([h4[i][docs[i]["eval_from"] : docs[i]["n"] - 1] for i, d in enumerate(docs) if d["role"] == "calib"]).float().cuda()
    colw = (cal**2).mean(0)
    print("calib positions", cal.shape[0], flush=True)
    ck = Checkpoint()
    W = ck.get("lm_head.weight").cuda()
    assert W.shape == (V, H) and W.dtype == torch.bfloat16
    heads = {}
    summary = {"arms": {}, "floor": {}, "weights": {}}
    for a in lmq.ARMS:
        heads[a] = lmq.quantise(W, a, colw)
        summary["weights"][a] = dict(rel_l2=lmq.weight_err(W, heads[a]), MB=heads[a].bytes() / 2**20)
        print(a, summary["weights"][a], f"{time.time() - t0:.0f}s", flush=True)
    RESCORE_ARMS = ("int8_rtn", "int4_g128_rtn", "int4_g128_awclip")  # post-hoc, exploratory
    RESCORE_M = (4, 8, 16)
    names = list(heads) + ["bf16_logits_rounded"] + [f"{a}+rescore{M}" for a in RESCORE_ARMS for M in RESCORE_M]
    acc = {n: {} for n in names}
    for c0 in range(0, len(D), chunk):
        sl = slice(c0, c0 + chunk)
        Lr = lmq.logits_from(X[sl], W)
        def put(n, Lq):
            for k, v in lmq.metrics(Lr, Lq, tgt[sl]).items():
                acc[n].setdefault(k, []).append(v.cpu())

        for n in list(heads) + ["bf16_logits_rounded"]:
            Lq = Lr.bfloat16().float() if n == "bf16_logits_rounded" else lmq.logits_from(X[sl], qh=heads[n])
            put(n, Lq)
            if n in RESCORE_ARMS:
                for M in RESCORE_M:
                    put(f"{n}+rescore{M}", lmq.rescore(Lr, Lq, M))
            del Lq
        del Lr
        print(f"chunk {c0}/{len(D)} {time.time() - t0:.0f}s", flush=True)
    per = {n: {k: torch.cat(v).numpy() for k, v in acc[n].items()} for n in names}
    del heads

    # noise floors on the floor docs
    el = torch.load(RUNS / "engine_logits.pt")
    fl = {"engine_T1_vs_fp32": ([], []), "engine_T4_vs_fp32": ([], []), "hidden_T1_vs_T4": ([], [])}
    f_meta = {k: [] for k in fl}
    for di_s, pos in el["pos"].items():
        di = int(di_s)
        if not pos:
            continue
        ids = docs[di]["ids"]
        t = torch.tensor([ids[p + 1] for p in pos], device="cuda")
        for name, hs, key in (("engine_T1_vs_fp32", h1[di], "l1"), ("engine_T4_vs_fp32", h4[di], "l4")):
            x = torch.stack([hs[p] for p in pos]).float().cuda()
            Lr = lmq.logits_from(x, W)
            m = lmq.metrics(Lr, el[key][di].cuda(), t)
            fl[name][0].append(m)
            f_meta[name] += [di] * len(pos)
    for di_s in h1:
        di = int(di_s)
        d = docs[di]
        ps = list(range(d["eval_from"], d["n"] - 1))
        t = torch.tensor([d["ids"][p + 1] for p in ps], device="cuda")
        for c0 in range(0, len(ps), 256):
            q = ps[c0 : c0 + 256]
            x4 = torch.stack([h4[di][p] for p in q]).float().cuda()
            x1 = torch.stack([h1[di][p] for p in q]).float().cuda()
            m = lmq.metrics(lmq.logits_from(x4, W), lmq.logits_from(x1, W), t[c0 : c0 + 256])
            fl["hidden_T1_vs_T4"][0].append(m)
            f_meta["hidden_T1_vs_T4"] += [di] * len(q)
    floor_per = {n: {k: torch.cat([m[k].cpu() for m in fl[n][0]]).numpy() for k in fl[n][0][0]} for n in fl}
    # relative L2 of T1 vs T4 final-norm hidden (context for the floor)
    rel = [float(((h1[i].float() - h4[i].float()) ** 2).sum() / (h4[i].float() ** 2).sum()) ** 0.5 for i in h1]
    summary["floor_hidden_rel_l2_T1_vs_T4"] = rel

    # ---- aggregate
    cats = np.array([docs[d]["cat"] for d in D])

    def agg(pp, dd, pidx, cat_arr):
        r = {}
        for tag, mask in [("all", np.ones(len(dd), bool))] + [(c, cat_arr == c) for c in ("code", "prose", "math", "greedy")] + [("natural", cat_arr != "greedy")]:
            if mask.sum() < 30:
                continue
            e = {}
            for k in ("top1", "kl_full", "kl_top20", "mass_top20", "tv_full", "tv_k20p95_T1", "tv_k20p95_T07", "dnll"):
                if k in pp:
                    mean, lo, hi = lmq.doc_bootstrap(pp[k], dd, mask=mask)
                    e[k] = dict(mean=mean, lo=lo, hi=hi)
            flips = mask & (pp["top1"] == 0)
            e["n"] = int(mask.sum())
            e["n_flips"] = int(flips.sum())
            e["flip_ref_margin_mean"] = float(pp["ref_margin"][flips].mean()) if flips.sum() else None
            e["flip_ref_margin_median"] = float(np.median(pp["ref_margin"][flips])) if flips.sum() else None
            e["nll_ref_mean"] = float(pp["nll_ref"][mask].mean()) if "nll_ref" in pp else None
            r[tag] = e
        r["blocks_all"] = lmq.block_stats(pp["_aq"], pp["_ar"], dd, pidx)
        return r

    for n in names:
        summary["arms"][n] = agg(per[n], D, P, cats)
    for n in fl:
        dd = np.array(f_meta[n])
        # positions for floors: pos index recomputed (all p; block stats not meaningful for the sparse engine-logit subsets)
        r = {}
        pp = floor_per[n]
        cat_arr = np.array([docs[d]["cat"] for d in dd])
        for tag, mask in [("all", np.ones(len(dd), bool))]:
            e = {}
            for k in ("top1", "kl_full", "kl_top20", "mass_top20", "tv_full", "tv_k20p95_T1", "tv_k20p95_T07", "dnll"):
                mean, lo, hi = lmq.doc_bootstrap(pp[k], dd, mask=mask)
                e[k] = dict(mean=mean, lo=lo, hi=hi)
            e["n"] = len(dd)
            e["n_flips"] = int((pp["top1"] == 0).sum())
            e["flip_ref_margin_mean"] = float(pp["ref_margin"][pp["top1"] == 0].mean()) if (pp["top1"] == 0).sum() else None
            r[tag] = e
        summary["floor"][n] = r
    summary["n_docs"] = {c: int(len({d for d in D if docs[d]["cat"] == c})) for c in ("code", "prose", "math", "greedy")}
    RESULTS.mkdir(exist_ok=True)
    json.dump(summary, open(RESULTS / "summary.json", "w"), indent=1)
    np.savez_compressed(RUNS / "per_position.npz", doc=D, pos=P, tgt=T, **{f"{n}/{k}": v for n in per for k, v in per[n].items()},
                        **{f"floor:{n}/{k}": v for n in floor_per for k, v in floor_per[n].items()})
    print("analyze done", time.time() - t0)


if __name__ == "__main__":
    st = sys.argv[1]
    {"corpus": stage_corpus, "collect": stage_collect, "analyze": stage_analyze}[st]()
