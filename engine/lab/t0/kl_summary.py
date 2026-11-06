"""Tables from the real-model KL runs (lab/traces/kl_eval.py jsonl).  python kl_summary.py -> results/kl/KL.md"""
import collections
import glob
import json
from pathlib import Path

import numpy as np

R = Path("/media/fatman/data/workspace/research/experiments/2026-11-06-sparsity-probes/results/kl")


def load(preset):
    rows = []
    for f in sorted(glob.glob(str(R / f"*_{preset}.jsonl"))):
        rows += [json.loads(l) for l in open(f)]
    return rows


def by_name(rows):
    d = collections.defaultdict(list)
    for r in rows:
        d[r["name"]].append(r)
    return d


def cell(rs):
    v = [r["kl"] * 100 for r in rs]
    return f"{np.mean(v):.3f} ({min(v):.3f}-{max(v):.3f})"


def interp_budget(xs, ys, target):
    """smallest x (log-log interpolation) with y <= target, y decreasing in x"""
    for i in range(len(xs)):
        if ys[i] <= target:
            if i == 0:
                return xs[0]
            lx0, lx1, ly0, ly1 = np.log(xs[i - 1]), np.log(xs[i]), np.log(ys[i - 1]), np.log(ys[i])
            return float(np.exp(lx0 + (np.log(target) - ly0) * (lx1 - lx0) / (ly1 - ly0)))
    return float("nan")


sp_kinds = ("qkv", "o", "gu", "down")


def main():
    L = ["# Real-model logit KL (nats x100 = %), 128 teacher-forced decode rows of the model's own greedy continuation; cells = mean (min-max over docs)\n"]
    A = load("calib_attn")
    d = by_name(A)
    d_attn = d
    docs = sorted({r["doc"] for r in A})
    L.append(f"docs: {', '.join(docs)}.  dense torch re-implementation of the attention (self check): {cell(d['attn_dense_torch'])} % (numerical floor of the override).\n")
    buds = (512, 1024, 2048, 4096, 8192)
    L.append("## attention sparsity, ALL 16 full-attention layers at once, 32k context (budget = tokens read per row/head incl. sink 4 + ring 256)\n")
    L.append("| rule | " + " | ".join(str(b) for b in buds) + " |")
    L.append("|---|" + "---|" * len(buds))
    for pre, lab in (("attn_tok_oracle_{}", "token oracle top-k (unrealisable)"), ("attn_oracle_b64_{}", "block-64 oracle mass (unrealisable)"), ("attn_quest_b64_{}", "Quest min/max, block 64, per head"), ("attn_quest_rot_b64_{}", "Quest min/max in rotated basis"),
                     ("attn_quest_b64_{}_kv", "Quest, one selection per KV head"), ("attn_quest_b64_{}_row8", "Quest, one selection per 8 rows x KV head")):
        L.append(f"| {lab} | " + " | ".join(cell(d[pre.format(b)]) if pre.format(b) in d else "-" for b in buds) + " |")
    for tgt in (0.05, 0.15, 0.5):
        L.append("")
    xs = list(buds)
    L.append("\nBudget (tokens) needed for mean KL <= target (log-log interpolation of the doc-mean curve; nan = not reached by 8192):\n")
    L.append("| rule | 0.05% | 0.15% | 0.5% |")
    L.append("|---|---|---|---|")
    for pre, lab in (("attn_tok_oracle_{}", "token oracle"), ("attn_oracle_b64_{}", "block-64 oracle"), ("attn_quest_b64_{}", "Quest b64")):
        ys = [np.mean([r["kl"] * 100 for r in d[pre.format(b)]]) for b in xs]
        L.append(f"| {lab} | " + " | ".join(f"{interp_budget(xs, ys, t):.0f}" for t in (0.05, 0.15, 0.5)) + " |")
    # 128k
    A128 = load("calib_attn128")
    if A128:
        d1 = by_name(A128)
        L.append("\n## 128k context (1 doc), all 16 attention layers\n")
        b2 = (1024, 2048, 4096, 8192, 16384)
        L.append("| rule | " + " | ".join(str(b) for b in b2) + " |")
        L.append("|---|" + "---|" * len(b2))
        L.append(f"| dense torch self-check | {d1['attn_dense_torch'][0]['kl'] * 100:.3f} |")
        for pre, lab in (("attn_tok_oracle_{}", "token oracle"), ("attn_oracle_b64_{}", "block-64 oracle"), ("attn_quest_b64_{}", "Quest b64"), ("attn_quest_b64_{}_kv", "Quest b64 per KV head")):
            L.append(f"| {lab} | " + " | ".join(f"{d1[pre.format(b)][0]['kl'] * 100:.3f}" if pre.format(b) in d1 else "-" for b in b2) + " |")
        L.append("| top-1 agreement (token oracle) | " + " | ".join(f"{d1['attn_tok_oracle_%d' % b][0]['top1']:.3f}" if 'attn_tok_oracle_%d' % b in d1 else '-' for b in b2) + " |")
    # sensitivity
    S = load("sens_attn")
    if S:
        ds = by_name(S)
        L.append("\n## per-layer attention sensitivity (ONE attention layer made sparse, others dense-torch), KL% mean over docs\n")
        L.append("| layer | token oracle 1024 | oracle b64 2048 | Quest b64 4096 |")
        L.append("|---|---|---|---|")
        for l in range(3, 64, 4):
            L.append(f"| {l} | " + " | ".join(f"{np.mean([r['kl'] * 100 for r in ds[f'sensattn_L{l}_{k}']]):.3f}" for k in ("tok1024", "oracle64_2048", "quest64_4096")) + " |")
    S = load("sens_act")
    if S:
        ds = by_name(S)
        kl = np.array([np.mean([r["kl"] * 100 for r in ds[f"sensact_L{l}_all_f0.8"]]) for l in range(64)])
        order = np.argsort(-kl)
        L.append("\n## per-layer activation-sparsity sensitivity (all 4 linear kinds of ONE layer at kept fraction 0.8, per-channel top-|x|), KL% mean over docs\n")
        L.append(f"sum over layers {kl.sum():.3f}% (all layers at once at f=0.8: see calib table); top 8 layers: " + ", ".join(f"L{l} {kl[l]:.3f}" for l in order[:8]) + f"; median layer {np.median(kl):.4f}; layers with KL < 0.002%: {(kl < 0.002).sum()}/64\n")
        L.append("| layers | " + " | ".join(f"{a}-{a + 7}" for a in range(0, 64, 8)) + " |")
        L.append("|---|" + "---|" * 8)
        L.append("| sum KL% | " + " | ".join(f"{kl[a:a + 8].sum():.3f}" for a in range(0, 64, 8)) + " |")
    # act tables
    for preset, title in (("calib_act", "activation sparsity, per-channel |x| top-k (b=1), one kind at a time in all 64 layers (T=1 rows)"), ("calib_act_blocks", "activation sparsity with K-blocks of b columns (T=1 rows) and T=8 union / max-over-rows (b=1 or 8)"), ("calib_mlp", "MLP intermediate-channel gating (up row + down column skipped), 8-channel blocks / 32")):
        A = load(preset)
        if not A:
            continue
        d = by_name(A)
        L.append(f"\n## {title}\n")
        L.append("| config | KL% mean (min-max over docs) | top-1 agreement | bytes frac of the kind's rows (mean) |")
        L.append("|---|---|---|---|")
        for n in sorted(d, key=lambda n: (n.split("_f")[0], -float(n.split("_f")[1].split("_")[0]) if "_f" in n and n.split("_f")[1].split("_")[0].replace(".", "").isdigit() else 0, n)):
            rs = d[n]
            bf = [list(r["bytes_frac"].values())[0] for r in rs if r.get("bytes_frac")]
            L.append(f"| {n} | {cell(rs)} | {np.mean([r['top1'] for r in rs]):.3f} | {np.mean(bf):.3f} |" if bf else f"| {n} | {cell(rs)} | {np.mean([r['top1'] for r in rs]):.3f} | - |")
    # ---- granularity of the selection (32k 3 docs and 128k)
    G = load("calib_attn_gran")
    if G:
        dg = by_name(G)
        L.append("\n## selection granularity / sharing at 32k (mean over 3 docs; budget = tokens per row/head incl. sink + ring); token oracle shared per KV head / per 8 rows x KV head; block-8/16/32 oracle\n")
        L.append("| rule | 1024 | 2048 | 4096 | 8192 |")
        L.append("|---|---|---|---|---|")
        rows = [("attnT_tok_oracle_kv_{}", "token oracle, one set per KV head"), ("attnT_tok_oracle_row8_{}", "token oracle, one set per KV head and 8 rows")] + [(f"attnT_oracle_b{b}_{{}}", f"block-{b} oracle, per head") for b in (8, 16, 32)]
        base = {"token oracle, per head": ("attn_tok_oracle_{}", d_attn), "block-64 oracle, per head": ("attn_oracle_b64_{}", d_attn)}
        for lab, (pre, dd) in base.items():
            L.append(f"| {lab} | " + " | ".join(cell(dd[pre.format(b)]) for b in (1024, 2048, 4096, 8192)) + " |")
        for pre, lab in rows:
            L.append(f"| {lab} | " + " | ".join(cell(dg[pre.format(b)]) if pre.format(b) in dg else "-" for b in (1024, 2048, 4096, 8192)) + " |")
        import glob as _g

        G1 = [json.loads(l) for f in _g.glob(str(R / "tr128_mixed_calib_attn_gran.jsonl")) for l in open(f)]
        if G1:
            d1g = {r["name"]: r for r in G1}
            L.append("\n128k (1 doc), KL%:\n")
            L.append("| rule | 1024 | 2048 | 4096 | 8192 |")
            L.append("|---|---|---|---|---|")
            for pre, lab in rows:
                L.append(f"| {lab} | " + " | ".join(f"{d1g[pre.format(b)]['kl'] * 100:.3f}" if pre.format(b) in d1g else "-" for b in (1024, 2048, 4096, 8192)) + " |")
    # ---- shifted window (generated tokens 128..255 instead of 0..127)
    sh = []
    for f in sorted(glob.glob(str(R / "shift128/*_calib_attn.jsonl"))):
        sh += [json.loads(l) for l in open(f)]
    if sh:
        ds_ = by_name(sh)
        L.append("\n## window robustness: generated tokens 128..255 (window 1) vs 0..127 (window 0), attention rules, 3 docs, KL% mean\n")
        L.append("| rule | window 0: 1024 / 4096 / 8192 | window 1: 1024 / 4096 / 8192 |")
        L.append("|---|---|---|")
        for pre, lab in (("attn_tok_oracle_{}", "token oracle"), ("attn_oracle_b64_{}", "block-64 oracle"), ("attn_quest_b64_{}", "Quest b64")):
            L.append(f"| {lab} | " + " / ".join(f"{np.mean([r['kl'] * 100 for r in d_attn[pre.format(b)]]):.3f}" for b in (1024, 4096, 8192)) + " | " + " / ".join(f"{np.mean([r['kl'] * 100 for r in ds_[pre.format(b)]]):.3f}" for b in (1024, 4096, 8192)) + " |")
    sa = []
    for f in sorted(glob.glob(str(R / "shift128/*_calib_act.jsonl"))):
        sa += [json.loads(l) for l in open(f)]
    if sa:
        da = by_name(sa)
        d0 = by_name(load("calib_act"))
        L.append("\n## window robustness, activation sparsity (per-channel top-|x|, one kind, all layers), KL% mean: window 0 / window 1\n")
        L.append("| kind | f=0.9 | f=0.8 | f=0.7 | f=0.6 |")
        L.append("|---|---|---|---|---|")
        for kind in (*sp_kinds, "all"):
            L.append(f"| {kind} | " + " | ".join(f"{np.mean([r['kl'] * 100 for r in d0[f'act_{kind}_f{f}_b1']]):.3f} / {np.mean([r['kl'] * 100 for r in da[f'act_{kind}_f{f}_b1']]):.3f}" for f in (0.9, 0.8, 0.7, 0.6)) + " |")
    # ---- B1d sequence-level masks
    SQ = load("seq")
    if SQ:
        dq = by_name(SQ)
        L.append("\n## B1d: sequence-level (GRIFFIN-style) frozen masks from the statistics (RMS) of the last N PROMPT rows, applied to all 128 decode rows (T=1 and T=8 alike); KL% mean over 3 docs; bytes = weight-byte fraction of the kind (down: columns; gu_rows = gate|up rows dropped with the MLP channels, 8-channel tile occupancy)\n")
        for kind in sp_kinds:
            L.append(f"\n### {kind}\n")
            L.append("| N | b | " + " | ".join(f"f={f} KL% (bytes)" for f in (0.8, 0.7, 0.6, 0.5)) + " |")
            L.append("|---|---|---|---|---|---|")
            for N in (256, 1024, 4096):
                for b in (1, 8, 64, 128):
                    cells = []
                    for f in (0.8, 0.7, 0.6, 0.5):
                        rs = dq[f"seq_{kind}_N{N}_f{f}_b{b}"]
                        bf = np.mean([r["bytes_frac"][kind] for r in rs if r.get("bytes_frac")])
                        cells.append(f"{np.mean([r['kl'] * 100 for r in rs]):.2f} ({bf:.2f})")
                    L.append(f"| {N} | {b} | " + " | ".join(cells) + " |")
        L.append("\n### GRIFFIN normalised statistic (down, N=1024) vs plain RMS\n")
        L.append("| b | f | KL% seq_nrm | KL% seq_rms |")
        L.append("|---|---|---|---|")
        for b in (8, 128):
            for f in (0.8, 0.7, 0.6, 0.5):
                L.append(f"| {b} | {f} | {np.mean([r['kl'] * 100 for r in dq[f'seqn_down_N1024_f{f}_b{b}']]):.2f} | {np.mean([r['kl'] * 100 for r in dq[f'seq_down_N1024_f{f}_b{b}']]):.2f} |")
    # ---- G0
    G0 = load("g0")
    if G0:
        dz = by_name(G0)
        L.append("\n## G0: DeltaNet state drift (256 teacher-forced rows: sparsity on rows 0-127 only, rows 128-255 dense; 3 docs mean)\n")
        L.append("| config | KL% rows 0-127 | KL% rows 128-255 (dense again) | state rel-err after 128 rows (mean / max over 48 layers) | after 256 rows (mean / max) | KL% by 16-row block |")
        L.append("|---|---|---|---|---|---|")
        for n in dz:
            rs = dz[n]
            m = lambda k: np.mean([r[k] for r in rs])
            blocks = np.mean([r["kl_rows_8"] for r in rs], 0) * 100
            L.append(f"| {n} | {m('kl_first128') * 100:.3f} | {m('kl_second128') * 100:.3f} | {m('state_err_after128_mean'):.4f} / {m('state_err_after128_max'):.4f} | {m('state_err_after256_mean'):.4f} / {m('state_err_after256_max'):.4f} | " + " ".join(f"{x:.2f}" for x in blocks) + " |")
    (R / "KL.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:70]))


if __name__ == "__main__":
    main()
