#!/usr/bin/env python
"""DSpine block drafter, end to end (engine/dspine.py, src/dspine.cu): the output is exactly the target's greedy output, the drafter keeps its quality, nothing goes non-finite, and
the CUDA drafter computes what the independent torch reference (tests/dspine_ref.py) computes.

  python tests/test_dspine.py [--n 192] [--update-reference]
Needs the target checkpoint ($SWIFT_MODEL), the DSpine checkpoint ($DSPINE_PATH), the golden file tests/data/dspine_golden.npz and the GPU (about 22 GB: stop the serving container).  ~6 min.

Data (tests/data/dspine_prompts.json): code (Python, Java, SQL), English and Vietnamese / Chinese / French prose, three TOOL-CALLING prompts (typed arguments, an array and a boolean,
a Vietnamese request) and two long-context prompts built from this repository's docs: ~2k tokens, and ~8.5k tokens which is past the drafter's 8,192-token attention window.
What each check is for (and where the method comes from):
  1. LOSSLESS  per prompt the device loop's ids equal the plain greedy decode through the SAME T = 8 verify kernel (verify([anchor] + 7 junk) + commit(1)) up to the first difference, and
     a difference is allowed only where the TARGET's own top-2 logits are within NEAR of each other (and are exactly the two tokens in question).  Measured: prompt code_sql flips at token
     164 with a margin of 0.14 (`),` / `)` at logit 29, one bf16 step there is 0.125): partial accepts are committed through a different chunking of the recurrent state, so a rounding
     differs.  A real drafter / verify bug diverges far from a tie and fails here.  (vLLM tests/v1/e2e/spec_decode require 60-90% exact matches instead.)
  2. TOOL CALLS  for the tool prompts the output must contain a well-formed call: the right function, every required argument, no unknown argument, every value of the type the schema
     declares (integer, boolean, array parsed as JSON), and the values the request implies.  The ids are equal to the reference by check 1, so a failure here is the TARGET path (engine
     numerics, KV cache, chat template) producing a worse call: tool calls are the first thing that breaks when something is slightly off (docs/USAGE.md: 39% -> 92% from a key order).
  3. DETERMINISM  the same request twice: identical ids and identical per-cycle accept histogram (llama.cpp tests/test-backend-ops.cpp runs each graph again and compares bytes).
  4. QUALITY FLOOR  tokens per cycle per prompt against tests/data/dspine_reference.json: mean >= 95% (vLLM tests/v1/e2e/spec_decode/acceptance_rates/utils.py uses 95%), every prompt >= 85%.
     A wrong rounding in the drafter leaves the output correct and silently costs acceptance: this is the check that sees it.  Per-language / per-length numbers expose data regressions.
  5. NUMERICS  the drafter's hidden block after the prefill of a fixed prompt against the golden reference computed with the same dequantised weights (NMSE, tight: a kernel bug) and
     against the bf16 weights (cosine: the quantisation quality), the drafts chain against the reference chain.
  6. FINITE  no NaN / inf in the drafter's hidden block, drafts are valid token ids.
--update-reference rewrites dspine_reference.json from this run (do it on purpose, review the diff).
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "engine"))
sys.path.insert(0, str(HERE))
import dspine_ref as D  # noqa: E402
from toolcall_check import validate_tool_call  # noqa: E402

MODEL = os.environ.get("SWIFT_MODEL", os.path.expanduser("~/models/swift-1.5-int4"))
DATA = HERE / "data"
MEAN_FLOOR, PROMPT_FLOOR = 0.95, 0.85
VOCAB = 248320
NEAR = 0.3  # a divergence is excused only if the target's own top-2 logits are this close; measured flips: 0.005-0.136 over 7 of 13 prompts, one bf16 step at |logit| ~ 30 is 0.125
COS_BF16_MIN = 0.985
GOLDEN_CHAIN_MIN = 5  # of 7 drafts equal to the reference chain (a near-tie flip changes the chain after the flip)
CTX_MAX = 12288  # > 8,192 + the longest prompt: the ~8.5k-token prompt runs past the drafter's attention window


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("PASS", msg, flush=True)


def doc_text(tok, n_tokens):
    """the first n_tokens of this repository's docs (a fixed, real, technical text; sorted file order)."""
    text = "\n\n".join(
        Path(f).read_text()
        for f in sorted(glob.glob(str(REPO / "docs" / "*.md")))
        + [str(REPO / "README.md")]
    )
    return tok.decode(tok(text, add_special_tokens=False)["input_ids"][:n_tokens])


def encode(tok, p):
    user = (
        p["user"]
        if p["kind"] != "doc"
        else doc_text(tok, p["doc_tokens"]) + "\n\n" + p["user"]
    )
    kw = {"tools": p["tools"], "enable_thinking": False} if p["kind"] == "tool" else {}
    o = tok.apply_chat_template(
        [{"role": "user", "content": user}],
        add_generation_prompt=True,
        tokenize=True,
        **kw,
    )
    return list(o["input_ids"] if hasattr(o, "keys") else o)


def t8_greedy(e, ids, n):
    """plain greedy through the T = 8 verify kernel: the same kernel the speculative loop verifies with, so the ids must be equal."""
    e.reset()
    m, i = len(ids) - 1, 0
    while i < m:
        t = min(8, m - i)
        e._step(ids[i : i + t], 0)
        i += t
    e._step(ids[-1:], 1)
    first = e.read_out_tok()
    ref, anchor = [first], first
    while len(ref) < n:
        y = e.verify([anchor] + [D.MASK_ID] * 7)
        e.commit(1, 8)
        ref.append(y[0])
        anchor = y[0]
    return ref


def top2_at(e, ids, prefix):
    """the target's two best next tokens after ids + prefix, through the same T = 8 / head path: -> ([id1, id2], margin = logit1 - logit2)."""
    ctx = ids + prefix
    e.reset()
    m, i = len(ctx) - 1, 0
    while i < m:
        t = min(8, m - i)
        e._step(ctx[i : i + t], 0)
        i += t
    e._step(ctx[-1:], 1)
    top = torch.topk(e.logits(1)[0].float(), 2)
    return [int(x) for x in top.indices], float(top.values[0] - top.values[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=192)
    ap.add_argument("--update-reference", action="store_true")
    args = ap.parse_args()
    from dspine import DspineEngine
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    prompts = json.loads((DATA / "dspine_prompts.json").read_text())
    ref_path = DATA / "dspine_reference.json"
    reference = (
        json.loads(ref_path.read_text())
        if ref_path.exists() and not args.update_reference
        else {}
    )
    e = DspineEngine(ctx_max=CTX_MAX, kv_fp8=0, verbose=False)
    e.disable_pld()

    measured, runs = {}, {}
    for p in prompts:
        ids = encode(tok, p)
        ref = t8_greedy(e, ids, args.n)
        toks, st = e.generate_sp(ids, args.n)
        toks = toks[: args.n]
        first_diff = next(
            (i for i, (a, b) in enumerate(zip(toks, ref)) if a != b), None
        )
        tag = f"{p['name']} [{p['lang']}, {len(ids)} prompt tokens]"
        if first_diff is None:
            check(True, f"{tag}: DSpine ids == T=8 greedy decode over {args.n} tokens")
        else:
            top, margin = top2_at(e, ids, ref[:first_diff])
            check(
                {ref[first_diff], toks[first_diff]} == set(top) and margin <= NEAR,
                f"{tag}: ids equal the T=8 greedy decode up to token {first_diff}; there the target's top-2 are {top} (margin {margin:.3f} <= {NEAR}): a rounding near-tie, not a drafter error",
            )
        check(
            all(0 <= t < VOCAB for t in toks),
            f"{p['name']}: every emitted id is a valid token id",
        )
        measured[p["name"]] = round(st["tokens_per_cycle"], 3)
        runs[p["name"]] = (ids, toks, st)
        print(
            f"     {p['name']:13s} tokens/cycle {st['tokens_per_cycle']:.2f}  accepted-drafts histogram {st['hist']}",
            flush=True,
        )
        if p["kind"] == "tool":
            ok, detail = validate_tool_call(
                p, tok.decode(toks, skip_special_tokens=False)
            )
            check(ok, f"{p['name']}: well-formed, schema-valid tool call: {detail}")

    # determinism: the same request again (short prompts only: the long ones take a minute each)
    for name in [p["name"] for p in prompts if p["kind"] != "doc"][:3]:
        ids, toks, st = runs[name]
        toks2, st2 = e.generate_sp(ids, args.n)
        check(
            toks2[: args.n] == toks
            and st2["hist"] == st["hist"]
            and st2["cycles"] == st["cycles"],
            f"{name}: second request identical (ids, cycles, accept histogram): deterministic, state reset",
        )

    # quality floor against the committed reference
    if args.update_reference:
        ref_path.write_text(
            json.dumps({"n": args.n, "tokens_per_cycle": measured}, indent=1)
        )
        print("wrote", ref_path, measured)
    else:
        rt = reference["tokens_per_cycle"]
        mean = sum(measured[k] for k in rt) / sum(rt.values())
        check(
            mean >= MEAN_FLOOR,
            f"tokens per cycle over all prompts {mean * 100:.1f}% of the reference (floor {MEAN_FLOOR * 100:.0f}%)",
        )
        for k, r in ((k, measured[k] / rt[k]) for k in rt if k in measured):
            check(
                r >= PROMPT_FLOOR,
                f"{k}: tokens per cycle {measured[k]:.2f} = {r * 100:.1f}% of the reference {rt[k]:.2f} (floor {PROMPT_FLOOR * 100:.0f}%)",
            )

    # numerics against the golden reference, finite
    g = np.load(DATA / "dspine_golden.npz", allow_pickle=False)
    meta = json.loads(str(g["meta"]))
    check(
        meta["quant_signature"] == D.quant_signature(),
        "golden file matches the checkpoint and the quantiser (regenerate with tests/make_dspine_golden.py if this fails)",
    )
    p0 = next(x for x in prompts if x["name"] == meta["prompt"])
    ids0 = encode(tok, p0)
    check(
        list(g["ids"]) == ids0,
        "golden prompt ids equal the tokenizer's ids (tokenizer / chat template unchanged)",
    )
    anchor = e.sp_prefill(ids0)
    check(
        anchor == int(g["anchor"]),
        f"first generated token {anchor} equals the golden anchor",
    )
    hid = e.sp_hid().float()
    check(
        bool(torch.isfinite(hid).all()), "drafter hidden block is finite (no NaN / inf)"
    )
    resid = (
        e.sp_read(3, 8 * D.H, torch.bfloat16).float().view(8, D.H)
    )  # residual stream after the last layer (before the final norm)
    nmse_max = float(os.environ.get("DSPINE_NMSE_MAX", "5e-4"))
    e_res = D.nmse(torch.from_numpy(g["layers_w4"][-1]), resid)
    check(
        e_res <= nmse_max,
        f"CUDA drafter residual stream after layer 5 vs reference with identical weights: NMSE {e_res:.2e}",
    )
    drafts = [int(x) for x in e._read(0, 20, torch.int32)[3:10]]
    check(all(0 <= d < VOCAB for d in drafts), f"drafts {drafts} are valid token ids")
    e_w4 = D.nmse(torch.from_numpy(g["hid_w4"]), hid)
    cos_bf = float(
        torch.nn.functional.cosine_similarity(
            torch.from_numpy(g["hid_bf16"]).flatten()[None], hid.flatten()[None]
        )
    )
    print(
        f"     hidden block vs reference with the same W4 weights: NMSE {e_w4:.3e};  vs bf16 weights: cosine {cos_bf:.5f}",
        flush=True,
    )
    check(
        e_w4 <= nmse_max,
        f"CUDA drafter vs reference with identical weights: NMSE {e_w4:.2e} (bf16 rounding of 5 layers)",
    )
    check(
        cos_bf >= COS_BF16_MIN,
        f"quantisation quality: cosine to the bf16-weight reference {cos_bf:.5f} >= {COS_BF16_MIN}",
    )
    same = sum(int(a == b) for a, b in zip(drafts, g["drafts_w4"].tolist()))
    check(
        same >= GOLDEN_CHAIN_MIN,
        f"draft chain: {same}/7 drafts equal the W4 reference chain {g['drafts_w4'].tolist()} (engine {drafts})",
    )
    print("OK")


if __name__ == "__main__":
    main()
