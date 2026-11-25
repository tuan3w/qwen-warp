"""The CUDA backend's capability table: what each kernel of kernels/ absorbs, its limits, and how a match binds to the
launcher's roles.

Nothing here hard-codes a model: head counts, head dim, rotary dim, rope frequencies, hidden and FFN widths come from
the graph and reach the kernels as arguments. A match is declined only where a kernel's real limits are not met (K a
multiple of 256 for bf16 weights and of 32 for Q4_0, head dim 64 / 128 / 256, at most 8 query heads per KV head, at
most STEP_ROWS rows, a KV cache format with a kernel).
"""

from __future__ import annotations

from ..compiler import Absorb, Cap, CapsTable, Pack, Region, Target, Verdict
from ..ir import Q4_0_BLOCK_ELEMS
from ..kvcache import FORMATS, RotatedInt4Format
from . import inkling, schedule

# GeForce RTX 3090 (sm_86): measured launch cost inside a CUDA graph, DRAM bandwidth, SM count. The cost model uses
# these on any GPU; the kernels themselves are built for the device the runtime runs on.
RTX3090 = Target(launch_us=4.0, dram_bandwidth=936e9, sms=82)

NUM_GEMV = "bf16 weights, fp32 accumulation, 32-lane butterfly sum (order depends on K only), one bf16 rounding"
NUM_Q4GEMV = NUM_GEMV.replace("bf16 weights", "Q4_0 weights dequantized in the kernel (fp32)")
HEAD_NUMERICS = (
    "fp32 logits = the fp32 accumulator, not rounded to bf16 (the interpreter rounds first),"
    " argmax lowest index on ties"
)


def _lin_shape(A, i):
    return A.graph.weights[A.node(i).attrs["weight"]].shape


def _norm_offset(m, A) -> float:
    """The offset of the rmsnorm(s) a match folds in front of its anchor (0.0 without one)."""
    return next((A.node(i).attrs.get("offset", 0.0) for i in m.pre if A.node(i).op == "rmsnorm"), 0.0)


def _norms_weighted(m, A) -> bool:
    """Every rmsnorm a match folds in front of its anchor carries a weight (the kernels have no weightless norm)."""
    return all("weight" in A.node(i).attrs for i in m.pre if A.node(i).op == "rmsnorm")


ACT_CODES = {"silu": 0, "gelu_tanh": 1}  # the GEMV kernels' `act`


def _act_of(m, A) -> str:
    """The gate activation of the silu_mul / gelu_mul a pair or row match absorbs."""
    n = A.node(m.pair if m.pair is not None else m.post[0])
    return {"silu_mul": "silu", "gelu_mul": "gelu_tanh"}[n.op]


def _gemv_consts(mode: int):
    def f(m, A):
        shapes = [_lin_shape(A, i) for i in m.anchors]
        # the pair kernel's N is the width of ONE of gate / up
        n = shapes[0][0] if mode == 1 else sum(s[0] for s in shapes)
        return {
            "N": n,
            "K": shapes[0][1],
            "T": A.rows_max,
            "mode": mode,
            "dtype": A.graph.types[A.node(m.anchors[0]).ins[0]].dtype,
            "offset": _norm_offset(m, A),
            "weighted": _norms_weighted(m, A),
            **({"act": _act_of(m, A)} if mode in (1, 4) else {}),
        }

    return f


def _gemv_info(c: dict) -> Verdict:
    return Verdict(ctas=schedule.gemv_ctas(c["N"], c["K"], c["mode"]))


def _norm_prologue_check(m, A):
    """The norm folded into a GEMV must normalise exactly the K columns the GEMV reads (the kernel computes row
    statistics over K)."""
    for i in m.pre:
        nn = A.node(i)
        if nn.op == "rmsnorm" and A.graph.types[nn.ins[0]].shape[-1] != m.consts["K"]:
            return "the norm's width is not the GEMV's K"
    return None


def _gemv_check(m, A):
    """A linear whose result is cast to fp32 is the lm_head: it belongs to g_head (fp32 logits + argmax)."""
    for i in m.anchors:
        if A.consumers(A.node(i).outs[0], "cast"):
            return "a linear feeding a cast is the lm_head: g_head's, not a bf16 GEMV"
    return _norm_prologue_check(m, A)


def _dual_check(m, A):
    if _lin_shape(A, m.anchors[0]) != _lin_shape(A, m.anchors[1]):
        return "gate and up differ in shape"
    return _norm_prologue_check(m, A)


def _row_check(m, A):
    """The linear is the gate (first input) of the gelu_mul; the other operand is a value computed outside the region
    that the kernel reads row by row."""
    gm = A.node(m.post[0])
    if gm.ins[0] != A.node(m.anchors[0]).outs[0] or gm.ins[1] == gm.ins[0]:
        return "the linear must be the first input of the gelu_mul, the other operand a value from outside"
    return _gemv_check(m, A)


def _head_check(m, A):
    """With a softcap the kernel stores only the capped logits: nothing else may read the cast's fp32 values."""
    if len(m.post) == 3:
        cast = A.node(m.post[0]).outs[0]
        if A.consumers(cast) != [m.post[1]] or cast in A.outputs:
            return "the logits before the softcap have other readers: the kernel writes the capped logits only"
    return None


def _post_norm_check(m, A):
    """x = add(h, rmsnorm(y)) [then scale by the layer's learned [1] weight]: the kernel stores the final stream, so a
    scale must be the only reader of the sum."""
    if len(m.post) == 2:
        sc, add = A.node(m.post[1]), A.node(m.post[0])
        if "weight" not in sc.attrs or A.graph.weights[sc.attrs["weight"]].shape != (1,):
            return "the scale is not by a learned [1] weight"
        if A.consumers(add.outs[0]) != [m.post[1]] or add.outs[0] in A.outputs:
            return "the sum has readers besides the scale: the kernel writes only the scaled stream"
    return None


def _ple_nodes(m, A):
    """(scale in, norm, embed, add, scale out, split) of a PLE match."""
    pre = [A.node(i) for i in m.pre]
    first = {n.op: n for n in reversed(pre)}
    return first["scale"], first["rmsnorm"], first["embed"], A.node(m.anchors[0]), A.node(m.post[0]), A.node(m.post[1])


def _ple_consts(m, A):
    _, norm, _, _, _, _ = _ple_nodes(m, A)
    x = A.graph.types[norm.ins[0]]
    return {
        "L": x.shape[-2] if len(x.shape) == 3 else 0,
        "D": x.shape[-1],
        "rank": len(x.shape),
        "T": A.rows_max,
        "dtype": x.dtype,
        "offset": norm.attrs.get("offset", 0.0),
        "weighted": "weight" in norm.attrs,
    }


def _ple_check(m, A):
    """The per-layer-embedding prelude: add(norm(reshape(scale(ctx))) reshaped back, embed(tokens)), scaled and split
    into one [T, D] slice per layer; both scales are constant factors."""
    g = A.graph
    sin, norm, emb, add, sout, split = _ple_nodes(m, A)
    L, D = m.consts["L"], m.consts["D"]
    if "factor" not in sin.attrs or "factor" not in sout.attrs:
        return "the prelude's scales are constant factors"
    r1, r2 = A.node(A.prod[norm.ins[0]]), A.node(A.prod[add.ins[0] if add.ins[0] != emb.outs[0] else add.ins[1]])
    if r1.ins != sin.outs or r2.ins != norm.outs or {*add.ins} != {r2.outs[0], emb.outs[0]}:
        return "not scale -> reshape -> norm -> reshape -> add with the embedding"
    if g.types[r2.outs[0]].shape[-1] != L * D or g.weights[emb.attrs["weight"]].shape[1] != L * D:
        return "the embedding row is not L x D wide"
    if emb.ins != ["tokens"] or list(split.attrs["sizes"]) != [D] * L:
        return "the embedding is not of the step's tokens, or the split is not L slices of D"
    return None


def _norm_consts(m, A):
    n = A.node(m.anchors[0])
    x = A.graph.types[n.ins[0]]
    # the kernel normalises rows of [T][H]: a [T][heads][hd] input is not its shape
    return {
        "H": x.shape[-1],
        "T": A.rows_max,
        "rank": len(x.shape),
        "dtype": x.dtype,
        "offset": n.attrs.get("offset", 0.0),
        "weighted": "weight" in n.attrs,
    }


def _attn_norms(m, A) -> tuple[list[int], list[int]]:
    """The rmsnorm nodes a match folds in: (those on the q / k the rope rotates, those on the attention's v)."""
    att = A.node(m.anchors[0])
    rp = A.prod.get(att.ins[0])
    rotated = set(A.node(rp).ins[:-1]) if rp is not None and A.node(rp).op == "rope" else set()
    v = att.ins[2] if att.op == "attention" else None
    norms = [i for i in m.pre if A.node(i).op == "rmsnorm"]
    return [i for i in norms if A.node(i).outs[0] in rotated], [i for i in norms if A.node(i).outs[0] == v]


def _attn_consts(m, A):
    g = A.graph
    att = A.node(m.anchors[0])
    st = g.states[att.attrs["state"]]
    _, ctx, hkv, hd = st.shape
    hq = g.types[att.ins[0]].shape[1]
    rp = A.prod.get(att.ins[0])
    rd = A.node(rp).attrs["rotary_dim"] if rp is not None and A.node(rp).op == "rope" else 0
    window = att.attrs.get("window")
    nc = schedule.attn_chunks(ctx, window)
    qk, vn = _attn_norms(m, A)
    return {
        "hd": hd,
        "Hq": hq,
        "Hkv": hkv,
        "G": hq // hkv if hkv and hq % hkv == 0 else 0,
        "ctx": ctx,
        "rd": rd,
        "T": A.rows_max,
        "cache": st.cache,
        "dtype": g.types[att.ins[0]].dtype,
        "nc": nc,
        "nsplit": schedule.attn_splits(ctx, nc, window),
        # the q / k norms' (the v norm has no weight and no offset)
        "offset": next((A.node(i).attrs.get("offset", 0.0) for i in qk), 0.0),
        "weighted": all("weight" in A.node(i).attrs for i in qk),
        "window": window,
        "v_norm": bool(vn),
    }


def _attn_info(c: dict) -> Verdict:
    why = []
    if c["G"] == 0:
        why.append("query heads are not a multiple of KV heads")
    if c["rd"] > c["hd"] or c["rd"] % 2:
        why.append("rotary dim must be even and at most the head dim")
    cap = schedule.attn_split_cap(c["cache"], c["nc"])
    if c["nsplit"] > cap:
        why.append(f"{c['nsplit']} split blocks per head (ctx {c['ctx']}, nc {c['nc']}) exceed the kernel's {cap}")
    # the split kernel's grid (Hkv, splits) at one row
    return Verdict(why, c["Hkv"] * min(c["nsplit"], cap))


def _attn_check(m, A):
    g = A.graph
    att = A.node(m.anchors[0])
    read = att.op == "attention_read"
    q = att.ins[0]
    rp = A.prod.get(q)
    if rp is None or A.node(rp).op != "rope" or A.node(rp).outs != ([q] if read else [q, att.ins[1]]):
        return "q must come from a rope node" if read else "q and k must come from one rope node"
    if not read and A.prod.get(att.ins[1]) != rp:
        return "q and k must come from one rope node"
    qk, vn = _attn_norms(m, A)
    norms = [A.node(i) for i in qk]
    if len(norms) not in (0, len(A.node(rp).ins) - 1):
        return "qk-norm must apply to both q and k or neither"
    if len({n.attrs["eps"] for n in (*norms, *(A.node(i) for i in vn))}) > 1:
        return "the q / k / v norms differ in eps"
    if len({n.attrs.get("offset", 0.0) for n in norms}) > 1:
        return "q and k norms differ in offset"
    if any("weight" in A.node(i).attrs for i in vn):
        return "the v norm carries a weight (the kernel's is weightless)"

    # the launcher reads the GEMV outputs through the reshapes: q and k reach the rope as (qk-norm of)
    # reshape(projection) and v as (a weightless norm of) reshape(projection); a norm anywhere else (full width before
    # the reshape) has no place in the kernel
    def reshaped(v_):
        pn = A.prod.get(v_)
        ok = (
            pn is not None
            and pn in m.covers
            and A.node(pn).op == "reshape"
            and A.prod.get(A.node(pn).ins[0]) not in m.covers
        )
        return pn if ok else None

    direct = []  # the norms that sit between a reshape and the rope (or the attention's v)
    sources = [*A.node(rp).ins[:-1], *([] if read else [att.ins[2]])]
    for v_ in sources:
        pn = A.prod.get(v_)
        if pn in m.covers and A.node(pn).op == "rmsnorm":
            direct.append(pn)
            v_ = A.node(pn).ins[0]
        if reshaped(v_) is None:
            return "q, k and v must be (a norm of) a reshape of a projection output: a norm elsewhere is no kernel's"
    if sorted(direct) != sorted(i for i in m.pre if A.node(i).op == "rmsnorm"):
        return "a norm in front of the attention does not sit between the reshape and the rope / attention"
    st = g.states[att.attrs["state"]]
    if st.cache not in FORMATS:
        return f"KV cache format {st.cache} has no kernel here"
    if vn and FORMATS[st.cache].quantized:
        return f"KV format {st.cache} has no weightless v-norm kernel"
    window, ctx = m.consts["window"], m.consts["ctx"]
    ctx_max = max(s.shape[1] for s in g.states.values() if s.kind == "kv")
    if window is not None and ctx < ctx_max and ctx < window + A.rows_max - 1:
        return f"a ring of {ctx} rows cannot hold a window of {window} keys and a step of {A.rows_max} rows"
    return schedule.kv_kernel_reason(FORMATS[st.cache], m.consts["hd"], window)


def _unfused_ctas(n, A) -> int:
    """CTAs of the kernel an unfused node would launch (the cost model's utilisation term)."""
    if n.op == "linear":
        w = A.graph.weights[n.attrs["weight"]]
        return schedule.gemv_ctas(w.shape[0], w.shape[1], 0)
    if n.op in ("attention", "attention_read"):
        _, ctx, hkv, _ = A.graph.states[n.attrs["state"]].shape
        return hkv * min(schedule.ATTN_MAX_SPLITS, max(1, schedule.attn_splits(ctx, 1, n.attrs.get("window"))))
    if n.op in ("rmsnorm", "embed"):
        return 1  # one CTA per row, priced at one decode row
    # ops this backend only runs fused: priced as an elementwise kernel of 256 threads x 8 bf16 per CTA over the
    # widest operand (a reduction is as wide as its input)
    return max(1, -(-max(A.elements(v) for v in (*n.ins, *n.outs)) // 2048))


def _with_norm(m, A, r: Region) -> Region:
    """Fold the matched rmsnorm into the GEMV's prologue: the raw residual becomes the input, the norm weight a second
    pack."""
    nn = next((A.node(i) for i in m.pre if A.node(i).op == "rmsnorm"), None)
    if nn is None:
        return r
    w = nn.attrs["weight"]
    r.ins = {**r.ins, "x": nn.ins[0]}
    r.weights = {**r.weights, "nw": w}
    r.packs = [*r.packs, Pack(w, "dense", [w])]
    r.attrs = {**r.attrs, "eps": nn.attrs["eps"]}
    return r


def _emit_embed(m, A):
    n = A.node(m.anchors[0])
    w = n.attrs["weight"]
    return Region(
        "g_embed",
        set(m.covers),
        ins={"tok": "@tokens"},
        outs={"x": n.outs[0]},
        weights={"table": w},
        attrs={
            "H": A.graph.weights[w].shape[1],
            **({"scale": n.attrs["scale"]} if n.attrs.get("scale", 1.0) != 1.0 else {}),
        },
        packs=[Pack(w, "dense", [w])],
        numerics="row gather, scaled by bf16(scale) when the node carries one",
    )


def _emit_rmsnorm(m, A):
    n = A.node(m.anchors[0])
    w = n.attrs["weight"]
    attrs = {"H": A.graph.types[n.ins[0]].shape[-1], "eps": n.attrs["eps"]}
    if not m.pre:
        return Region(
            "g_rmsnorm",
            set(m.covers),
            ins={"x": n.ins[0]},
            outs={"xn": n.outs[0]},
            weights={"w": w},
            packs=[Pack(w, "dense", [w])],
            attrs=attrs,
            numerics="fp32 rmsnorm, one rounding",
        )
    an = A.node(m.inplace_node)
    y = an.ins[1] if an.ins[0] == m.h else an.ins[0]
    return Region(
        "g_rmsnorm_add",
        set(m.covers),
        ins={"h": m.h, "y": y},
        outs={"h_out": an.outs[0], "xn": n.outs[0]},
        weights={"w": w},
        packs=[Pack(w, "dense", [w])],
        attrs=attrs,
        inplace=("h_out", "h"),
        numerics="h = bf16(h + y) in place, then fp32 rmsnorm with one rounding",
    )


def _emit_rmsnorm_post(m, A):
    rn = A.node(m.anchors[0])
    w = rn.attrs["weight"]
    weights, packs = {"w": w}, [Pack(w, "dense", [w])]
    numerics = "h = bf16(h + bf16(fp32 rmsnorm)) in place"
    if len(m.post) == 2:
        s = A.node(m.post[1]).attrs["weight"]
        weights["s"] = s
        packs.append(Pack(s, "dense", [s]))
        numerics += ", then h = bf16(h * the layer scalar)"
    return Region(
        "g_rmsnorm_post",
        set(m.covers),
        ins={"x": rn.ins[0], "h": m.h},
        outs={"h_out": A.node(m.post[-1]).outs[0]},
        weights=weights,
        packs=packs,
        attrs={"H": A.graph.types[rn.ins[0]].shape[-1], "eps": rn.attrs["eps"]},
        inplace=("h_out", "h"),
        numerics=numerics,
    )


def _emit_ple(m, A):
    sin, norm, emb, _, sout, split = _ple_nodes(m, A)
    w, table = norm.attrs["weight"], emb.attrs["weight"]
    return Region(
        "g_ple",
        set(m.covers),
        ins={"ctx": sin.ins[0], "tok": "@tokens"},
        outs={f"p{j}": v for j, v in enumerate(split.outs)},
        weights={"w": w, "table": table},
        packs=[Pack(w, "dense", [w]), Pack(table, "dense", [table])],
        attrs={
            "L": m.consts["L"],
            "D": m.consts["D"],
            "eps": norm.attrs["eps"],
            "ctx_scale": sin.attrs["factor"],
            "out_scale": sout.attrs["factor"],
            "scale": emb.attrs.get("scale", 1.0),
        },
        numerics="fp32 rmsnorm over each D-wide slice (thread partials of 8 in order, butterfly, warp sums in order);"
        " every op boundary rounded to bf16 as the interpreter does",
    )


def _emit_segments(kernel: str, cat_kind: str, one_kind: str, numerics: str):
    def emit(m, A):
        lins = [A.node(i) for i in m.anchors]
        ws = [n.attrs["weight"] for n in lins]
        ns = [A.graph.weights[w].shape[0] for w in ws]
        name = "|".join(ws)
        r = Region(
            kernel,
            set(m.covers),
            ins={"x": lins[0].ins[0]},
            outs={f"o{j}": n.outs[0] for j, n in enumerate(lins)},
            weights={"w": name},
            packs=[Pack(name, cat_kind if len(ws) > 1 else one_kind, ws)],
            attrs={"N": sum(ns), "K": A.graph.weights[ws[0]].shape[1], "segs": ns},
            numerics=numerics,
        )
        return _with_norm(m, A, r)

    return emit


def _emit_dual(kernel: str, kind: str, numerics: str):
    def emit(m, A):
        lg, lu = A.node(m.anchors[0]), A.node(m.anchors[1])
        wg, wu = lg.attrs["weight"], lu.attrs["weight"]
        r = Region(
            kernel,
            set(m.covers),
            ins={"x": lg.ins[0]},
            outs={"out": A.node(m.pair).outs[0]},
            weights={"w": wg, "w2": wu},
            packs=[Pack(wg, kind, [wg]), Pack(wu, kind, [wu])],
            attrs={
                "N": A.graph.weights[wg].shape[0],
                "K": A.graph.weights[wg].shape[1],
                **({"act": ACT_CODES[_act_of(m, A)]} if _act_of(m, A) != "silu" else {}),
            },
            numerics=numerics + f"; {_act_of(m, A)} emulated at bf16 between the two roundings",
        )
        return _with_norm(m, A, r)

    return emit


def _emit_resadd(kernel: str, kind: str, numerics: str):
    def emit(m, A):
        ln, ad = A.node(m.anchors[0]), A.node(m.post[0])
        w = A.graph.weights[ln.attrs["weight"]]
        return Region(
            kernel,
            set(m.covers),
            ins={"x": ln.ins[0], "h": m.h},
            outs={"h_out": ad.outs[0]},
            weights={"w": w.name},
            packs=[Pack(w.name, kind, [w.name])],
            attrs={"N": w.shape[0], "K": w.shape[1]},
            inplace=("h_out", "h"),
            numerics=numerics + "; residual add in the epilogue",
        )

    return emit


def _emit_row(kernel: str, kind: str, numerics: str):
    def emit(m, A):
        lin, gm = A.node(m.anchors[0]), A.node(m.post[0])
        w = A.graph.weights[lin.attrs["weight"]]
        return Region(
            kernel,
            set(m.covers),
            ins={"x": lin.ins[0], "row": gm.ins[1]},
            outs={"out": gm.outs[0]},
            weights={"w": w.name},
            packs=[Pack(w.name, kind, [w.name])],
            attrs={"N": w.shape[0], "K": w.shape[1], "act": ACT_CODES[_act_of(m, A)]},
            numerics=numerics + f"; {_act_of(m, A)} emulated at bf16, then the product with the row operand",
        )

    return emit


def _emit_head(m, A):
    ln, cast, amax = A.node(m.anchors[0]), A.node(m.post[0]), A.node(m.post[-1])
    cap = A.node(m.post[1]) if len(m.post) == 3 else None
    w = A.graph.weights[ln.attrs["weight"]]
    return Region(
        "g_head",
        set(m.covers),
        ins={"x": ln.ins[0]},
        outs={"logits": (cap or cast).outs[0], "ids": amax.outs[0]},
        weights={"w": w.name},
        packs=[Pack(w.name, "dense", [w.name])],
        attrs={"N": w.shape[0], "K": w.shape[1], **({"cap": cap.attrs["cap"]} if cap else {})},
        numerics=HEAD_NUMERICS + ("; softcap in fp32 on the accumulator" if cap else ""),
    )


def _emit_attn(m, A):
    g = A.graph
    n = A.node(m.anchors[0])
    read = n.op == "attention_read"
    rpn = A.node(A.prod[n.ins[0]])
    norms, srcs = [], []
    for v_ in rpn.ins[:-1]:
        rn = A.prod.get(v_)
        if rn in m.covers and A.node(rn).op == "rmsnorm":
            norms.append(A.node(rn))
            v_ = A.node(rn).ins[0]
        srcs.append(A.node(A.prod[v_]).ins[0])  # the reshape's input: the GEMV output the kernel reads in place
    ins, vnorm = {"q": srcs[0]}, None
    if not read:
        v_ = n.ins[2]
        if A.prod.get(v_) in m.covers and A.node(A.prod[v_]).op == "rmsnorm":
            vnorm = A.node(A.prod[v_])
            v_ = vnorm.ins[0]
        ins = {**ins, "k": srcs[1], "v": A.node(A.prod[v_]).ins[0]}
    st = g.states[n.attrs["state"]]
    fmt = FORMATS[st.cache]
    weights = {"q_norm": norms[0].attrs["weight"], "k_norm": norms[1].attrs["weight"]} if norms and not read else {}
    if norms and read:
        weights = {"q_norm": norms[0].attrs["weight"]}
    prep = "q prep (norm, rope) + " if read else "prep (qk norm, rope, KV write) + "
    numerics = prep + "split-K flash decode + reduce; fp32 softmax, one bf16 rounding of the output"
    if vnorm is not None:
        numerics += "; weightless v norm in the prep"
    if n.attrs.get("window") is not None:
        numerics += f"; window {n.attrs['window']}: chunks below it are skipped, ring slot = position mod rows"
    if isinstance(fmt, RotatedInt4Format):
        numerics += (
            f"; KV cache {st.cache}: the hand engine's rotated int4 (WHT, 16-level companded grid, 8-bit log2 scale per"
            " 64 rotated coefficients, the engine's writer bit for bit), fp16 query, levels decoded in fp16,"
            f" first {fmt.sink} positions and the {fmt.window} before the step's last row exact (fp16)"
        )
    elif fmt.quantized:
        numerics += (
            f"; KV cache {st.cache}: rotated (WHT) int{fmt.k_bits} K and int{fmt.v_bits} V"
            " with one fp16 scale per (token, head), quantised on write in prep"
            " and dequantised in registers in the split kernel"
        )
        if fmt.exact_rows:
            numerics += f", first {fmt.sink} and newest {fmt.window} positions exact (fp16)"
    eps = next((x.attrs["eps"] for x in (*norms, *([vnorm] if vnorm is not None else []))), None)
    return Region(
        "g_attn_read" if read else "g_attn",
        set(m.covers),
        ins=ins,
        outs={"y": n.outs[0]},
        weights=weights,
        states={"kv": n.attrs["state"]},
        constants={"inv_freq": rpn.attrs["inv_freq"]},
        attrs={
            "Hq": m.consts["Hq"],
            "Hkv": m.consts["Hkv"],
            "hd": m.consts["hd"],
            "rd": rpn.attrs["rotary_dim"],
            "ctx": m.consts["ctx"],
            "scale": n.attrs["scale"],
            "qk_norm": bool(norms),
            **({"eps": eps} if eps is not None else {}),
            "cache": st.cache,
            "nc": m.consts["nc"],
            **({"window": n.attrs["window"]} if n.attrs.get("window") is not None else {}),
            **({"v_norm": True} if vnorm is not None else {}),
        },
        packs=[Pack(w, "dense", [w]) for w in weights.values()],
        numerics=numerics,
    )


_NORM_PRO = Absorb("rmsnorm", min=0, max=1)
# h = bf16(h + bf16(y)), the stream written back in place
_RESADD = Absorb("add", min=1, max=1, exposes=True, inplace=True)
_ROWS = schedule.STEP_ROWS
# K: k_gemv reads 8 bf16 per lane and 32 lanes per iteration; k_q4_gemv one Q4_0 block per lane. offset: the norms
# compute x-hat * w (a folded prologue norm included), not the zero-centred x-hat * (1 + w)
_GEMV_LIMITS = {
    "K_multiple": 256,
    "T_max": _ROWS,
    "dtype_in": ("bf16",),
    "offset_in": (0.0,),
    "weighted_in": (True,),
}
_Q4GEMV_LIMITS = {"K_multiple": Q4_0_BLOCK_ELEMS, "T_max": _ROWS, "dtype_in": ("bf16",)}
_NORM_LIMITS = {
    "H_multiple": 8,
    "T_max": _ROWS,
    "rank_in": (2,),
    "dtype_in": ("bf16",),
    "offset_in": (0.0,),
    "weighted_in": (True,),
}
# kernels/gemma_rows.cu PLE_MAX_LAYERS: the layer slices k_ple writes through a by-value pointer array
PLE_MAX_LAYERS = 64

# hd 512, a window or a ring only with a bf16 cache (schedule.kv_kernel_reason); offset: the q / k norms compute x-hat*w
_ATTN_LIMITS = {
    "hd_in": schedule.ATTN_HEAD_DIMS,
    "G_max": 8,
    "T_max": _ROWS,
    "dtype_in": ("bf16",),
    "offset_in": (0.0,),
    "weighted_in": (True,),
}

# attention `launches`: prep + split + merge, as a prompt step (T > 1) runs them. A decode row (T = 1) of head dim
# 256 / 512 is ONE launch (k_attn_decode1: prep, split and a last-block merge; two when the state has more than 32
# chunks of 32 keys, i.e. past 1024 positions).
# order = tie-break of the cost model: the kernels with the most absorbed work first
CAPS = [
    *inkling.CAPS,
    Cap(
        "g_attn",
        "attention",
        # q, k, v reshapes; q / k norms (weighted) and v norm (weightless); rope on q and k
        pre=(Absorb("reshape", min=3, max=3), Absorb("rmsnorm", min=0, max=3), Absorb("rope", min=1, max=1)),
        limits=_ATTN_LIMITS,
        rounds={"rmsnorm": ("out",), "rope": ("out",), "attention": ("out",)},
        tolerance=("rope.trig", "rope.prod", "attention.p"),
        launches=3,
        constants=_attn_consts,
        info=_attn_info,
        check=_attn_check,
        emit=_emit_attn,
    ),
    Cap(
        "g_attn_read",
        "attention_read",
        # the q reshape, q norm (weighted) and rope; no KV write: the state is an earlier task's
        pre=(Absorb("reshape", min=1, max=1), Absorb("rmsnorm", min=0, max=1), Absorb("rope", min=1, max=1)),
        limits=_ATTN_LIMITS,
        rounds={"rmsnorm": ("out",), "rope": ("out",), "attention_read": ("out",)},
        tolerance=("rope.trig", "rope.prod", "attention_read.p"),
        launches=3,
        constants=_attn_consts,
        info=_attn_info,
        check=_attn_check,
        emit=_emit_attn,
    ),
    Cap(
        "g_gemv_resadd",
        "linear",
        anchor_schemes=("dense",),
        post=(_RESADD,),
        limits=_GEMV_LIMITS,
        rounds={"linear": ("out",), "add": ("out",)},
        constants=_gemv_consts(3),
        info=_gemv_info,
        emit=_emit_resadd("g_gemv_resadd", "dense", NUM_GEMV),
    ),
    Cap(
        "g_gemv_dual",
        "linear",
        anchor_schemes=("dense",),
        pair=("silu_mul", "gelu_mul"),
        pre=(_NORM_PRO,),
        limits={**_GEMV_LIMITS, "act_in": tuple(ACT_CODES)},
        rounds={
            "linear": ("out",),
            "rmsnorm": ("out",),
            "silu_mul": ("silu", "out"),
            "gelu_mul": ("gelu", "out"),
        },
        constants=_gemv_consts(1),
        info=_gemv_info,
        check=_dual_check,
        emit=_emit_dual("g_gemv_dual", "dense", NUM_GEMV),
    ),
    Cap(
        "g_gemv_row",
        "linear",
        anchor_schemes=("dense",),
        post=(Absorb("gelu_mul", min=1, max=1),),
        limits={**_GEMV_LIMITS, "act_in": ("gelu_tanh",)},
        rounds={"linear": ("out",), "gelu_mul": ("gelu", "out")},
        constants=_gemv_consts(4),
        info=_gemv_info,
        check=_row_check,
        emit=_emit_row("g_gemv_row", "dense", NUM_GEMV),
    ),
    Cap(
        "g_head",
        "linear",
        anchor_schemes=("dense",),
        post=(
            Absorb("cast", min=1, max=1, exposes=True, attrs={"dtype": "f32"}),
            Absorb("softcap", min=0, max=1, exposes=True),
            Absorb("argmax", min=1, max=1, exposes=True, reduction_ok=True),
        ),
        limits=_GEMV_LIMITS,
        rounds={"cast": ("out",)},
        # GEMV mode 2 stores the fp32 accumulator unrounded; the interpreter rounds it to bf16 first
        tolerance=("linear.out",),
        launches=3,
        constants=_gemv_consts(2),
        info=_gemv_info,
        check=_head_check,
        emit=_emit_head,
    ),
    Cap(
        "g_q4gemv_resadd",
        "linear",
        anchor_schemes=("q4_0",),
        post=(_RESADD,),
        limits=_Q4GEMV_LIMITS,
        rounds={"linear": ("out",), "add": ("out",)},
        constants=_gemv_consts(3),
        info=_gemv_info,
        emit=_emit_resadd("g_q4gemv_resadd", "q4_0", NUM_Q4GEMV),
    ),
    Cap(
        "g_q4gemv_dual",
        "linear",
        anchor_schemes=("q4_0",),
        pair=("silu_mul",),
        limits=_Q4GEMV_LIMITS,
        rounds={"linear": ("out",), "silu_mul": ("silu", "out")},
        constants=_gemv_consts(1),
        info=_gemv_info,
        check=_dual_check,
        emit=_emit_dual("g_q4gemv_dual", "q4_0", NUM_Q4GEMV),
    ),
    Cap(
        "g_q4gemv",
        "linear",
        anchor_schemes=("q4_0",),
        segments=3,
        limits=_Q4GEMV_LIMITS,
        rounds={"linear": ("out",)},
        constants=_gemv_consts(0),
        info=_gemv_info,
        check=_gemv_check,
        emit=_emit_segments("g_q4gemv", "q4_0_cat", "q4_0", NUM_Q4GEMV),
    ),
    Cap(
        "g_gemv",
        "linear",
        anchor_schemes=("dense",),
        segments=3,
        pre=(_NORM_PRO,),
        limits=_GEMV_LIMITS,
        rounds={"linear": ("out",), "rmsnorm": ("out",)},
        constants=_gemv_consts(0),
        info=_gemv_info,
        check=_gemv_check,
        emit=_emit_segments("g_gemv", "dense_cat", "dense", NUM_GEMV),
    ),
    Cap(
        "g_rmsnorm_post",
        "rmsnorm",
        # x = add(h, rmsnorm(y)), then the layer's learned scalar: h is updated in place
        post=(_RESADD, Absorb("scale", min=0, max=1, exposes=True)),
        limits={**_NORM_LIMITS, "H_max": 8192},
        rounds={"rmsnorm": ("out",), "add": ("out",), "scale": ("out",)},
        constants=_norm_consts,
        check=_post_norm_check,
        emit=_emit_rmsnorm_post,
    ),
    Cap(
        "g_rmsnorm_add",
        "rmsnorm",
        pre=(_RESADD,),
        limits=_NORM_LIMITS,
        rounds={"add": ("out",), "rmsnorm": ("out",)},
        constants=_norm_consts,
        emit=_emit_rmsnorm,
    ),
    Cap(
        "g_rmsnorm",
        "rmsnorm",
        limits=_NORM_LIMITS,
        rounds={"rmsnorm": ("out",)},
        constants=_norm_consts,
        emit=_emit_rmsnorm,
    ),
    Cap(
        "g_embed",
        "embed",
        limits={"H_multiple": 8},
        rounds={"embed": ("out",)},
        constants=lambda m, A: {"H": A.graph.weights[A.node(m.anchors[0]).attrs["weight"]].shape[1]},
        emit=_emit_embed,
    ),
    Cap(
        "g_ple",
        "add",
        # the per-layer-embedding prelude: the projection's scale, its reshape / norm / reshape, the scaled token
        # embedding, their sum, the sum's scale and the split into one slice per layer
        pre=(
            Absorb("scale", min=1, max=1),
            Absorb("reshape", min=2, max=2),
            Absorb("rmsnorm", min=1, max=1),
            Absorb("embed", min=1, max=1),
        ),
        post=(Absorb("scale", min=1, max=1), Absorb("split", min=1, max=1, exposes=True)),
        limits={
            "D_multiple": 256,
            "D_max": 1024,
            "L_max": PLE_MAX_LAYERS,
            "rank_in": (3,),
            "T_max": _ROWS,
            "dtype_in": ("bf16",),
            "offset_in": (0.0,),
            "weighted_in": (True,),
        },
        rounds={"scale": ("out",), "rmsnorm": ("out",), "embed": ("out",), "add": ("out",)},
        operand_cap=PLE_MAX_LAYERS + 4,  # the slices, the projection and the tokens, the norm weight and the table
        constants=_ple_consts,
        check=_ple_check,
        emit=_emit_ple,
    ),
]

# an unfused attention is still prep + split + merge; a reshape is a view
TABLE = CapsTable(CAPS, RTX3090, _unfused_ctas, {"attention": 3, "attention_read": 3, "reshape": 0, "split": 0})
