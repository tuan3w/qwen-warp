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
from . import schedule

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


def _norm_consts(m, A):
    x = A.graph.types[A.node(m.anchors[0]).ins[0]]
    # the kernel normalises rows of [T][H]: a [T][heads][hd] input is not its shape
    return {"H": x.shape[-1], "T": A.rows_max, "rank": len(x.shape), "dtype": x.dtype}


def _attn_consts(m, A):
    g = A.graph
    att = A.node(m.anchors[0])
    st = g.states[att.attrs["state"]]
    _, ctx, hkv, hd = st.shape
    hq = g.types[att.ins[0]].shape[1]
    rp = A.prod.get(att.ins[0])
    rd = A.node(rp).attrs["rotary_dim"] if rp is not None and A.node(rp).op == "rope" else 0
    nc = schedule.attn_chunks(ctx)
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
        "nsplit": schedule.attn_splits(ctx, nc),
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
    q, k = att.ins[0], att.ins[1]
    rp = A.prod.get(q)
    if rp is None or A.prod.get(k) != rp or A.node(rp).outs != [q, k]:
        return "q and k must come from one rope node"
    norms = [A.node(i) for i in m.pre if A.node(i).op == "rmsnorm"]
    if len(norms) not in (0, 2):
        return "qk-norm must apply to both q and k or neither"
    if norms and norms[0].attrs["eps"] != norms[1].attrs["eps"]:
        return "q and k norms differ in eps"

    # the launcher reads the GEMV outputs through the reshapes: q and k reach the rope as (qk-norm of)
    # reshape(projection) and v as reshape(projection); a norm anywhere else (full width before the reshape, or on v)
    # has no place in the kernel
    def reshaped(v_):
        pn = A.prod.get(v_)
        ok = (
            pn is not None
            and pn in m.covers
            and A.node(pn).op == "reshape"
            and A.prod.get(A.node(pn).ins[0]) not in m.covers
        )
        return pn if ok else None

    direct = []  # the norms that sit between a reshape and the rope
    for v_ in A.node(rp).ins[:2]:
        pn = A.prod.get(v_)
        if pn in m.covers and A.node(pn).op == "rmsnorm":
            direct.append(pn)
            v_ = A.node(pn).ins[0]
        if reshaped(v_) is None:
            return (
                "q and k must be (a qk-norm of) a reshape of a projection output: a norm elsewhere is not the kernel's"
            )
    if reshaped(att.ins[2]) is None:
        return "v must be a reshape of a projection output"
    if sorted(direct) != sorted(i for i in m.pre if A.node(i).op == "rmsnorm"):
        return "a norm in front of the attention does not sit between the q / k reshape and the rope"
    cache = g.states[att.attrs["state"]].cache
    if cache not in FORMATS:
        return f"KV cache format {cache} has no kernel here"
    return schedule.kv_kernel_reason(FORMATS[cache], m.consts["hd"])


def _unfused_ctas(n, A) -> int:
    """CTAs of the kernel an unfused node would launch (the cost model's utilisation term)."""
    if n.op == "linear":
        w = A.graph.weights[n.attrs["weight"]]
        return schedule.gemv_ctas(w.shape[0], w.shape[1], 0)
    if n.op == "attention":
        _, ctx, hkv, _ = A.graph.states[n.attrs["state"]].shape
        return hkv * min(schedule.ATTN_MAX_SPLITS, max(1, schedule.attn_splits(ctx, 1)))
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
        attrs={"H": A.graph.weights[w].shape[1]},
        packs=[Pack(w, "dense", [w])],
        numerics="row gather",
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
            attrs={"N": A.graph.weights[wg].shape[0], "K": A.graph.weights[wg].shape[1]},
            numerics=numerics + "; silu emulated at bf16 between the two roundings",
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


def _emit_head(m, A):
    ln, cast, amax = A.node(m.anchors[0]), A.node(m.post[0]), A.node(m.post[1])
    w = A.graph.weights[ln.attrs["weight"]]
    return Region(
        "g_head",
        set(m.covers),
        ins={"x": ln.ins[0]},
        outs={"logits": cast.outs[0], "ids": amax.outs[0]},
        weights={"w": w.name},
        packs=[Pack(w.name, "dense", [w.name])],
        attrs={"N": w.shape[0], "K": w.shape[1]},
        numerics=HEAD_NUMERICS,
    )


def _emit_attn(m, A):
    g = A.graph
    n = A.node(m.anchors[0])
    q, _k, v, _pos = n.ins
    rpn = A.node(A.prod[q])
    norms, srcs = [], []
    for v_ in rpn.ins[:2]:
        rn = A.prod.get(v_)
        if rn in m.covers and A.node(rn).op == "rmsnorm":
            norms.append(A.node(rn))
            v_ = A.node(rn).ins[0]
        srcs.append(A.node(A.prod[v_]).ins[0])  # the reshape's input: the GEMV output the kernel reads in place
    st = g.states[n.attrs["state"]]
    fmt = FORMATS[st.cache]
    weights = {"q_norm": norms[0].attrs["weight"], "k_norm": norms[1].attrs["weight"]} if norms else {}
    numerics = (
        "prep (qk norm, rope, KV write) + split-K flash decode + reduce; fp32 softmax, one bf16 rounding of the output"
    )
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
    return Region(
        "g_attn",
        set(m.covers),
        ins={"q": srcs[0], "k": srcs[1], "v": A.node(A.prod[v]).ins[0]},
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
            **({"eps": norms[0].attrs["eps"]} if norms else {}),
            "cache": st.cache,
            "nc": m.consts["nc"],
        },
        packs=[Pack(w, "dense", [w]) for w in weights.values()],
        numerics=numerics,
    )


_NORM_PRO = Absorb("rmsnorm", min=0, max=1)
# h = bf16(h + bf16(y)), the stream written back in place
_RESADD = Absorb("add", min=1, max=1, exposes=True, inplace=True)
_ROWS = schedule.STEP_ROWS
# K: k_gemv reads 8 bf16 per lane and 32 lanes per iteration; k_q4_gemv one Q4_0 block per lane
_GEMV_LIMITS = {"K_multiple": 256, "T_max": _ROWS, "dtype_in": ("bf16",)}
_Q4GEMV_LIMITS = {"K_multiple": Q4_0_BLOCK_ELEMS, "T_max": _ROWS, "dtype_in": ("bf16",)}
_NORM_LIMITS = {"H_multiple": 8, "T_max": _ROWS, "rank_in": (2,), "dtype_in": ("bf16",)}

# order = tie-break of the cost model: the kernels with the most absorbed work first
CAPS = [
    Cap(
        "g_attn",
        "attention",
        pre=(Absorb("reshape", min=3, max=3), Absorb("rmsnorm", min=0, max=2), Absorb("rope", min=1, max=1)),
        limits={"hd_in": schedule.ATTN_HEAD_DIMS, "G_max": 8, "T_max": _ROWS, "dtype_in": ("bf16",)},
        rounds={"rmsnorm": ("out",), "rope": ("out",), "attention": ("out",)},
        tolerance=("rope.trig", "rope.prod", "attention.p"),
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
        pair=("silu_mul",),
        pre=(_NORM_PRO,),
        limits=_GEMV_LIMITS,
        rounds={"linear": ("out",), "rmsnorm": ("out",), "silu_mul": ("silu", "out")},
        constants=_gemv_consts(1),
        info=_gemv_info,
        check=_dual_check,
        emit=_emit_dual("g_gemv_dual", "dense", NUM_GEMV),
    ),
    Cap(
        "g_head",
        "linear",
        anchor_schemes=("dense",),
        post=(
            Absorb("cast", min=1, max=1, exposes=True, attrs={"dtype": "f32"}),
            Absorb("argmax", min=1, max=1, exposes=True, reduction_ok=True),
        ),
        limits=_GEMV_LIMITS,
        rounds={"cast": ("out",)},
        # GEMV mode 2 stores the fp32 accumulator unrounded; the interpreter rounds it to bf16 first
        tolerance=("linear.out",),
        launches=3,
        constants=_gemv_consts(2),
        info=_gemv_info,
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
    Cap("g_embed", "embed", rounds={"embed": ("out",)}, emit=_emit_embed),
]

# an unfused attention is still prep + split + merge; a reshape is a view
TABLE = CapsTable(CAPS, RTX3090, _unfused_ctas, {"attention": 3, "reshape": 0})
