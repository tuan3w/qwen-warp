"""Stage 2 family 6: the Gated DeltaNet decode of the Qwen3.5 hybrid.

A family module exports `CAPS` (capabilities appended to the CUDA table), `LAUNCHERS` (kernel name -> `(lib, mem, task,
T, stream)`), `SIGNATURES` (ctypes signatures of its extern "C" entries, merged into glue.SIGNATURES) and `PACKS` (pack
kind -> `(packer, pack) -> device tensor`, merged into pack.Packer).

`g_deltanet` is the whole mixer of a linear-attention layer in one launch (`kernels/qwen35_deltanet*.cu`): conv1d +
SiLU, the l2 norms, the gates, the delta rule on the in-place fp32 state [Hv, dk, dv] and the gated RMSNorm, for 1
to 8 rows with the state read and written once per launch. Its numerics are the hand engine's (`s_deltanet`), not the
interpreter's: conv and SiLU stay fp32, beta and o are not rounded to bf16, the gated norm rounds once at the end.
"""

from __future__ import annotations

import ctypes as C

from ..compiler import Absorb, Cap, Pack, Region, Verdict

ROWS_MAX = 8  # kernels/qwen35_deltanet.cuh TMAX
# (dk, dv, W) of the instantiated kernels: DN_CONFIGS in kernels/qwen35_deltanet.cuh (test_deltanet checks they agree)
CONFIGS = ((128, 128, 4),)
WARPS = 4  # NW_TUNED of the header: warps per CTA


def _ctas(T: int, dk: int, dv: int, hv: int) -> int:
    """CTAs of a launch (the header's lpr_for / nw_for): a warp owns 128 / LPR columns of dv."""
    cols = 128 // (16 if T <= 2 else 8)
    n = WARPS
    while n > 1 and (dv % (cols * n) or (dk // 32) % n or (dv // 32) % n):
        n //= 2
    return hv * (dv // (cols * n))


_PTR, _INT, _FLOAT, _SIZE = C.c_void_p, C.c_int, C.c_float, C.c_size_t
SIGNATURES: dict[str, tuple[list, type]] = {
    "warpc_deltanet_ws_bytes": ([_INT, _INT, _INT], _SIZE),
    "warpc_deltanet_step": ([_INT, *[_PTR] * 12, *[_INT] * 6, *[_FLOAT] * 3, _PTR], _INT),
}


def _w(A, i: int, key: str):
    return A.graph.weights[A.node(i).attrs[key]]


def _ttype(A, v: str):
    return A.graph.types[v]


def _nodes(m, A) -> dict:
    """The covered nodes by role: conv, silu, split, gdr (the anchor), gnorm, out (the final reshape)."""
    by = {A.node(i).op: i for i in m.covers if A.node(i).op != "reshape"}
    return {
        "conv": by["causal_conv1d"],
        "silu": by["silu"],
        "split": by["split"],
        "gdr": m.anchors[0],
        "gnorm": m.post[0],
        "out": m.post[1],
    }


def _consts(m, A):
    r = _nodes(m, A)
    gdr, gn = A.node(r["gdr"]), A.node(r["gnorm"])
    cw = _w(A, r["conv"], "weight")
    q, v = _ttype(A, gdr.ins[0]), _ttype(A, gdr.ins[2])
    return {
        "C": cw.shape[0],
        "W": cw.shape[1],
        "Hk": q.shape[1],
        "dk": q.shape[2],
        "Hv": v.shape[1],
        "dv": v.shape[2],
        "split": tuple(A.node(r["split"]).attrs["sizes"]),
        "eps": gdr.attrs["eps"],
        "norm_eps": gn.attrs["eps"],
        "scale": gdr.attrs["scale"],
        "T": A.rows_max,
        "dtype": _ttype(A, A.node(r["conv"]).ins[0]).dtype,
    }


def _info(c: dict) -> Verdict:
    why = []
    if c["Hv"] % c["Hk"]:
        why.append(f"Hv={c['Hv']} is not a multiple of Hk={c['Hk']}")
    if (c["dk"], c["dv"], c["W"]) not in CONFIGS:
        why.append(f"(dk, dv, W)=({c['dk']}, {c['dv']}, {c['W']}) not in {CONFIGS}")
    if c["split"] != (c["Hk"] * c["dk"], c["Hk"] * c["dk"], c["Hv"] * c["dv"]) or c["C"] != sum(c["split"]):
        why.append(f"the conv channels {c['C']} are not split {c['split']} into q | k | v of the head shapes")
    return Verdict(why, _ctas(c["T"], c["dk"], c["dv"], c["Hv"]))


def _check(m, A):
    """The kernel reads qkv as the conv input and q | k | v as consecutive column ranges of it, z as the gated norm's
    gate before its head reshape, and writes the flat [T, Hv*dv] output."""
    r = _nodes(m, A)
    sp, gdr, gn = A.node(r["split"]), A.node(r["gdr"]), A.node(r["gnorm"])
    if A.node(r["silu"]).ins[0] != A.node(r["conv"]).outs[0] or sp.ins[0] != A.node(r["silu"]).outs[0]:
        return "the split must read silu(causal_conv1d(x))"
    for j, v in enumerate(gdr.ins[:3]):
        p = A.prod.get(v)
        if p not in m.covers or A.node(p).op != "reshape" or A.node(p).ins[0] != sp.outs[j]:
            return "q, k, v must be the head reshapes of the split's outputs, in order"
    if gn.ins[0] != gdr.outs[0]:
        return "the gated norm must normalise the delta rule's output"
    zp = A.prod.get(gn.ins[1])
    if zp not in m.covers or A.node(zp).op != "reshape" or A.prod.get(A.node(zp).ins[0]) in m.covers:
        return "the gated norm's gate must be a head reshape of an outside value"
    _, hv, dv = _ttype(A, gdr.ins[2]).shape
    if _ttype(A, A.node(r["out"]).outs[0]).shape[1:] != (hv * dv,):
        return "the output must be flattened to [T, Hv * dv]"
    return None


def _emit(m, A):
    r = _nodes(m, A)
    conv, gdr, gn = A.node(r["conv"]), A.node(r["gdr"]), A.node(r["gnorm"])
    weights = {
        "conv_w": conv.attrs["weight"],
        "A_log": gdr.attrs["a_log"],
        "dt_bias": gdr.attrs["dt_bias"],
        "norm_w": gn.attrs["weight"],
    }
    c = m.consts
    return Region(
        "g_deltanet",
        set(m.covers),
        ins={"qkv": conv.ins[0], "z": A.node(A.prod[gn.ins[1]]).ins[0], "a": gdr.ins[3], "b": gdr.ins[4]},
        outs={"y": A.node(r["out"]).outs[0]},
        weights=weights,
        states={"conv": conv.attrs["state"], "rec": gdr.attrs["state"]},
        packs=[Pack(w, "dense", [w]) for w in weights.values()],
        attrs={
            **{k: c[k] for k in ("C", "W", "Hk", "Hv", "dk", "dv")},
            "eps": gdr.attrs["eps"],
            "norm_eps": gn.attrs["eps"],
            "scale": gdr.attrs["scale"],
        },
        numerics=(
            "one launch: conv + silu in fp32 (no rounding between), l2-norm, gates (beta unrounded), the delta rule on"
            " the in-place fp32 state [Hv, dk, dv] read and written once per launch, o unrounded, gated rmsnorm with"
            " one bf16 rounding"
        ),
    )


# the hand kernel's (s_deltanet) limits and tolerances, without its fixed shape and its ring-slot bound
CAPS = [
    Cap(
        "g_deltanet",
        "gated_delta_rule",
        pre=(
            Absorb("reshape", min=4, max=4),
            Absorb("split", min=1, max=1),
            Absorb("silu", min=1, max=1),
            Absorb("causal_conv1d", min=1, max=1),
        ),
        post=(Absorb("gated_rmsnorm", min=1, max=1, reduction_ok=True), Absorb("reshape", min=1, max=1)),
        limits={"T_max": ROWS_MAX, "dtype_in": ("bf16",)},
        rounds={"gated_rmsnorm": ("out",)},
        tolerance=(
            "causal_conv1d.out",
            "silu.out",
            "gated_delta_rule.beta",
            "gated_delta_rule.out",
            "gated_rmsnorm.norm",
            "gated_rmsnorm.scale",
        ),
        constants=_consts,
        info=_info,
        check=_check,
        emit=_emit,
    )
]


def _deltanet(L: C.CDLL, mem, t, T: int, st: int) -> None:
    a = t.attrs
    ws = mem.scratch("g_deltanet_ws", L.warpc_deltanet_ws_bytes(a["Hk"], a["Hv"], a["dv"]))
    ins = [mem.buffer(t.ins[r]) for r in ("qkv", "z", "a", "b")]
    weights = [mem.packs[t.weights[r]] for r in ("conv_w", "A_log", "dt_bias", "norm_w")]
    rc = L.warpc_deltanet_step(
        T,
        *ins,
        *weights,
        mem.states[t.states["rec"]],
        mem.states[t.states["conv"]],
        mem.buffer(t.outs["y"]),
        ws,
        *(a[k] for k in ("C", "W", "Hk", "Hv", "dk", "dv")),
        a["eps"],
        a["scale"],
        a["norm_eps"],
        st,
    )
    if rc != 0:
        raise RuntimeError(f"{t.kernel} failed: {L.warpc_error_string(rc).decode()} (CUDA error {rc})")


LAUNCHERS: dict = {"g_deltanet": _deltanet}
PACKS: dict = {}
