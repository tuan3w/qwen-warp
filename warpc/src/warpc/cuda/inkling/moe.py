"""Inkling-2 kernel family `moe`: the routed-expert layer in one persistent launch (kernels/inkling_moe*.cu).

g_moe        moe_experts, pre moe_route, post causal_conv1d(residual, divisor) + cast f32 + add f32 + cast bf16: the
             whole MLP half of a visit; the visit output h bf16 is written, the expert output never is
g_moe_plain  moe_experts, pre moe_route: writes the expert output (the conv chain then belongs to another kernel)

Numerics: router logits fp32 from the stored bf16 weights, top-8 by descending s + bias (lowest id on ties), weights
normalised by a left fold in idx order; per expert g, u, the activation and y are rounded to bf16 as in the interpreter;
the contributions of a row are added in ascending expert id (`fl(w * y)` then an add), the conv as `fma` chain over
[state | x]. The kernel's blocks wait for each other: nothing may run beside it (its Launch carries no side work).
"""

from __future__ import annotations

import ctypes as C
from typing import NamedTuple

import torch

from ...compiler import Absorb, Cap, Pack, Region, Verdict

_PTR, _INT, _FLOAT, _SIZE = C.c_void_p, C.c_int, C.c_float, C.c_size_t

HIDDEN, WIDTH = 2304, 1152  # the instantiated (H, I)
TOP_K, MAX_EXPERTS, MAX_ROWS = 8, 128, 8
GATE_CAP, UP_CAP = 4.0, 25.0  # the kernel's activation
CONV_WIDTH = 4  # taps of the MLP short convolution
GROUP = 128

SIGNATURES = {
    "warpc_moe_ws_bytes": ([_INT], _SIZE),
    "warpc_moe_ws_offsets": ([_INT, _PTR], _INT),
    "warpc_moe_grid": ([_INT, _INT], _INT),
    # x, router, bias, pg, sg, pu, su, pd, sd, out, idx, w, ws, res, cw, cst, h, T, E, route_scale, divisor, stream
    "warpc_moe_forward": ([*[_PTR] * 17, _INT, _INT, _FLOAT, _FLOAT, _PTR], _INT),
}


class StackLayout(NamedTuple):
    """Byte offsets inside the `w4g128_experts` pack: packed int32 and bf16 scales of gate, up, down, 256-aligned."""

    gate: tuple[int, int]
    up: tuple[int, int]
    down: tuple[int, int]
    total: int


def stack_layout(experts: int, hidden: int = HIDDEN, width: int = WIDTH) -> StackLayout:
    def span(rows: int, k: int) -> tuple[int, int]:
        return experts * rows * k // 8 * 4, experts * rows * (k // GROUP) * 2

    def align(n: int) -> int:
        return -(-n // 256) * 256

    at, parts = 0, []
    for rows, k in ((width, hidden), (width, hidden), (hidden, width)):
        packed, scale = span(rows, k)
        parts.append((at, align(at + packed)))
        at = align(align(at + packed) + scale)
    return StackLayout(*parts, at)


# ---------------------------------------------------------------------------------------------------- capability


def _chain(m, A):
    """(conv, cast f32, add, cast bf16) of a match with the epilogue, else None."""
    return [A.node(i) for i in m.chains[0]] if m.chains and m.chains[0] else None


def _route(m, A):
    return next(A.node(i) for i in m.pre if A.node(i).op == "moe_route")


def _constants(m, A) -> dict:
    g = A.graph
    ex, rt = A.node(m.anchors[0]), _route(m, A)
    gate = g.weights[ex.attrs["gate"]]
    experts, width, hidden = gate.shape
    c = {
        "H": hidden,
        "I": width,
        "E": experts,
        "k": rt.attrs["top_k"],
        "T": A.rows_max,
        "dtype": g.types[ex.ins[0]].dtype,
        "gate_cap": ex.attrs.get("gate_cap"),
        "up_cap": ex.attrs.get("up_cap"),
        "scheme": "/".join(sorted({g.weights[ex.attrs[r]].scheme for r in ("gate", "up", "down")})),
    }
    chain = _chain(m, A)
    if chain:
        w = g.weights[chain[0].attrs["weight"]]
        c["W"] = w.shape[1]
        c["conv_channels"] = w.shape[0]
    return c


def _info(c: dict) -> Verdict:
    # the grid is 288 blocks (kernels/inkling_moe_inst.cu): every SM is busy
    return Verdict(ctas=288)


def _check(m, A) -> str | None:
    ex, rt = A.node(m.anchors[0]), _route(m, A)
    if rt.ins[0] != ex.ins[0]:
        return "the router and the experts read different inputs"
    if ex.ins[1:] != rt.outs:
        return "the experts do not read the router's idx and w"
    chain = _chain(m, A)
    if chain is None:
        return None
    conv, cast, add, _ = chain
    if not conv.attrs.get("residual"):
        return "the conv has no residual: the kernel applies (conv + x) / divisor"
    if m.consts["conv_channels"] != m.consts["H"]:
        return f"conv channels {m.consts['conv_channels']} != hidden {m.consts['H']}"
    if cast.outs[0] not in add.ins:
        return "the add does not read the cast of the conv"
    return None


_LIMITS = {
    "H_in": (HIDDEN,),
    "I_in": (WIDTH,),
    "E_min": TOP_K,
    "E_max": MAX_EXPERTS,
    "k_in": (TOP_K,),
    "T_max": MAX_ROWS,
    "dtype_in": ("bf16",),
    "gate_cap_in": (GATE_CAP,),
    "up_cap_in": (UP_CAP,),
    "scheme_in": ("w4g128",),
}
# the interpreter's rounding points of moe_experts, the conv and the closing cast are all reproduced; the fp32 sum
# orders (router, expert partials, conv) differ from the interpreter's without dropping a point
_ROUNDS = {"moe_experts": ("gate", "up", "act", "down", "out"), "causal_conv1d": ("out",), "cast": ("out",)}


def _emit(conv: bool):
    def emit(m, A) -> Region:
        g = A.graph
        ex, rt = A.node(m.anchors[0]), _route(m, A)
        gate, up, down = (ex.attrs[r] for r in ("gate", "up", "down"))
        stack = f"{gate}|{up}|{down}"
        router, bias = rt.attrs["weight"], rt.attrs["bias"]
        r = Region(
            "g_moe" if conv else "g_moe_plain",
            set(m.covers),
            ins={"x": ex.ins[0]},
            outs={"out": ex.outs[0]},
            weights={"router": router, "bias": bias, "experts": stack},
            packs=[
                Pack(router, "dense", [router]),
                Pack(bias, "f32", [bias]),
                Pack(stack, "w4g128_experts", [gate, up, down]),
            ],
            attrs={
                "H": g.weights[gate].shape[2],
                "I": g.weights[gate].shape[1],
                "E": g.weights[gate].shape[0],
                "route_scale": rt.attrs["route_scale"],
            },
            numerics="router fp32 from bf16 weights, expert sums fp32 in a fixed order, contributions added in"
            " ascending expert id",
        )
        if conv:
            cv, cast, add, last = _chain(m, A)
            res = add.ins[0] if add.ins[1] == cast.outs[0] else add.ins[1]
            r.ins["res"] = res
            r.outs = {"h": last.outs[0]}
            r.weights["cw"] = cv.attrs["weight"]
            r.packs.append(Pack(cv.attrs["weight"], "dense", [cv.attrs["weight"]]))
            r.states = {"conv": cv.attrs["state"]}
            r.attrs["divisor"] = float(cv.attrs.get("divisor", 1.0))
        return r

    return emit


CAPS = [
    Cap(
        "g_moe",
        "moe_experts",
        pre=(Absorb("moe_route", min=1, max=1),),
        post=(
            Absorb("causal_conv1d", min=1, max=1),
            Absorb("cast", min=1, max=1, attrs={"dtype": "f32"}),
            Absorb("add", min=1, max=1),
            Absorb("cast", min=1, max=1, attrs={"dtype": "bf16"}),
        ),
        limits={**_LIMITS, "W_in": (CONV_WIDTH,)},
        rounds=_ROUNDS,
        constants=_constants,
        info=_info,
        check=_check,
        emit=_emit(True),
    ),
    Cap(
        "g_moe_plain",
        "moe_experts",
        pre=(Absorb("moe_route", min=1, max=1),),
        limits=_LIMITS,
        rounds=_ROUNDS,
        constants=_constants,
        info=_info,
        check=_check,
        emit=_emit(False),
    ),
]

# ---------------------------------------------------------------------------------------------------- launch


def _forward(L: C.CDLL, mem, t, T: int, st: int) -> None:
    a = t.attrs
    conv = "h" in t.outs
    E, base = a["E"], mem.packs[t.weights["experts"]]
    lay = stack_layout(E, a["H"], a["I"])
    ws = mem.scratch("moe_ws", L.warpc_moe_ws_bytes(E))
    res = mem.buffer(t.ins["res"]) if conv else None
    rc = L.warpc_moe_forward(
        mem.buffer(t.ins["x"]),
        mem.packs[t.weights["router"]],
        mem.packs[t.weights["bias"]],
        base + lay.gate[0],
        base + lay.gate[1],
        base + lay.up[0],
        base + lay.up[1],
        base + lay.down[0],
        base + lay.down[1],
        None if conv else mem.buffer(t.outs["out"]),
        None,
        None,
        ws,
        res,
        mem.packs[t.weights["cw"]] if conv else None,
        mem.states[t.states["conv"]] if conv else None,
        mem.buffer(t.outs["h"]) if conv else None,
        T,
        E,
        a["route_scale"],
        a.get("divisor", 1.0),
        st,
    )
    if rc != 0:
        raise RuntimeError(f"{t.kernel} failed: {L.warpc_error_string(rc).decode()} (CUDA error {rc})")


LAUNCHERS = {"g_moe": _forward, "g_moe_plain": _forward}

# ---------------------------------------------------------------------------------------------------- packs


def _stack(packer, pack: Pack) -> torch.Tensor:
    """gate, up and down of one layer as ONE uint8 tensor: for each, the packed int32 words [E, N, K / 8] then the bf16
    scales [E, N, K / 128] (offsets: `stack_layout`), as the kernel's pointers address them."""
    decls = [packer.weights[n] for n in pack.parts]
    experts, width, hidden = decls[0].shape
    lay = stack_layout(experts, hidden, width)
    buf = torch.empty(lay.total, dtype=torch.uint8, device=packer.device)
    shapes = ((width, hidden), (width, hidden), (hidden, width))
    for d, (poff, soff), (rows, k) in zip(decls, (lay.gate, lay.up, lay.down), shapes, strict=True):
        if d.scheme != "w4g128" or d.shape != (experts, rows, k):
            raise ValueError(
                f"{d.name}: a {d.scheme} weight of shape {d.shape} in an w4g128_experts of [{experts}, {rows}, {k}]"
            )
        raw = packer.source.raw(d.source)
        packed, scale = raw["packed"], raw["scale"]
        if tuple(packed.shape) != (experts, rows, k // 8) or tuple(scale.shape) != (experts, rows, k // GROUP):
            raise ValueError(f"{d.name}: W4G128 codes {tuple(packed.shape)} / scales {tuple(scale.shape)}")
        nbytes = packed.numel() * 4
        buf[poff : poff + nbytes].view(torch.int32).copy_(packed.reshape(-1))
        buf[soff : soff + scale.numel() * 2].view(torch.bfloat16).copy_(scale.reshape(-1))
    return buf


PACKS = {"w4g128_experts": _stack}
