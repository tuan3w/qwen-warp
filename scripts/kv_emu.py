#!/usr/bin/env python
"""KV-cache scheme lab: measure a KV quantisation scheme written in Python/torch against the bf16 cache on the real model, with no kernel work.

  python scripts/kv_dkl.py ref --text a.py b.py --tokens 16384 --out ref.pt                      # bf16 reference, once (same file as for the real kernels)
  python scripts/kv_emu.py --scheme scripts/kv_schemes/shipped.py --ref ref.pt --calib calib.json  # dKL of the scheme (calib.json: ~4k token ids; only used by calibrate())

The engine runs with a bf16 KV cache.  After every 8-row step, the rows that have aged out of `scheme.window` (and are not one of the first `scheme.sink`) are passed in chunks of 32
rows through scheme.encode -> payload tensors -> scheme.decode and the cache rows are overwritten with the decoded values, so the model sees what a cache holding `payload` would
give it.  Bytes are counted from the actual payload tensors, not reported by the scheme.  `scripts/kv_schemes/shipped.py` is a torch port of the shipped format (engine/src/kvrot.cuh);
on 6.4k tokens it gave dKL 0.0931% against 0.0905% for the real kernel.  Differences from a real cache: rows of the same step are exact for each other, and the exact window is
`window` to `window + 31` rows (accounted for in the fixed bytes below).

Scheme contract (a module with `make() -> scheme`):
    scheme.window : int          newest rows kept exact (bf16)             scheme.sink : int   leading rows kept exact
    scheme.calibrate(layers)     optional.  layers[l] (l = 0..15, the attention layers) = dict(K=[N,4,256], V=[N,4,256], Q=[N,24,256]) bf16, post-rope, of the calibration text
                                 (needs the engine's query tap, SMK_QTAP=1, which this script sets).  Store what it learns as tensors on the scheme: they are charged.
    scheme.encode(layer, k, v, pos0) -> dict[str, Tensor]     k, v bf16 [32, 4, 256]: 32 consecutive tokens of one layer
    scheme.decode(layer, payload)    -> (k, v) bf16 [32, 4, 256]   reads only the payload and tensors on the scheme
encode must be a pure function of its arguments (checked at the end by re-encoding the first chunk).
Budget printed next to the result: payload bytes per token per layer (shipped: 1056) and fixed bytes per layer = (window + sink + 32) exact rows x 4096 B + scheme tensors / 16
(shipped: 2,113,536 for the 512-slot ring).  A scheme is comparable to the shipped format only at no more of either.
"""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kv_dkl  # noqa: E402

CH = 32
LAYERS, NKV, HD, NH = 16, 4, 256, 24
ROW_BYTES = 2 * NKV * HD * 2  # K and V rows of one token, bf16


def _tensors(o, seen=None):
    import torch

    seen = set() if seen is None else seen
    if id(o) in seen:
        return
    seen.add(id(o))
    if isinstance(o, torch.Tensor):
        yield o
    elif isinstance(o, dict):
        for v in o.values():
            yield from _tensors(v, seen)
    elif isinstance(o, (list, tuple)):
        for v in o:
            yield from _tensors(v, seen)
    elif hasattr(o, "__dict__"):
        for v in vars(o).values():
            yield from _tensors(v, seen)


class _Buf:
    def __init__(self, ptr, nbytes):
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 3,
        }


def _view(ptr, nbytes, dtype):
    import torch

    return torch.as_tensor(_Buf(ptr, nbytes), device="cuda").view(dtype)


def _cache(e):
    import torch
    from prefill import setup_prefill_api

    setup_prefill_api(e.lib)
    base, lay = e.lib.smkp_engine_ptr(e.h, 3), e.lib.smkp_engine_ptr(e.h, 4)
    return [
        _view(base + i * lay, lay, torch.bfloat16)[: 2 * NKV * e.ctx_max * HD].view(
            2, NKV, e.ctx_max, HD
        )
        for i in range(LAYERS)
    ]


def calibrate(e, scheme, ids):
    import torch

    cache = _cache(e)  # also declares the ctypes signatures of smkp_engine_ptr (pointer-sized results)
    qtap = e.lib.smkp_engine_ptr(e.h, 8)
    assert qtap, (
        "the engine has no query tap: set SMK_QTAP=1 before building it (this script does) and use an engine built from this tree"
    )
    qt = _view(qtap, LAYERS * 8 * NH * HD * 2, torch.bfloat16).view(LAYERS, 8, NH, HD)
    Ks, Vs, Qs = ([[] for _ in range(LAYERS)] for _ in range(3))
    e.reset()
    i, n = 0, len(ids) - 1
    while i < n:
        T = min(8, n - i)
        e._step(ids[i : i + T], 0)
        for ly in range(LAYERS):
            Ks[ly].append(cache[ly][0, :, i : i + T].transpose(0, 1).clone())
            Vs[ly].append(cache[ly][1, :, i : i + T].transpose(0, 1).clone())
            Qs[ly].append(qt[ly, :T].clone())
        i += T
    scheme.calibrate(
        [
            dict(K=torch.cat(Ks[ly]), V=torch.cat(Vs[ly]), Q=torch.cat(Qs[ly]))
            for ly in range(LAYERS)
        ]
    )
    torch.cuda.empty_cache()
    e.reset()


class Emu:
    def __init__(self, e, scheme):
        import torch

        self.s, self.cache = scheme, _cache(e)
        self.window, self.sink = int(scheme.window), int(scheme.sink)
        self.done, self.max_bpt, self.first = self.sink, 0.0, None
        self.torch = torch

    def after(self, upto):
        """rows [0, upto) exist: replace every full chunk that is at least `window` rows old."""
        torch = self.torch
        while self.done + CH <= upto - self.window:
            a, b = self.done, self.done + CH
            for ly in range(LAYERS):
                c = self.cache[ly]
                k, v = (
                    c[0, :, a:b].transpose(0, 1).contiguous(),
                    c[1, :, a:b].transpose(0, 1).contiguous(),
                )
                p = self.s.encode(ly, k, v, a)
                self.max_bpt = max(
                    self.max_bpt,
                    sum(t.numel() * t.element_size() for t in p.values()) / CH,
                )
                if self.first is None and ly == 0:
                    self.first = (
                        a,
                        k.clone(),
                        v.clone(),
                        {n: t.clone() for n, t in p.items()},
                    )
                k2, v2 = self.s.decode(ly, p)
                assert k2.shape == k.shape and v2.shape == v.shape, "decode shape"
                c[0, :, a:b] = k2.to(torch.bfloat16).transpose(0, 1)
                c[1, :, a:b] = v2.to(torch.bfloat16).transpose(0, 1)
            self.done = b

    def stats(self):
        torch = self.torch
        fixed = (self.window + self.sink + CH) * ROW_BYTES + sum(
            t.numel() * t.element_size() for t in _tensors(self.s)
        ) / LAYERS
        if self.first is not None:
            a, k, v, p0 = self.first
            p1 = self.s.encode(0, k, v, a)
            assert p0.keys() == p1.keys() and all(
                torch.equal(p0[n], p1[n]) for n in p0
            ), "encode is not a pure function of its inputs"
        return {
            "payload_bytes_per_token_layer": self.max_bpt,
            "fixed_bytes_per_layer": fixed,
        }


def main():
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", required=True, help="python file defining make()")
    ap.add_argument(
        "--ref", default="ref.pt", help="reference written by scripts/kv_dkl.py ref"
    )
    ap.add_argument(
        "--calib",
        help="JSON list of ~4k token ids for scheme.calibrate (default: the first 4096 tokens of the reference text)",
    )
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("kv_scheme", a.scheme)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    scheme = mod.make()
    os.environ["SMK_QTAP"] = "1"
    ref = torch.load(a.ref)
    ids = ref["ids_text"]
    if hasattr(scheme, "calibrate") and not a.calib:
        ap.error("this scheme calibrates: pass --calib with token ids of a text other than the reference text")
    calib = json.load(open(a.calib)) if a.calib else []
    assert len(calib) <= len(ids) + 64, (
        "calibration text longer than the engine's context"
    )
    box = {}

    def setup(e):
        if hasattr(scheme, "calibrate"):
            calibrate(e, scheme, calib)
        box["emu"] = Emu(e, scheme)
        return box["emu"].after

    kl, nll = kv_dkl.sweep(0, ids, ref, setup=setup)
    q99 = torch.quantile(kl, torch.tensor([0.99], device=kl.device)).item()
    print(
        json.dumps(
            {
                "scheme": Path(a.scheme).name,
                "tokens": ref["n"],
                "dkl_pct": 100 * kl.mean().item(),
                "p99_kl_pct": 100 * q99,
                "nll": nll,
                **box["emu"].stats(),
            }
        )
    )


if __name__ == "__main__":
    main()
