"""The Inkling-2 kernel families (message reads, GEMVs and convolutions, attention, MoE): each submodule exports
`CAPS` (capabilities of the CUDA table), `LAUNCHERS` (kernel name -> launcher), `SIGNATURES` (ctypes signatures of its
extern "C" entries); this package merges them for the table, the launchers and the kernel library. `PACKS` (pack kind ->
device tensor builder) holds the one layout only a family reads: the MoE's stacked experts."""

from __future__ import annotations

from . import attn, gemv, moe, res

FAMILIES = (res, gemv, attn, moe)
CAPS = [cap for f in FAMILIES for cap in f.CAPS]
LAUNCHERS = {k: v for f in FAMILIES for k, v in f.LAUNCHERS.items()}
SIGNATURES = {k: v for f in FAMILIES for k, v in f.SIGNATURES.items()}
PACKS = moe.PACKS
