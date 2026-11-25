"""The data-movement kernels of the MTP draft graph (g_concat, g_select_row, g_take; kernels/move.cu) against torch
indexing, bit for bit, and a small graph lowered onto the CUDA table and run through the launchers.

  uv run --package warpc python warpc/tests/test_glue.py

The lowering tests need no GPU; the kernel tests skip without a CUDA device.
"""

import torch

from harness import Skip, run
from warpc.compiler import lower
from warpc.cuda import CudaBackend, glue, launch
from warpc.errors import LoweringError
from warpc.ir import Graph, TType, WeightDecl

VOCAB = 68864  # the draft vocabulary of the 27B's MTP head
H = 5120


def _lib():
    if not torch.cuda.is_available():
        raise Skip("no CUDA device")
    major, minor = torch.cuda.get_device_capability()
    return glue.load(10 * major + minor)


def _stream() -> int:
    return torch.cuda.current_stream().cuda_stream


def _check(L, rc: int) -> None:
    assert rc == 0, L.warpc_error_string(rc).decode()


def _rand(gen, *shape, dtype=torch.bfloat16):
    x = torch.randn(*shape, generator=gen) * 4
    x[..., ::97] *= 1e3  # outliers: every bit pattern class, not just the small values
    return x.to(dtype).cuda()


def _bits(x: torch.Tensor) -> torch.Tensor:
    return x.view(torch.int16 if x.element_size() == 2 else torch.int32)


def test_concat_is_exact_for_every_row_count():
    L = _lib()
    gen = torch.Generator().manual_seed(0)
    for a, b in ((H, H), (H, 2 * H), (8, H), (H, 8), (6144, 2048)):
        for T in range(1, 9):
            x, y = _rand(gen, T, a), _rand(gen, T, b)
            out = torch.zeros(8, a + b, dtype=torch.bfloat16, device="cuda")
            _check(L, L.warpc_concat(x.data_ptr(), y.data_ptr(), out.data_ptr(), T, a, b, 2, _stream()))
            torch.cuda.synchronize()
            assert torch.equal(_bits(out[:T]), _bits(torch.cat([x, y], -1))), (a, b, T)
            assert not out[T:].any(), f"a={a} b={b} T={T}: rows past T were written"


def test_select_row_is_exact_for_every_row_and_dtype():
    L = _lib()
    gen = torch.Generator().manual_seed(1)
    for dtype, esz in ((torch.bfloat16, 2), (torch.float32, 4)):
        for D in (H, 2 * H, 8):
            x = _rand(gen, 8, D, dtype=dtype)
            for T in range(1, 9):
                for r in range(T):
                    i = torch.tensor([r], dtype=torch.int32, device="cuda")
                    out = torch.zeros(1, D, dtype=dtype, device="cuda")
                    _check(L, L.warpc_select_row(x.data_ptr(), i.data_ptr(), out.data_ptr(), D, esz, _stream()))
                    torch.cuda.synchronize()
                    assert torch.equal(_bits(out), _bits(x[i.long()])), (dtype, D, T, r)


def test_take_is_exact_over_the_whole_table():
    L = _lib()
    gen = torch.Generator().manual_seed(2)
    table = torch.randint(0, 1 << 24, (VOCAB,), generator=gen, dtype=torch.int64)
    table[:3] = torch.tensor([0, (1 << 24) - 1, 1 << 23])
    t32 = table.float().cuda()
    assert torch.equal(t32.long().cpu(), table), "the fp32 constant must hold every id exactly"
    idx = torch.arange(VOCAB, dtype=torch.int32, device="cuda")
    out = torch.full((VOCAB,), -1, dtype=torch.int32, device="cuda")
    st = _stream()
    for j in range(VOCAB):  # one launch per index: i and out are offset views of the two arrays
        _check(L, L.warpc_take(idx.data_ptr() + 4 * j, t32.data_ptr(), out.data_ptr() + 4 * j, st))
    torch.cuda.synchronize()
    assert torch.equal(out.long().cpu(), table)


def test_kernels_capture_into_a_cuda_graph():
    """No allocation, no synchronisation: the three launches replay from a graph on a new index."""
    L = _lib()
    gen = torch.Generator().manual_seed(3)
    x, y = _rand(gen, 4, H), _rand(gen, 4, H)
    cat = torch.zeros(4, 2 * H, dtype=torch.bfloat16, device="cuda")
    row = torch.zeros(1, 2 * H, dtype=torch.bfloat16, device="cuda")
    table = torch.arange(VOCAB, dtype=torch.float32, device="cuda").flip(0).contiguous()
    i = torch.zeros(1, dtype=torch.int32, device="cuda")
    tok = torch.zeros(1, dtype=torch.int32, device="cuda")

    def enqueue() -> None:
        st = _stream()
        _check(L, L.warpc_concat(x.data_ptr(), y.data_ptr(), cat.data_ptr(), 4, H, H, 2, st))
        _check(L, L.warpc_select_row(cat.data_ptr(), i.data_ptr(), row.data_ptr(), 2 * H, 2, st))
        _check(L, L.warpc_take(i.data_ptr(), table.data_ptr(), tok.data_ptr(), st))

    enqueue()  # warm-up
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        enqueue()
    for r in range(4):
        i.fill_(r)
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(_bits(row), _bits(torch.cat([x, y], -1)[r : r + 1])), r
        assert int(tok) == VOCAB - 1 - r, (r, int(tok))


def _draft_glue_graph(rows_x: int = H, vocab: int = VOCAB):
    """embed(tokens) | hidden -> concat -> select_row(sel) ; take(sel)."""
    table = tuple((7 * j + 3) % (1 << 24) for j in range(vocab))
    g = Graph("glue")
    tokens = g.input("tokens", TType(("T",), "i32"))
    hidden = g.input("hidden", TType(("T", rows_x), "bf16"))
    sel = g.input("sel", TType((1,), "i32"))
    g.weight(WeightDecl("embed", (32, rows_x), "dense", "embed"))
    x = g.op("embed", [tokens], weight="embed")
    cat = g.op("concat", [x, hidden])
    g.output(cat=cat, row=g.op("select_row", [cat, sel]), tok=g.op("take", [sel], table=table))
    g.verify()
    return g, table


def test_lowered_glue_graph_claims_one_task_per_op():
    g, table = _draft_glue_graph()
    plan = lower(g, CudaBackend(), rows_max=8)
    assert [t.kernel for t in plan.tasks] == ["g_embed", "g_concat", "g_select_row", "g_take"], plan.summary()
    cat, row, take = plan.tasks[1:]
    assert cat.attrs == {"a": H, "b": H} and set(cat.ins) == {"x", "y"}
    assert row.attrs == {"D": 2 * H, "esz": 2} and row.ins["i"] == "@sel" and row.ins["x"] == cat.outs["out"]
    assert take.ins["i"] == "@sel" and plan.constants[take.constants["table"]] == tuple(float(v) for v in table)
    assert all(t.gather == (("table",) if t.kernel == "g_embed" else ()) for t in plan.tasks)


def test_glue_limits():
    g, _ = _draft_glue_graph()
    try:
        lower(g, CudaBackend(), rows_max=9)
    except LoweringError as e:
        assert "T=9 > 8" in str(e), str(e)
    else:
        raise AssertionError("concat past STEP_ROWS rows lowered")
    g, _ = _draft_glue_graph(rows_x=4)
    try:
        lower(g, CudaBackend(), rows_max=8)
    except LoweringError as e:
        assert "a=4 is not a multiple of 8" in str(e), str(e)
    else:
        raise AssertionError("a concat width that is not a multiple of 8 lowered")


_DT = {"bf16": torch.bfloat16, "f32": torch.float32, "i32": torch.int32}


class _Mem:
    """What the launchers read of a runtime, backed by torch tensors."""

    def __init__(self, plan, inputs: dict):
        self.rows_max = plan.rows_max
        self.tensors = {
            b.name: torch.zeros(plan.rows_max * b.dim, dtype=_DT[b.dtype], device="cuda") for b in plan.buffers.values()
        }
        self.inputs = inputs
        self.constants_t = {n: torch.tensor(v, dtype=torch.float32, device="cuda") for n, v in plan.constants.items()}
        self.constants = {n: t.data_ptr() for n, t in self.constants_t.items()}
        self.packs = {}

    def buffer(self, name: str) -> int:
        return (self.inputs[name[1:]] if name.startswith("@") else self.tensors[name]).data_ptr()

    def scratch(self, name: str, nbytes: int) -> int:
        raise AssertionError(name)


def test_lowered_glue_graph_runs_through_the_launchers():
    L = _lib()
    g, table = _draft_glue_graph()
    plan = lower(g, CudaBackend(), rows_max=8)
    gen = torch.Generator().manual_seed(4)
    emb = _rand(gen, 32, H)
    tokens = torch.zeros(8, dtype=torch.int32, device="cuda")
    hidden = _rand(gen, 8, H)
    sel = torch.zeros(1, dtype=torch.int32, device="cuda")
    mem = _Mem(plan, {"tokens": tokens, "hidden": hidden, "sel": sel})
    mem.packs["embed"] = emb.data_ptr()
    by_out = {v: plan.buffer_of[v] for v in g.outputs.values()}
    for T in range(1, 9):
        tokens.copy_(torch.randint(0, 32, (8,), generator=gen, dtype=torch.int32))
        sel.fill_(T - 1)  # the last row of the step, as the draft graph selects it
        for t in plan.tasks:
            launch.LAUNCHERS[t.kernel](L, mem, t, T, _stream())
        torch.cuda.synchronize()
        cat = torch.cat([emb[tokens[:T].long()], hidden[:T]], -1)
        got = mem.tensors[by_out[g.outputs["cat"]]][: T * 2 * H].view(T, 2 * H)
        assert torch.equal(_bits(got), _bits(cat)), T
        row = mem.tensors[by_out[g.outputs["row"]]][: 2 * H].view(1, 2 * H)
        assert torch.equal(_bits(row), _bits(cat[T - 1 : T])), T
        tok = mem.tensors[by_out[g.outputs["tok"]]][:1]
        assert int(tok) == table[T - 1], (T, int(tok))


if __name__ == "__main__":
    run(globals())
