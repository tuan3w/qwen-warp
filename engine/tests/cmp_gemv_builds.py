"""Old-vs-new GEMV build comparison (K7 follow-up): same random W4 weights / bf16 activations through smk_w4_gemv of two builds, bitwise diff per engine shape and T=1..4.
  ~/llm-lab/venv/bin/python tests/cmp_gemv_builds.py   (edit the two SMK_BUILD dirs below; needs the GPU lease)"""
import sys, os, ctypes as C
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, smk
os.environ["SMK_BUILD"] = "build/k7"; new = smk.load("libswiftmk.so")
os.environ["SMK_BUILD"] = "/tmp/k7o/build"; old = smk.load("libswiftmk.so")
assert new._name != old._name
dev = "cuda"; g = torch.Generator(device=dev).manual_seed(0)
st = smk.stream()
def run(lib, kind, N, K, T, w, s, x):
    if kind == "segs":
        outs = [torch.zeros(T, N, dtype=torch.bfloat16, device=dev)]
        epi = smk.epi_segs(N, outs)
    elif kind == "silu":
        outs = [torch.zeros(T, N // 2, dtype=torch.bfloat16, device=dev)]; epi = smk.epi_simple(smk.SMK_EPI_SILU, outs[0])
    else:
        outs = [torch.zeros(T, N, dtype=torch.float32, device=dev)]; epi = smk.epi_simple(smk.SMK_EPI_F32, outs[0])
    smk.check(lib.smk_w4_gemv(w.data_ptr(), s.data_ptr(), x.data_ptr(), N, K, T, C.byref(epi), st), "gemv")
    torch.cuda.synchronize()
    return outs[0].float()
shapes = [("qkvz", "segs", 16384, 5120), ("attn_qkv", "segs", 14336, 5120), ("o_proj", "segs", 5120, 6144), ("gate_up", "silu", 34816, 5120), ("down", "segs", 5120, 17408),
          ("mtp_fc", "segs", 5120, 10240), ("draft_head", "f32", 65536, 5120)]
for name, kind, N, K in shapes:
    w = torch.randint(-2**31, 2**31 - 1, (N, K // 8), dtype=torch.int32, device=dev, generator=g)
    s = (torch.rand(N, K // 128, device=dev, generator=g) * 0.02 + 0.001).bfloat16()
    row = []
    for T in (1, 2, 3, 4):
        x = torch.randn(T, K, device=dev, generator=g).bfloat16()
        a, b = run(new, kind, N, K, T, w, s, x), run(old, kind, N, K, T, w, s, x)
        d = (a - b).abs()
        row.append(f"T={T}: {'IDENT' if d.max() == 0 else f'diff n={(d>0).sum().item()} max {d.max():.3g} relL2 {(d.norm()/b.norm()).item():.2e}'}")
    print(f"{name:10s} N={N} K={K}: " + " | ".join(row), flush=True)
