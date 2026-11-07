"""Inline-PTX building blocks shared by the CuTe DSL prototypes (sm_86)."""
import cutlass, cutlass.cute as cute
from cutlass import Int32, Int64, Uint32, Float32
from cutlass.cute.arch import inline_ptx

def addr(tensor, off):
    """byte address (Int64) of tensor.iterator + off elements"""
    return (tensor.iterator + off).toint()

def ldw4(a):
    """streaming weight load: 16 B, non-coherent path, do not allocate in L1"""
    return inline_ptx("ld.global.nc.L1::no_allocate.v4.u32 {{$w0}, {$w1}, {$w2}, {$w3}}, [{$r0}];",
                      write_only_types=[Uint32] * 4, read_only_args=[a])

def ldx4(a):
    """activation load: 16 B through the read-only/L1 path"""
    return inline_ptx("ld.global.nc.v4.u32 {{$w0}, {$w1}, {$w2}, {$w3}}, [{$r0}];",
                      write_only_types=[Uint32] * 4, read_only_args=[a])

def ld32(a):
    return inline_ptx("ld.global.nc.u32 {$w0}, [{$r0}];", write_only_types=[Uint32], read_only_args=[a])

def bits2f(u):
    return inline_ptx("mov.b32 {$w0}, {$r0};", write_only_types=[Float32], read_only_args=[u])

def st_bf16(a, f):
    inline_ptx("{\n.reg .b16 h;\ncvt.rn.bf16.f32 h, {$r1};\nst.global.b16 [{$r0}], h;\n}", read_only_args=[a, f])

def _dq(src, d0, d1, d2, d3):
    """PTX lines: nibbles of u32 `src` -> 4 bf16x2 regs d0..d3 = (n_j, n_{j+4}) - 8, exact."""
    lines = []
    for j, d in enumerate((d0, d1, d2, d3)):
        s = src if j == 0 else "s"
        if j:
            lines.append(f"shr.u32 s, {src}, {4*j};")
        lines.append(f"lop3.b32 q, {s}, m, c, 0xEA;")
        lines.append(f"fma.rn.bf16x2 {d}, q, one, off;")
    return lines

def group_mma_ptx():
    """One 128-k group of one thread: 8 weight words (wa[4]: row g, wb[4]: row g+8) x 16 activation regs -> 4 fp32 partials.
    Operands: $r0..$r3 = wa, $r4..$r7 = wb, $r8..$r23 = x (u: x[4u..4u+3]), $w0..$w3 = out.  Same instruction sequence as k_w4 (RP=1, NT=1)."""
    L = ["{", ".reg .b32 m, c, one, off, q, s;", ".reg .b32 pa<4>, pb<4>;",
         "mov.b32 m, 0x000F000F;", "mov.b32 c, 0x43004300;", "mov.b32 one, 0x3F803F80;", "mov.b32 off, 0xC308C308;",
         "mov.f32 {$w0}, 0f00000000;", "mov.f32 {$w1}, 0f00000000;", "mov.f32 {$w2}, 0f00000000;", "mov.f32 {$w3}, 0f00000000;"]
    for u in range(4):
        L += _dq(f"{{$r{u}}}", "pa0", "pa1", "pa2", "pa3")
        L += _dq(f"{{$r{4+u}}}", "pb0", "pb1", "pb2", "pb3")
        x = 8 + 4 * u
        L.append("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%s, %s, %s, %s}, {pa0, pb0, pa1, pb1}, {%s, %s}, {%s, %s, %s, %s};" % (
            "{$w0}", "{$w1}", "{$w2}", "{$w3}", f"{{$r{x}}}", f"{{$r{x+1}}}", "{$w0}", "{$w1}", "{$w2}", "{$w3}"))
        L.append("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%s, %s, %s, %s}, {pa2, pb2, pa3, pb3}, {%s, %s}, {%s, %s, %s, %s};" % (
            "{$w0}", "{$w1}", "{$w2}", "{$w3}", f"{{$r{x+2}}}", f"{{$r{x+3}}}", "{$w0}", "{$w1}", "{$w2}", "{$w3}"))
    L.append("}")
    return "\n".join(L)

_GROUP_PTX = None
def group_mma(wa, wb, xv):
    """wa, wb: 4 Uint32 each; xv: 16 Uint32. Returns 4 Float32."""
    global _GROUP_PTX
    if _GROUP_PTX is None: _GROUP_PTX = group_mma_ptx()
    return inline_ptx(_GROUP_PTX, write_only_types=[Float32] * 4, read_only_args=list(wa) + list(wb) + list(xv))

# ---- megakernel sync / memory primitives
def ldg4(a):
    """plain 16 B load (coherent with in-kernel writes once an acquire has been observed)"""
    return inline_ptx("ld.global.v4.u32 {{$w0}, {$w1}, {$w2}, {$w3}}, [{$r0}];",
                      write_only_types=[Uint32] * 4, read_only_args=[a])

def ld_acq(a):
    return inline_ptx("ld.global.acquire.gpu.b32 {$w0}, [{$r0}];", write_only_types=[Int32], read_only_args=[a])

def atom_add_rel(a, v):
    inline_ptx("{\n.reg .b32 o;\natom.add.release.gpu.u32 o, [{$r0}], {$r1};\n}", read_only_args=[a, v])

def st32(a, v):
    inline_ptx("st.global.b32 [{$r0}], {$r1};", read_only_args=[a, v])

def prefetch_l2(a):
    inline_ptx("prefetch.global.L2 [{$r0}];", read_only_args=[a])

def gtimer32():
    return Uint32(cute.arch.clock64() & 0xFFFFFFFF)
