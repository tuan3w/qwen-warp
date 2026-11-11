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

def mma_u_ptx(NT, first):
    """One k16 step (u) of one thread, NT n8 token tiles sharing the dequantised weight fragments.
    Operands: $r0 = wa word, $r1 = wb word, x regs of tile n: $r{2+4n..5+4n}, then (unless `first`) the 4*NT incoming partials; $w = partials."""
    L = ["{", ".reg .b32 m, c, one, off, q, s;", ".reg .b32 pa<4>, pb<4>;",
         "mov.b32 m, 0x000F000F;", "mov.b32 c, 0x43004300;", "mov.b32 one, 0x3F803F80;", "mov.b32 off, 0xC308C308;"]
    for n in range(4 * NT):
        L.append(f"mov.f32 {{$w{n}}}, 0f00000000;" if first else f"mov.b32 {{$w{n}}}, {{$r{2 + 4 * NT + n}}};")
    L += _dq("{$r0}", "pa0", "pa1", "pa2", "pa3")
    L += _dq("{$r1}", "pb0", "pb1", "pb2", "pb3")
    for n in range(NT):
        x = 2 + 4 * n
        w = ", ".join(f"{{$w{4*n+c}}}" for c in range(4))
        L.append(f"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {{{w}}}, {{pa0, pb0, pa1, pb1}}, {{{{$r{x}}}, {{$r{x+1}}}}}, {{{w}}};")
        L.append(f"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {{{w}}}, {{pa2, pb2, pa3, pb3}}, {{{{$r{x+2}}}, {{$r{x+3}}}}}, {{{w}}};")
    L.append("}")
    return "\n".join(L)

_MMA_U_PTX = {}
def mma_u(wa, wb, xs, acc, NT=1):
    """wa, wb: Uint32 weight words; xs: 4*NT Uint32 activation regs; acc: 4*NT Float32 partials, or None to start from zero. Returns 4*NT Float32."""
    key = (NT, acc is None)
    if key not in _MMA_U_PTX: _MMA_U_PTX[key] = mma_u_ptx(NT, acc is None)
    return inline_ptx(_MMA_U_PTX[key], write_only_types=[Float32] * (4 * NT),
                      read_only_args=[wa, wb] + list(xs) + ([] if acc is None else list(acc)))

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

def stg4(a, v0, v1, v2, v3):
    inline_ptx("st.global.v4.u32 [{$r0}], {{$r1}, {$r2}, {$r3}, {$r4}};", read_only_args=[a, v0, v1, v2, v3])

def pack_bf16x2(lo, hi):
    """two fp32 -> bf16x2 (lo in the low half), round to nearest even"""
    return inline_ptx("cvt.rn.bf16x2.f32 {$w0}, {$r1}, {$r0};", write_only_types=[Uint32], read_only_args=[lo, hi])

# ---- fused GEMM building blocks (shared-memory x tile, mma with per-row scale folded into the bf16 weights)
def lds4(a):
    """16 B shared-memory load, `a` = 32-bit shared address"""
    return inline_ptx("ld.shared.v4.u32 {{$w0}, {$w1}, {$w2}, {$w3}}, [{$r0}];",
                      write_only_types=[Uint32] * 4, read_only_args=[a])

def lds_f32(a):
    return inline_ptx("ld.shared.f32 {$w0}, [{$r0}];", write_only_types=[Float32], read_only_args=[a])

def sts_f32(a, v):
    inline_ptx("st.shared.f32 [{$r0}], {$r1};", read_only_args=[a, v])

def cp_async16(dst, src):
    """16 B global -> shared async copy (L2 only); dst 32-bit shared address, src 64-bit global address"""
    inline_ptx("cp.async.cg.shared.global [{$r0}], [{$r1}], 16;", read_only_args=[dst, src])

def prmt(a, sel):
    return inline_ptx("prmt.b32 {$w0}, {$r0}, {$r0}, " + str(sel) + ";", write_only_types=[Uint32], read_only_args=[a])

def _R(i): return "{$r%d}" % i
def _W(i): return "{$w%d}" % i

def wmma_u_ptx(NT):
    """One 8-k slice (word u of the 128-k group) of one 16-row weight tile against NT n8 token tiles.
    $r0 = row-g word, $r1 = row-(g+8) word, $r2/$r3 = bf16x2 scale (row g / row g+8, broadcast),
    $r{4+4n+c} = x regs of tile n, $r{4+4NT+4n+c} = accumulator in; $w{4n+c} = accumulator out.
    Weights are dequantised to bf16 exactly as the baseline does: bf16(round((nibble-8)*scale))."""
    L = ["{", ".reg .b32 m, c, one, off, nz, q, s;", ".reg .b32 pa<4>, pb<4>;",
         "mov.b32 m, 0x000F000F;", "mov.b32 c, 0x43004300;", "mov.b32 one, 0x3F803F80;",
         "mov.b32 off, 0xC308C308;", "mov.b32 nz, 0x80008000;"]
    for p, src, sc in (("pa", _R(0), _R(2)), ("pb", _R(1), _R(3))):
        for j in range(4):
            s = src
            if j:
                L.append("shr.u32 s, %s, %d;" % (src, 4 * j)); s = "s"
            L.append("lop3.b32 q, %s, m, c, 0xEA;" % s)
            L.append("fma.rn.bf16x2 q, q, one, off;")
            L.append("fma.rn.bf16x2 %s%d, q, %s, nz;" % (p, j, sc))
    ab = 4 + 4 * NT
    for n in range(NT):
        acc_in = ", ".join(_R(ab + 4 * n + c) for c in range(4))
        w = ", ".join(_W(4 * n + c) for c in range(4))
        xa = "{%s, %s}" % (_R(4 + 4 * n), _R(5 + 4 * n))
        L.append("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%s}, {pa0, pb0, pa1, pb1}, %s, {%s};" % (w, xa, acc_in))
    for n in range(NT):
        w = ", ".join(_W(4 * n + c) for c in range(4))
        xb = "{%s, %s}" % (_R(6 + 4 * n), _R(7 + 4 * n))
        L.append("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%s}, {pa2, pb2, pa3, pb3}, %s, {%s};" % (w, xb, w))
    L.append("}")
    return "\n".join(L)

_WMMA_U = {}
def wmma_u(wa, wb, sa, sb, xv, acc, NT):
    """wa, wb, sa, sb: Uint32; xv: 4*NT Uint32; acc: 4*NT Float32 -> 4*NT Float32 (new accumulators)"""
    if NT not in _WMMA_U: _WMMA_U[NT] = wmma_u_ptx(NT)
    return inline_ptx(_WMMA_U[NT], write_only_types=[Float32] * (4 * NT),
                      read_only_args=[wa, wb, sa, sb] + list(xv) + list(acc))
