// E2: persistent kernel with (A) fused MLP phases, (B) dataflow readiness counters + gather warp, (C) balanced assignment,
// (D) cheap activation staging + L2 prefetch.  Builds on E1's kernels.cuh (copied unchanged except the L2 prefetch cursor).
// See README.md.  One kernel template: k_main<C, DF, CHEAP>.
//   DF=false: grid barrier between dependent phases (like E1 V4) but with the new phase kinds / assignment / prefetch.
//   DF=true : no grid barrier; a 9th "gather" warp stages activation slices as soon as their producer counters say so;
//             compute warps consume K-tiles in order and wait on a CTA-local staged-slice counter; RMSNorm is applied late
//             (dot product on bf16(x), scaled by rsqrt(mean(x^2)+eps) in the epilogue).
#pragma once
#include "kernels.cuh"

namespace e1 {

constexpr int SL = 1024;               // activation slice (elements) = one W4 tile along K
constexpr int MLP_CHUNKS = INTER / 8;  // 2176 chunks of 8 intermediate channels
enum : int { PH_MLPB = 21 };           // fused-MLP down-projection slice (the gate|up half is a PH_W4 phase with EPI_MLPA)
enum : int { EPI_MLPA = 3 };           // gate|up -> silu*mul -> CTA-local smem vector (consumed by PH_MLPB)

struct Ph2 {
  Phase b;           // E1 fields. MLPB: N = HID rows, K = INTER, w/s = repacked down weights/scales, out = x
  int sync;          // 1: grid-wide dependency after this phase, 0: CTA-local only (MLPA -> MLPB)
  int src;           // producer phase whose slice counters gate this phase's input (-1 none / CTA-local)
  int sig;           // first counter index of this phase's output slice counters (-1 none)
  int nout;          // output elements (slices = ceil(nout / SL))
  int need_mult;     // readiness multiplicity per output element (1, or #CTA for the atomic MLPB accumulation)
  int abuf;          // smem byte offset of this phase's activation buffer (DF)
  int reuse_after;   // gemv ordinal whose compute must have finished before the gather may overwrite abuf (-1 none)
  int gord;          // ordinal among GEMV-like phases (W4, BF16, MLPB)
  int xread;         // counter index: every CTA staged this phase's input x (MLPA); MLPB waits for it before its atomics; -1 none
  int asg;           // 0: strided warp tasks (E1), 1: balanced contiguous ranges
  int ticks;         // staged-slice ticks the gather warp publishes for this phase
  int pad;
};
static_assert(sizeof(Ph2) % 16 == 0, "Ph2 layout");

struct CtaTab { int wlo, nwc, nseg, pad; unsigned long long sc_off; };  // fused-MLP slice owned by a CTA (units: 8-channel chunks)

__host__ __device__ inline bool is_gemv2(int t) { return t == PH_W4 || t == PH_BF16 || t == PH_MLPB; }
__host__ __device__ inline bool is_staged(const Ph2& d) { return (d.b.type == PH_W4 || d.b.type == PH_BF16); }

// ------------------------------------------------------------------ small helpers
__device__ __forceinline__ unsigned ld_acq(const unsigned* p) { unsigned v; asm volatile("ld.acquire.gpu.global.u32 %0,[%1];\n" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ void red_rel(unsigned* p, unsigned v) { asm volatile("red.release.gpu.global.add.u32 [%0], %1;\n" ::"l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ unsigned ld_vs(const unsigned* p) { unsigned v; asm volatile("ld.volatile.shared.u32 %0,[%1];\n" : "=r"(v) : "r"((uint32_t)__cvta_generic_to_shared(p)) : "memory"); return v; }
__device__ __forceinline__ void st_vs(unsigned* p, unsigned v) { asm volatile("st.volatile.shared.u32 [%0],%1;\n" ::"r"((uint32_t)__cvta_generic_to_shared(p)), "r"(v) : "memory"); }
__device__ __forceinline__ void bar_named(int id, int n) { asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ float4 ldcg4(const float* p) { return __ldcg((const float4*)p); }

__device__ __forceinline__ void cp_async16_nh(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((uint32_t)__cvta_generic_to_shared(dst)), "l"(src) : "memory");
}

// byte offset of element k inside a W4-permuted activation vector (slot t*128+c*32+l, 16 B per slot of 8 bf16)
__device__ __forceinline__ size_t perm_off(int k) { return ((size_t)((k >> 10) * 128 + ((k >> 3) & 3) * 32 + ((k >> 5) & 31))) * 16 + (k & 7) * 2; }

// ------------------------------------------------------------------ cursor over (task, tile) of the warp's stream
struct TP { int q0, qstep, cnt, tiles; };
template <class C>
__device__ __forceinline__ TP task_params(const Ph2& d, const CtaTab& ct, int wl, int gw, int W) {
  TP t;
  if (d.b.type == PH_MLPB) {
    const int nrb = d.b.N >> 7;
    t.q0 = wl; t.qstep = C::NW; t.cnt = wl < nrb ? (nrb - wl + C::NW - 1) / C::NW : 0; t.tiles = ct.nwc;
  } else if (d.b.epi == EPI_MLPA) {
    const int chlo = ct.wlo * 8, nch = ct.nwc * 8;
    const int lo = chlo + (int)((long long)wl * nch / C::NW), hi = chlo + (int)((long long)(wl + 1) * nch / C::NW);
    t.q0 = lo; t.qstep = 1; t.cnt = hi - lo; t.tiles = d.b.tiles;
  } else {
    const int nt = d.b.N / C::R;
    if (d.asg) {
      const int lo = (int)((long long)gw * nt / W), hi = (int)((long long)(gw + 1) * nt / W);
      t.q0 = lo; t.qstep = 1; t.cnt = hi - lo;
    } else {
      t.q0 = gw; t.qstep = W; t.cnt = gw < nt ? (nt - gw + W - 1) / W : 0;
    }
    t.tiles = d.b.tiles;
  }
  return t;
}

struct Cur {
  const Ph2* ph; int nph;
  int p, m, t;             // p < 0: exhausted
  TP tp;
  int gw, W, wl;
  CtaTab ct;
  const char* wbase; const char* sbase;
  uint32_t rowbytes, srowbytes;
  bool mlpb, w4;
  int pfmode;  // diagnostics for the L2 prefetch: bit0 no instruction, bit1 data lines only, bit2 evict_last
  int pslot, cslot;
  uint64_t pol;
};

template <class C>
__device__ __forceinline__ void cur_load(Cur& c, int p) {
  const Ph2& d = c.ph[p];
  c.p = p; c.m = 0; c.t = 0;
  c.tp = task_params<C>(d, c.ct, c.wl, c.gw, c.W);
  c.mlpb = d.b.type == PH_MLPB;
  if (c.mlpb) {
    c.wbase = (const char*)d.b.w + (size_t)20480 * c.ct.wlo;
    c.w4 = false; c.rowbytes = c.srowbytes = 0;
  } else {
    c.wbase = (const char*)d.b.w; c.sbase = (const char*)d.b.s;
    c.w4 = d.b.type == PH_W4;
    c.rowbytes = c.w4 ? (uint32_t)d.b.K / 2 : (uint32_t)d.b.K * 2;
    c.srowbytes = (uint32_t)d.b.K / 64;
  }
}
template <class C>
__device__ __forceinline__ void cur_seek(Cur& c, int p) {
  while (p >= 0 && p < c.nph) {
    const Ph2& d = c.ph[p];
    if (is_gemv2(d.b.type)) {
      cur_load<C>(c, p);
      if (c.tp.cnt > 0) return;
    }
    p = c.ph[p].b.next_gemv;
  }
  c.p = -1;
}
template <class C>
__device__ __forceinline__ void cur_advance(Cur& c) {
  if (++c.t == c.tp.tiles) {
    c.t = 0;
    if (++c.m == c.tp.cnt) cur_seek<C>(c, c.ph[c.p].b.next_gemv);
  }
}
template <class C>
__device__ __forceinline__ void cur_issue(Cur& c, char* ring_w, int lane) {
  char* slot = ring_w + c.pslot * C::STG;
  const long long q = c.tp.q0 + (long long)c.m * c.tp.qstep;
  if (c.mlpb) {
    cp_async16(slot + lane * 16, c.wbase + ((size_t)q * c.ct.nwc + c.t) * 512 + lane * 16, c.pol);
  } else {
    const size_t row0 = (size_t)q * C::R;
    const char* src = c.wbase + row0 * c.rowbytes + (size_t)c.t * 512 + lane * 16;
#pragma unroll
    for (int r = 0; r < C::R; r++) cp_async16(slot + r * 512 + lane * 16, src + (size_t)r * c.rowbytes, c.pol);
    if (c.w4 && lane < C::R) cp_async16(slot + C::R * 512 + lane * 16, c.sbase + (row0 + lane) * c.srowbytes + (size_t)c.t * 16, c.pol);
  }
  c.pslot = (c.pslot + 1 == C::D) ? 0 : c.pslot + 1;
}
template <class C>
__device__ __forceinline__ void cur_produce(Cur& c, char* ring_w, int lane) {
  if (c.p >= 0) { cur_issue<C>(c, ring_w, lane); cur_advance<C>(c); }
  cp_commit();
}
template <class C>
__device__ __forceinline__ void cur_prime(Cur& c, char* ring_w, int lane) {
#pragma unroll
  for (int i = 0; i < C::D; i++) cur_produce<C>(c, ring_w, lane);
}
__device__ __forceinline__ void l2_prefetch_last(const void* p) { asm volatile("prefetch.global.L2::evict_last [%0];\n" ::"l"(p)); }
template <class C>
__device__ __forceinline__ void cur_pf(Cur& c, int lane) {
  if (c.p < 0) return;
  const long long q = c.tp.q0 + (long long)c.m * c.tp.qstep;
  if (!(c.pfmode & 1)) {
    if (c.mlpb) {
      if (lane < 4) { const void* a = c.wbase + ((size_t)q * c.ct.nwc + c.t) * 512 + lane * 128; if (c.pfmode & 4) l2_prefetch_last(a); else l2_prefetch(a); }
    } else {
      const size_t row0 = (size_t)q * C::R;
      if (lane < 4 * C::R) { const void* a = c.wbase + (row0 + (lane >> 2)) * c.rowbytes + (size_t)c.t * 512 + (lane & 3) * 128; if (c.pfmode & 4) l2_prefetch_last(a); else l2_prefetch(a); }
      else if (c.w4 && lane >= 32 - C::R && !(c.pfmode & 2)) l2_prefetch(c.sbase + (row0 + lane - (32 - C::R)) * c.srowbytes + (size_t)c.t * 16);
    }
  }
  cur_advance<C>(c);
}

// ------------------------------------------------------------------ dequant+FMA tile math (E1 compute_w4 with a CHEAP ablation: no dequant ALU, results invalid)
template <bool CHEAP, int P>
__device__ __forceinline__ float nibx(uint32_t u) {
  if (CHEAP) return __uint_as_float(u >> (P & 15));  // ablation: 1 cheap op instead of LOP3+FADD (results invalid)
  return nib<P>(u);
}
template <int R, bool CHEAP>
__device__ __forceinline__ void compute_w4x(const uint4 (&w)[R], const uint32_t (&sc)[R], const uint4 (&a)[4], float (&acc)[R]) {
  float af[32];
#pragma unroll
  for (int c = 0; c < 4; c++) {
    af[c * 8 + 0] = bflo(a[c].x); af[c * 8 + 1] = bfhi(a[c].x);
    af[c * 8 + 2] = bflo(a[c].y); af[c * 8 + 3] = bfhi(a[c].y);
    af[c * 8 + 4] = bflo(a[c].z); af[c * 8 + 5] = bfhi(a[c].z);
    af[c * 8 + 6] = bflo(a[c].w); af[c * 8 + 7] = bfhi(a[c].w);
  }
#pragma unroll
  for (int r = 0; r < R; r++) {
    float S0 = 0.f, S1 = 0.f;
    const uint32_t ww[4] = {w[r].x, w[r].y, w[r].z, w[r].w};
#pragma unroll
    for (int c = 0; c < 4; c++) {
      const uint32_t u = ww[c], u2 = u >> 20;
      S0 = fmaf(nibx<CHEAP, 0>(u), af[c * 8 + 0], S0);
      S1 = fmaf(nibx<CHEAP, 4>(u), af[c * 8 + 1], S1);
      S0 = fmaf(nibx<CHEAP, 8>(u), af[c * 8 + 2], S0);
      S1 = fmaf(nibx<CHEAP, 12>(u), af[c * 8 + 3], S1);
      S0 = fmaf(nibx<CHEAP, 16>(u), af[c * 8 + 4], S0);
      S1 = fmaf(nibx<CHEAP, 0>(u2), af[c * 8 + 5], S1);
      S0 = fmaf(nibx<CHEAP, 4>(u2), af[c * 8 + 6], S0);
      S1 = fmaf(nibx<CHEAP, 8>(u2), af[c * 8 + 7], S1);
    }
    acc[r] = fmaf(bflo(sc[r]), S0 + S1, acc[r]);
  }
}
// dot of one W4 word (8 weights) with 8 prescaled fp32 activations
template <bool CHEAP>
__device__ __forceinline__ float dot8(uint32_t u, const float4& h0, const float4& h1) {
  const uint32_t u2 = u >> 20;
  float S0 = nibx<CHEAP, 0>(u) * h0.x;
  float S1 = nibx<CHEAP, 4>(u) * h0.y;
  S0 = fmaf(nibx<CHEAP, 8>(u), h0.z, S0);
  S1 = fmaf(nibx<CHEAP, 12>(u), h0.w, S1);
  S0 = fmaf(nibx<CHEAP, 16>(u), h1.x, S0);
  S1 = fmaf(nibx<CHEAP, 0>(u2), h1.y, S1);
  S0 = fmaf(nibx<CHEAP, 4>(u2), h1.z, S0);
  S1 = fmaf(nibx<CHEAP, 8>(u2), h1.w, S1);
  return S0 + S1;
}

// ------------------------------------------------------------------ core / embed elementwise stand-ins writing the new activation formats
// c (bf16) is written directly in the W4-permuted, prescaled layout the consumer GEMV wants (bit-identical math to E1: bf16(v)*2^k exact)
__device__ __forceinline__ void store_act(void* base, int k, float v) {
  *(uint16_t*)((char*)base + perm_off(k)) = (uint16_t)f2bf(v * act_prescale(k & 7));
}
__device__ __forceinline__ void core_elem2(const Ph2& d, int i) {
  switch (d.b.type) {
    case PH_EMBED: ((float*)d.b.out)[i] = ((const float*)d.b.in)[i]; break;
    case PH_CORE_GDN: {
      const float* y = (const float*)d.b.in;
      store_act(d.b.out, i, __ldcg(y + i) * sigm_f(__ldcg(y + 16383 - i)) * 2.0f);
    } break;
    case PH_CORE_A1: {
      const float* y = (const float*)d.b.in;
      ((float*)d.b.out)[i] = 0.7f * __ldcg(y + i) + 0.3f * __ldcg(y + 8192 + i);
    } break;
    case PH_CORE_A2: {
      const float* t = (const float*)d.b.in;
      store_act(d.b.out, i, __ldcg(t + i) * sigm_f(__ldcg(t + (i * 7 + 1) % 6144)) * 2.0f);
    } break;
  }
}

// ------------------------------------------------------------------ barrier-mode prologue (all threads of the CTA)
template <class C, bool W4>
__device__ __forceinline__ void prologue2(const Ph2& d, uint4* act, float* red) {
  if (d.b.pro == PRO_RMS) {
    prologue<C, W4>(d.b, act, red);  // E1 code path, unchanged numerics
  } else {
    const uint4* src = (const uint4*)d.b.in;
    const int nslot = d.b.K >> 3;
    for (int j = threadIdx.x; j < nslot; j += C::NT) act[j] = __ldcg(src + j);  // already prescaled + permuted by the producer
    __syncthreads();
  }
}

// ------------------------------------------------------------------ GEMV-like phase body for warp tasks (all kinds except MLPB)
template <class C, bool DF, bool W4, bool CHEAP>
__device__ __forceinline__ void run_gemv(const Ph2& d, Cur& s, Cur& pf, int pfd, const uint4* act, char* ring_w, int lane, const TP& tp,
                                         float* hS, int chlo, unsigned* cnt, unsigned ep, unsigned* s_stg, const float* s_inv, int stg_base, int abl) {
  constexpr int R = C::R;
  const int tiles = d.b.tiles;
  const bool late = DF && d.b.pro == PRO_RMS;
  const int ns = tiles;  // slices staged for this phase (W4: tile == slice; BF16 tile = 1/4 slice, nslices = K/1024 == tiles/4)
  (void)ns;
  for (int m = 0; m < tp.cnt; ++m) {
    const size_t row0 = (size_t)(tp.q0 + (long long)m * tp.qstep) * R;
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; r++) acc[r] = 0.f;
    for (int t = 0; t < tiles; ++t) {
      if (DF && m == 0 && !(abl & 1)) {
        if (W4 || (t & 3) == 0) {
          const unsigned need = (unsigned)(stg_base + (W4 ? t : (t >> 2)) + 1);
          while ((int)(ld_vs(s_stg) - need) < 0) {}
          __threadfence_block();
        }
      }
      cp_wait<C::D - 1>();
      __syncwarp();
      const char* slot = ring_w + s.cslot * C::STG;
      uint4 w[R];
#pragma unroll
      for (int r = 0; r < R; r++) w[r] = *(const uint4*)(slot + r * 512 + lane * 16);
      uint32_t sc[R];
      uint4 a4[4];
      uint4 a1;
      if (W4) {
#pragma unroll
        for (int r = 0; r < R; r++) sc[r] = *(const uint16_t*)(slot + R * 512 + r * 16 + (lane >> 2) * 2);
#pragma unroll
        for (int c = 0; c < 4; c++) a4[c] = act[t * 128 + c * 32 + lane];
      } else {
        a1 = act[t * 32 + lane];
      }
      __syncwarp();
      s.cslot = (s.cslot + 1 == C::D) ? 0 : s.cslot + 1;
      cur_produce<C>(s, ring_w, lane);
      if (pfd) cur_pf<C>(pf, lane);
      if (W4) compute_w4x<R, CHEAP>(w, sc, a4, acc); else compute_bf16<R>(w, a1, acc);
    }
    float tot[R];
#pragma unroll
    for (int r = 0; r < R; r++) tot[r] = warp_sum(acc[r]);
    if (late) {  // the 1/rms scalar is published by the gather warp one tick after the last slice
      if (m == 0 && !(abl & 1)) {
        const unsigned need = (unsigned)(stg_base + (W4 ? tiles : (tiles >> 2)) + 1);
        while ((int)(ld_vs(s_stg) - need) < 0) {}
      }
      const float inv = s_inv[d.gord & 3];
#pragma unroll
      for (int r = 0; r < R; r++) tot[r] *= inv;
    }
    // ---- epilogue (lane 0 writes; R == 2 for GATEUP/MLPA)
    if (d.b.epi == EPI_MLPA) {
      if (lane == 0) {
        const int k = (int)(row0 >> 1);
        const float h = __uint_as_float(f2bf(silu_f(tot[0]) * tot[1]) << 16);
        hS[k - chlo] = h * act_prescale((k - chlo) & 7);
      }
      continue;
    }
    if (lane == 0) {
      if (d.b.epi == EPI_STORE) {
        float* o = (float*)d.b.out;
        if (R == 2) *(float2*)(o + row0) = make_float2(tot[0], tot[1]);
        else {
#pragma unroll
          for (int r = 0; r < R; r++) o[row0 + r] = tot[r];
        }
      } else if (d.b.epi == EPI_RESID) {
        float* o = (float*)d.b.out;
        if (R == 2) { float2 v = __ldcg((const float2*)(o + row0)); *(float2*)(o + row0) = make_float2(v.x + tot[0], v.y + tot[1]); }
        else {
#pragma unroll
          for (int r = 0; r < R; r++) o[row0 + r] = __ldcg(o + row0 + r) + tot[r];
        }
      } else {  // EPI_GATEUP -> h in the consumer's prescaled + permuted layout
        const int k = (int)(row0 >> 1);
#pragma unroll
        for (int j = 0; j < R / 2; j++) store_act(d.b.out, k + j, __uint_as_float(f2bf(silu_f(tot[2 * j]) * tot[2 * j + 1]) << 16));
      }
      if (DF && d.sig >= 0) {
        asm volatile("fence.acq_rel.gpu;\n" ::: "memory");
        const int units = d.b.epi == EPI_GATEUP ? R / 2 : R;
        const int idx = d.b.epi == EPI_GATEUP ? (int)(row0 >> 1) : (int)row0;
        red_rel(cnt + d.sig + (idx >> 10), (unsigned)units);
      }
    }
  }
}

// ------------------------------------------------------------------ fused-MLP down slice: acc[row] += sum over this CTA's channels (lane-parallel over rows, no reductions)
template <class C, bool DF, bool CHEAP>
__device__ __forceinline__ void run_mlpb(const Ph2& d, Cur& s, Cur& pf, int pfd, char* ring_w, int lane, const TP& tp, const CtaTab& ct, const float* hS,
                                         int wl, unsigned* cnt, unsigned ep, int ncta, int abl) {
  float* out = (float*)d.b.out;
  const int nwc = ct.nwc, g0 = ct.wlo >> 4;
  const char* sbase = (const char*)d.b.s + ct.sc_off;
  bool first = true;
  for (int m = 0; m < tp.cnt; ++m) {
    const int rb = tp.q0 + m * tp.qstep;
    uint2 sc[3];
#pragma unroll
    for (int sg = 0; sg < 3; sg++) {
      sc[sg] = make_uint2(0, 0);
      if (sg < ct.nseg) sc[sg] = __ldcg((const uint2*)(sbase + (((size_t)rb * ct.nseg + sg) * 32 + lane) * 8));
    }
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    int w = 0;
#pragma unroll
    for (int sg = 0; sg < 3; sg++) {
      if (sg < ct.nseg) {
        const int wend = min(nwc, (g0 + sg + 1) * 16 - ct.wlo);
        float S[4] = {0.f, 0.f, 0.f, 0.f};
        for (; w < wend; ++w) {
          cp_wait<C::D - 1>();
          __syncwarp();
          const uint4 v = *(const uint4*)(ring_w + s.cslot * C::STG + lane * 16);
          const float4 h0 = *(const float4*)(hS + w * 8), h1 = *(const float4*)(hS + w * 8 + 4);
          __syncwarp();
          s.cslot = (s.cslot + 1 == C::D) ? 0 : s.cslot + 1;
          cur_produce<C>(s, ring_w, lane);
          if (pfd) cur_pf<C>(pf, lane);
          S[0] += dot8<CHEAP>(v.x, h0, h1);
          S[1] += dot8<CHEAP>(v.y, h0, h1);
          S[2] += dot8<CHEAP>(v.z, h0, h1);
          S[3] += dot8<CHEAP>(v.w, h0, h1);
        }
        acc[0] = fmaf(bflo(sc[sg].x), S[0], acc[0]);
        acc[1] = fmaf(bfhi(sc[sg].x), S[1], acc[1]);
        acc[2] = fmaf(bflo(sc[sg].y), S[2], acc[2]);
        acc[3] = fmaf(bfhi(sc[sg].y), S[3], acc[3]);
      }
    }
    // every CTA must have staged x for this MLP before anyone accumulates into it
    if (first) {
      first = false;
      if (d.xread >= 0 && !(abl & 1)) {
        const unsigned target = (unsigned)ncta * ep;
        while ((int)(ld_acq(cnt + d.xread) - target) < 0) {}
      }
    }
#pragma unroll
    for (int i = 0; i < 4; i++) atomicAdd(out + rb * 128 + i * 32 + lane, acc[i]);
    if (DF) {
      __threadfence();
      __syncwarp();
      if (lane == 0) red_rel(cnt + d.sig + (rb >> 3), 128u);
    }
  }
}

// ------------------------------------------------------------------ DF gather warp
__device__ __forceinline__ bool slice_ready(const unsigned* cnt, const Ph2& src, int s, unsigned ep) {
  const int rem = src.nout - s * SL;
  const unsigned need = (unsigned)((rem < SL ? rem : SL) * src.need_mult) * ep;
  return (int)(ld_acq(cnt + src.sig + s) - need) >= 0;
}

template <class C, bool CHEAP>
__device__ void gather_main(const Ph2* ph, int nph, const CtaTab& ct, unsigned* cnt, unsigned ep, int ncta, char* smem_raw, unsigned* s_stg,
                            unsigned* s_cdone, float* s_inv, int abl) {
  const int lane = threadIdx.x & 31, cta = blockIdx.x;
  unsigned ticks = 0;
  for (int p = 0; p < nph; p++) {
    const Ph2 d = ph[p];
    const int t = d.b.type;
    if (t == PH_MLPB) continue;
    if (t == PH_EMBED || t == PH_CORE_GDN || t == PH_CORE_A1 || t == PH_CORE_A2) {
      // distributed elementwise phase: this CTA's contiguous element range, gated on the whole input vector
      if (d.src >= 0 && !(abl & 1)) {
        const Ph2& sp = ph[d.src];
        const int nsl = (sp.nout + SL - 1) / SL;
        while (true) {
          bool ok = true;
          if (lane < nsl) ok = slice_ready(cnt, sp, lane, ep);
          if (__all_sync(0xffffffffu, ok)) break;
        }
        __syncwarp();
      }
      const int e0 = (int)((long long)d.b.N * cta / ncta), e1 = (int)((long long)d.b.N * (cta + 1) / ncta);
      for (int i = e0 + lane; i < e1; i += 32) core_elem2(d, i);
      __threadfence();
      __syncwarp();
      for (int s0 = e0 >> 10; s0 <= (e1 - 1) >> 10; s0++) {
        if (lane == 0) {
          const int a = max(e0, s0 * SL), b = min(e1, (s0 + 1) * SL);
          red_rel(cnt + d.sig + s0, (unsigned)(b - a));
        }
      }
      continue;
    }
    // staged GEMV phase
    const bool w4 = t == PH_W4;
    const int nsl = d.b.K / SL;
    if (d.reuse_after >= 0) {
      const unsigned need = (unsigned)(C::NW * (d.reuse_after + 1));
      while ((int)(ld_vs(s_cdone) - need) < 0) {}
    }
    uint4* act = (uint4*)(smem_raw + d.abuf);
    const Ph2* sp = d.src >= 0 ? &ph[d.src] : nullptr;
    float ss = 0.f;
    int done = 0;
    while (done < nsl) {
      int rdy = nsl;
      if (sp && !(abl & 1)) {
        bool ok = true;
        if (lane < nsl) ok = slice_ready(cnt, *sp, lane, ep);
        const unsigned notok = ~__ballot_sync(0xffffffffu, ok) & ((1u << nsl) - 1u);
        rdy = notok ? (__ffs(notok) - 1) : nsl;
        if (rdy <= done) continue;
      }
      const int n = rdy - done > 2 ? 2 : rdy - done;  // up to 2 slices per batch
      if (d.b.pro == PRO_RMS) {
        float4 x[2][8];
#pragma unroll
        for (int b = 0; b < 2; b++)
          if (b < n) {
            const float* src = (const float*)d.b.in + (size_t)(done + b) * SL + lane * 32;
#pragma unroll
            for (int i = 0; i < 8; i++) x[b][i] = ldcg4(src + i * 4);
          }
#pragma unroll
        for (int b = 0; b < 2; b++)
          if (b < n) {
            const int tt = done + b;
#pragma unroll
            for (int c = 0; c < 4; c++) {
              const float4 a = x[b][c * 2], bb = x[b][c * 2 + 1];
              ss = fmaf(a.x, a.x, ss); ss = fmaf(a.y, a.y, ss); ss = fmaf(a.z, a.z, ss); ss = fmaf(a.w, a.w, ss);
              ss = fmaf(bb.x, bb.x, ss); ss = fmaf(bb.y, bb.y, ss); ss = fmaf(bb.z, bb.z, ss); ss = fmaf(bb.w, bb.w, ss);
              uint4 q;
              if (w4) {
                q.x = pack2(a.x * act_prescale(0), a.y * act_prescale(1)); q.y = pack2(a.z * act_prescale(2), a.w * act_prescale(3));
                q.z = pack2(bb.x * act_prescale(4), bb.y * act_prescale(5)); q.w = pack2(bb.z * act_prescale(6), bb.w * act_prescale(7));
                act[tt * 128 + c * 32 + lane] = q;
              } else {
                q.x = pack2(a.x, a.y); q.y = pack2(a.z, a.w); q.z = pack2(bb.x, bb.y); q.w = pack2(bb.z, bb.w);
                act[tt * 128 + lane * 4 + c] = q;
              }
            }
          }
      } else {  // producer wrote the slice already prescaled + permuted: plain copy
        for (int b = 0; b < n; b++) {
          const uint4* src = (const uint4*)d.b.in + (size_t)(done + b) * 128;
#pragma unroll
          for (int i = 0; i < 4; i++) cp_async16_nh(act + (done + b) * 128 + lane + 32 * i, src + lane + 32 * i);
        }
        cp_commit();
        cp_wait<0>();
      }
      __threadfence_block();
      __syncwarp();
      done += n;
      if (lane == 0) st_vs(s_stg, ticks + done);
    }
    ticks += nsl;
    if (d.b.pro == PRO_RMS) {
      ss = warp_sum(ss);
      if (lane == 0) {
        s_inv[d.gord & 3] = rsqrtf(ss * (1.0f / HID) + 1e-6f);
        __threadfence_block();
        st_vs(s_stg, ticks + 1);
      }
      ticks += 1;
      if (d.xread >= 0) {
        __threadfence();
        __syncwarp();
        if (lane == 0) red_rel(cnt + d.xread, 1u);
      }
    }
  }
}

// ------------------------------------------------------------------ the persistent kernel
template <class C, bool DF, bool CHEAP>
__global__ void __launch_bounds__(C::NT + (DF ? 32 : 0), 1)
    k_main(const Ph2* ph, int nph, const CtaTab* tab, unsigned* cnt, unsigned bar_target0, unsigned ep, int ring_off, int first_gemv, int abl, int pfd,
           unsigned long long* stamps) {
  extern __shared__ __align__(128) char smem_raw[];
  __shared__ unsigned s_stg, s_cdone;
  __shared__ float s_inv[4];
  __shared__ float s_red[C::NW];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, cta = blockIdx.x, ncta = gridDim.x;
  if (threadIdx.x == 0) { s_stg = 0; s_cdone = 0; }
  __syncthreads();
  const CtaTab ct = tab[cta];
  float* hS = (float*)(smem_raw + ring_off - 1024);
  if (DF && warp == C::NW) { gather_main<C, CHEAP>(ph, nph, ct, cnt, ep, ncta, smem_raw, &s_stg, &s_cdone, s_inv, abl); return; }

  char* ring_w = smem_raw + ring_off + warp * (C::D * C::STG);
  Cur s;
  s.ph = ph; s.nph = nph; s.gw = cta * C::NW + warp; s.W = ncta * C::NW; s.wl = warp; s.ct = ct;
  s.pslot = 0; s.cslot = 0; s.p = -1;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(s.pol));
  cur_seek<C>(s, first_gemv);
  cur_prime<C>(s, ring_w, lane);
  Cur pf = s;
  pf.pfmode = (abl >> 3) & 7;
  for (int i = 0; i < pfd; i++) cur_pf<C>(pf, lane);

  unsigned bar_t = bar_target0;
  int stg_base = 0;
  const int gtid = cta * C::NT + threadIdx.x, nthr = ncta * C::NT;
  for (int p = 0; p < nph; p++) {
    const Ph2 d = ph[p];
    unsigned long long* st4 = stamps ? stamps + ((size_t)p * ncta + cta) * 4 : nullptr;
    if (st4 && threadIdx.x == 0) { st4[0] = gtimer(); st4[1] = st4[0]; }
    const int t = d.b.type;
    if (!is_gemv2(t)) {
      if (!DF && !(abl & 4)) {
        for (int i = gtid; i < d.b.N; i += nthr) core_elem2(d, i);
      }
    } else {
      const TP tp = task_params<C>(d, ct, warp, s.gw, s.W);
      uint4* act = (uint4*)(smem_raw + (DF ? d.abuf : 0));
      if (t == PH_MLPB) {
        if (st4 && threadIdx.x == 0) st4[1] = gtimer();
        run_mlpb<C, DF, CHEAP>(d, s, pf, pfd, ring_w, lane, tp, ct, hS, warp, cnt, ep, ncta, abl);
      } else {
        if (!DF) {
          if (!(abl & 2)) { if (t == PH_W4) prologue2<C, true>(d, act, s_red); else prologue2<C, false>(d, act, s_red); }
          if (d.xread >= 0 && threadIdx.x == 0) red_rel(cnt + d.xread, 1u);
        }
        if (st4 && threadIdx.x == 0) st4[1] = gtimer();
        if (t == PH_W4) run_gemv<C, DF, true, CHEAP>(d, s, pf, pfd, act, ring_w, lane, tp, hS, ct.wlo * 8, cnt, ep, &s_stg, s_inv, stg_base, abl);
        else run_gemv<C, DF, false, CHEAP>(d, s, pf, pfd, act, ring_w, lane, tp, hS, ct.wlo * 8, cnt, ep, &s_stg, s_inv, stg_base, abl);
        if (DF && is_staged(d)) stg_base += d.ticks;
      }
      if (DF) {
        __syncwarp();
        if (lane == 0) atomicAdd(&s_cdone, 1u);
        if (d.b.epi == EPI_MLPA) bar_named(1, C::NT);
      } else if (d.b.epi == EPI_MLPA) {
        __syncthreads();
      }
    }
    if (st4) { if (!DF) __syncthreads(); if (threadIdx.x == 0) st4[2] = gtimer(); }
    if (!DF && d.sync && p + 1 < nph && !(abl & 1)) {
      bar_t += ncta;
      grid_barrier<false>(cnt, bar_t);
    }
    if (st4 && threadIdx.x == 0) st4[3] = gtimer();
  }
}

// ------------------------------------------------------------------ repack of the down projection into per-CTA row-block-major slices (offline in a real engine)
// dst block of CTA c (word units, 4 B): [rb][w][128 words]: word (rb*nwc + w)*128 + lane*4 + wh  <-  natural word (row = rb*128 + wh*32 + lane, col = wlo + w)
__global__ void k_repack_down(const uint32_t* __restrict__ src, uint32_t* __restrict__ dst, const CtaTab* tab, int n_cta_tab) {
  const int c = blockIdx.y;
  const CtaTab ct = tab[c];
  const size_t nwords = (size_t)HID * ct.nwc;
  uint32_t* d = dst + (size_t)HID * ct.wlo;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < nwords; i += (size_t)gridDim.x * blockDim.x) {
    const int wh = i & 3, lane = (i >> 2) & 31;
    const size_t rest = i >> 7;
    const int w = (int)(rest % ct.nwc), rb = (int)(rest / ct.nwc);
    const int row = rb * 128 + wh * 32 + lane;
    d[i] = src[(size_t)row * (INTER / 8) + ct.wlo + w];
  }
}
// scales: block region [rb][seg][32 lanes] x 4 bf16 (rows lane, lane+32, lane+64, lane+96); natural: bf16 [HID][INTER/128]
__global__ void k_repack_down_scales(const uint16_t* __restrict__ src, uint16_t* __restrict__ dst, const CtaTab* tab) {
  const int c = blockIdx.y;
  const CtaTab ct = tab[c];
  const size_t n = (size_t)(HID / 128) * ct.nseg * 128;
  uint16_t* d = dst + ct.sc_off / 2;
  const int g0 = ct.wlo >> 4;
  for (size_t j = (size_t)blockIdx.x * blockDim.x + threadIdx.x; j < n; j += (size_t)gridDim.x * blockDim.x) {
    const int wh = j & 3, lane = (j >> 2) & 31;
    const size_t rest = j >> 7;
    const int sg = (int)(rest % ct.nseg), rb = (int)(rest / ct.nseg);
    const int row = rb * 128 + wh * 32 + lane;
    d[j] = src[(size_t)row * (INTER / 128) + g0 + sg];
  }
}

}  // namespace e1
