// Clean-room W4 (g128, symmetric uint4b8) GEMV phase kernels + per-phase-kernel (V1) and persistent (V3/V4) execution styles.
// See README.md. Layout (= compressed-tensors pack-quantized natural layout):
//   W4 : uint32 [N][K/8], nibble i of word j holds k = 8*j+i, value (nib-8)*scale[n][k/128] (bf16 scale)
//   BF16: bf16 [N][K]
// One "tile" = 512 contiguous bytes of one row (32 lanes x 16 B): 1024 int4 weights (= 8 groups, 16 B of scales)
// or 256 bf16 weights.  A warp task = R consecutive rows; the warp streams tiles through a private cp.async ring.
#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace e1 {

constexpr int HID = 5120;
constexpr int INTER = 17408;
constexpr int ACT_MAX_BYTES = INTER * 2;  // largest activation vector (bf16) = 34816 B

enum PhaseType : int { PH_EMBED = 0, PH_W4, PH_BF16, PH_CORE_GDN, PH_CORE_A1, PH_CORE_A2 };
enum ProType : int { PRO_RMS = 0, PRO_VEC = 1 };
enum EpiType : int { EPI_STORE = 0, EPI_RESID = 1, EPI_GATEUP = 2 };

struct Phase {
  int type, N, K, pro, epi, next_gemv, tiles, pad;
  const void* w;
  const void* s;
  const void* in;
  void* out;
};
static_assert(sizeof(Phase) == 64, "Phase layout");

__host__ __device__ inline bool is_gemv(int t) { return t == PH_W4 || t == PH_BF16; }

// ------------------------------------------------------------------ small helpers
__device__ __forceinline__ float bflo(uint32_t u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float bfhi(uint32_t u) { return __uint_as_float(u & 0xffff0000u); }
__device__ __forceinline__ uint32_t f2bf(float f) { return (uint32_t)__bfloat16_as_ushort(__float2bfloat16_rn(f)); }
__device__ __forceinline__ uint32_t pack2(float lo, float hi) { return f2bf(lo) | (f2bf(hi) << 16); }
__device__ __forceinline__ float silu_f(float g) { return g / (1.0f + __expf(-g)); }
__device__ __forceinline__ float sigm_f(float g) { return 1.0f / (1.0f + __expf(-g)); }

// prescale applied to the activation slot e (k mod 8) so that the mantissa-window dequant is exact: 2^(23-P_e)
__device__ __forceinline__ float act_prescale(int e) {
  const float t[8] = {8388608.f, 524288.f, 32768.f, 2048.f, 128.f, 8388608.f, 524288.f, 32768.f};
  return t[e];
}

// (n-8) * 2^(P-23) exactly, n = nibble at bit P of u (other bits of u are masked off)
template <int P>
__device__ __forceinline__ float nib(uint32_t u) {
  return __uint_as_float(0x3F800000u | (u & (0xFu << P))) - __uint_as_float(0x3F800000u | (1u << (P + 3)));
}

__device__ __forceinline__ void cp_async16(void* dst, const void* src, uint64_t pol) {
  uint32_t d = (uint32_t)__cvta_generic_to_shared(dst);
#ifdef E1_NO_EVICT_HINT
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(d), "l"(src) : "memory");
#else
  asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;\n" ::"r"(d), "l"(src), "l"(pol) : "memory");
#endif
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }

__device__ __forceinline__ unsigned long long gtimer() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// ------------------------------------------------------------------ config
// NW warps/CTA, R rows per warp task, D ring depth (tiles in flight per warp), MINB CTAs/SM
template <int NW_, int R_, int D_, int MINB_ = 1>
struct Cfg {
  static constexpr int NW = NW_, R = R_, D = D_, MINB = MINB_, NT = NW_ * 32;
  static constexpr int STG = R_ * 512 + R_ * 16;  // stage bytes: R row-tiles + R scale vectors
  static constexpr int RING_BYTES = NW_ * D_ * STG;
};

// ------------------------------------------------------------------ per-warp tile stream
struct Stream {
  const Phase* ph;
  int nph, gw, W;
  int p, m, t;  // producer cursor; p < 0 == exhausted
  const char* wbase;
  const char* sbase;
  uint32_t rowbytes, srowbytes;
  int tiles, ntasks;
  bool w4;
  int pslot, cslot;
  uint64_t pol;
};

template <class C>
__device__ __forceinline__ void stream_load(Stream& s, int p) {
  const Phase& d = s.ph[p];
  s.p = p; s.m = 0; s.t = 0;
  s.wbase = (const char*)d.w;
  s.sbase = (const char*)d.s;
  s.w4 = d.type == PH_W4;
  s.rowbytes = s.w4 ? (uint32_t)d.K / 2 : (uint32_t)d.K * 2;
  s.srowbytes = (uint32_t)d.K / 64;
  s.tiles = d.tiles;
  s.ntasks = d.N / C::R;
}
// first phase >= p (following next_gemv links) in which this warp owns a task
template <class C>
__device__ __forceinline__ void stream_seek(Stream& s, int p) {
  while (p >= 0 && p < s.nph) {
    if (s.gw < s.ph[p].N / C::R) { stream_load<C>(s, p); return; }
    p = s.ph[p].next_gemv;
  }
  s.p = -1;
}
template <class C>
__device__ __forceinline__ void stream_begin(Stream& s, int p) {  // this phase only
  if (s.gw < s.ph[p].N / C::R) stream_load<C>(s, p); else s.p = -1;
}
template <class C, bool CROSS>
__device__ __forceinline__ void stream_advance(Stream& s) {
  if (++s.t == s.tiles) {
    s.t = 0; ++s.m;
    if (s.gw + (long long)s.m * s.W >= s.ntasks) {
      if (CROSS) stream_seek<C>(s, s.ph[s.p].next_gemv); else s.p = -1;
    }
  }
}
template <class C>
__device__ __forceinline__ void issue_tile(Stream& s, char* ring_w, int lane) {
  char* slot = ring_w + s.pslot * C::STG;
  const size_t row0 = (size_t)(s.gw + (long long)s.m * s.W) * C::R;
  const char* src = s.wbase + row0 * s.rowbytes + (size_t)s.t * 512 + lane * 16;
#pragma unroll
  for (int r = 0; r < C::R; r++) cp_async16(slot + r * 512 + lane * 16, src + (size_t)r * s.rowbytes, s.pol);
  if (s.w4 && lane < C::R)
    cp_async16(slot + C::R * 512 + lane * 16, s.sbase + (row0 + lane) * s.srowbytes + (size_t)s.t * 16, s.pol);
  s.pslot = (s.pslot + 1 == C::D) ? 0 : s.pslot + 1;
}
template <class C, bool CROSS>
__device__ __forceinline__ void produce_one(Stream& s, char* ring_w, int lane) {
  if (s.p >= 0) { issue_tile<C>(s, ring_w, lane); stream_advance<C, CROSS>(s); }
  cp_commit();
}
template <class C, bool CROSS>
__device__ __forceinline__ void prime(Stream& s, char* ring_w, int lane) {
#pragma unroll
  for (int i = 0; i < C::D; i++) produce_one<C, CROSS>(s, ring_w, lane);
}

// ------------------------------------------------------------------ prologue: activation -> smem (bf16, permuted)
// W4 layout: element k = t*1024 + l*32 + c*8 + e lives in 16B slot t*128 + c*32 + l  (conflict-free lane reads)
// BF16 layout: natural, slot = k/8
__device__ __forceinline__ int slot_perm(int j) { return (j & ~127) | ((j & 3) << 5) | ((j >> 2) & 31); }

template <int NW>
__device__ __forceinline__ float block_sum(float v, float* red) {
  v = warp_sum(v);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
  __syncthreads();
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < NW; i++) s += red[i];
  __syncthreads();
  return s;
}

template <class C, bool W4>
__device__ __forceinline__ void prologue(const Phase& d, uint4* act, float* red) {
  constexpr int NT = C::NT;
  const int tid = threadIdx.x;
  const int nslot = d.K >> 3;
  if (d.pro == PRO_RMS) {
    constexpr int SPT = (HID / 8 + NT - 1) / NT;
    const float* x = (const float*)d.in;
    float v[SPT][8];
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < SPT; i++) {
      int j = tid + i * NT;
      if (j < nslot) {
        float4 a = __ldcg((const float4*)(x + 8 * j));
        float4 b = __ldcg((const float4*)(x + 8 * j + 4));
        v[i][0] = a.x; v[i][1] = a.y; v[i][2] = a.z; v[i][3] = a.w;
        v[i][4] = b.x; v[i][5] = b.y; v[i][6] = b.z; v[i][7] = b.w;
#pragma unroll
        for (int e = 0; e < 8; e++) ss = fmaf(v[i][e], v[i][e], ss);
      } else {
#pragma unroll
        for (int e = 0; e < 8; e++) v[i][e] = 0.f;
      }
    }
    const float tot = block_sum<C::NW>(ss, red);
    const float inv = rsqrtf(tot * (1.0f / HID) + 1e-6f);
#pragma unroll
    for (int i = 0; i < SPT; i++) {
      int j = tid + i * NT;
      if (j < nslot) {
        uint4 q;
        float f[8];
#pragma unroll
        for (int e = 0; e < 8; e++) f[e] = v[i][e] * inv;
        if (W4) {
#pragma unroll
          for (int e = 0; e < 8; e++) f[e] *= act_prescale(e);
        }
        // round to bf16 first (the activation is a bf16 vector), prescale is a power of two so order is irrelevant
        q.x = pack2(f[0], f[1]); q.y = pack2(f[2], f[3]); q.z = pack2(f[4], f[5]); q.w = pack2(f[6], f[7]);
        act[W4 ? slot_perm(j) : j] = q;
      }
    }
  } else {
    const uint4* src = (const uint4*)d.in;
    for (int j = tid; j < nslot; j += NT) {
      uint4 q = __ldcg(src + j);
      if (W4) {
        q.x = pack2(bflo(q.x) * act_prescale(0), bfhi(q.x) * act_prescale(1));
        q.y = pack2(bflo(q.y) * act_prescale(2), bfhi(q.y) * act_prescale(3));
        q.z = pack2(bflo(q.z) * act_prescale(4), bfhi(q.z) * act_prescale(5));
        q.w = pack2(bflo(q.w) * act_prescale(6), bfhi(q.w) * act_prescale(7));
      }
      act[W4 ? slot_perm(j) : j] = q;
    }
  }
  __syncthreads();
}

// ------------------------------------------------------------------ tile math
template <int R>
__device__ __forceinline__ void compute_w4(const uint4 (&w)[R], const uint32_t (&sc)[R], const uint4 (&a)[4], float (&acc)[R]) {
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
      S0 = fmaf(nib<0>(u), af[c * 8 + 0], S0);
      S1 = fmaf(nib<4>(u), af[c * 8 + 1], S1);
      S0 = fmaf(nib<8>(u), af[c * 8 + 2], S0);
      S1 = fmaf(nib<12>(u), af[c * 8 + 3], S1);
      S0 = fmaf(nib<16>(u), af[c * 8 + 4], S0);
      S1 = fmaf(nib<0>(u2), af[c * 8 + 5], S1);
      S0 = fmaf(nib<4>(u2), af[c * 8 + 6], S0);
      S1 = fmaf(nib<8>(u2), af[c * 8 + 7], S1);
    }
    acc[r] = fmaf(bflo(sc[r]), S0 + S1, acc[r]);
  }
}

template <int R>
__device__ __forceinline__ void compute_bf16(const uint4 (&w)[R], const uint4& a, float (&acc)[R]) {
  const float a0 = bflo(a.x), a1 = bfhi(a.x), a2 = bflo(a.y), a3 = bfhi(a.y);
  const float a4 = bflo(a.z), a5 = bfhi(a.z), a6 = bflo(a.w), a7 = bfhi(a.w);
#pragma unroll
  for (int r = 0; r < R; r++) {
    float s0 = acc[r], s1 = 0.f;
    s0 = fmaf(bflo(w[r].x), a0, s0); s1 = fmaf(bfhi(w[r].x), a1, s1);
    s0 = fmaf(bflo(w[r].y), a2, s0); s1 = fmaf(bfhi(w[r].y), a3, s1);
    s0 = fmaf(bflo(w[r].z), a4, s0); s1 = fmaf(bfhi(w[r].z), a5, s1);
    s0 = fmaf(bflo(w[r].w), a6, s0); s1 = fmaf(bfhi(w[r].w), a7, s1);
    acc[r] = s0 + s1;
  }
}

template <int R>
__device__ __forceinline__ void epilogue(const Phase& d, size_t row0, int lane, const float (&tot)[R]) {
  if (d.epi == EPI_STORE) {
    float* o = (float*)d.out;
#pragma unroll
    for (int r = 0; r < R; r++) if (lane == r) o[row0 + r] = tot[r];
  } else if (d.epi == EPI_RESID) {
    float* o = (float*)d.out;
#pragma unroll
    for (int r = 0; r < R; r++) if (lane == r) o[row0 + r] = o[row0 + r] + tot[r];
  } else {  // EPI_GATEUP: rows interleaved (gate_j, up_j)
    uint16_t* o = (uint16_t*)d.out;
#pragma unroll
    for (int j = 0; j < R / 2; j++)
      if (lane == j) o[(row0 >> 1) + j] = (uint16_t)f2bf(silu_f(tot[2 * j]) * tot[2 * j + 1]);
  }
}

// ------------------------------------------------------------------ the GEMV phase body (consumer)
template <class C, bool CROSS, bool W4>
__device__ __forceinline__ void gemv_tasks(const Phase& d, Stream& s, const uint4* act, char* ring_w, int lane) {
  constexpr int R = C::R;
  const int ntasks = d.N / R;
  const int tiles = d.tiles;
  for (int m = 0; s.gw + (long long)m * s.W < ntasks; ++m) {
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; r++) acc[r] = 0.f;
    for (int t = 0; t < tiles; ++t) {
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
      produce_one<C, CROSS>(s, ring_w, lane);
      if (W4) compute_w4<R>(w, sc, a4, acc); else compute_bf16<R>(w, a1, acc);
    }
    float tot[R];
#pragma unroll
    for (int r = 0; r < R; r++) tot[r] = warp_sum(acc[r]);
    epilogue<R>(d, (size_t)(s.gw + (long long)m * s.W) * R, lane, tot);
  }
}

// ------------------------------------------------------------------ non-GEMV (tiny) phases; shared with the reference
__device__ __forceinline__ void core_elem(const Phase& d, int i) {
  switch (d.type) {
    case PH_EMBED: ((float*)d.out)[i] = ((const float*)d.in)[i]; break;
    case PH_CORE_GDN: {  // y1[16384] -> c[6144]: value * sigmoid(gate)
      const float* y = (const float*)d.in;
      ((uint16_t*)d.out)[i] = (uint16_t)f2bf(__ldcg(y + i) * sigm_f(__ldcg(y + 16383 - i)) * 2.0f);
    } break;
    case PH_CORE_A1: {  // y1[14336] -> t[6144]
      const float* y = (const float*)d.in;
      ((float*)d.out)[i] = 0.7f * __ldcg(y + i) + 0.3f * __ldcg(y + 8192 + i);
    } break;
    case PH_CORE_A2: {  // t[6144] -> c[6144]
      const float* t = (const float*)d.in;
      ((uint16_t*)d.out)[i] = (uint16_t)f2bf(__ldcg(t + i) * sigm_f(__ldcg(t + (i * 7 + 1) % 6144)) * 2.0f);
    } break;
  }
}
__device__ __forceinline__ void core_phase(const Phase& d, int gtid, int nthr) {
  for (int i = gtid; i < d.N; i += nthr) core_elem(d, i);
}

// ------------------------------------------------------------------ per-phase kernels (V1 / V2)
// PREF=true (V1b): issue the first D weight tiles BEFORE the activation prologue (weights do not depend on the previous phase)
// PREF=false (V1a): plain kernel, weight loads start only after the activation vector has been staged
template <class C, bool PREF>
__global__ void __launch_bounds__(C::NT, C::MINB) k_gemv(const Phase* ph, int p, int ring_off) {
  extern __shared__ __align__(128) char smem_raw[];
  __shared__ float red[C::NW];
  uint4* act = (uint4*)smem_raw;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  char* ring_w = smem_raw + ring_off + warp * (C::D * C::STG);
  Stream s;
  s.ph = ph; s.nph = p + 1; s.gw = blockIdx.x * C::NW + warp; s.W = gridDim.x * C::NW;
  s.pslot = 0; s.cslot = 0;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(s.pol));
  const Phase d = ph[p];
  stream_begin<C>(s, p);
  if (PREF) prime<C, false>(s, ring_w, lane);
  if (d.type == PH_W4) {
    prologue<C, true>(d, act, red);
    if (!PREF) prime<C, false>(s, ring_w, lane);
    gemv_tasks<C, false, true>(d, s, act, ring_w, lane);
  } else {
    prologue<C, false>(d, act, red);
    if (!PREF) prime<C, false>(s, ring_w, lane);
    gemv_tasks<C, false, false>(d, s, act, ring_w, lane);
  }
}

__global__ void k_core(const Phase* ph, int p) {
  const Phase d = ph[p];
  core_phase(d, blockIdx.x * blockDim.x + threadIdx.x, gridDim.x * blockDim.x);
}

// ------------------------------------------------------------------ grid barriers
// monotone-counter barrier: one release-add per CTA, acquire-spin until counter reaches target (wrap-safe)
template <bool SLEEP>
__device__ __forceinline__ void grid_barrier(unsigned* cnt, unsigned target) {
  __syncthreads();
  if (threadIdx.x == 0) {
    asm volatile("fence.acq_rel.gpu;\n" ::: "memory");
    asm volatile("red.release.gpu.global.add.u32 [%0], 1;\n" ::"l"(cnt) : "memory");
    unsigned v;
    while (true) {
      asm volatile("ld.acquire.gpu.global.u32 %0, [%1];\n" : "=r"(v) : "l"(cnt) : "memory");
      if ((int)(v - target) >= 0) break;
      if (SLEEP) __nanosleep(32);
    }
  }
  __syncthreads();
}

// classic sense-reversing barrier (centralised counter + release flag); local sense kept by caller
__device__ __forceinline__ void grid_barrier_sense(unsigned* count, unsigned* flag, unsigned nblocks, unsigned& local) {
  __syncthreads();
  if (threadIdx.x == 0) {
    local ^= 1u;
    asm volatile("fence.acq_rel.gpu;\n" ::: "memory");
    unsigned a;
    asm volatile("atom.add.acq_rel.gpu.global.u32 %0, [%1], 1;\n" : "=r"(a) : "l"(count), "r"(1u) : "memory");
    if (a == nblocks - 1) {
      *(volatile unsigned*)count = 0u;
      asm volatile("fence.acq_rel.gpu;\n" ::: "memory");
      asm volatile("st.release.gpu.global.u32 [%0], %1;\n" ::"l"(flag), "r"(local) : "memory");
    } else {
      unsigned v;
      do {
        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];\n" : "=r"(v) : "l"(flag) : "memory");
      } while (v != local);
    }
  }
  __syncthreads();
}

// ------------------------------------------------------------------ persistent kernel (V3: CROSS=false, V4: CROSS=true)
template <class C, bool CROSS>
__global__ void __launch_bounds__(C::NT, C::MINB)
    k_persist(const Phase* ph, int nph, unsigned* cnt, unsigned epoch, int ring_off, int first_gemv, int abl,
               unsigned long long* stamps) {
  // abl (ablation only, results invalid when != 0): bit0 skip grid barrier, bit1 skip activation prologue, bit2 skip core phases
  extern __shared__ __align__(128) char smem_raw[];
  __shared__ float red[C::NW];
  uint4* act = (uint4*)smem_raw;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  char* ring_w = smem_raw + ring_off + warp * (C::D * C::STG);
  Stream s;
  s.ph = ph; s.nph = nph; s.gw = blockIdx.x * C::NW + warp; s.W = gridDim.x * C::NW;
  s.pslot = 0; s.cslot = 0; s.p = -1;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(s.pol));
  if (CROSS) {
    stream_seek<C>(s, first_gemv);
    prime<C, true>(s, ring_w, lane);
  }
  unsigned target = epoch;
  const int gtid = blockIdx.x * C::NT + threadIdx.x, nthr = gridDim.x * C::NT;
  for (int p = 0; p < nph; p++) {
    const Phase d = ph[p];
    unsigned long long* st4 = stamps ? stamps + ((size_t)p * gridDim.x + blockIdx.x) * 4 : nullptr;
    if (st4 && threadIdx.x == 0) { st4[0] = gtimer(); st4[1] = st4[0]; }
    if (d.type == PH_W4 || d.type == PH_BF16) {
      if (!CROSS) {
        stream_begin<C>(s, p);
        prime<C, false>(s, ring_w, lane);
      }
      if (d.type == PH_W4) {
        if (!(abl & 2)) prologue<C, true>(d, act, red);
        if (st4 && threadIdx.x == 0) st4[1] = gtimer();
        gemv_tasks<C, CROSS, true>(d, s, act, ring_w, lane);
      } else {
        if (!(abl & 2)) prologue<C, false>(d, act, red);
        if (st4 && threadIdx.x == 0) st4[1] = gtimer();
        gemv_tasks<C, CROSS, false>(d, s, act, ring_w, lane);
      }
    } else if (!(abl & 4)) {
      core_phase(d, gtid, nthr);
    }
    if (st4) { __syncthreads(); if (threadIdx.x == 0) st4[2] = gtimer(); }
    if (p + 1 < nph && !(abl & 1)) {
      target += gridDim.x;
      grid_barrier<false>(cnt, target);
    }
    if (st4 && threadIdx.x == 0) st4[3] = gtimer();
  }
}

// ------------------------------------------------------------------ V0: read-only sum over all buffers
struct Span { const uint4* p; size_t n; };  // n = number of uint4
__global__ void k_sumread(const Span* spans, int nspans, uint32_t* out) {
  const size_t gt = (size_t)blockIdx.x * blockDim.x + threadIdx.x, nt = (size_t)gridDim.x * blockDim.x;
  uint32_t acc = 0;
  for (int b = 0; b < nspans; b++) {
    const uint4* p = spans[b].p;
    const size_t n = spans[b].n;
    size_t i = gt;
    for (; i + 3 * nt < n; i += 4 * nt) {
      uint4 a = __ldcs(p + i), b1 = __ldcs(p + i + nt), c = __ldcs(p + i + 2 * nt), e = __ldcs(p + i + 3 * nt);
      acc ^= a.x ^ a.y ^ a.z ^ a.w ^ b1.x ^ b1.y ^ b1.z ^ b1.w ^ c.x ^ c.y ^ c.z ^ c.w ^ e.x ^ e.y ^ e.z ^ e.w;
    }
    for (; i < n; i += nt) { uint4 a = __ldcs(p + i); acc ^= a.x ^ a.y ^ a.z ^ a.w; }
  }
  if (acc == 0x9e3779b9u) out[0] = acc;  // keep the loads alive without a store per thread
}
// one read kernel per phase matrix (independent, no activation dependency): isolates per-kernel overhead
__global__ void k_readone(const uint4* p, size_t n, const uint4* p2, size_t n2, uint32_t* out) {
  const size_t gt = (size_t)blockIdx.x * blockDim.x + threadIdx.x, nt = (size_t)gridDim.x * blockDim.x;
  uint32_t acc = 0;
  for (int pass = 0; pass < 2; pass++) {
    const uint4* q = pass ? p2 : p;
    const size_t m = pass ? n2 : n;
    size_t i = gt;
    for (; i + 3 * nt < m; i += 4 * nt) {
      uint4 a = __ldcs(q + i), b1 = __ldcs(q + i + nt), c = __ldcs(q + i + 2 * nt), e = __ldcs(q + i + 3 * nt);
      acc ^= a.x ^ a.y ^ a.z ^ a.w ^ b1.x ^ b1.y ^ b1.z ^ b1.w ^ c.x ^ c.y ^ c.z ^ c.w ^ e.x ^ e.y ^ e.z ^ e.w;
    }
    for (; i < m; i += nt) { uint4 a = __ldcs(q + i); acc ^= a.x ^ a.y ^ a.z ^ a.w; }
  }
  if (acc == 0x9e3779b9u) out[0] = acc;
}

// ------------------------------------------------------------------ naive reference (simple fp32, one warp per row)
__global__ void k_ref_prologue(const Phase d, float* actf) {
  // actf[k] = float(bf16(act_k)), unscaled; single block of 1024 threads
  __shared__ float red[32];
  const int tid = threadIdx.x;
  if (d.pro == PRO_RMS) {
    const float* x = (const float*)d.in;
    float ss = 0.f;
    for (int k = tid; k < d.K; k += blockDim.x) ss += x[k] * x[k];
    ss = warp_sum(ss);
    if ((tid & 31) == 0) red[tid >> 5] = ss;
    __syncthreads();
    float tot = 0.f;
    for (int i = 0; i < (int)(blockDim.x >> 5); i++) tot += red[i];
    const float inv = rsqrtf(tot * (1.0f / HID) + 1e-6f);
    for (int k = tid; k < d.K; k += blockDim.x) actf[k] = __uint_as_float(f2bf(x[k] * inv) << 16);
  } else {
    const uint16_t* v = (const uint16_t*)d.in;
    for (int k = tid; k < d.K; k += blockDim.x) actf[k] = __uint_as_float((uint32_t)v[k] << 16);
  }
}
__global__ void k_ref_gemv(const Phase d, const float* actf, int rev) {
  const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (row >= d.N) return;
  float acc = 0.f;
  if (d.type == PH_W4) {
    const uint32_t* w = (const uint32_t*)d.w + (size_t)row * (d.K / 8);
    const uint16_t* sc = (const uint16_t*)d.s + (size_t)row * (d.K / 128);
    for (int jj = lane; jj < d.K / 8; jj += 32) {
      const int j = rev ? (d.K / 8 - 1 - jj) : jj;
      const uint32_t u = w[j];
      const float scale = __uint_as_float((uint32_t)sc[j / 16] << 16);
      for (int i = 0; i < 8; i++) acc = fmaf((float)((int)((u >> (4 * i)) & 15u) - 8) * scale, actf[8 * j + i], acc);
    }
  } else {
    const uint16_t* w = (const uint16_t*)d.w + (size_t)row * d.K;
    for (int kk = lane; kk < d.K; kk += 32) { const int k = rev ? (d.K - 1 - kk) : kk; acc = fmaf(__uint_as_float((uint32_t)w[k] << 16), actf[k], acc); }
  }
  acc = warp_sum(acc);
  if (lane == 0) {
    if (d.epi == EPI_STORE) ((float*)d.out)[row] = acc;
    else if (d.epi == EPI_RESID) ((float*)d.out)[row] += acc;
  }
}
// gateup reference: raw g/u rows go to a scratch fp32 buffer, then fused here
__global__ void k_ref_gemv_raw(const Phase d, const float* actf, float* raw, int rev) {
  const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (row >= d.N) return;
  float acc = 0.f;
  const uint32_t* w = (const uint32_t*)d.w + (size_t)row * (d.K / 8);
  const uint16_t* sc = (const uint16_t*)d.s + (size_t)row * (d.K / 128);
  for (int jj = lane; jj < d.K / 8; jj += 32) {
      const int j = rev ? (d.K / 8 - 1 - jj) : jj;
    const uint32_t u = w[j];
    const float scale = __uint_as_float((uint32_t)sc[j / 16] << 16);
    for (int i = 0; i < 8; i++) acc = fmaf((float)((int)((u >> (4 * i)) & 15u) - 8) * scale, actf[8 * j + i], acc);
  }
  acc = warp_sum(acc);
  if (lane == 0) raw[row] = acc;
}
__global__ void k_ref_gateup(const Phase d, const float* raw) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < d.N / 2) ((uint16_t*)d.out)[j] = (uint16_t)f2bf(silu_f(raw[2 * j]) * raw[2 * j + 1]);
}
__global__ void k_ref_core(const Phase d) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < d.N) core_elem(d, i);
}

// ------------------------------------------------------------------ data generation
__device__ __forceinline__ uint32_t hash32(uint64_t x) {
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return (uint32_t)((x ^ (x >> 31)) >> 16) ^ (uint32_t)(x >> 40) * 2654435761u;
}
__global__ void k_fill_u32(uint32_t* p, size_t n, uint64_t seed) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    p[i] = hash32(i * 0x2545F4914F6CDD1Dull + seed);
}
// W4 words: random nibbles with 0 remapped to 8 => values (n-8) in [-7,7], exactly zero mean (plain uniform has mean -0.5,
// which makes the residual stream blow up through the common-mode feedback over 64 layers)
__global__ void k_fill_w4(uint32_t* p, size_t n, uint64_t seed) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint32_t h = hash32(i * 0x2545F4914F6CDD1Dull + seed), h2 = hash32(i * 0x9E3779B97F4A7C15ull + seed + 99);
    uint32_t w = (h & 0xFFFFu) | (h2 << 16);
    uint32_t o = 0;
#pragma unroll
    for (int q = 0; q < 8; q++) { uint32_t v = (w >> (4 * q)) & 15u; o |= (v ? v : 8u) << (4 * q); }
    p[i] = o;
  }
}
// bf16 values: uniform(0.5,1.5)*s0 (scales)
__global__ void k_fill_scale(uint16_t* p, size_t n, uint64_t seed, float s0) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    float u = (hash32(i * 0x2545F4914F6CDD1Dull + seed) & 0xffffff) * (1.0f / 16777216.0f);
    p[i] = (uint16_t)f2bf((0.5f + u) * s0);
  }
}
// bf16 weights uniform(-1,1)*s0
__global__ void k_fill_bf16(uint16_t* p, size_t n, uint64_t seed, float s0) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    float u = (hash32(i * 0x2545F4914F6CDD1Dull + seed) & 0xffffff) * (1.0f / 8388608.0f) - 1.0f;
    p[i] = (uint16_t)f2bf(u * s0);
  }
}
__global__ void k_fill_f32(float* p, size_t n, uint64_t seed) {  // ~N(0,1)-ish (sum of 4 uniforms)
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint32_t h = hash32(i * 0x2545F4914F6CDD1Dull + seed);
    float s = 0.f;
    for (int q = 0; q < 4; q++) s += ((h >> (8 * q)) & 255) * (1.0f / 255.0f) - 0.5f;
    p[i] = s * 1.7320508f;
  }
}
__global__ void k_fill_bf16_vec(uint16_t* p, size_t n, uint64_t seed) {  // random bf16 activations ~U(-2,2)
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    float u = (hash32(i * 0x2545F4914F6CDD1Dull + seed) & 0xffffff) * (1.0f / 4194304.0f) - 2.0f;
    p[i] = (uint16_t)f2bf(u);
  }
}

}  // namespace e1
