#pragma once
#include "qwen35_attn_common.cuh"

// The gated hd-256 attention of the Qwen3.5 hybrid (warpc/cuda/attn_gated.py): q_proj rows are [Hq][query HD | gate HD], q / k get the zero-centred per-head RMSNorm
// x * rsqrt(mean + eps) * (offset + w), partial rope over the first RD = 64 dims (rotate_half), the output is multiplied by sigmoid(gate). The prep reproduces the hand engine's
// writers (engine/src/attn.cu attn_prep_kernel / attn_prep_i4_kernel) bit for bit, KV bytes included: the norm and the rope stay in fp32 (no bf16 rounding between them), the
// rope angle is pos * theta^(-l/32) in fp64 (an fp32 angle carries ~pos * 6e-8 rad of error and flips bf16 roundings of the cached rows), the cache rows are rounded once.
// The split kernel is qwen35_attn_split.cuh; the merge (here) applies the gate like the hand's reduce:
// out = bf16(o / (1 + expf(-gate))), the sigmoid unrounded (the interpreter's sigmoid_mul rounds it to bf16 first).
namespace g256 {


// theta^(-l/32), theta = 1e7, l = 0..31 (correctly rounded doubles, the hand engine's table)
__device__ __constant__ double c_inv_freq[32] = {
    1.0, 0.6042963902381329, 0.3651741272548377, 0.220673406908459, 0.1333521432163324, 0.08058421877614819, 0.04869675251658631,
    0.029427271762092817, 0.01778279410038923, 0.010746078283213174, 0.006493816315762113, 0.003924189758484536, 0.0023713737056616554,
    0.0014330125702369627, 0.0008659643233600654, 0.0005232991146814947, 0.00031622776601683794, 0.00019109529749704405,
    0.00011547819846894582, 6.978305848598664e-05, 4.216965034285822e-05, 2.5482967479793464e-05, 1.539926526059492e-05,
    9.30572040929699e-06, 5.623413251903491e-06, 3.3982083289425593e-06, 2.053525026457146e-06, 1.2409377607517195e-06,
    7.498942093324558e-07, 4.531583637600818e-07, 2.7384196342643614e-07, 1.6548170999431814e-07};

// head h of token t: h < Hq the query half of the q_proj head, then K heads, then V heads. Lane holds d = lane + 32 i; after the norm and the rope x is the row as the cache stores
// it (before the rotation of int4_rot).
__device__ __forceinline__ void load_row(float (&x)[NI], int h, int t, int pos, int lane, const bf16* __restrict__ q_proj, const bf16* __restrict__ kk, const bf16* __restrict__ vv,
                                         const bf16* __restrict__ qn, const bf16* __restrict__ kn, int Hq, int Hkv, float eps, float offset) {
  const bf16* src;
  const bf16* nw = nullptr;
  if (h < Hq) {
    src = q_proj + (size_t)t * (2 * Hq * HD) + (size_t)h * 2 * HD;
    nw = qn;
  } else if (h < Hq + Hkv) {
    src = kk + (size_t)t * (Hkv * HD) + (size_t)(h - Hq) * HD;
    nw = kn;
  } else {
    src = vv + (size_t)t * (Hkv * HD) + (size_t)(h - Hq - Hkv) * HD;
  }
#pragma unroll
  for (int i = 0; i < NI; i++) x[i] = bf(src[lane + 32 * i]);
  if (nw) {
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < NI; i++) ss += x[i] * x[i];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    const float inv = rsqrtf(ss * (1.f / HD) + eps);
#pragma unroll
    for (int i = 0; i < NI; i++) x[i] = x[i] * inv * (offset + bf(nw[lane + 32 * i]));
    // rope on dims 0..63: lane l holds d = l (x[0]) and d = l + 32 (x[1]), a rotate_half pair. Cody-Waite reduction mod 2 pi in fp64, then fp32 sincos.
    const double ang = (double)pos * c_inv_freq[lane];
    const double kq = rint(ang * 0.15915494309189535);
    const double rr = fma(-kq, 2.4492935982947064e-16, fma(-kq, 6.283185307179586, ang));
    float sn, cs;
    sincosf((float)rr, &sn, &cs);
    const float a = x[0], b = x[1];
    x[0] = a * cs - b * sn;
    x[1] = b * cs + a * sn;
  }
}

// bf16 cache. One warp per (token, head): Hq query heads (bf16 to qout [T][Hq][HD]), Hkv K heads and Hkv V heads (bf16 into kv [2][Hkv][ctx][HD]). Four warps per block.
__global__ void __launch_bounds__(128) k_prep(int T, const bf16* __restrict__ q_proj, const bf16* __restrict__ kk, const bf16* __restrict__ vv, const bf16* __restrict__ qn,
                                              const bf16* __restrict__ kn, bf16* __restrict__ qout, bf16* __restrict__ kv, const int* __restrict__ pos0, int Hq, int Hkv, int ctx,
                                              float eps, float offset) {
  const int lane = threadIdx.x & 31;
  const int gw = blockIdx.x * 4 + (threadIdx.x >> 5);
  const int nh = Hq + 2 * Hkv;
  const int t = gw / nh, h = gw % nh;
  if (t >= T) return;
  const int pos = *pos0 + t;
  float x[NI];
  load_row(x, h, t, pos, lane, q_proj, kk, vv, qn, kn, Hq, Hkv, eps, offset);
  if (h < Hq) {
    bf16* dst = qout + ((size_t)t * Hq + h) * HD;
#pragma unroll
    for (int i = 0; i < NI; i++) dst[lane + 32 * i] = tb(x[i]);
  } else {
    const int isv = h >= Hq + Hkv;
    const int kvh = isv ? h - Hq - Hkv : h - Hq;
    bf16* dst = kv + (((size_t)isv * Hkv + kvh) * ctx + pos) * HD;
#pragma unroll
    for (int i = 0; i < NI; i++) dst[lane + 32 * i] = tb(x[i]);
  }
}

// int4_rot cache. Grid: T * (ceil(Hq / 8) + 2 Hkv) blocks of 8 warps. Per token the first ceil(Hq / 8) blocks hold 8 query heads each (one warp per head: rotated, fp16 to qout), then one
// block per (K | V, kv head) row: every warp rotates the row redundantly, warp w evaluates scale candidate w of the 8 (the serial part of r4::quant_store), the winners are
// exchanged through shared memory and warp i stores code row i. Same bytes as r4::quant_store and the hand's writers.
__global__ void __launch_bounds__(256) k_prep_i4(int T, const bf16* __restrict__ q_proj, const bf16* __restrict__ kk, const bf16* __restrict__ vv, const bf16* __restrict__ qn,
                                                 const bf16* __restrict__ kn, __half* __restrict__ qout, uint8_t* kvb, QKV L, const int* __restrict__ pos0, int Hq, int Hkv, int ctx,
                                                 float eps, float offset) {
  __shared__ float s_err[2][2][8];  // [chunk c][half h][candidate k]: squared error of the 16-lane group (2 c + h)
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int nq = (Hq + 7) / 8, nb = nq + 2 * Hkv;
  const int t = blockIdx.x / nb, b = blockIdx.x % nb;
  const int pos = *pos0 + t;
  float x[NI];
  if (b < nq) {
    const int h = b * 8 + w;
    if (h >= Hq) return;
    load_row(x, h, t, pos, lane, q_proj, kk, vv, qn, kn, Hq, Hkv, eps, offset);
    r4::wht<HD>(x, lane);
    __half* dst = qout + ((size_t)t * Hq + h) * HD;
#pragma unroll
    for (int i = 0; i < NI; i++) dst[lane + 32 * i] = __float2half_rn(x[i]);
    return;
  }
  const int h = Hq + (b - nq);
  const int isv = h >= Hq + Hkv;
  const int kvh = isv ? h - Hq - Hkv : h - Hq;
  load_row(x, h, t, pos, lane, q_proj, kk, vv, qn, kn, Hq, Hkv, eps, offset);
  r4::wht<HD>(x, lane);
  const size_t row = (size_t)kvh * ctx + pos;
  if (w == 1) {
    __half* ring = reinterpret_cast<__half*>(kvb + (isv ? L.vr : L.kr)) + ((size_t)kvh * L.ring + pos % L.ring) * HD;
#pragma unroll
    for (int i = 0; i < NI; i++) ring[lane + 32 * i] = __float2half_rn(x[i]);
  }
  if (w == 2 && pos < L.sink) {
    __half* sink = reinterpret_cast<__half*>(kvb + (isv ? L.vx : L.kx)) + ((size_t)kvh * L.sink + pos) * HD;
#pragma unroll
    for (int i = 0; i < NI; i++) sink[lane + 32 * i] = __float2half_rn(x[i]);
  }
  const int hb = (lane >> 3) & 1;
  int e0[2];
#pragma unroll
  for (int c = 0; c < 2; c++) {
    float a = fmaxf(fmaxf(fabsf(x[4 * c]), fabsf(x[4 * c + 1])), fmaxf(fabsf(x[4 * c + 2]), fabsf(x[4 * c + 3])));
    a = r4::gmax<HD>(a);
    e0[c] = __float2int_rn(r4::E_RES * log2f(fmaxf(a, 1e-30f) * (1.f / r4::level(15))) + (float)r4::E_BIAS);
    e0[c] = min(max(e0[c], 0), 255);
    const int ec = min(max(e0[c] + r4::cand_off(w), 0), 255);
    const float s = r4::scale_of(ec), inv = 1.f / s;
    float er = 0.f;
#pragma unroll
    for (int q = 0; q < 4; q++) {
      const float r = x[4 * c + q] - s * r4::level(r4::quant_code(x[4 * c + q], inv));
      er = fmaf(r, r, er);
    }
    er = r4::gsum<HD>(er);
    if ((lane & 7) == 0) s_err[c][hb][w] = er;
  }
  __syncthreads();
  int eb[2];
#pragma unroll
  for (int c = 0; c < 2; c++) {
    int best = 0;
    float be = s_err[c][hb][0];
#pragma unroll
    for (int k = 1; k < 8; k++) {
      const float ek = s_err[c][hb][k];
      if (ek < be) {
        be = ek;
        best = k;
      }
    }
    eb[c] = min(max(e0[c] + r4::cand_off(best), 0), 255);
  }
  // codes: warp w stores code row i = w (16 * i + lane / 2: two lanes share one byte)
  float xi = x[0];
#pragma unroll
  for (int i = 1; i < NI; i++)
    if (i == w) xi = x[i];
  const int code = r4::quant_code(xi, 1.f / r4::scale_of(eb[w >> 2]));
  const int hi_c = __shfl_xor_sync(0xffffffffu, code, 1);
  uint8_t* codes = kvb + (isv ? L.vc : L.kc) + row * (HD / 2);
  if (!(lane & 1)) codes[16 * w + (lane >> 1)] = (uint8_t)((hi_c << 4) | code);
  if (w == 0) {
    uint32_t* meta = reinterpret_cast<uint32_t*>(kvb + (isv ? L.vs : L.ks)) + row;
    const uint32_t mine = ((uint32_t)eb[0] << (8 * hb)) | ((uint32_t)eb[1] << (16 + 8 * hb));
    const uint32_t other = __shfl_sync(0xffffffffu, mine, 8);
    if (lane == 0) *meta = mine | other;
  }
}

// merge of the split partials with the output gate: ROT 0 bf16 cache, ROT 2 int4_rot (the output of the rotated V is un-rotated first); `win` is the exact ring of int4_rot (0: none).
// q_proj: the gate of head `head` is the second half of its [query | gate] row. One block of 64 threads per (head, row), thread t owns d = 4 t .. 4 t + 3. The arithmetic is the
// hand engine's reduce kernel's, in its order: the weights exp2(m_s - M) / L through one warp (smax <= 32), the splits summed in four interleaved groups (s = g, g + 4, ...) that
// are then added in order, the Walsh-Hadamard stages in ascending bit order (bits 0, 1 in registers, 2..6 by warp shuffles, 7 through shared memory).
template <int ROT>
__global__ void __launch_bounds__(64) k_merge(const float* part, const int* pos0, const bf16* __restrict__ q_proj, bf16* out, int Hq, int smax, int chmin, int win) {
  constexpr int PS = HD + 4;
  __shared__ float wgt[32];
  __shared__ float4 xch[64];
  const int head = blockIdx.x, t = blockIdx.y, tid = threadIdx.x;
  const int pe = *pos0 + gridDim.y - 1;
  int ns, chunk;
  const int ctxq = win ? max(pe + 1 - win, 0) : pe + 1;
  geom(ctxq, smax, chmin, ns, chunk);
  const float* base = part + ((size_t)t * Hq + head) * smax * PS;
  const bf16* gate = q_proj + (size_t)t * (2 * Hq * HD) + (size_t)head * 2 * HD + HD;
  // the partials of every split (and the gate) are requested first: they do not depend on the weights
  float4 v[32];
#pragma unroll
  for (int s = 0; s < 32; s++)
    if (s < ns) v[s] = reinterpret_cast<const float4*>(base + (size_t)s * PS)[tid];
  const uint2 gb = *reinterpret_cast<const uint2*>(gate + 4 * tid);
  if (tid < 32) {
    const float m = tid < ns ? base[tid * PS + HD] : -INFINITY, l = tid < ns ? base[tid * PS + HD + 1] : 0.f;
    float M = m;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, o));
    const float w = m == -INFINITY ? 0.f : exp2f(m - M);
    float L = 0.f;
    L += w * l;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) L += __shfl_xor_sync(0xffffffffu, L, o);
    const float invL = 1.f / L;
    wgt[tid] = w * invL;
  }
  __syncthreads();
  float4 grp[4];
#pragma unroll
  for (int g = 0; g < 4; g++) {
    grp[g] = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int s = g; s < 32; s += 4)
      if (s < ns) {
        const float w = wgt[s];
        grp[g].x += w * v[s].x;
        grp[g].y += w * v[s].y;
        grp[g].z += w * v[s].z;
        grp[g].w += w * v[s].w;
      }
  }
  float x[4] = {grp[0].x, grp[0].y, grp[0].z, grp[0].w};
#pragma unroll
  for (int g = 1; g < 4; g++) {
    x[0] += grp[g].x;
    x[1] += grp[g].y;
    x[2] += grp[g].z;
    x[3] += grp[g].w;
  }
  if (ROT) {
#pragma unroll
    for (int bit = 0; bit < 2; bit++)
#pragma unroll
      for (int j = 0; j < 4; j++)
        if (!(j & (1 << bit))) {  // (lo, hi) -> (lo + hi, lo - hi)
          const float u = x[j], v = x[j | (1 << bit)];
          x[j] = u + v;
          x[j | (1 << bit)] = u - v;
        }
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const bool up = (tid & o) != 0;
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const float y = __shfl_xor_sync(0xffffffffu, x[j], o);
        x[j] = up ? y - x[j] : x[j] + y;
      }
    }
    xch[tid] = make_float4(x[0], x[1], x[2], x[3]);
    __syncthreads();
    const float4 p = xch[tid ^ 32];
    const bool up = (tid & 32) != 0;
    x[0] = up ? p.x - x[0] : x[0] + p.x;
    x[1] = up ? p.y - x[1] : x[1] + p.y;
    x[2] = up ? p.z - x[2] : x[2] + p.z;
    x[3] = up ? p.w - x[3] : x[3] + p.w;
#pragma unroll
    for (int j = 0; j < 4; j++) x[j] = x[j] * 0.0625f;
  }
  const float g[4] = {__uint_as_float(gb.x << 16), __uint_as_float(gb.x & 0xffff0000u), __uint_as_float(gb.y << 16), __uint_as_float(gb.y & 0xffff0000u)};
  __nv_bfloat162 lo = __floats2bfloat162_rn(x[0] / (1.f + expf(-g[0])), x[1] / (1.f + expf(-g[1])));
  __nv_bfloat162 hi = __floats2bfloat162_rn(x[2] / (1.f + expf(-g[2])), x[3] / (1.f + expf(-g[3])));
  uint2 st;
  st.x = *reinterpret_cast<uint32_t*>(&lo);
  st.y = *reinterpret_cast<uint32_t*>(&hi);
  *reinterpret_cast<uint2*>(out + ((size_t)t * Hq + head) * HD + 4 * tid) = st;
}

}  // namespace g256
