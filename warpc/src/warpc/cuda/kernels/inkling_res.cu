#include "inkling_res.cuh"

// ---------------------------------------------------------------------------------------------------------------- message_gate
// n = [rms(x); rms(m)] unweighted, rounded to bf16; h = bf16(n . down^T) (rank 64); f = tanh(h . fw^T), g = tanh(h . gw^T) in fp32;
// out = (1 - f) * x + g * m in fp32, every product and sum a separate rounding (the reference's torch ops are not contracted).
// Two launches: k_ink_mg_down (grid (16, nseg * T): 4 rows of down per CTA, each CTA recomputes the norms of its row) and
// k_ink_mg_out (grid (ceil(H / 128), nseg * T): one output column per thread). With outn the last CTA of a row to finish (a
// counter in the workspace, left zero) normalises the row: rmsnorm(out) * nw -> bf16, so the norm that follows the gate in the
// graph costs no launch.
// Both kernels are one memory round trip deep: every global load of a thread (the weights first, they do not depend on anything)
// is issued before the first use. HC is H at compile time (2304: the model's) so the loads unroll; 0 is any H.
template <int HC>
__global__ void __launch_bounds__(128) k_ink_mg_down(const __grid_constant__ MgP p) {
  extern __shared__ float nv[];  // [2H]
  __shared__ float sm[8];
  constexpr int NK = HC / 128, NX = (HC / 4 + 127) / 128;  // uint4 of a down row per lane, float4 of a row per thread
  const int c = blockIdx.y / p.T, t = blockIdx.y % p.T, H = HC ? HC : p.H, tid = threadIdx.x;
  const int row = blockIdx.x * 4 + (tid >> 5), lane = tid & 31;
  const uint4* dr = reinterpret_cast<const uint4*>(p.down[c] + (size_t)row * 2 * H);
  const float* x = p.x[c] + (size_t)t * H;
  const float* m = p.m[c] + (size_t)t * H;
  float acc = 0.f;
  if constexpr (HC > 0) {
    uint4 wv[NK];
    float4 xv[NX], mv[NX];
#pragma unroll
    for (int k = 0; k < NK; k++) wv[k] = __ldg(dr + lane + 32 * k);
#pragma unroll
    for (int k = 0; k < NX; k++) {
      const int i = tid + 128 * k;
      xv[k] = mv[k] = make_float4(0.f, 0.f, 0.f, 0.f);
      if (i < H / 4) {
        xv[k] = reinterpret_cast<const float4*>(x)[i];
        mv[k] = reinterpret_cast<const float4*>(m)[i];
      }
    }
    float v2[2] = {0.f, 0.f};
#pragma unroll
    for (int k = 0; k < NX; k++) {
      v2[0] += xv[k].x * xv[k].x + xv[k].y * xv[k].y + xv[k].z * xv[k].z + xv[k].w * xv[k].w;
      v2[1] += mv[k].x * mv[k].x + mv[k].y * mv[k].y + mv[k].z * mv[k].z + mv[k].w * mv[k].w;
    }
    block_sum_vec<2, 4>(v2, sm);
    const float rx = rsqrtf(v2[0] / (float)H + p.eps), rm = rsqrtf(v2[1] / (float)H + p.eps);
#pragma unroll
    for (int k = 0; k < NX; k++) {
      const int i = tid + 128 * k;
      if (i < H / 4) {
        reinterpret_cast<float4*>(nv)[i] = make_float4(rb(xv[k].x * rx), rb(xv[k].y * rx), rb(xv[k].z * rx), rb(xv[k].w * rx));
        reinterpret_cast<float4*>(nv + H)[i] = make_float4(rb(mv[k].x * rm), rb(mv[k].y * rm), rb(mv[k].z * rm), rb(mv[k].w * rm));
      }
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < NK; k++) {
      float f[8];
      unpack8(wv[k], f);
#pragma unroll
      for (int e = 0; e < 8; e++) acc = fmaf(f[e], nv[(lane + 32 * k) * 8 + e], acc);
    }
  } else {
    float sx = 0.f, sm_ = 0.f;
    for (int i = tid; i < H; i += 128) {
      sx += x[i] * x[i];
      sm_ += m[i] * m[i];
    }
    float v2[2] = {sx, sm_};
    block_sum_vec<2, 4>(v2, sm);
    const float rx = rsqrtf(v2[0] / (float)H + p.eps), rm = rsqrtf(v2[1] / (float)H + p.eps);
    for (int i = tid; i < H; i += 128) {
      nv[i] = rb(x[i] * rx);
      nv[H + i] = rb(m[i] * rm);
    }
    __syncthreads();
#pragma unroll 6
    for (int k = lane; k < 2 * H / 8; k += 32) {
      float f[8];
      unpack8(__ldg(dr + k), f);
#pragma unroll
      for (int e = 0; e < 8; e++) acc = fmaf(f[e], nv[k * 8 + e], acc);
    }
  }
  acc = warp_sum(acc);
  if (lane == 0) p.hid[((size_t)c * p.T + t) * MG_R + row] = rb(acc);
}

template <int HC>
__global__ void __launch_bounds__(128) k_ink_mg_out(const __grid_constant__ MgP p) {
  __shared__ float hs[MG_R];
  __shared__ float sm[4];
  __shared__ float rn;
  __shared__ int last;
  constexpr int NX = (HC / 4 + 127) / 128;
  const int c = blockIdx.y / p.T, t = blockIdx.y % p.T, tid = threadIdx.x, H = HC ? HC : p.H;
  const int j = blockIdx.x * 128 + tid;
  const size_t o = (size_t)t * H + j;
  uint4 fv[MG_R / 8], gv[MG_R / 8];
  float xj = 0.f, mj = 0.f;
  if (j < H) {
    const uint4* fr = reinterpret_cast<const uint4*>(p.fw[c] + (size_t)j * MG_R);
    const uint4* gr = reinterpret_cast<const uint4*>(p.gw[c] + (size_t)j * MG_R);
#pragma unroll
    for (int k = 0; k < MG_R / 8; k++) {
      fv[k] = __ldg(fr + k);
      gv[k] = __ldg(gr + k);
    }
    xj = p.x[c][o];
    mj = p.m[c][o];
  }
  if (tid < MG_R) hs[tid] = p.hid[((size_t)c * p.T + t) * MG_R + tid];
  __syncthreads();
  float val = 0.f;
  if (j < H) {
    float a = 0.f, b = 0.f;
#pragma unroll
    for (int k = 0; k < MG_R / 8; k++) {
      float f[8], g[8];
      unpack8(fv[k], f);
      unpack8(gv[k], g);
#pragma unroll
      for (int e = 0; e < 8; e++) {
        a = fmaf(f[e], hs[k * 8 + e], a);
        b = fmaf(g[e], hs[k * 8 + e], b);
      }
    }
    val = __fadd_rn(__fmul_rn(1.f - tanhf(a), xj), __fmul_rn(tanhf(b), mj));
    p.y[c][o] = val;
  }
  bf16* on = p.outn[c];
  if (!on) return;
  // the norm of the row: per-CTA sums of squares in a fixed order, finished by the last CTA to arrive
  float s = val * val;
  block_sum_vec<1, 4>(&s, sm);
  const int item = c * p.T + t;
  if (tid == 0) p.part[(size_t)item * MG_MAXNB + blockIdx.x] = s;
  __threadfence();
  __syncthreads();
  if (tid == 0) {
    const unsigned ticket = atomicAdd(p.cnt + item, 1u);
    last = ticket == gridDim.x - 1;
    if (last) p.cnt[item] = 0;
  }
  __syncthreads();
  if (!last) return;
  __threadfence();
  const float* y = p.y[c] + (size_t)t * H;
  float4 yv[NX ? NX : 1];
  uint2 wv[NX ? NX : 1];
  if constexpr (HC > 0) {
#pragma unroll
    for (int k = 0; k < NX; k++) {
      const int i = tid + 128 * k;
      if (i < H / 4) {
        yv[k] = __ldcg(reinterpret_cast<const float4*>(y) + i);
        wv[k] = __ldg(reinterpret_cast<const uint2*>(p.nw[c]) + i);
      }
    }
  }
  if (tid < 32) {  // the row's sum of squares: the CTAs' partials in lane order, then a butterfly (fixed, so deterministic)
    const int nb = gridDim.x;
    float v = (tid < nb ? __ldcg(p.part + (size_t)item * MG_MAXNB + tid) : 0.f) + (tid + 32 < nb ? __ldcg(p.part + (size_t)item * MG_MAXNB + tid + 32) : 0.f);
    v = warp_sum(v);
    if (tid == 0) rn = rsqrtf(v / (float)H + p.eps_n);
  }
  __syncthreads();
  const float r = rn;
  if constexpr (HC > 0) {
#pragma unroll
    for (int k = 0; k < NX; k++) {
      const int i = tid + 128 * k;
      if (i < H / 4) {
        const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(&wv[k]);
        const float2 w01 = __bfloat1622float2(w2[0]), w23 = __bfloat1622float2(w2[1]);
        uint2 u;
        __nv_bfloat162* o2 = reinterpret_cast<__nv_bfloat162*>(&u);
        o2[0] = __floats2bfloat162_rn(__fmul_rn(__fmul_rn(yv[k].x, r), w01.x), __fmul_rn(__fmul_rn(yv[k].y, r), w01.y));
        o2[1] = __floats2bfloat162_rn(__fmul_rn(__fmul_rn(yv[k].z, r), w23.x), __fmul_rn(__fmul_rn(yv[k].w, r), w23.y));
        reinterpret_cast<uint2*>(on + (size_t)t * H)[i] = u;
      }
    }
  } else {
    for (int i = tid; i < H; i += 128) on[(size_t)t * H + i] = tb(__fmul_rn(__fmul_rn(__ldcg(y + i), r), bf(p.nw[c][i])));
  }
}

// ---------------------------------------------------------------------------------------------------------------- short convolution
// y[t][c] = bf16((sum_j w[c][j] u[t + j][c] + x[t][c]) / div) over u = [state (the 3 inputs before the window, oldest first) | x]
// in the reference's order: the four products and their left-to-right sum in fp32 (bf16 x bf16 products are exact), the residual,
// then the division, which torch runs on the GPU as a multiplication by the fp32 reciprocal of the divisor. The state then
// holds the last 3 rows of u (bf16, in place: a thread owns its channel's row, reads it before it writes it).
// res != NULL: out = res + f32(y) (f32 add); out_f32: out is f32 (f32(y) without res, the sum with it) else bf16.
__global__ void __launch_bounds__(128) k_ink_conv_res(const bf16* x, const float* res, const bf16* w, bf16* st, void* out, int out_f32, int T, int C, float inv_div) {
  const int c = blockIdx.x * 128 + threadIdx.x;
  if (c >= C) return;
  float s0 = bf(st[(size_t)c * 3]), s1 = bf(st[(size_t)c * 3 + 1]), s2 = bf(st[(size_t)c * 3 + 2]);
  float wf[4];
  const uint2 wu = *reinterpret_cast<const uint2*>(w + (size_t)c * 4);
  const __nv_bfloat162* wp = reinterpret_cast<const __nv_bfloat162*>(&wu);
  const float2 w01 = __bfloat1622float2(wp[0]), w23 = __bfloat1622float2(wp[1]);
  wf[0] = w01.x, wf[1] = w01.y, wf[2] = w23.x, wf[3] = w23.y;
  for (int t = 0; t < T; t++) {
    const size_t i = (size_t)t * C + c;
    const float xv = bf(x[i]);
    float y = __fadd_rn(__fadd_rn(__fadd_rn(__fmul_rn(wf[0], s0), __fmul_rn(wf[1], s1)), __fmul_rn(wf[2], s2)), __fmul_rn(wf[3], xv));
    y = rb(__fmul_rn(__fadd_rn(y, xv), inv_div));
    if (res) y = __fadd_rn(res[i], y);
    if (out_f32)
      reinterpret_cast<float*>(out)[i] = y;
    else
      reinterpret_cast<bf16*>(out)[i] = tb(y);
    s0 = s1, s1 = s2, s2 = xv;
  }
  st[(size_t)c * 3] = tb(s0), st[(size_t)c * 3 + 1] = tb(s1), st[(size_t)c * 3 + 2] = tb(s2);
}

// ---------------------------------------------------------------------------------------------------------------- entry points
extern "C" {

// rows[0..nrows): [T][H] bf16 or f32 (bit n of f32mask); nseg queries: q[j] fp32 [H]; outf[j] f32 [T][H] or NULL; outn[j] bf16 [T][H]
// or NULL with nw[j] bf16 [H] (outf, outn, nw themselves may be NULL: no outputs of that kind). nseg <= 48.
int warpc_ink_attn_res(int nrows, const void* const* rows, unsigned f32mask, int nseg, const void* const* q, void* const* outf, void* const* outn, const void* const* nw, int T, int H,
                       float eps, float eps_n, cudaStream_t st) {
  if (nrows < 1 || nrows > AR_MAXROWS || nseg < 1 || nseg > AR_MAXSEG || H < 1 || H > 256 * AR_KE || T < 1 || T > STEP_ROWS) return cudaErrorInvalidValue;
  const int nq = nseg < AR_MAXQ ? nseg : AR_MAXQ, G = (nseg + nq - 1) / nq;
  ArP p;
  for (int n = 0; n < AR_MAXROWS; n++) p.rows[n] = n < nrows ? rows[n] : nullptr;
  for (int j = 0; j < AR_MAXSEG; j++) {
    const bool live = j < nseg;
    p.q[j] = reinterpret_cast<const float*>(live ? q[j] : q[0]);
    p.outf[j] = live && outf ? reinterpret_cast<float*>(outf[j]) : nullptr;
    p.outn[j] = live && outn ? reinterpret_cast<bf16*>(outn[j]) : nullptr;
    p.nw[j] = live && nw ? reinterpret_cast<const bf16*>(nw[j]) : nullptr;
  }
  p.f32mask = f32mask;
  p.H = H;
  p.T = T;
  p.eps = eps;
  p.eps_n = eps_n;
  switch (nrows) {
    case 1: return ink_ar_run<1>(p, nq, G, T, st);
    case 2: return ink_ar_run<2>(p, nq, G, T, st);
    case 3: return ink_ar_run<3>(p, nq, G, T, st);
    case 4: return ink_ar_run<4>(p, nq, G, T, st);
    case 5: return ink_ar_run<5>(p, nq, G, T, st);
    default: return ink_ar_run<6>(p, nq, G, T, st);
  }
}

// workspace of warpc_ink_msg_gate: counters [4][8] u32, hid [4][8][64] f32, part [4][8][MG_MAXNB] f32, y [4][8][H] f32. Zeroed once; every launch leaves the counters zero.
size_t warpc_ink_mg_ws_bytes(int H) {
  return sizeof(unsigned) * MG_MAXSEG * STEP_ROWS + sizeof(float) * (MG_MAXSEG * STEP_ROWS * (MG_R + MG_MAXNB) + (size_t)MG_MAXSEG * STEP_ROWS * H);
}

// nseg gates: x[c], m[c] f32 [T][H]; down[c] bf16 [64][2H], fw[c], gw[c] bf16 [H][64]; outf[c] f32 [T][H] or NULL; outn[c] bf16 [T][H] or NULL with nw[c]
// bf16 [H]; ws: warpc_ink_mg_ws_bytes(H) bytes, device memory shared by nothing else that runs concurrently.
int warpc_ink_msg_gate(int nseg, const void* const* x, const void* const* m, const void* const* down, const void* const* fw, const void* const* gw, void* const* outf,
                       void* const* outn, const void* const* nw, void* ws, int T, int H, float eps, float eps_n, cudaStream_t st) {
  if (nseg < 1 || nseg > MG_MAXSEG || H % 8 || H > 128 * MG_MAXNB || T < 1 || T > STEP_ROWS) return cudaErrorInvalidValue;
  MgP p;
  unsigned* cnt = reinterpret_cast<unsigned*>(ws);
  float* hid = reinterpret_cast<float*>(cnt + MG_MAXSEG * STEP_ROWS);
  float* part = hid + MG_MAXSEG * STEP_ROWS * MG_R;
  float* y = part + MG_MAXSEG * STEP_ROWS * MG_MAXNB;
  for (int c = 0; c < MG_MAXSEG; c++) {
    const int s = c < nseg ? c : 0;
    p.x[c] = reinterpret_cast<const float*>(x[s]);
    p.m[c] = reinterpret_cast<const float*>(m[s]);
    p.down[c] = reinterpret_cast<const bf16*>(down[s]);
    p.fw[c] = reinterpret_cast<const bf16*>(fw[s]);
    p.gw[c] = reinterpret_cast<const bf16*>(gw[s]);
    p.outf[c] = c < nseg && outf ? reinterpret_cast<float*>(outf[c]) : nullptr;
    p.outn[c] = c < nseg && outn ? reinterpret_cast<bf16*>(outn[c]) : nullptr;
    p.nw[c] = c < nseg && nw ? reinterpret_cast<const bf16*>(nw[c]) : nullptr;
    p.y[c] = p.outf[c] ? p.outf[c] : y + (size_t)c * STEP_ROWS * H;
    if (c < nseg && !p.outf[c] && !p.outn[c]) return cudaErrorInvalidValue;
  }
  p.hid = hid;
  p.part = part;
  p.cnt = cnt;
  p.nseg = nseg;
  p.T = T;
  p.H = H;
  p.eps = eps;
  p.eps_n = eps_n;
  const dim3 gd(MG_R / 4, nseg * T), go((H + 127) / 128, nseg * T);
  if (H == 2304) {
    k_ink_mg_down<2304><<<gd, 128, 2 * H * sizeof(float), st>>>(p);
    k_ink_mg_out<2304><<<go, 128, 0, st>>>(p);
  } else {
    k_ink_mg_down<0><<<gd, 128, 2 * H * sizeof(float), st>>>(p);
    k_ink_mg_out<0><<<go, 128, 0, st>>>(p);
  }
  return cudaGetLastError();
}

// x bf16 [T][C], res f32 [T][C] or NULL, w bf16 [C][4] (tap 3 = the current row), state bf16 [C][3] updated in place, out bf16 or f32 [T][C] (see k_ink_conv_res)
int warpc_ink_conv_res(const void* x, const void* res, const void* w, void* state, void* out, int out_f32, int T, int C, float div, cudaStream_t st) {
  if (T < 1 || T > STEP_ROWS || C < 1 || !(div > 0.f)) return cudaErrorInvalidValue;
  k_ink_conv_res<<<(C + 127) / 128, 128, 0, st>>>(reinterpret_cast<const bf16*>(x), reinterpret_cast<const float*>(res), reinterpret_cast<const bf16*>(w), reinterpret_cast<bf16*>(state), out,
                                                   out_f32, T, C, 1.f / div);
  return cudaGetLastError();
}

}  // extern "C"
