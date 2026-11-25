// The persistent MoE kernel k_moe<T, CONV> of Inkling-2-Mini (sm_86): top-8 sigmoid router (bf16 weights, fp32 logits),
// INT4 g128 routed experts (gate|up with the softcapped SwiGLU fused, down with the weighted fp32 accumulation), rows
// 1..T where different rows may route to different experts. Included by inkling_moe_inst.cu only (one object per
// (T, CONV)). Same conventions as the dense kernels: raw device pointers and a stream, no allocation, no host
// synchronisation, graph-capturable.
//
// Math (the reference interpreter's moe_route / moe_experts, i.e. transformers' Inkling2TopKRouter / Inkling2Experts):
//   logits[t][e] = x_f32[t] . gate_w[e]  (bf16 weights widened, fp32 accumulation)        scores = sigmoid(logits)
//   idx[t] = top-8 of (scores + bias) by descending key, the LOWEST expert id winning ties;  w = scores[idx];
//   w = w / (sum(w) + 1e-20) * route_scale, the sum a left fold in idx order
//   per (token t, selected expert e):  g, u = x . Wg[e]^T, x . Wu[e]^T   (fp32 accumulation, both rounded to bf16)
//        a = bf16( tanh(g/4)*4 * sigmoid(g) * tanh(u/25)*25 )                    (fp32 math)
//        y = bf16( Wd[e] . a )                                                    (fp32 accumulation)
//   out[t] = bf16( sum over the selected experts in ASCENDING EXPERT ID of fl(w * y) )   fp32 adds (the order of the
//   official index_add_; the product is rounded before the add, as the interpreter's `y.float() * w` is)
// The INT4 scale is applied once per 128-column group to the group's partial sum, so the products are (nibble - 8) * x
// exactly and only the fp32 sums round.
//
// Determinism: no floating-point atomics. The partial sums of one output are formed in a fixed order that depends only
// on the shapes (lane partials combined with a fixed butterfly, the 8 expert contributions of a row added in ascending
// expert id by exactly one warp), so equal inputs give bit-equal outputs run to run. Which warp computes an item is
// decided by an atomic work counter, but an item's result does not depend on that.
//
// One persistent kernel, blocks of 4 warps, grid = SMs x 4 blocks (128 registers each), ALL BLOCKS RESIDENT: blocks wait
// for each other, so the grid must fit the GPU (the launch clamps it to the occupancy limit; it deadlocks if another
// kernel keeps blocks from being scheduled, so nothing may run beside it). Phases, per block:
//   1. router: block b computes the fp32 logits of expert b (a quarter of the row per warp), all blocks convert x to
//      fp32 in the layout the GEMVs read; then arrive.
//   2. every block waits for all arrivals, reads the logits, and redundantly computes top-8, the weights and the table of
//      distinct experts ("slots", ascending id, each with the (token, k) pairs that chose it; T = 1: slot k = k-th best
//      expert) in shared memory, so no serial last-block stage sits between router and expert loads.
//   3. one work queue (atomic counter, items claimed in order): gate|up items (4 rows of one (expert, token) pair)
//      followed by down items (16 rows). Each warp keeps 1 KB unit of weights in flight in registers, and the loads of
//      an item flow into the next one's (the next item is claimed while the current one is being computed). A down item
//      starts when the gate|up items of its pair are done (counter per pair): every claimed item belongs to a running
//      warp and the queue is ordered, so this cannot deadlock, and the down weights of early pairs stream while the last
//      gate|up items finish (no phase barrier). The last warp to finish a 16-row chunk of one pair's down items (ticket)
//      adds the 8 contributions of every token in ascending expert id and, with CONV, applies the MLP short convolution.
//      Pairs that chose the same expert are consecutive in the queue and chunk-major, so the same weight rows are
//      requested by adjacent items at the same time and the second request is served by L2: the expert is read from DRAM
//      once.
// Counters live in the workspace, zero-initialised once and left zero by every launch (the last warp to exit resets them).
//
// ONE copy of the unrolled K loop serves gate|up and down items (lanes per row group a run-time value): half the
// instruction footprint of two copies, which is what keeps the kernel warm in the instruction caches in a full model.
#pragma once
#include "dense_common.cuh"
#include "inkling_moe.cuh"

namespace moe {

constexpr int POLL_NS = 20;  // back-off of the spin waits
constexpr int BPS = 4;       // resident blocks (128 threads) per SM = the register budget

__device__ __forceinline__ void poll_sleep() { __nanosleep(POLL_NS); }

struct Slot {
  int e;           // expert id
  int cnt;         // number of (token, k) pairs that selected it
  int pair[TOPK];  // pair = t * 8 + k, ascending t
};
struct Table {
  int nslots;
  int pad[3];
  Slot s[MAXPAIRS];
};

struct Ws {
  unsigned* ctr;  // [0] router arrivals, [1] item counter, [2] exits, [8 + chunk] down tickets, [256 + group] gate|up items done
  float* logits;  // [MAXT][E]
  float* xf;      // [MAXT][H]     x as fp32, xpos<8> layout, pre-scaled (xscale)
  float* abuf;    // [MAXPAIRS][I] activations a per (token, k) pair (bf16-rounded values as fp32), xpos<4> layout, pre-scaled
  bf16* ybuf;     // [MAXPAIRS][H] y per pair
};
__device__ __forceinline__ Ws ws_ptrs(void* base, int E) {
  const WsLayout L = ws_layout(E);
  char* b = reinterpret_cast<char*>(base);
  Ws w;
  w.ctr = reinterpret_cast<unsigned*>(b + L.ctr);
  w.logits = reinterpret_cast<float*>(b + L.logits);
  w.xf = reinterpret_cast<float*>(b + L.xf);
  w.abuf = reinterpret_cast<float*>(b + L.abuf);
  w.ybuf = reinterpret_cast<bf16*>(b + L.ybuf);
  return w;
}

// Position of column c inside an activation vector stored for a row group of LPR lanes: iteration block i = c / (32 LPR);
// within it the 16-byte piece ((word w * 2 + half h) * LPR + lane l) holds the 4 floats q that lane l multiplies with the
// (w, h) quarter of its 16 weight bytes, so the LPR lanes of a row group read 16 * LPR contiguous bytes in one load.
template <int LPR>
__device__ __forceinline__ int xpos(int c) {
  constexpr int B = 32 * LPR;
  const int i = c / B, m = c % B;
  const int l = m >> 5, w = (m >> 3) & 3, h = (m >> 2) & 1, q = m & 3;
  return i * B + ((((w * 2 + h) * LPR + l)) << 2) + q;
}
// 2^(-4 (c & 3)): scale applied to column c of an activation vector (the nibble position of column c within its weight
// word is c & 7, its shift 4 (c & 3))
__device__ __forceinline__ float xscale(int c) { return __uint_as_float((unsigned)(127 - 4 * (c & 3)) << 23); }

__device__ __forceinline__ unsigned ld_acquire(const unsigned* p) {
  unsigned v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

// ------------------------------------------------------------------------------------------------------ router
__device__ __forceinline__ unsigned ordered_key(float f) {
  const unsigned u = __float_as_uint(f);
  return u ^ ((unsigned)((int)u >> 31) | 0x80000000u);
}

struct SmemRoute {
  Table tab;
  unsigned key[MAXE];  // T = 1 path
  int sel_e[TOPK];
  float sel_s[TOPK];
  float sum;
  int ngrp;                 // groups = (slot, one token of the slot): the unit of the work queue
  int grp_slot[MAXPAIRS];
  int grp_tok0[MAXPAIRS];   // index of the group's token in the slot's pair list
  int slot_g0[MAXPAIRS];    // first group of a slot (its groups are consecutive)
  int slot_ng[MAXPAIRS];    // number of groups of a slot
  float w[MAXPAIRS];
  int idx[MAXPAIRS];
  unsigned char ord[MAXPAIRS];  // ord[t * 8 + j] = the k of token t whose expert has the j-th smallest id
  int mask[MAXE];
  int slot[MAXE];
  int wc[4];
  float part[4][MAXT];
};

// logits of expert e: warp = quarter of the row, all T tokens against it; the four partial sums are added in a fixed order
__device__ __forceinline__ void router_logits(SmemRoute& sm, const Ws& ws, const bf16* __restrict__ x, const bf16* __restrict__ gw, int e, int T, int E, int lane,
                                              int warp) {
  constexpr int Q = HID / 4 / 4;  // 4-element pieces per warp = 144
  const uint2* row = reinterpret_cast<const uint2*>(gw + (size_t)e * HID) + warp * Q;
  uint2 g[5];
  uint2 xv[5][MAXT];
#pragma unroll
  for (int i = 0; i < 5; i++)
    if (lane + 32 * i < Q) g[i] = __ldg(row + lane + 32 * i);
#pragma unroll
  for (int t = 0; t < MAXT; t++)
#pragma unroll
    for (int i = 0; i < 5; i++)
      if (t < T && lane + 32 * i < Q) xv[i][t] = __ldg(reinterpret_cast<const uint2*>(x + (size_t)t * HID) + warp * Q + lane + 32 * i);
#pragma unroll
  for (int t = 0; t < MAXT; t++) {
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < 5; i++)
      if (t < T && lane + 32 * i < Q) {
        const float2 ga = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&g[i].x));
        const float2 gb = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&g[i].y));
        const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xv[i][t].x));
        const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xv[i][t].y));
        acc = fmaf(ga.x, a.x, acc);
        acc = fmaf(ga.y, a.y, acc);
        acc = fmaf(gb.x, b.x, acc);
        acc = fmaf(gb.y, b.y, acc);
      }
    acc = warp_sum(acc);
    if (lane == 0) sm.part[warp][t] = acc;
  }
  __syncthreads();
  if (threadIdx.x < T) ws.logits[threadIdx.x * E + e] = (sm.part[0][threadIdx.x] + sm.part[1][threadIdx.x]) + (sm.part[2][threadIdx.x] + sm.part[3][threadIdx.x]);
  __syncthreads();
}

// T = 1: thread e owns expert e and ranks it against all 128 keys (key descending, lowest index first on ties); the 8 best
// write themselves to slot = rank. Cheaper than 8 rounds of warp reductions, and no table sort is needed: slot k = k-th
// best expert, one pair each. Ends with __syncthreads. write_out: also store idx / w to global.
__device__ __forceinline__ void topk_one(SmemRoute& sm, const Ws& ws, const float* __restrict__ bias, int* __restrict__ idx_out, float* __restrict__ w_out, bool write_out, int E,
                                         float route_scale) {
  const int e = threadIdx.x;
  unsigned key = 0u;
  float sc = 0.f;
  if (e < E) {
    const float l = __ldcg(ws.logits + e);
    sc = 1.f / (1.f + expf(-l));
    key = ordered_key(sc + __ldg(bias + e));
  }
  sm.key[e] = key;
  __syncthreads();
  int rank = 0;
#pragma unroll
  for (int j4 = 0; j4 < MAXE / 4; j4++) {
    const uint4 kk = reinterpret_cast<const uint4*>(sm.key)[j4];
    rank += (kk.x > key) | ((kk.x == key) & (4 * j4 + 0 < e));
    rank += (kk.y > key) | ((kk.y == key) & (4 * j4 + 1 < e));
    rank += (kk.z > key) | ((kk.z == key) & (4 * j4 + 2 < e));
    rank += (kk.w > key) | ((kk.w == key) & (4 * j4 + 3 < e));
  }
  if (e < E && rank < TOPK) {
    sm.sel_e[rank] = e;
    sm.sel_s[rank] = sc;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float sum = 0.f;
#pragma unroll
    for (int r = 0; r < TOPK; r++) sum += sm.sel_s[r];
    sm.sum = sum;
  }
  __syncthreads();
  if (threadIdx.x < TOPK) {
    const int r = threadIdx.x;
    const int er = sm.sel_e[r];
    const float wv = sm.sel_s[r] / (sm.sum + 1e-20f) * route_scale;
    int pos = 0;  // the position of expert er among the 8 in ascending id
#pragma unroll
    for (int j = 0; j < TOPK; j++) pos += sm.sel_e[j] < er;
    sm.ord[pos] = (unsigned char)r;
    sm.idx[r] = er;
    sm.w[r] = wv;
    sm.tab.s[r].e = er;
    sm.tab.s[r].cnt = 1;
    sm.tab.s[r].pair[0] = r;
    sm.grp_slot[r] = r;
    sm.grp_tok0[r] = 0;
    sm.slot_g0[r] = r;
    sm.slot_ng[r] = 1;
    if (write_out) {
      idx_out[r] = er;
      w_out[r] = wv;
    }
  }
  if (threadIdx.x == 0) {
    sm.tab.nslots = TOPK;
    sm.ngrp = TOPK;
  }
  __syncthreads();
}

// T > 1: top-8 per token (warp t, 8 rounds of warp reductions, each lane holds experts lane + 32 i, E <= 128), weights, the
// ascending-id order of each token's experts and the table of distinct experts in ascending id in shared memory; ends with
// __syncthreads.
template <int T>
__device__ __forceinline__ void topk_many(SmemRoute& sm, const Ws& ws, const float* __restrict__ bias, int* __restrict__ idx_out, float* __restrict__ w_out, bool write_out, int E,
                                          float route_scale, int lane, int warp) {
  for (int t = warp; t < T; t += 4) {
    float sc[4];
    unsigned key[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const int ee = lane + 32 * i;
      if (ee < E) {
        const float l = __ldcg(ws.logits + t * E + ee);
        sc[i] = 1.f / (1.f + expf(-l));
        key[i] = ordered_key(sc[i] + __ldg(bias + ee));
      } else {
        sc[i] = 0.f;
        key[i] = 0u;  // below every real key (the smallest, -inf, maps to 0x007fffff)
      }
    }
    float sel_s = 0.f;
    int sel_e = 0;
    float sum = 0.f;
#pragma unroll
    for (int r = 0; r < TOPK; r++) {
      const unsigned m = max(max(key[0], key[1]), max(key[2], key[3]));
      const unsigned gm = __reduce_max_sync(0xffffffffu, m);
      unsigned cand = 0xffffffffu;
#pragma unroll
      for (int i = 3; i >= 0; i--)
        if (key[i] == gm) cand = lane + 32 * i;
      const unsigned ce = __reduce_min_sync(0xffffffffu, cand);  // lowest expert index among the maxima
      const int ol = ce & 31, oi = ce >> 5;
      float v = sc[0];
      if (oi == 1) v = sc[1];
      if (oi == 2) v = sc[2];
      if (oi == 3) v = sc[3];
      v = __shfl_sync(0xffffffffu, v, ol);
      if (lane == ol) {
#pragma unroll
        for (int i = 0; i < 4; i++)
          if (i == oi) key[i] = 0u;
      }
      sum += v;
      if (lane == r) {
        sel_s = v;
        sel_e = (int)ce;
      }
    }
    if (lane < TOPK) {
      const float wv = sel_s / (sum + 1e-20f) * route_scale;
      sm.idx[t * TOPK + lane] = sel_e;
      sm.w[t * TOPK + lane] = wv;
      if (write_out) {
        idx_out[t * TOPK + lane] = sel_e;
        w_out[t * TOPK + lane] = wv;
      }
    }
  }
  for (int i = threadIdx.x; i < MAXE; i += blockDim.x) sm.mask[i] = 0;
  __syncthreads();
  const int NP = T * TOPK;
  if (threadIdx.x < NP) atomicOr(&sm.mask[sm.idx[threadIdx.x]], 1 << (threadIdx.x / TOPK));
  __syncthreads();
  // distinct experts in ascending id: rank by ballot over the experts (thread = expert id, E <= 128)
  const int ex = threadIdx.x;
  const int m = ex < E ? sm.mask[ex] : 0;
  const unsigned bal = __ballot_sync(0xffffffffu, m != 0);
  if (lane == 0) sm.wc[warp] = __popc(bal);
  __syncthreads();
  int off = 0;
  for (int i = 0; i < warp; i++) off += sm.wc[i];
  const int slot = off + __popc(bal & ((1u << lane) - 1u));
  if (ex < E && m != 0) {
    sm.slot[ex] = slot;
    sm.tab.s[slot].e = ex;
    sm.tab.s[slot].cnt = __popc(m);
  }
  if (threadIdx.x == 0) sm.tab.nslots = sm.wc[0] + sm.wc[1] + sm.wc[2] + sm.wc[3];
  __syncthreads();
  if (threadIdx.x < NP) {
    const int p = threadIdx.x, t = p / TOPK, ee = sm.idx[p];
    const int u = __popc(sm.mask[ee] & ((1 << t) - 1));
    sm.tab.s[sm.slot[ee]].pair[u] = p;
    int pos = 0;  // the position of expert ee among the 8 of its token in ascending id
#pragma unroll
    for (int j = 0; j < TOPK; j++) pos += sm.idx[t * TOPK + j] < ee;
    sm.ord[t * TOPK + pos] = (unsigned char)(p % TOPK);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    int g = 0;
    for (int sl = 0; sl < sm.tab.nslots; sl++) {
      sm.slot_g0[sl] = g;
      sm.slot_ng[sl] = sm.tab.s[sl].cnt;
      for (int t0 = 0; t0 < sm.tab.s[sl].cnt; t0++) {
        sm.grp_slot[g] = sl;
        sm.grp_tok0[g] = t0;
        g++;
      }
    }
    sm.ngrp = g;
  }
  __syncthreads();
}

template <int T>
__device__ __forceinline__ void topk_table(SmemRoute& sm, const Ws& ws, const float* __restrict__ bias, int* __restrict__ idx_out, float* __restrict__ w_out, bool write_out, int E,
                                           float route_scale, int lane, int warp) {
  if constexpr (T == 1) topk_one(sm, ws, bias, idx_out, w_out, write_out, E, route_scale);
  else topk_many<T>(sm, ws, bias, idx_out, w_out, write_out, E, route_scale, lane, warp);
}

// ------------------------------------------------------------------------------------------------------ INT4 GEMV core
// A row group of LPR lanes (LPR = 8: 4 groups per warp, LPR = 4: 8 per warp) reads R = 2 rows with the same column slice
// for both. Iteration (unit) i covers 32 * LPR columns of each row: lane l loads the 16 bytes (32 columns) j = LPR * i + l,
// i.e. the LPR lanes read 16 * LPR contiguous bytes of a row. NI = 9 units.
//
// Dequantisation in place: for nibble e (p = 4 (e & 3)) the word bits (w & (0xF << p)) | 0x4B000000 are the float
// 2^23 + nibble * 2^p exactly; subtracting 2^23 + 8 * 2^p leaves (nibble - 8) * 2^p exactly, and the activation is stored
// pre-multiplied by 2^-p (xscale, an exact power of two), so the product equals (nibble - 8) * x with one rounding in the
// FMA, as if the nibble had been shifted down first. One shift per word (for nibbles 4..7) instead of seven.
// (a & m) | 0x4B000000 as ONE lop3 (a & b | c = 0xEA); the plain C++ form compiles to two
__device__ __forceinline__ uint32_t and_or_magic(uint32_t a, uint32_t m) {
  uint32_t d;
  asm("lop3.b32 %0, %1, %2, 0x4B000000, 0xEA;" : "=r"(d) : "r"(a), "r"(m));
  return d;
}
__device__ __forceinline__ void dq8(uint32_t w, float* f, const uint32_t (&msk)[4]) {
  const uint32_t hi = w >> 16;
#pragma unroll
  for (int k = 0; k < 4; k++) {
    const float c = 8388608.f + 8.f * (float)(1 << (4 * k));
    f[k] = __uint_as_float(and_or_magic(w, msk[k])) - c;
    f[4 + k] = __uint_as_float(and_or_magic(hi, msk[k])) - c;
  }
}

// raw scale words of one row for this lane (loaded one item ahead): LPR = 8: words l and 8 of the 9-word row (18 bf16,
// 4-byte aligned); LPR = 4: bf16 l, l + 4, 8
template <int LPR>
__device__ __forceinline__ void load_scw(const bf16* srow, int l, uint32_t (&scw)[3]) {
  if (LPR == 8) {
    const uint32_t* p = reinterpret_cast<const uint32_t*>(srow);
    scw[0] = __ldg(p + l);
    scw[1] = __ldg(p + 8);
    scw[2] = 0;
  } else {
    const uint16_t* p = reinterpret_cast<const uint16_t*>(srow);
    scw[0] = __ldg(p + l);
    scw[1] = __ldg(p + l + 4);
    scw[2] = __ldg(p + 8);
  }
}
// the 9 scales (bf16 -> fp32) iteration i of this lane needs: the row group redistributes the words with shuffles
template <int LPR>
__device__ __forceinline__ void decode_scales(const uint32_t (&scw)[3], int l, float (&sc)[9]) {
#pragma unroll
  for (int i = 0; i < 9; i++) {
    if (LPR == 8) {  // iteration i needs word i, half l >> 2
      const uint32_t ww = i < 8 ? __shfl_sync(0xffffffffu, scw[0], i, 8) : scw[1];
      sc[i] = __uint_as_float((l & 4) ? (ww & 0xffff0000u) : (ww << 16));
    } else {  // iteration i needs bf16 i for all 4 lanes
      const uint32_t v = i < 8 ? __shfl_sync(0xffffffffu, i < 4 ? scw[0] : scw[1], i & 3, 4) : scw[2];
      sc[i] = __uint_as_float(v << 16);
    }
  }
}

constexpr int R = 2;   // weight rows per lane (gate + up of one row n, or two rows of down)
constexpr int PF = 1;  // units (of 9 per item; 16 bytes x 32 lanes x 2 rows each = 1 KB) of weights kept in flight per warp (measured 1 >= 3 >> 9)
constexpr int PD = 1;  // x prefetch distance (steps of 8 columns)
struct RowPtrs {
  const uint4* w[R];  // packed weights of each row, already offset by the lane
  const bf16* s[R];   // scale rows
  int stride;         // uint4 distance between consecutive units: LPR
};

struct NextItem {
  RowPtrs P;
  bool has;
};

// first PF units of an item
__device__ __forceinline__ void load_units(uint4 (&wv)[PF][R], const RowPtrs& P) {
#pragma unroll
  for (int i = 0; i < PF; i++)
#pragma unroll
    for (int r = 0; r < R; r++) wv[i][r] = __ldcs(P.w[r] + P.stride * i);
}

// acc[r] += sum over the K columns of (nibble - 8) * scale * x for the item with pointers cur whose first PF units sit in
// the ring wv (unit i in slot i % PF) and whose scales are decoded in sc. A unit slot is refilled right after use with
// the unit PF later of the stream of units: of this item, and for the last PF units of the NEXT item (nx, filled by prep()
// when unit 9 - PF is reached), so the loads of one item flow into the next and about PF units per warp are in flight at
// any time. xg is in xpos<LPR> layout; its loads run PD steps (one step = 8 columns of the lane's 32) ahead of their
// use, in registers. LPR (lanes per row group, 8 or 4) is a run-time value: ONE copy of the unrolled loop serves
// gate|up and down items.
template <class Prep>
__device__ __forceinline__ void gemv_core(uint4 (&wv)[PF][R], const float (&sc)[R][9], const RowPtrs& cur, NextItem& nx, Prep&& prep, const float* xg, int LPR, int l,
                                          float (&acc)[R], const uint32_t (&msk)[4]) {
  constexpr int NI = 9, NS = NI * 4;
  float4 xr[PD + 1][2];
#define LOAD_X(S)                                                                                         \
  {                                                                                                       \
    const float4* xp = reinterpret_cast<const float4*>(xg) + ((S) >> 2) * (8 * LPR) + (((S)&3) * 2) * LPR + l; \
    xr[(S) % (PD + 1)][0] = __ldg(xp);                                                                    \
    xr[(S) % (PD + 1)][1] = __ldg(xp + LPR);                                                              \
  }
#pragma unroll
  for (int st = 0; st < PD; st++) LOAD_X(st)
  float part[R];
#pragma unroll
  for (int st = 0; st < NS; st++) {
    if (st + PD < NS) LOAD_X(st + PD)
    const int i = st >> 2, w = st & 3;
    if (st == 4 * (NI - PF)) prep(nx);
    if (w == 0) {
#pragma unroll
      for (int r = 0; r < R; r++) part[r] = 0.f;
    }
    float f[R][8];
#pragma unroll
    for (int r = 0; r < R; r++) dq8(reinterpret_cast<const uint32_t*>(&wv[i % PF][r])[w], f[r], msk);
    const float4 xa = xr[st % (PD + 1)][0], xb = xr[st % (PD + 1)][1];
    const float xv[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
#pragma unroll
    for (int r = 0; r < R; r++)
#pragma unroll
      for (int e = 0; e < 8; e++) part[r] = fmaf(f[r][e], xv[e], part[r]);
    if (w == 3) {
#pragma unroll
      for (int r = 0; r < R; r++) acc[r] = fmaf(sc[r][i], part[r], acc[r]);
      if (i + PF < NI) {
#pragma unroll
        for (int r = 0; r < R; r++) wv[i % PF][r] = __ldcs(cur.w[r] + cur.stride * (i + PF));
      } else if (nx.has) {
#pragma unroll
        for (int r = 0; r < R; r++) wv[i % PF][r] = __ldcs(nx.P.w[r] + nx.P.stride * (i + PF - NI));
      }
    }
  }
#undef LOAD_X
}

// ------------------------------------------------------------------------------------------------------ the kernel
constexpr int GU_RPI = 4;   // gate|up rows per item: 4 row groups of 8 lanes, one row n each (R = 2: gate and up)
constexpr int DN_RPI = 16;  // down rows per item: 8 row groups of 4 lanes, two rows each
constexpr int CH_GU = INTER / GU_RPI;
constexpr int CH_DN = HID / DN_RPI;
static_assert(9 % PF == 0, "the ring of PF unit slots must repeat exactly over the 9 units of an item");

// sum over the 8 experts of token t at column n in ascending expert id: fl(w * y) is rounded before the add (the
// reference's `y.float() * w` then index_add_), no fma
__device__ __forceinline__ float ordered_sum(const SmemRoute& sm, const Ws& ws, int t, int n) {
  float o = 0.f;
#pragma unroll
  for (int j = 0; j < TOPK; j++) {
    const int k = sm.ord[t * TOPK + j];
    o = __fadd_rn(o, __fmul_rn(sm.w[t * TOPK + k], bf(__ldcg(ws.ybuf + (size_t)(t * TOPK + k) * HID + n))));
  }
  return o;
}

template <int T, int CONV>
__global__ void __launch_bounds__(128, BPS)
    k_moe(const bf16* __restrict__ x, const bf16* __restrict__ gw, const float* __restrict__ bias, const uint32_t* __restrict__ pg, const bf16* __restrict__ sg,
          const uint32_t* __restrict__ pu, const bf16* __restrict__ su, const uint32_t* __restrict__ pd, const bf16* __restrict__ sd, bf16* __restrict__ out,
          int* __restrict__ idx_out, float* __restrict__ w_out, void* wsbase, int E, float route_scale, Conv cv) {
  __shared__ SmemRoute sm;
  const Ws ws = ws_ptrs(wsbase, E);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int nw_tot = gridDim.x * 4;
  // nibble masks kept in registers (a LOP3 takes one immediate; the compiler would otherwise re-materialise a mask before
  // every use): E >> 20 is 0 but not a constant
  const uint32_t mz = (uint32_t)E >> 20;
  const uint32_t msk[4] = {0xFu | mz, 0xF0u | mz, 0xF00u | mz, 0xF000u | mz};

  // ---- 1. router logits + x -> fp32
  for (int e = blockIdx.x; e < E; e += gridDim.x) router_logits(sm, ws, x, gw, e, T, E, lane, warp);
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < T * HID; i += gridDim.x * blockDim.x) {
    const int t = i / HID, c = i - t * HID;
    ws.xf[t * HID + xpos<8>(c)] = bf(x[i]) * xscale(c);
  }
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) atomicAdd(&ws.ctr[0], 1u);

  // ---- the first queue item of every warp is claimed before the wait (it does not depend on the table)
  unsigned p0 = 0;
  if (lane == 0) p0 = atomicAdd(&ws.ctr[1], 1u);
  // ---- 2. wait for every block's logits, then top-8 + table per block
  if (threadIdx.x == 0)
    while (ld_acquire(&ws.ctr[0]) < gridDim.x) poll_sleep();
  __syncthreads();
  topk_table<T>(sm, ws, bias, idx_out, w_out, idx_out != nullptr && blockIdx.x == 0, E, route_scale, lane, warp);

  // ---- 3. the queue
  const int ngrp = sm.ngrp;
  const int gu_total = ngrp * CH_GU;
  const int total = gu_total + ngrp * CH_DN;
  int cur = (int)__shfl_sync(0xffffffffu, p0, 0);

  // item q of a kind -> (group, chunk). The groups of one slot run chunk-major (chunk 0 of every group of the slot, then
  // chunk 1, ...), so the items that read the same weight rows for different tokens are adjacent in the queue, run at the
  // same time and are served by one DRAM read (the second request hits L2)
  auto locate = [&](int q, int ch_per_grp, int& grp, int& chunk) {
    if (T == 1) {
      grp = q / ch_per_grp;
      chunk = q - grp * ch_per_grp;
      return;
    }
    const int slot = sm.grp_slot[q / ch_per_grp];  // q / ch_per_grp is a group of the slot's run
    const int g0 = sm.slot_g0[slot], ng = sm.slot_ng[slot];
    const int j = q - g0 * ch_per_grp;
    chunk = j / ng;
    grp = g0 + (j - chunk * ng);
  };
  // pointers of an item: kind (gate|up / down), slot, first row n0 of this lane's row group, chunk
  auto make = [&](int item, RowPtrs& P, int& grp, int& n0, int& chunk) {
    if (item < gu_total) {
      locate(item, CH_GU, grp, chunk);
      const int slot = sm.grp_slot[grp];
      const int l = lane & 7, rg = lane >> 3;
      n0 = chunk * GU_RPI + rg;
      const size_t row = (size_t)sm.tab.s[slot].e * INTER + n0;
      P.w[0] = reinterpret_cast<const uint4*>(pg + row * (HID / 8)) + l;
      P.w[1] = reinterpret_cast<const uint4*>(pu + row * (HID / 8)) + l;
      P.s[0] = sg + row * (HID / 128);
      P.s[1] = su + row * (HID / 128);
      P.stride = 8;
    } else {
      locate(item - gu_total, CH_DN, grp, chunk);
      const int slot = sm.grp_slot[grp];
      const int l = lane & 3, rg = lane >> 2;
      n0 = chunk * DN_RPI + rg * 2;
      const size_t row = (size_t)sm.tab.s[slot].e * HID + n0;
      P.w[0] = reinterpret_cast<const uint4*>(pd + row * (INTER / 8)) + l;
      P.w[1] = reinterpret_cast<const uint4*>(pd + (row + 1) * (INTER / 8)) + l;
      P.s[0] = sd + row * (INTER / 128);
      P.s[1] = sd + (row + 1) * (INTER / 128);
      P.stride = 4;
    }
  };
  uint4 wv[PF][R];
  uint32_t scw[R][3];
  auto load_scales_for = [&](const RowPtrs& P, bool gu) {
#pragma unroll
    for (int r = 0; r < R; r++) {
      if (gu) load_scw<8>(P.s[r], lane & 7, scw[r]);
      else load_scw<4>(P.s[r], lane & 3, scw[r]);
    }
  };
  if (cur < total) {
    RowPtrs P;
    int grp, n0, chunk;
    make(cur, P, grp, n0, chunk);
    load_scales_for(P, cur < gu_total);
    load_units(wv, P);
  }
  while (cur < total) {
    unsigned pn = 0;
    if (lane == 0) pn = atomicAdd(&ws.ctr[1], 1u);  // claim the next item now; its index is needed when unit 9 - PF is reached
    RowPtrs P;
    int grp, n0, chunk;
    make(cur, P, grp, n0, chunk);
    NextItem nx;
    int nxt = 0;
    auto prep = [&](NextItem& n) {
      nxt = (int)__shfl_sync(0xffffffffu, pn, 0);
      n.has = nxt < total;
      if (n.has) {
        int g2, n02, ch2;
        make(nxt, n.P, g2, n02, ch2);
        load_scales_for(n.P, nxt < gu_total);
      } else {
        n.P = P;
      }
    };
    const int slot = sm.grp_slot[grp], tok0 = sm.grp_tok0[grp];
    const bool gu = cur < gu_total;
    const int LPR = gu ? 8 : 4, l = lane & (LPR - 1);
    float sc[R][9];
    if (gu) {
#pragma unroll
      for (int r = 0; r < R; r++) decode_scales<8>(scw[r], l, sc[r]);
    } else {
#pragma unroll
      for (int r = 0; r < R; r++) decode_scales<4>(scw[r], l, sc[r]);
      if (lane == 0)
        while (ld_acquire(&ws.ctr[256 + grp]) < (unsigned)CH_GU) poll_sleep();  // all gate|up items of this group are done
      __syncwarp();
    }
    const int pr = sm.tab.s[slot].pair[tok0];
    const float* xg = gu ? ws.xf + (size_t)(pr / TOPK) * HID : ws.abuf + (size_t)pr * INTER;
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; r++) acc[r] = 0.f;
    gemv_core(wv, sc, P, nx, prep, xg, LPR, l, acc, msk);
#pragma unroll
    for (int r = 0; r < R; r++) {
      acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], 1);
      acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], 2);
      if (gu) acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], 4);
    }
    if (gu) {
      // ---------------------------------------------------------------- gate | up item: 4 rows n, gate and up
      if (l == 0) {
        const float g = rb(acc[0]), up = rb(acc[1]);
        const float sg_ = 1.f / (1.f + expf(-g));
        const float act = rb(tanhf(g * 0.25f) * 4.f * sg_ * (tanhf(up * (1.f / 25.f)) * 25.f));
        ws.abuf[(size_t)pr * INTER + xpos<4>(n0)] = act * xscale(n0);
      }
      __threadfence();
      __syncwarp();
      if (lane == 0) atomicAdd(&ws.ctr[256 + grp], 1u);
    } else {
      // ---------------------------------------------------------------- down item: 16 rows n of one slot
      if (l == 0) {
#pragma unroll
        for (int r = 0; r < R; r++) ws.ybuf[(size_t)pr * HID + n0 + r] = tb(acc[r]);
      }
      if (CONV && lane < MAXT + 3) {  // the conv inputs of this chunk's 16 columns (res rows, the taps, the state) are pulled into L2 while the chunk is still being finished
        const char* p;
        if (lane < MAXT) p = reinterpret_cast<const char*>(cv.res + (size_t)(lane < T ? lane : 0) * HID + chunk * DN_RPI);
        else if (lane == MAXT) p = reinterpret_cast<const char*>(cv.cw + (size_t)chunk * DN_RPI * CONV_TAPS);
        else p = reinterpret_cast<const char*>(cv.cst + (size_t)chunk * DN_RPI * (CONV_TAPS - 1)) + (lane - MAXT - 1) * (DN_RPI * (CONV_TAPS - 1) * 2 - 1);
        asm volatile("prefetch.global.L2 [%0];" ::"l"(p));
      }
      __threadfence();
      __syncwarp();
      unsigned old = 0;
      if (lane == 0) old = atomicAdd(&ws.ctr[8 + chunk], 1u);
      old = __shfl_sync(0xffffffffu, old, 0);
      if (old == (unsigned)ngrp - 1) {  // last group to finish this chunk: add the 8 contributions of every token in ascending expert id
        __threadfence();
        if (lane == 0) ws.ctr[8 + chunk] = 0;
        if (!CONV) {
          for (int i = lane; i < T * DN_RPI; i += 32) {
            const int t = i / DN_RPI, n = chunk * DN_RPI + i % DN_RPI;
            out[(size_t)t * HID + n] = tb(ordered_sum(sm, ws, t, n));
          }
        } else {
          // lane = (hf, c): channel n = chunk * 16 + c; the half hf sums the tokens t with t & 1 == hf and then every lane
          // holds all T outputs x_t (bf16 values) of its channel; y_t = sum_j w_j u[t + j] over u = [state (3) | x_0 .. x_{T-1}]
          constexpr int NJ = (T + 1) / 2;
          const int c = lane & 15, hf = lane >> 4, n = chunk * DN_RPI + c;
          float xo[NJ];
#pragma unroll
          for (int j = 0; j < NJ; j++) xo[j] = 0.f;
#pragma unroll
          for (int j = 0; j < NJ; j++)
            if (2 * j + hf < T) xo[j] = rb(ordered_sum(sm, ws, 2 * j + hf, n));
          float xin[T];
#pragma unroll
          for (int t = 0; t < T; t++) xin[t] = __shfl_sync(0xffffffffu, xo[t >> 1], ((t & 1) << 4) | c);
          const bf16* sp = cv.cst + (size_t)n * (CONV_TAPS - 1);
          const uint2 wraw = __ldg(reinterpret_cast<const uint2*>(cv.cw + (size_t)n * CONV_TAPS));
          const float2 wa = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&wraw.x));
          const float2 wb = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&wraw.y));
          float u[T + CONV_TAPS - 1];
#pragma unroll
          for (int j = 0; j < CONV_TAPS - 1; j++) u[j] = bf(sp[j]);
#pragma unroll
          for (int t = 0; t < T; t++) u[CONV_TAPS - 1 + t] = xin[t];
          __syncwarp();  // every lane has read the old state before the state is overwritten
#pragma unroll
          for (int t = 0; t < T; t++) {
            if ((t & 1) == hf) {
              float y = fmaf(wa.x, u[t], __fmul_rn(wa.y, u[t + 1]));
              y = fmaf(wb.x, u[t + 2], y);
              y = fmaf(wb.y, u[t + 3], y);
              const size_t o = (size_t)t * HID + n;
              cv.h[o] = tb(rb(cv.res[o] + rb((y + xin[t]) / cv.div)));
            }
          }
          if (hf == 0) {
#pragma unroll
            for (int j = 0; j < CONV_TAPS - 1; j++) cv.cst[(size_t)n * (CONV_TAPS - 1) + j] = tb(u[T + j]);
          }
        }
      }
    }
    cur = nxt;
  }

  // ---- exit: the last warp to leave resets the counters
  {
    unsigned last = 0;
    if (lane == 0) last = (atomicAdd(&ws.ctr[2], 1u) == (unsigned)nw_tot - 1);
    last = __shfl_sync(0xffffffffu, last, 0);
    if (last && lane == 0) {
      ws.ctr[1] = 0;
      ws.ctr[2] = 0;
      ws.ctr[0] = 0;
      for (int g = 0; g < ngrp; g++) ws.ctr[256 + g] = 0;
    }
  }
}

}  // namespace moe
