#pragma once
#include "qwen35_attn_split.cuh"

// Tensor-core split-KV flash decode of the gated hd-256 attention with the int4_rot cache, the hand engine's attn_split_i4_kernel (engine/src/attn.cu) operation for operation,
// so the output is bit for bit its. One block is (split, KV head, n-tile of 8 query columns) (the hand's n-tile groups are blocks: every column is its own softmax and its own dot
// products); its 2 KS warps are the hand's teams: team ks = the key slice (tiles ks, ks + KS, ...), its two warps h split d in halves (K chunks 2 h, 2 h + 1 and V chunk h) and
// exchange the half scores through shared memory (one named barrier per tile). What fixes the bits:
//  - keys per online softmax: the splits of `geom` (>= 128 keys per split), tiles of 16 keys from the split's first key (the sink keys 0..3 are replaced inside tile 0 of split 0),
//    then the exact ring tiles rt = s + nsplit ks, + nsplit KS, ... of the 31 (RING_W / 16) in the same online softmax;
//  - scores: the q fragments are q / 16 in fp16; per half h of d (chunks of 16 codes at d = 64 b + 16 j) and scale group (word of the chunk) the four k steps accumulate in fp16
//    (mma f16 accumulate), the group sums (fp32) are weighed by the K scale and added: x_h = g0 k0 + g1 k1, s = (x_0 + x_1) log2(e); the sink keys' scores are an fp32 fma chain over
//    the unscaled q and the fp16 sink rows;
//  - P rounded to fp16 after the V scale of the key and d group (the sink keys' P zeroed there, their exact V added with fma afterwards), fp32 accumulation;
//  - the key slices merged o = sum_k exp2(m_k - M) o_k in order; k_merge sums the splits in four groups and un-rotates (WHT, ascending bits).
// A lane (row group r = lane >> 2, quad j = lane & 3) holds the accumulators O[h' * 4 + mt] of the rows r, r + 8 = elements mt, mt + 4 of the 8-code word d = 128 h + 16 r + 8 h' + e
// (each output d is its own dot product over the keys, so the row assignment does not change a bit).
namespace g256 {

constexpr float LOG2E_F = 1.4426950408889634f;
constexpr int NSINK = 4;  // kvrot::NSINK

template <int KS>
__global__ void __launch_bounds__(64 * KS, 1) k_split_i4(SplitArgs a) {
  constexpr float NEG = -INFINITY;
  extern __shared__ __align__(16) float smem[];
  const int T = a.T, Hq = a.Hq, Hkv = a.Hkv, G = Hq / Hkv, ctx = a.ctx;
  const int pos0 = *a.pos0, pe = pos0 + T - 1;
  const int ctxq = max(pe + 1 - a.L.win, 0);  // keys of the split loop; the newest `win` keys are the exact ring
  int nsl, chunk;
  geom(ctxq, a.smax, a.chmin, nsl, chunk);
  const int s = blockIdx.x, kvh = blockIdx.y, ntb = blockIdx.z;
  if (s >= nsl) return;
  const int wid = threadIdx.x >> 5, lane = threadIdx.x & 31, r = lane >> 2, j = lane & 3;
  const int h = wid & 1, ks = wid >> 1;  // half of d, key slice
  const int ncols = T * G;
  const int kstart = s * chunk, kend = min(ctxq, kstart + chunk);
  const int ntiles = (kend - kstart + 15) / 16;
  const uint16_t* qbuf = reinterpret_cast<const uint16_t*>(a.q);

  // q fragments (B operand) of this half: k step sl = 4 bb + u, d = 128 h + 64 bb + 16 j + 8 (u >> 1) + 2 (u & 1) + {0, 4} and +1, scaled by 1/16 in fp16
  u32 qs[8][2];
  {
    const int c = ntb * 8 + r;
    const bool valid = c < ncols;
    const uint16_t* qp = qbuf + ((size_t)(valid ? c / G : 0) * Hq + kvh * G + (valid ? c % G : 0)) * HD;
#pragma unroll
    for (int sl = 0; sl < 8; sl++) {
      const int bb = sl >> 2, u = sl & 3;
      const uint16_t* qq = qp + (2 * h + bb) * 64 + j * 16 + 8 * (u >> 1) + 2 * (u & 1);
      qs[sl][0] = qs[sl][1] = 0u;
      if (valid) {
        const __half2 lo = __hmul2(__halves2half2(__ushort_as_half(qq[0]), __ushort_as_half(qq[4])), __float2half2_rn(0.0625f));
        const __half2 hi = __hmul2(__halves2half2(__ushort_as_half(qq[1]), __ushort_as_half(qq[5])), __float2half2_rn(0.0625f));
        qs[sl][0] = *reinterpret_cast<const u32*>(&lo);
        qs[sl][1] = *reinterpret_cast<const u32*>(&hi);
      }
    }
  }
  // keys visible to the lane's columns 8 ntb + 2 j + e: key <= lim
  int limq[2], limr[2];
#pragma unroll
  for (int e = 0; e < 2; e++) {
    const int tc = min((ntb * 8 + 2 * j + e) / G, T - 1);
    limr[e] = pos0 + tc;
    limq[e] = min(pos0 + tc, kend - 1);
  }

  float O[8][4];
#pragma unroll
  for (int i = 0; i < 8; i++)
#pragma unroll
    for (int c = 0; c < 4; c++) O[i][c] = 0.f;
  float m_run[2] = {NEG, NEG}, l_run[2] = {0.f, 0.f};

  // the team's score exchange: [KS][2 parities][2 halves][4][32]
  float* xw = smem + ((ks * 2) * 2 + h) * 128 + lane;
  float* xr = smem + (ks * 2) * 2 * 128 + lane;
  int par = 0;
  // sc = (x_0 + x_1) log2(e) from the two halves' sums of a tile: xwrite stores this half's sum, xread (one named barrier per team and tile) returns the score
  auto xwrite = [&](const float (&x)[4]) {
#pragma unroll
    for (int i = 0; i < 4; i++) xw[par * 256 + i * 32] = x[i];
  };
  auto xread = [&](float (&sc)[4]) {
    asm volatile("bar.sync %0, %1;\n" ::"r"(1 + ks), "r"(64) : "memory");
#pragma unroll
    for (int i = 0; i < 4; i++) sc[i] = (xr[par * 256 + i * 32] + xr[par * 256 + 128 + i * 32]) * LOG2E_F;
    par ^= 1;
  };

  // mask, online softmax of one tile (hand order), rescale of the accumulators; p: the probabilities
  auto softmax_h = [&](float (&sc)[4], int base, int lim0, int lim1, float (&p)[4]) {
    const int kp_lo = base + r, kp_hi = base + r + 8;
    if (kp_lo > lim0) sc[0] = NEG;
    if (kp_lo > lim1) sc[1] = NEG;
    if (kp_hi > lim0) sc[2] = NEG;
    if (kp_hi > lim1) sc[3] = NEG;
    float mx0 = fmaxf(sc[0], sc[2]), mx1 = fmaxf(sc[1], sc[3]);
#pragma unroll
    for (int o = 4; o < 32; o <<= 1) {
      mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, o));
      mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, o));
    }
    const float mn0 = fmaxf(m_run[0], mx0), mn1 = fmaxf(m_run[1], mx1);
    const float off0 = (mn0 == NEG) ? 0.f : mn0, off1 = (mn1 == NEG) ? 0.f : mn1;
    const float al0 = exp2f(m_run[0] - off0), al1 = exp2f(m_run[1] - off1);
    m_run[0] = mn0;
    m_run[1] = mn1;
    p[0] = exp2f(sc[0] - off0);
    p[1] = exp2f(sc[1] - off1);
    p[2] = exp2f(sc[2] - off0);
    p[3] = exp2f(sc[3] - off1);
    l_run[0] = l_run[0] * al0 + p[0] + p[2];
    l_run[1] = l_run[1] * al1 + p[1] + p[3];
    if (__any_sync(0xffffffffu, (al0 != 1.f) | (al1 != 1.f))) {
#pragma unroll
      for (int i = 0; i < 8; i++) {
        O[i][0] *= al0;
        O[i][1] *= al1;
        O[i][2] *= al0;
        O[i][3] *= al1;
      }
    }
  };
  // sums of the two f16 accumulator chains of this half: x[i] = g0[i] k0[row] + g1[i] k1[row] (i: c0..c3, row = i >> 1)
  auto group_sum = [&](const u32 (&sg)[2][2], float k00, float k01, float k10, float k11, float (&x)[4]) {
    const float2 s00 = h2f(sg[0][0]), s01 = h2f(sg[0][1]), s10 = h2f(sg[1][0]), s11 = h2f(sg[1][1]);
    const float g0[4] = {s00.x, s00.y, s01.x, s01.y}, g1[4] = {s10.x, s10.y, s11.x, s11.y};
    const float ksc[2][2] = {{k00, k01}, {k10, k11}};
#pragma unroll
    for (int i = 0; i < 4; i++) x[i] = g0[i] * ksc[i >> 1][0] + g1[i] * ksc[i >> 1][1];
  };

  // ---- tiles of 4-bit codes
  struct CodeRegs {
    uint2 K[2][2];  // K rows r, r + 8: chunks b = 2 h + bb of lane j (16 codes at d = 64 b + 16 j)
    uint2 V[4];     // V rows of the key slots {2 j, 2 j + 1, 2 j + 8, 2 j + 9}: 16 codes at d = 128 h + 16 r
    u32 km[2], vm[2];
  };
  auto code_load = [&](CodeRegs& t, int base) {
    const int klast = kend - 1;
    const uint8_t* kcb = a.kv + a.L.kc + (size_t)kvh * ctx * (HD / 2);
    const uint8_t* vcb = a.kv + a.L.vc + (size_t)kvh * ctx * (HD / 2);
    const int r0 = min(base + r, klast), r1 = min(base + r + 8, klast);
#pragma unroll
    for (int bb = 0; bb < 2; bb++) {
      t.K[0][bb] = ld8<1>(kcb + (size_t)r0 * (HD / 2) + j * 8 + 64 * h + 32 * bb);
      t.K[1][bb] = ld8<1>(kcb + (size_t)r1 * (HD / 2) + j * 8 + 64 * h + 32 * bb);
    }
    const u32* km = reinterpret_cast<const u32*>(a.kv + a.L.ks) + (size_t)kvh * ctx;
    const u32* vm = reinterpret_cast<const u32*>(a.kv + a.L.vs) + (size_t)kvh * ctx;
    t.km[0] = __ldg(km + r0);
    t.km[1] = __ldg(km + r1);
    t.vm[0] = __ldg(vm + r0);
    t.vm[1] = __ldg(vm + r1);
#pragma unroll
    for (int sl = 0; sl < 4; sl++) {
      const int kk = min(base + 2 * j + (sl & 1) + (sl >> 1) * 8, klast);
      t.V[sl] = ld8<1>(vcb + (size_t)kk * (HD / 2) + 8 * r + 64 * h);
    }
  };
  const uint16_t* sinkK = reinterpret_cast<const uint16_t*>(a.kv + a.L.kx) + (size_t)kvh * NSINK * HD;
  const uint16_t* sinkV = reinterpret_cast<const uint16_t*>(a.kv + a.L.vx) + (size_t)kvh * NSINK * HD;
  // QK stage of a tile: this half's sum x
  auto code_qk = [&](const CodeRegs& t, float (&x)[4]) {
    u32 sg[2][2] = {{0u, 0u}, {0u, 0u}};
#pragma unroll
    for (int sl = 0; sl < 8; sl++) {
      const int bb = sl >> 2, u = sl & 3;
      const uint2 K0 = t.K[0][bb], K1 = t.K[1][bb];
      const u32 w0 = (u >> 1) ? K0.y : K0.x, w1 = (u >> 1) ? K1.y : K1.x;
      mma_h(sg[u >> 1], lvp(w0, 2 * (u & 1)), lvp(w1, 2 * (u & 1)), lvp(w0, 2 * (u & 1) + 1), lvp(w1, 2 * (u & 1) + 1), qs[sl][0], qs[sl][1]);
    }
    float ksc[2][2];
#pragma unroll
    for (int row = 0; row < 2; row++)
#pragma unroll
      for (int gl = 0; gl < 2; gl++) ksc[row][gl] = r4::scale_read((t.km[row] >> (8 * (2 * h + gl))) & 255u);
    group_sum(sg, ksc[0][0], ksc[0][1], ksc[1][0], ksc[1][1], x);
  };
  // softmax and PV stage of a tile whose scores are sc
  auto code_finish = [&](const CodeRegs& t, float (&sc)[4], int base, auto sink_tag) {
    constexpr bool sink_tile = decltype(sink_tag)::value;
    if (sink_tile) {  // exact scores of the sink keys: lane (col = lane >> 2, key = lane & 3) takes q . k_sink in fp32, the holders (key r, columns 2 j, 2 j + 1) fetch them
      const int c = ntb * 8 + (lane >> 2);
      const int tq = c < ncols ? c / G : 0, hq = c < ncols ? c % G : 0;
      const uint4* qv = reinterpret_cast<const uint4*>(qbuf + ((size_t)tq * Hq + kvh * G + hq) * HD);
      const uint4* kv4 = reinterpret_cast<const uint4*>(sinkK + (size_t)(lane & 3) * HD);
      float dv = 0.f;
#pragma unroll 4
      for (int x8 = 0; x8 < HD / 8; x8++) {
        const uint4 qq = qv[x8], kk8 = kv4[x8];
        const u32 qw[4] = {qq.x, qq.y, qq.z, qq.w}, kw[4] = {kk8.x, kk8.y, kk8.z, kk8.w};
#pragma unroll
        for (int w = 0; w < 4; w++) {
          const float2 qf = h2f(qw[w]), kf = h2f(kw[w]);
          dv = fmaf(qf.x, kf.x, dv);
          dv = fmaf(qf.y, kf.y, dv);
        }
      }
      float ex[2];
#pragma unroll
      for (int e = 0; e < 2; e++) ex[e] = __shfl_sync(0xffffffffu, dv, (2 * j + e) * 4 + (r & 3));
      if (r < NSINK) {
        sc[0] = ex[0] * ((1.f / 16.f) * LOG2E_F);
        sc[1] = ex[1] * ((1.f / 16.f) * LOG2E_F);
      }
    }
    float p[4];
    softmax_h(sc, base, limq[0], limq[1], p);
    float pk[NSINK][2];
    if (sink_tile) {  // the sink keys' exact probabilities (held by lanes 4 k + j) go to the exact V update below; the quantised mma sees 0 for them
#pragma unroll
      for (int k = 0; k < NSINK; k++) {
        pk[k][0] = __shfl_sync(0xffffffffu, p[0], 4 * k + j);
        pk[k][1] = __shfl_sync(0xffffffffu, p[1], 4 * k + j);
      }
    }
#pragma unroll
    for (int h2 = 0; h2 < 2; h2++) {
      const int gg = 2 * h + h2;
      const float s0 = r4::scale_read((t.vm[0] >> (8 * gg)) & 255u), s1 = r4::scale_read((t.vm[1] >> (8 * gg)) & 255u);
      const bool zs = sink_tile && r < NSINK;
      const u32 b0 = movT(pack2<true>(zs ? 0.f : p[0] * s0, zs ? 0.f : p[1] * s0)), b1 = movT(pack2<true>(p[2] * s1, p[3] * s1));
      // the words of the key slots {0, 1} and {2, 3}, their nibbles 0..3 (low halves) and 4..7 (high halves) interleaved: lvp(pair, mt) = the levels of the elements mt / mt + 4 of both keys
      const u32 wk[4] = {h2 ? t.V[0].y : t.V[0].x, h2 ? t.V[1].y : t.V[1].x, h2 ? t.V[2].y : t.V[2].x, h2 ? t.V[3].y : t.V[3].x};
      const u32 ab_lo = __byte_perm(wk[0], wk[1], 0x5410), ab_hi = __byte_perm(wk[0], wk[1], 0x7632);
      const u32 cd_lo = __byte_perm(wk[2], wk[3], 0x5410), cd_hi = __byte_perm(wk[2], wk[3], 0x7632);
#pragma unroll
      for (int mt = 0; mt < 4; mt++) mma16<true>(O[h2 * 4 + mt], lvp(ab_lo, mt), lvp(ab_hi, mt), lvp(cd_lo, mt), lvp(cd_hi, mt), b0, b1);
    }
    if (sink_tile) {
#pragma unroll
      for (int h2 = 0; h2 < 2; h2++)
#pragma unroll
        for (int mt = 0; mt < 4; mt++) {
          const int d = 128 * h + 16 * r + 8 * h2 + mt;
          float* o = O[h2 * 4 + mt];
#pragma unroll
          for (int k = 0; k < NSINK; k++) {
            const float v0 = __half2float(__ushort_as_half(sinkV[(size_t)k * HD + d])), v1 = __half2float(__ushort_as_half(sinkV[(size_t)k * HD + d + 4]));
            o[0] = fmaf(pk[k][0], v0, o[0]);
            o[1] = fmaf(pk[k][1], v0, o[1]);
            o[2] = fmaf(pk[k][0], v1, o[2]);
            o[3] = fmaf(pk[k][1], v1, o[3]);
          }
        }
    }
  };
  {  // two register sets: the loads of the next tile fly while the current one is computed
    CodeRegs ra, rb;
    int it = ks;
    if (it < ntiles) code_load(ra, kstart + it * 16);
    while (it < ntiles) {
      const int it1 = it + KS, it2 = it1 + KS;
      if (it1 < ntiles) code_load(rb, kstart + it1 * 16);
      const int base = kstart + it * 16;
      float x[4], sc[4];
      code_qk(ra, x);
      xwrite(x);
      xread(sc);
      if (base == 0) code_finish(ra, sc, base, std::true_type{});
      else code_finish(ra, sc, base, std::false_type{});
      if (it1 >= ntiles) break;
      if (it2 < ntiles) code_load(ra, kstart + it2 * 16);
      code_qk(rb, x);
      xwrite(x);
      xread(sc);
      code_finish(rb, sc, kstart + it1 * 16, std::false_type{});
      it = it2;
    }
  }

  // ---- exact ring tiles: fp16 rows in the rotated basis
  const uint16_t* ringK = reinterpret_cast<const uint16_t*>(a.kv + a.L.kr) + (size_t)kvh * a.L.ring * HD;
  const uint16_t* ringV = reinterpret_cast<const uint16_t*>(a.kv + a.L.vr) + (size_t)kvh * a.L.ring * HD;
  for (int rt = s + nsl * ks; rt < a.L.win / 16; rt += nsl * KS) {
    const int base = ctxq + rt * 16;
    if (base > pe) break;  // a tile past the live keys changes nothing (every score masked)
    u32 kw[2][2][8];       // [row][bb][word]: d = 128 h + 64 bb + 16 j + 8 hf + 0..7
#pragma unroll
    for (int row = 0; row < 2; row++) {
      const uint16_t* kp = ringK + (size_t)((base + r + 8 * row) & (a.L.ring - 1)) * HD + 128 * h + j * 16;
#pragma unroll
      for (int bb = 0; bb < 2; bb++)
#pragma unroll
        for (int hf = 0; hf < 2; hf++) {
          const uint4 u = ld16<2>(kp + bb * 64 + hf * 8);
          kw[row][bb][4 * hf] = u.x;
          kw[row][bb][4 * hf + 1] = u.y;
          kw[row][bb][4 * hf + 2] = u.z;
          kw[row][bb][4 * hf + 3] = u.w;
        }
    }
    uint4 V[4][2];  // key slots {2 j, 2 j + 1, 2 j + 8, 2 j + 9}, chunk hf: d = 128 h + 16 r + 8 hf
#pragma unroll
    for (int sl = 0; sl < 4; sl++) {
      const uint16_t* vp = ringV + (size_t)((base + 2 * j + (sl & 1) + (sl >> 1) * 8) & (a.L.ring - 1)) * HD + 128 * h + r * 16;
#pragma unroll
      for (int hf = 0; hf < 2; hf++) V[sl][hf] = ld16<2>(vp + 8 * hf);
    }
    u32 sg[2][2] = {{0u, 0u}, {0u, 0u}};
#pragma unroll
    for (int sl = 0; sl < 8; sl++) {
      const int bb = sl >> 2, u = sl & 3, wi = 4 * (u >> 1) + (u & 1);  // a0 = (e[n], e[n + 4]) of row r, a2 = (e[n + 1], e[n + 5]), n = 8 (u >> 1) + 2 (u & 1)
      const u32 x0 = kw[0][bb][wi], y0 = kw[0][bb][wi + 2], x1 = kw[1][bb][wi], y1 = kw[1][bb][wi + 2];
      mma_h(sg[u >> 1], __byte_perm(x0, y0, 0x5410), __byte_perm(x1, y1, 0x5410), __byte_perm(x0, y0, 0x7632), __byte_perm(x1, y1, 0x7632), qs[sl][0], qs[sl][1]);
    }
    float x[4], sc[4], p[4];
    group_sum(sg, 1.f, 1.f, 1.f, 1.f, x);
    xwrite(x);
    xread(sc);
    softmax_h(sc, base, limr[0], limr[1], p);
    const u32 b0 = movT(pack2<true>(p[0] * 1.f, p[1] * 1.f)), b1 = movT(pack2<true>(p[2] * 1.f, p[3] * 1.f));
#pragma unroll
    for (int hf = 0; hf < 2; hf++)
#pragma unroll
      for (int mt = 0; mt < 4; mt++) {
        const int sel = (mt & 1) ? 0x7632 : 0x5410, lo = mt >> 1, hi = lo + 2;
        mma16<true>(O[hf * 4 + mt], __byte_perm(comp(V[0][hf], lo), comp(V[1][hf], lo), sel), __byte_perm(comp(V[0][hf], hi), comp(V[1][hf], hi), sel),
                    __byte_perm(comp(V[2][hf], lo), comp(V[3][hf], lo), sel), __byte_perm(comp(V[2][hf], hi), comp(V[3][hf], hi), sel), b0, b1);
      }
  }

  // ---- merge of the key slices: complete l over the 8 key-row lanes, dump (m, l, O) (the buffers alias the score exchange: every warp is done with it), weights exp2(m_k - M)
  // per column (k ascending, the hand engine's order)
#pragma unroll
  for (int o = 4; o < 32; o <<= 1) {
    l_run[0] += __shfl_xor_sync(0xffffffffu, l_run[0], o);
    l_run[1] += __shfl_xor_sync(0xffffffffu, l_run[1], o);
  }
  __syncthreads();
  // the dump of O pads one float per 16 d (rows r of a warp are 16 d apart: without it 16 lanes hit one bank)
  constexpr int PC = HD + HD / 16;
  float* ow = smem + ks * 8 * PC;
  float* mls = smem + KS * 8 * PC;  // [KS][8][2]
  float* wts = mls + KS * 16;       // [8][KS]
#pragma unroll
  for (int h2 = 0; h2 < 2; h2++)
#pragma unroll
    for (int mt = 0; mt < 4; mt++) {
      const int d = 128 * h + 16 * r + 8 * h2 + mt;
      ow[(2 * j) * PC + d + (d >> 4)] = O[h2 * 4 + mt][0];
      ow[(2 * j) * PC + (d + 4) + ((d + 4) >> 4)] = O[h2 * 4 + mt][2];
      ow[(2 * j + 1) * PC + d + (d >> 4)] = O[h2 * 4 + mt][1];
      ow[(2 * j + 1) * PC + (d + 4) + ((d + 4) >> 4)] = O[h2 * 4 + mt][3];
    }
  if (h == 0 && r == 0) {
    mls[(ks * 8 + 2 * j) * 2] = m_run[0];
    mls[(ks * 8 + 2 * j) * 2 + 1] = l_run[0];
    mls[(ks * 8 + 2 * j + 1) * 2] = m_run[1];
    mls[(ks * 8 + 2 * j + 1) * 2 + 1] = l_run[1];
  }
  __syncthreads();
  if (threadIdx.x < 8) {
    const int cc = threadIdx.x, col = ntb * 8 + cc;
    float M = NEG;
#pragma unroll
    for (int k = 0; k < KS; k++) M = fmaxf(M, mls[(k * 8 + cc) * 2]);
    float L = 0.f;
#pragma unroll
    for (int k = 0; k < KS; k++) {
      const float mk = mls[(k * 8 + cc) * 2];
      const float wk = M == NEG || mk == NEG ? 0.f : exp2f(mk - M);
      wts[cc * KS + k] = wk;
      L += wk * mls[(k * 8 + cc) * 2 + 1];
    }
    if (col < ncols) {
      float* dst = a.part + (((size_t)(col / G) * Hq + kvh * G + col % G) * a.smax + s) * (HD + 4);
      dst[HD] = M;
      dst[HD + 1] = L;
    }
  }
  __syncthreads();
  for (int cc = 0; cc < 8; cc++) {
    const int col = ntb * 8 + cc;
    if (col >= ncols) break;
    float* dst = a.part + (((size_t)(col / G) * Hq + kvh * G + col % G) * a.smax + s) * (HD + 4);
    for (int d = threadIdx.x; d < HD; d += 64 * KS) {
      float o = 0.f;
#pragma unroll
      for (int k = 0; k < KS; k++) o += wts[cc * KS + k] * smem[(k * 8 + cc) * PC + d + (d >> 4)];
      dst[d] = o;
    }
  }
}

template <int KS>
int run_split_i4(const SplitArgs& a, int nt, cudaStream_t st) {
  constexpr size_t smem = ((size_t)KS * (8 * (HD + HD / 16) + 16) + 8 * KS) * sizeof(float);
  if (a.L.sink != NSINK || a.L.win % 16 || a.L.ring & (a.L.ring - 1)) return cudaErrorInvalidValue;
  k_split_i4<KS><<<dim3(a.smax, a.Hkv, nt), 64 * KS, smem, st>>>(a);
  return cudaGetLastError();
}

}  // namespace g256
