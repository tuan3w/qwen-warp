// Token-granular gather copy of attn_split_i4_kernel (see attn_sparse.cuh for the 64-token-block version).  Included after attn_sparse.cuh.
#pragma once
constexpr int SP_MAXT = 1024;  // selected tokens one split can own (16384 tokens / 20 splits, tile padded = 848)
// lst[kvh * lst_stride] = count, then entries (block id | row mask << 16), ascending block id.  Row mask bit t = query row t attends the block (columns of the other rows get -inf).
template <int NTG, int NG, int DS, int KS>
__global__ void __launch_bounds__(32 * DS * KS * NG, 1)
    attn_split_i4t_kernel(int T, const void* __restrict__ kv, int ctx_max, const int* __restrict__ pos0p, const uint16_t* __restrict__ qbuf, float2* __restrict__ ml,
                          float* __restrict__ part, const uint32_t* __restrict__ lst, int lst_stride, int dbg, int bps) {
  static_assert(DS == 2 || DS == 4, "d-split warps per team");
  static_assert(KS * NG <= 15, "named barriers: one per team");
  constexpr int NT = NTG;       // n-tiles per warp (the group's)
  constexpr int NTT = NTG * NG;  // n-tiles of the block (>= ceil(6 T / 8); surplus tiles are all-invalid columns)
  constexpr int NW = DS * KS * NG;
  constexpr int ROWB = HD / 2;  // 128 B of codes per token row
  constexpr int NSTEP = 16;     // k16 steps over d
  constexpr int NI = 16 / DS;   // PV mma instances per warp: DS = 2: chunk c = h, g = 0..7; DS = 4: chunk c = h >> 1, g = 4 (h & 1) + 0..3
  constexpr int NBB = 4 / DS;   // K chunks (16 codes of a lane = 4 k16 steps each) per warp
  constexpr int NSL = 4 * NBB;  // k16 steps per warp
  constexpr int VW = DS == 2 ? 2 : 1;  // 32-bit words of a V chunk (16 codes = 2 words) this warp needs
  constexpr int GVN = NI / 4;          // V scale groups of this warp
  extern __shared__ __align__(16) float smem[];
  __shared__ uint32_t sent[SP_MAXT];
  __shared__ __half2 ssv[kvrot::NSINK * HD / 2];  // exact fp16 V sink rows of this kv head (only filled in the CTA that holds block 0)
  __shared__ float ssink[8 * GRP * kvrot::NSINK];  // exact q . k_sink scores [column][sink key] (only filled in the CTA that holds block 0)

  const int pos0 = *pos0p;
  const int ctx = max(pos0 + T - kvrot::RING_W, 0);
  const int s = blockIdx.x, kvh = blockIdx.y;
  const uint32_t* lh = lst + (size_t)kvh * lst_stride;
  const int ncnt = (int)lh[0];  // selected tokens of this kv head, lh[1..ncnt] ascending token ids (0,1,2,3 first: the exact sink keys)
  const int nsplit = sp_nsplit(ncnt, bps, ctx);  // bps = tokens per split here
  if (s >= nsplit) return;
  const int NTL = (ncnt + 15) / 16;  // tiles of 16 selected tokens (slot index = position in the list)
  const int e0 = (int)(((long long)s * NTL) / nsplit), e1 = (int)(((long long)(s + 1) * NTL) / nsplit);  // this split's tiles
  const int kend = ncnt;  // slot limit: slots >= ncnt are padding
  const int wid = threadIdx.x >> 5, lane = threadIdx.x & 31, r = lane >> 2, j = lane & 3;
  const int h = wid % DS, team = wid / DS, ks = team % KS, gi = team / KS, nt0 = gi * NTG;  // n-tile group gi: global n-tiles nt0 .. nt0 + NTG - 1
  const int ncols = GRP * T;

  const char* kbase = reinterpret_cast<const char*>(kv) + (size_t)(0 * NKV + kvh) * ctx_max * ROWB;
  const char* vbase = reinterpret_cast<const char*>(kv) + (size_t)(1 * NKV + kvh) * ctx_max * ROWB;
  const uint16_t* sinkb = reinterpret_cast<const uint16_t*>(reinterpret_cast<const char*>(kv) + kvrot::sink_off(ctx_max));
  const uint32_t* kmeta = reinterpret_cast<const uint32_t*>(reinterpret_cast<const char*>(kv) + (size_t)2 * NKV * ctx_max * ROWB) + (size_t)(0 * NKV + kvh) * ctx_max;
  const uint32_t* vmeta = reinterpret_cast<const uint32_t*>(reinterpret_cast<const char*>(kv) + (size_t)2 * NKV * ctx_max * ROWB) + (size_t)(1 * NKV + kvh) * ctx_max;
  const int ntiles = e1 - e0;
  struct Tile {
    uint2 k[2][NBB];  // K rows r and r+8, chunks b = NBB h .. NBB h + NBB - 1 of lane j (16 codes = one k16 step each)
    uint32_t v[4][VW];  // V key slots {2j, 2j+1, 2j+8, 2j+9}, the VW words of chunk c of lane r this warp needs
    uint32_t km[2], vm[2], rm;
  };
  // V chunk c = (h * NI) >> 3 of this warp (byte offset 64 c within the row slice of lane r) and, for DS = 4, its word (h & 1)
  const int voff = r * 8 + (DS == 2 ? h * 64 : (h >> 1) * 64 + (h & 1) * 4);
  auto issue = [&](Tile& t, int it) {
    const uint32_t tk0 = sent[it * 16 + r], tk1 = sent[it * 16 + r + 8];  // token ids of key slots r, r + 8 of this tile
    t.rm = 0xFFFFFFFFu;
    const char* k0 = kbase + (size_t)tk0 * ROWB + j * 8 + h * (NBB * 32);
    const char* k1 = kbase + (size_t)tk1 * ROWB + j * 8 + h * (NBB * 32);
#pragma unroll
    for (int bb = 0; bb < NBB; bb++) {
      t.k[0][bb] = ld_kv8<NTT>(k0 + bb * 32);
      t.k[1][bb] = ld_kv8<NTT>(k1 + bb * 32);
    }
    t.km[0] = __ldg(kmeta + tk0); t.km[1] = __ldg(kmeta + tk1);
    t.vm[0] = __ldg(vmeta + tk0); t.vm[1] = __ldg(vmeta + tk1);
#pragma unroll
    for (int sl = 0; sl < 4; sl++) {
      const uint32_t kk_ = sent[it * 16 + 2 * j + (sl & 1) + (sl >> 1) * 8];
      const char* vp = vbase + (size_t)kk_ * ROWB + voff;
      if constexpr (VW == 2) {
        const uint2 w = ld_kv8<NTT>(vp);
        t.v[sl][0] = w.x; t.v[sl][1] = w.y;
      } else {
        uint32_t w;
        asm volatile("ld.global.nc.u32 %0, [%1];\n" : "=r"(w) : "l"(vp));
        t.v[sl][0] = w;
      }
    }
  };
  // exact ring tile (DS = 2 only): fp16 rows of keys base .. base + 15 in the rotated basis, same lane layout as Tile
  struct RTile {
    uint4 k[2][NBB][2];  // K rows r, r+8: d = 128 h + 64 bb + 16 j .. +15 (2 x uint4)
    uint4 v[4][2];       // V key slots {2j, 2j+1, 2j+8, 2j+9}: d = 128 h + 16 r .. +15
  };
  static_assert(DS == 2, "ring tiles: V chunk = h");
  const uint16_t* ringK = reinterpret_cast<const uint16_t*>(reinterpret_cast<const char*>(kv) + kvrot::ring_off(ctx_max)) + (size_t)(0 * NKV + kvh) * kvrot::RING_R * HD;
  const uint16_t* ringV = ringK + (size_t)NKV * kvrot::RING_R * HD;
  auto issue_ring = [&](RTile& t, int base) {
#pragma unroll
    for (int row = 0; row < 2; row++) {
      const uint16_t* kp = ringK + (size_t)((base + r + 8 * row) & (kvrot::RING_R - 1)) * HD + h * 128 + j * 16;
#pragma unroll
      for (int bb = 0; bb < NBB; bb++)
#pragma unroll
        for (int hf = 0; hf < 2; hf++) t.k[row][bb][hf] = ld_kv<2>(kp + bb * 64 + hf * 8);
    }
#pragma unroll
    for (int sl = 0; sl < 4; sl++) {
      const uint16_t* vp = ringV + (size_t)((base + 2 * j + (sl & 1) + (sl >> 1) * 8) & (kvrot::RING_R - 1)) * HD + h * 128 + r * 16;
#pragma unroll
      for (int hf = 0; hf < 2; hf++) t.v[sl][hf] = ld_kv<2>(vp + hf * 8);
    }
  };

  Tile ta, tb;
  for (int i = threadIdx.x; i < (e1 - e0) * 16; i += NW * 32) { const int sl_ = e0 * 16 + i; sent[i] = lh[1 + min(sl_, ncnt - 1)]; }  // padding repeats the last token
  __syncthreads();
  auto tbase = [&](int it) { return (e0 + it) * 16; };  // global slot of the tile's first key
  if (ks < ntiles) issue(ta, ks);
  if (e1 > e0 && e0 == 0 && !(dbg & 2)) {  // this CTA owns block 0 (the sink tile): exact fp16 sink scores, all columns x NSINK keys, one warp per (column, key)
    const int wid_ = threadIdx.x >> 5;
    for (int i = threadIdx.x; i < kvrot::NSINK * HD / 2; i += NW * 32) ssv[i] = reinterpret_cast<const __half2*>(sinkb + (size_t)(1 * NKV + kvh) * kvrot::NSINK * HD)[i];
#pragma unroll 4
    for (int d = wid_; d < ncols * kvrot::NSINK; d += NW) {
      const int c = d / kvrot::NSINK, k = d % kvrot::NSINK, tq = c / GRP, hq = c % GRP;
      const uint4 qq = reinterpret_cast<const uint4*>(qbuf + ((size_t)tq * NH + kvh * GRP + hq) * HD)[lane];
      const uint4 kk8 = reinterpret_cast<const uint4*>(sinkb + ((size_t)(0 * NKV + kvh) * kvrot::NSINK + k) * HD)[lane];
      const uint32_t qw[4] = {qq.x, qq.y, qq.z, qq.w}, kw[4] = {kk8.x, kk8.y, kk8.z, kk8.w};
      float a = 0.f;
#pragma unroll
      for (int w = 0; w < 4; w++) {
        const float2 qf = __half22float2(*reinterpret_cast<const __half2*>(&qw[w])), kf = __half22float2(*reinterpret_cast<const __half2*>(&kw[w]));
        a = fmaf(qf.x, kf.x, a);
        a = fmaf(qf.y, kf.y, a);
      }
      a = warp_sum(a);
      if (lane == 0) ssink[c * kvrot::NSINK + k] = a;
    }
  }
  __syncthreads();

  // ---- Q B-fragments of the NSL steps this warp consumes: qs[nt][sidx][lane], sidx = NSL h .. NSL h + NSL - 1 (same fragment layout as attn_split_kernel)
  uint2* qs = reinterpret_cast<uint2*>(smem);
  float* xs = smem + NTT * NSTEP * 64;  // score exchange [KS NG teams][2 parity][DS][NT*4][32]
#pragma unroll
  for (int nt = 0; nt < NT; nt++) {
    const int c = (nt0 + nt) * 8 + r;
    const bool valid = c < ncols;
    const int tq = valid ? c / GRP : 0, hq = valid ? c - tq * GRP : 0;
    const uint16_t* qp = qbuf + ((size_t)tq * NH + kvh * GRP + hq) * HD;
    // one 16-element chunk (2 x uint4) per (n-tile, b) holds the fragments of the 4 steps u = 0..3: u -> words (w0, w2), (w1, w3), (w4, w6), (w5, w7)
#pragma unroll
    for (int bb = 0; bb < NBB; bb++) {
      const int b = h * NBB + bb;
      uint4 v0 = make_uint4(0, 0, 0, 0), v1 = v0;
      if (valid) {
        const uint4* src = reinterpret_cast<const uint4*>(qp + b * 64 + j * 16);
        v0 = src[0]; v1 = src[1];
      }
      const uint32_t w[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
#pragma unroll
      for (int u = 0; u < 4; u++) {
        const int wa = (u >> 1) * 4 + (u & 1), wb = wa + 2;
        qs[((nt0 + nt) * NSTEP + (h * NSL + bb * 4 + u)) * 32 + lane] = make_uint2(__byte_perm(w[wa], w[wb], 0x5410), __byte_perm(w[wa], w[wb], 0x7632));
      }
    }
  }
  __syncwarp();
  int lim[NT][2];
#pragma unroll
  for (int nt = 0; nt < NT; nt++)
#pragma unroll
    for (int e = 0; e < 2; e++) {
      const int c = (nt0 + nt) * 8 + 2 * j + e;
      const int tc = min(c / GRP, T - 1);
      lim[nt][e] = kend - 1; (void)tc;
    }
  const float scl = (1.f / 16.f) * LOG2E;
  float acc[NT][NI][4];
#pragma unroll
  for (int nt = 0; nt < NT; nt++)
#pragma unroll
    for (int i = 0; i < NI; i++) acc[nt][i][0] = acc[nt][i][1] = acc[nt][i][2] = acc[nt][i][3] = 0.f;
  float m_run[NT][2], l_run[NT][2];
#pragma unroll
  for (int nt = 0; nt < NT; nt++) { m_run[nt][0] = m_run[nt][1] = -INFINITY; l_run[nt][0] = l_run[nt][1] = 0.f; }

  auto compute = [&](const auto& t, int base, int par, auto ring_tag) {
    constexpr bool RING = decltype(ring_tag)::value;  // true: t is an RTile (exact fp16 ring rows, no scales)
    uint32_t rm = 0xFFFFFFFFu;
    if constexpr (!RING) rm = (dbg & 1) ? 0xFFFFFFFFu : t.rm;
    float ksc[2][2], vsc[2][GVN];  // [key row r / r+8][local scale group]: K groups 2 ((h NBB) >> 1) + gl (the 2 groups of this warp's K chunks), V groups GVN h + gl
#pragma unroll
    for (int row = 0; row < 2; row++) {
#pragma unroll
      for (int gl = 0; gl < 2; gl++) {
        if constexpr (RING) ksc[row][gl] = 1.f;
        else ksc[row][gl] = kvrot::scale_dec((t.km[row] >> (8 * (2 * ((h * NBB) >> 1) + gl))) & 255u, kvrot::LOG2_KGAIN);
      }
#pragma unroll
      for (int gl = 0; gl < GVN; gl++) {
        if constexpr (RING) vsc[row][gl] = 1.f;
        else vsc[row][gl] = kvrot::scale_dec((t.vm[row] >> (8 * (h * GVN + gl))) & 255u, kvrot::LOG2_VGAIN);
      }
    }
    // ---- S^T = K Q^T over this warp's half of d: K dequantised once, one mma per n-tile
    float sg[NT][2][4];
#pragma unroll
    for (int nt = 0; nt < NT; nt++)
#pragma unroll
      for (int gl = 0; gl < 2; gl++) sg[nt][gl][0] = sg[nt][gl][1] = sg[nt][gl][2] = sg[nt][gl][3] = 0.f;
#pragma unroll
    for (int sl = 0; sl < NSL; sl++) {
      const int bb = sl >> 2, u = sl & 3, sidx = h * NSL + sl;
      uint32_t a0, a1, a2, a3;
      if constexpr (RING) {  // fp16 rows: a0 = (e[n], e[n+4]) of row r, a2 = (e[n+1], e[n+5]), n = 8 (u >> 1) + 2 (u & 1) (the element order the q fragments were built for)
        const int wi = 4 * (u >> 1) + (u & 1);
        const uint32_t x0 = (&t.k[0][bb][0].x)[wi], y0 = (&t.k[0][bb][0].x)[wi + 2], x1 = (&t.k[1][bb][0].x)[wi], y1 = (&t.k[1][bb][0].x)[wi + 2];
        a0 = __byte_perm(x0, y0, 0x5410); a2 = __byte_perm(x0, y0, 0x7632);
        a1 = __byte_perm(x1, y1, 0x5410); a3 = __byte_perm(x1, y1, 0x7632);
      } else {
        const uint2 K0 = t.k[0][bb], K1 = t.k[1][bb];
        const uint32_t w0 = (u >> 1) ? K0.y : K0.x, w1 = (u >> 1) ? K1.y : K1.x;
        a0 = i4_pair(w0, 2 * (u & 1)); a2 = i4_pair(w0, 2 * (u & 1) + 1);
        a1 = i4_pair(w1, 2 * (u & 1)); a3 = i4_pair(w1, 2 * (u & 1) + 1);
      }
#pragma unroll
      for (int nt = 0; nt < NT; nt++) {
        const uint2 q = qs[((nt0 + nt) * NSTEP + sidx) * 32 + lane];
        mma_t<false, true>(sg[nt][u >> 1], a0, a1, a2, a3, q.x, q.y);
      }
    }
    // ---- exchange the partial scores (scaled partial sums over this warp's K chunks) with the other warps of the team, then add them in a fixed order
    float* xw = xs + ((size_t)((team * 2 + par) * DS + h) * (NT * 4)) * 32 + lane;
    const float* xr = xs + ((size_t)((team * 2 + par) * DS) * (NT * 4)) * 32 + lane;
#pragma unroll
    for (int nt = 0; nt < NT; nt++)
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const int row = i >> 1;
        xw[(nt * 4 + i) * 32] = sg[nt][0][i] * ksc[row][0] + sg[nt][1][i] * ksc[row][1];
      }
    asm volatile("bar.sync %0, %1;\n" ::"r"(1 + team), "r"(DS * 32) : "memory");

    const bool sink_tile = !RING && base == 0 && !(dbg & 2);  // dbg bit 1: timing diagnostic only (wrong numerics)  // tile 0 of split 0: keys 0..NSINK-1 use the exact fp16 sink rows instead of the 4-bit codes
    uint32_t pbg[NT][GVN][2];
    float pk[NT][kvrot::NSINK][2];
#pragma unroll
    for (int nt = 0; nt < NT; nt++) {
      float sc[4];
#pragma unroll
      for (int i = 0; i < 4; i++) {
        float x[DS];
#pragma unroll
        for (int hh = 0; hh < DS; hh++) x[hh] = xr[(hh * (NT * 4) + nt * 4 + i) * 32];
        if constexpr (DS == 2) sc[i] = (x[0] + x[1]) * scl;
        else sc[i] = ((x[0] + x[1]) + (x[2] + x[3])) * scl;
      }
      if (sink_tile) {  // exact sink scores were precomputed into ssink[column][key] (once per CTA, overlapped with the first tile's loads); holder lanes: key r, columns 2j, 2j+1
        if (r < kvrot::NSINK) {
          const int c0 = (nt0 + nt) * 8 + 2 * j;
          sc[0] = (c0 < ncols ? ssink[c0 * kvrot::NSINK + r] : 0.f) * scl;
          sc[1] = (c0 + 1 < ncols ? ssink[(c0 + 1) * kvrot::NSINK + r] : 0.f) * scl;
        }
      }
      int lim0 = lim[nt][0], lim1 = lim[nt][1];
      if constexpr (RING) {  // the ring is not clipped to the split's key range
        const int cc0 = (nt0 + nt) * 8 + 2 * j;
        lim0 = pos0 + min(cc0 / GRP, T - 1); lim1 = pos0 + min((cc0 + 1) / GRP, T - 1);
      }
      const int kp_lo = base + r, kp_hi = base + r + 8;
      if (kp_lo > lim0) sc[0] = -INFINITY;
      if (kp_lo > lim1) sc[1] = -INFINITY;
      if (kp_hi > lim0) sc[2] = -INFINITY;
      if (kp_hi > lim1) sc[3] = -INFINITY;
      {
        const int cc0 = (nt0 + nt) * 8 + 2 * j;
        if (!((rm >> min(cc0 / GRP, T - 1)) & 1u)) { sc[0] = -INFINITY; sc[2] = -INFINITY; }
        if (!((rm >> min((cc0 + 1) / GRP, T - 1)) & 1u)) { sc[1] = -INFINITY; sc[3] = -INFINITY; }
      }
      float mx0 = fmaxf(sc[0], sc[2]), mx1 = fmaxf(sc[1], sc[3]);
#pragma unroll
      for (int o = 4; o < 32; o <<= 1) {
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, o));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, o));
      }
      const float mn0 = fmaxf(m_run[nt][0], mx0), mn1 = fmaxf(m_run[nt][1], mx1);
      const float off0 = (mn0 == -INFINITY) ? 0.f : mn0, off1 = (mn1 == -INFINITY) ? 0.f : mn1;
      const float al0 = exp2f(m_run[nt][0] - off0), al1 = exp2f(m_run[nt][1] - off1);
      m_run[nt][0] = mn0;
      m_run[nt][1] = mn1;
      const float p0 = exp2f(sc[0] - off0), p1 = exp2f(sc[1] - off1), p2 = exp2f(sc[2] - off0), p3 = exp2f(sc[3] - off1);
      l_run[nt][0] = l_run[nt][0] * al0 + p0 + p2;
      l_run[nt][1] = l_run[nt][1] * al1 + p1 + p3;
      if (sink_tile) {  // exact probabilities of the sink keys (held by lanes 4k + j) go to the exact-V update below; the quantised mma sees p = 0 for them
#pragma unroll
        for (int k = 0; k < kvrot::NSINK; k++) {
          pk[nt][k][0] = __shfl_sync(0xffffffffu, p0, 4 * k + j);
          pk[nt][k][1] = __shfl_sync(0xffffffffu, p1, 4 * k + j);
        }
      }
#pragma unroll
      for (int i = 0; i < NI; i++) {
        acc[nt][i][0] *= al0; acc[nt][i][1] *= al1; acc[nt][i][2] *= al0; acc[nt][i][3] *= al1;
      }
      // fold the per-key, per-group V scale into P: (p_k * scale_kG) multiplies the integer codes of the d rows of group G
#pragma unroll
      for (int gl = 0; gl < GVN; gl++) {
        const bool zs = sink_tile && r < kvrot::NSINK;
        pbg[nt][gl][0] = movmatrix_t(pack16(zs ? 0.f : p0 * vsc[0][gl], zs ? 0.f : p1 * vsc[0][gl], true));
        pbg[nt][gl][1] = movmatrix_t(pack16(p2 * vsc[1][gl], p3 * vsc[1][gl], true));
      }
    }
    // ---- O^T += V^T P^T : V dequantised once per instance g (global instance i = NI h + g: d rows 128 (i >> 3) + r*16 + 2 (i & 7), +1), one mma per n-tile
#pragma unroll
    for (int g = 0; g < NI; g++) {
      uint32_t a0, a1, a2, a3;
      if constexpr (RING) {  // exact fp16 rows: a0 = (s0.e2g, s1.e2g), a1 = (s0.e2g+1, s1.e2g+1), a2 / a3 = slots 2, 3
        const uint32_t w0 = (&t.v[0][g >> 2].x)[g & 3], w1 = (&t.v[1][g >> 2].x)[g & 3], w2 = (&t.v[2][g >> 2].x)[g & 3], w3 = (&t.v[3][g >> 2].x)[g & 3];
        a0 = __byte_perm(w0, w1, 0x5410); a1 = __byte_perm(w0, w1, 0x7632); a2 = __byte_perm(w2, w3, 0x5410); a3 = __byte_perm(w2, w3, 0x7632);
      } else {
        const int widx = VW == 2 ? (g >> 2) : 0, n = (2 * g) & 7, nn = n & 3;  // compile-time: g is unrolled
        const uint32_t sel = n < 4 ? 0x5410u : 0x7632u;
        const uint32_t pab = __byte_perm(t.v[0][widx], t.v[1][widx], sel);
        const uint32_t pcd = __byte_perm(t.v[2][widx], t.v[3][widx], sel);
        a0 = i4_pair(pab, nn); a1 = i4_pair(pab, nn + 1); a2 = i4_pair(pcd, nn); a3 = i4_pair(pcd, nn + 1);
      }
#pragma unroll
      for (int nt = 0; nt < NT; nt++) mma_t<false, true>(acc[nt][g], a0, a1, a2, a3, pbg[nt][g >> 2][0], pbg[nt][g >> 2][1]);  // g >> 2 = local V scale group (0 for NI = 4)
    }
    if (sink_tile) {  // acc[nt][g] holds d0 = 128 (i >> 3) + 16 r + 2 (i & 7) with i = NI h + g (elements 0/2: d0, d0+1 of column 2j; 1/3: column 2j+1)
#pragma unroll
      for (int g = 0; g < NI; g++) {
        const int ig = NI * h + g;
        const int d0 = (ig >> 3) * 128 + r * 16 + 2 * (ig & 7);
#pragma unroll
        for (int nt = 0; nt < NT; nt++) {
          float* a = acc[nt][g];
#pragma unroll
          for (int k = 0; k < kvrot::NSINK; k++) {
            const __half2 vh = ssv[k * (HD / 2) + (d0 >> 1)];  // exact V sink rows staged in smem
            const float2 vf = __half22float2(vh);
            a[0] = fmaf(pk[nt][k][0], vf.x, a[0]);
            a[1] = fmaf(pk[nt][k][1], vf.x, a[1]);
            a[2] = fmaf(pk[nt][k][0], vf.y, a[2]);
            a[3] = fmaf(pk[nt][k][1], vf.y, a[3]);
          }
        }
      }
    }
  };

  // ---- key loop: two tile buffers per warp, the next tile's loads are in flight while the current one is computed
  int par = 0;
  if (ks < ntiles) {
    int it = ks;
    while (it < ntiles) {
      const int it1 = it + KS;
      if (it1 < ntiles) issue(tb, it1);
      compute(ta, tbase(it), par, std::false_type{});
      par ^= 1;
      if (it1 >= ntiles) break;
      const int it2 = it1 + KS;
      if (it2 < ntiles) issue(ta, it2);
      compute(tb, tbase(it1), par, std::false_type{});
      par ^= 1;
      it = it2;
    }
  }
  // ---- exact ring: the RING_W / 16 tiles of fp16 rotated rows are spread over the (split, team) pairs of the whole grid (pair index s + nsplit ks) and
  //      processed as extra tiles of the same online softmax (the old reduce kernel did the ring by itself, one block per query column)
  for (int rt = s + nsplit * ks; rt < ((dbg & 4) ? 0 : kvrot::RING_W / 16); rt += nsplit * KS) {  // dbg bit 2: skip the ring (timing diagnostic only)
    RTile rtl;
    issue_ring(rtl, ctx + rt * 16);
    compute(rtl, ctx + rt * 16, par, std::true_type{});
    par ^= 1;
  }

  // ---- finish: complete l over the 8 key-row lanes, dump (m, l, acc) to smem, merge the KS teams of each n-tile
#pragma unroll
  for (int nt = 0; nt < NT; nt++)
#pragma unroll
    for (int o = 4; o < 32; o <<= 1) {
      l_run[nt][0] += __shfl_xor_sync(0xffffffffu, l_run[nt][0], o);
      l_run[nt][1] += __shfl_xor_sync(0xffffffffu, l_run[nt][1], o);
    }
  __syncthreads();  // every warp is done with its Q fragments and the score exchange: the merge buffers alias them
  // the warps of one d chunk c (128 d) dump and merge in round c, so the buffer holds one chunk: sacc [KS][NT][8][128]
  float* sacc = smem;
  float* sm_m = smem + (size_t)KS * NTT * 8 * 128;  // [KS][NTT][8]
  float* sm_l = sm_m + KS * NTT * 8;
  if (h == 0 && r == 0) {
#pragma unroll
    for (int nt = 0; nt < NT; nt++) {
      const int w = ks * NTT + nt0 + nt;
      sm_m[w * 8 + 2 * j] = m_run[nt][0]; sm_m[w * 8 + 2 * j + 1] = m_run[nt][1];
      sm_l[w * 8 + 2 * j] = l_run[nt][0]; sm_l[w * 8 + 2 * j + 1] = l_run[nt][1];
    }
  }
  const int cw = (h * NI) >> 3;  // d chunk of this warp
#pragma unroll
  for (int rc = 0; rc < 2; rc++) {
    if (cw == rc) {
#pragma unroll
      for (int nt = 0; nt < NT; nt++) {
        const int w = ks * NTT + nt0 + nt;
#pragma unroll
        for (int g = 0; g < NI; g++) {
          const int d0 = r * 16 + 2 * ((NI * h + g) & 7);
          const float* a = acc[nt][g];
          *reinterpret_cast<float2*>(sacc + (size_t)(w * 8 + 2 * j) * 128 + d0) = make_float2(a[0], a[2]);
          *reinterpret_cast<float2*>(sacc + (size_t)(w * 8 + 2 * j + 1) * 128 + d0) = make_float2(a[1], a[3]);
        }
      }
    }
    __syncthreads();
    for (int e = threadIdx.x; e < NTT * 8 * 128; e += NW * 32) {
      const int dl = e & 127, col = e >> 7;
      if (col >= ncols) continue;
      const int nt_ = col >> 3, cc = col & 7;
      float M = -INFINITY;
#pragma unroll
      for (int k = 0; k < KS; k++) M = fmaxf(M, sm_m[(k * NTT + nt_) * 8 + cc]);
      float o = 0.f;
      if (M != -INFINITY) {
#pragma unroll
        for (int k = 0; k < KS; k++) {
          const int w = k * NTT + nt_;
          o += exp2f(sm_m[w * 8 + cc] - M) * sacc[(size_t)(w * 8 + cc) * 128 + dl];
        }
      }
      part[(((size_t)kvh * S_MAX + s) * ncols + col) * HD + rc * 128 + dl] = o;
    }
    if (rc == 0) __syncthreads();  // the next round overwrites sacc
  }
  if (threadIdx.x < NTT * 8) {
    const int col = threadIdx.x, nt_ = col >> 3, cc = col & 7;
    if (col < ncols) {
      float M = -INFINITY;
#pragma unroll
      for (int k = 0; k < KS; k++) M = fmaxf(M, sm_m[(k * NTT + nt_) * 8 + cc]);
      float L = 0.f;
      if (M != -INFINITY) {
#pragma unroll
        for (int k = 0; k < KS; k++) {
          const int w = k * NTT + nt_;
          L += exp2f(sm_m[w * 8 + cc] - M) * sm_l[w * 8 + cc];
        }
      }
      ml[((size_t)kvh * S_MAX + s) * ncols + col] = make_float2(M, L);
    }
  }
}

