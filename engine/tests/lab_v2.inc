// ---- v2 kernel: repacked (tile-contiguous, natural-order nibble pairs) weights, pointer-increment addressing, JIT activation loads
// Layout: unit(tile, kg) = 64 uint4 = 1 KB: [lane 0..31: rows g (16 B chunk t)] [32 + lane: rows g+8]; word nibble order: nib j = col 2j, nib j+4 = col 2j+1.
// Scales S2[(tile*nkg + kg)*8 + g] = bf16(row g) | bf16(row g+8) << 16.
namespace smk {
__device__ __forceinline__ uint32_t ld32(const uint32_t* p) { return __ldg(p); }

template <int NW, int KS, int U, int MINB, int NAT>
__global__ void __launch_bounds__(NW * 32, MINB)
    k_w4v2(const uint4* __restrict__ W, const uint32_t* __restrict__ S, const uint4* __restrict__ X, const smk_epi epi, int N, int K, int T) {
  constexpr int TPC = NW / KS;
  __shared__ float red[KS > 1 ? NW : 1][32][4];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  const int tin = warp / KS, ks = warp % KS;
  const int nkg = K >> 7, per = nkg / KS;
  const int groups = (N >> 4) / TPC;
  const int cnt = (groups - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x;
  const int total = cnt * per;
  const int xg = g < T ? g : T - 1;
  const uint4* xbase = X + (size_t)xg * (K >> 3) + ks * 16 + t * 4;
  auto tile_of = [&](int m) { return (((int)blockIdx.x + m * (int)gridDim.x) * TPC + tin); };

  const int rowu = K >> 5;
  const uint4* wp;
  const uint32_t* sp;
  const uint16_t* S16 = (const uint16_t*)S;
  uint4 wa[U], wb[U];
  uint32_t ss[U];
  int lm = 0, lk = 0;
  auto setp = [&](int m) {
    if (NAT) {
      const size_t row = (size_t)tile_of(m) * 16 + g;
      wp = W + row * rowu + ks * 4 + t;
      S16 = (const uint16_t*)S + row * nkg + ks;
    } else {
      const size_t u0 = (size_t)tile_of(m) * nkg + ks;
      wp = W + u0 * 64 + lane;
      sp = S + u0 * 8 + g;
    }
  };
  setp(0);
  auto issue = [&](int j) {
    if (NAT) {
      wa[j] = ldw(wp); wb[j] = ldw(wp + (size_t)8 * rowu);
      ss[j] = ld16(S16) | (ld16(S16 + (size_t)8 * nkg) << 16);
    } else {
      wa[j] = ldw(wp); wb[j] = ldw(wp + 32);
      ss[j] = ld32(sp);
    }
    if (++lk == per) {
      lk = 0; ++lm;
      setp(lm);
    } else if (NAT) {
      wp += KS * 4;
      S16 += KS;
    } else {
      wp += KS * 64;
      sp += KS * 8;
    }
  };
#pragma unroll
  for (int j = 0; j < U; j++)
    if (j < total) issue(j);
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  int ct = 0, cm = 0;
  const uint4* xp = xbase;
  for (int f = 0; f < total; f += U) {
#pragma unroll
    for (int j = 0; j < U; j++) {
      if (f + j < total) {
        const uint4 ca = wa[j], cb = wb[j];
        const uint32_t sv = ss[j];
        if (f + j + U < total) issue(j);
        uint4 xb[4];
#pragma unroll
        for (int u = 0; u < 4; u++) xb[u] = NAT == 1 ? perm_b(ldx(xp + u)) : ldx(xp + u);
        xp += KS * 16;
        float part[4] = {0.f, 0.f, 0.f, 0.f};
        const uint32_t wra[4] = {ca.x, ca.y, ca.z, ca.w}, wrb[4] = {cb.x, cb.y, cb.z, cb.w};
#pragma unroll
        for (int u = 0; u < 4; u++) {
          uint32_t pa[4], pb[4];
          dq_word(wra[u], pa);
          dq_word(wrb[u], pb);
          mma16816(part, pa[0], pb[0], pa[1], pb[1], xb[u].x, xb[u].y);
          mma16816(part, pa[2], pb[2], pa[3], pb[3], xb[u].z, xb[u].w);
        }
        const float fsa = bflo(sv), fsb = bfhi(sv);
        acc[0] = fmaf(fsa, part[0], acc[0]); acc[1] = fmaf(fsa, part[1], acc[1]);
        acc[2] = fmaf(fsb, part[2], acc[2]); acc[3] = fmaf(fsb, part[3], acc[3]);
        if (++ct == per) {
          ct = 0;
          xp = xbase;
          const int row = tile_of(cm) << 4;
          ++cm;
          if (KS > 1) {
            if (ks > 0) {
#pragma unroll
              for (int c = 0; c < 4; c++) red[warp][lane][c] = acc[c];
            }
            __syncthreads();
            if (ks == 0) {
#pragma unroll
              for (int q = 1; q < KS; q++)
#pragma unroll
                for (int c = 0; c < 4; c++) acc[c] += red[warp + q][lane][c];
              epilogue(acc, row, g, t, T, N, epi);
            }
            __syncthreads();
          } else {
            epilogue(acc, row, g, t, T, N, epi);
          }
          acc[0] = acc[1] = acc[2] = acc[3] = 0.f;
        }
      }
    }
  }
}

// packer: natural [N][K/8] u32 + bf16 [N][K/128]  ->  v2 layout
__global__ void k_pack_v2(const uint4* __restrict__ W, const uint16_t* __restrict__ S, uint4* __restrict__ dW, uint32_t* __restrict__ dS, int N, int K) {
  const int nkg = K >> 7, rowu = K >> 5, ntile = N >> 4;
  const size_t total = (size_t)ntile * nkg * 32;
  auto perm = [](uint32_t w) {
    uint32_t r = 0;
#pragma unroll
    for (int j = 0; j < 4; j++) r |= (((w >> (8 * j)) & 0xFu) << (4 * j)) | (((w >> (8 * j + 4)) & 0xFu) << (4 * (j + 4)));
    return r;
  };
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
    const int lane = i & 31;
    const size_t unit = i >> 5;
    const int kg = unit % nkg, tile = unit / nkg, g = lane >> 2, t = lane & 3;
    const size_t o = (size_t)(tile * 16 + g) * rowu + (size_t)kg * 4 + t;
    uint4 a = W[o], b = W[o + (size_t)8 * rowu];
    a.x = perm(a.x); a.y = perm(a.y); a.z = perm(a.z); a.w = perm(a.w);
    b.x = perm(b.x); b.y = perm(b.y); b.z = perm(b.z); b.w = perm(b.w);
    dW[unit * 64 + lane] = a;
    dW[unit * 64 + 32 + lane] = b;
    if (t == 0) dS[unit * 8 + g] = (uint32_t)S[(size_t)(tile * 16 + g) * nkg + kg] | ((uint32_t)S[(size_t)(tile * 16 + g + 8) * nkg + kg] << 16);
  }
}
}  // namespace smk

namespace smk {
// natural layout, nibble order permuted within each word only (nib j = col 2j, nib j+4 = col 2j+1)
__global__ void k_pack_perm(const uint32_t* __restrict__ W, uint32_t* __restrict__ D, size_t n) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    const uint32_t w = W[i];
    uint32_t r = 0;
#pragma unroll
    for (int j = 0; j < 4; j++) r |= (((w >> (8 * j)) & 0xFu) << (4 * j)) | (((w >> (8 * j + 4)) & 0xFu) << (4 * (j + 4)));
    D[i] = r;
  }
}
}
