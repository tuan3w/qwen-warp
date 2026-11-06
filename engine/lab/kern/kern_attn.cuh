// Part (iii) driver: sparse 64-token-block gather attention over the rotated int4 KV layout vs the engine's dense int4 path.  Included at the end of kern.cu.
#pragma once

// ---------------------------------------------------------------------------------------------------------------- synthetic cache through the engine's writer primitives
__device__ __forceinline__ uint64_t mix64(uint64_t z) {
  z += 0x9e3779b97f4a7c15ULL;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}
__device__ __forceinline__ float gauss_h(uint64_t key) {
  const uint64_t a = mix64(key), b = mix64(key ^ 0xabcdef12345ULL);
  const float u1 = ((a >> 40) + 0.5f) * (1.f / 16777216.f), u2 = ((b >> 40) + 0.5f) * (1.f / 16777216.f);
  return sqrtf(-2.f * logf(u1)) * cosf(6.2831853f * u2);
}
// grid (ceil(nctx / 8), 8 planes = K|V x kv head), 256 threads = 8 warps, one warp per (plane, token).  K: static per-channel lognormal profile x gaussian + a 64-token-block
// mean (so block selection changes the output); V: gaussian + block mean.  Written with kvrot::wht256 + kvrot::quant_store (the engine's writer), plus the exact sink / ring rows.
__global__ void __launch_bounds__(256) k_kvfill(uint8_t* kv, int ctx_max, int nctx) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int pos = blockIdx.x * 8 + warp, plane = blockIdx.y;
  if (pos >= nctx) return;
  const int blk = pos >> 6;
  float x[8];
#pragma unroll
  for (int i = 0; i < 8; i++) {
    const int d = lane + 32 * i;
    float v = gauss_h(((uint64_t)plane << 52) ^ ((uint64_t)pos << 12) ^ d);
    const float bm = 0.8f * gauss_h(0x5555000000000000ULL ^ ((uint64_t)plane << 40) ^ ((uint64_t)blk << 10) ^ d);
    if (plane < NKV) v = expf(0.6f * gauss_h(0x7777000000000000ULL ^ ((uint64_t)plane << 20) ^ d)) * v;
    x[i] = v + bm;
  }
  kvrot::wht256(x, lane);
  uint8_t* codes = kv + ((size_t)plane * ctx_max + pos) * (HD / 2);
  uint32_t* meta = reinterpret_cast<uint32_t*>(kv + (size_t)2 * NKV * ctx_max * (HD / 2)) + (size_t)plane * ctx_max + pos;
  kvrot::quant_store(x, lane, codes, meta);
  if (pos < kvrot::NSINK) kvrot::sink_store(x, lane, reinterpret_cast<uint16_t*>(kv + kvrot::sink_off(ctx_max)) + ((size_t)plane * kvrot::NSINK + pos) * HD);
  if (pos >= nctx - kvrot::RING_R) {
    uint16_t* ring = reinterpret_cast<uint16_t*>(kv + kvrot::ring_off(ctx_max)) + ((size_t)plane * kvrot::RING_R + (pos & (kvrot::RING_R - 1))) * HD;
#pragma unroll
    for (int i = 0; i < 8; i++) ring[lane + 32 * i] = __half_as_ushort(__float2half_rn(x[i]));
  }
}

// ---------------------------------------------------------------------------------------------------------------- fp32 reference over an explicit selection
// one warp per (query row t, q head h): exact sink rows (keys 0..3), the selected 64-token blocks of the 4-bit region (kvrot::dequant_row, the decoder gains), the exact ring keys
// (causal), softmax in fp32, un-rotate, gate.  rowm[kvh][blk] bit t = row t attends block blk.
__global__ void k_ref_attn(const uint8_t* __restrict__ kv, int ctx_max, int pos0, int T, const uint16_t* __restrict__ qbuf, const uint16_t* __restrict__ qproj, const uint8_t* __restrict__ rowm,
                           int nbt, uint16_t* __restrict__ out, float* __restrict__ dbgml, const uint8_t* __restrict__ tokm = nullptr, int tokstride = 0) {
  const int lane = threadIdx.x, t = blockIdx.x / NH, h = blockIdx.x % NH, kvh = h / GRP;
  const int ctx = max(pos0 + T - kvrot::RING_W, 0);
  float q[8];
#pragma unroll
  for (int i = 0; i < 8; i++) q[i] = __half2float(__ushort_as_half(qbuf[((size_t)t * NH + h) * HD + lane + 32 * i]));
  const uint8_t* kc = kv + (size_t)(0 * NKV + kvh) * ctx_max * (HD / 2);
  const uint8_t* vc = kv + (size_t)(1 * NKV + kvh) * ctx_max * (HD / 2);
  const uint32_t* km = reinterpret_cast<const uint32_t*>(kv + (size_t)2 * NKV * ctx_max * (HD / 2)) + (size_t)(0 * NKV + kvh) * ctx_max;
  const uint32_t* vm = reinterpret_cast<const uint32_t*>(kv + (size_t)2 * NKV * ctx_max * (HD / 2)) + (size_t)(1 * NKV + kvh) * ctx_max;
  const uint16_t* sink = reinterpret_cast<const uint16_t*>(kv + kvrot::sink_off(ctx_max));
  const uint16_t* ring = reinterpret_cast<const uint16_t*>(kv + kvrot::ring_off(ctx_max));
  float m = -INFINITY, l = 0.f, acc[8];
#pragma unroll
  for (int i = 0; i < 8; i++) acc[i] = 0.f;
  auto step = [&](const float (&yk)[8], const float (&yv)[8]) {
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < 8; i++) s = fmaf(yk[i], q[i], s);
    s = warp_sum(s) * 0.0625f;
    const float mn = fmaxf(m, s), a = expf(m - mn), p = expf(s - mn);
    l = l * a + p;
#pragma unroll
    for (int i = 0; i < 8; i++) acc[i] = acc[i] * a + p * yv[i];
    m = mn;
  };
  float yk[8], yv[8];
  if (ctx > 0) {
    for (int k = 0; k < kvrot::NSINK; k++) {
#pragma unroll
      for (int i = 0; i < 8; i++) {
        yk[i] = __half2float(__ushort_as_half(sink[((size_t)(0 * NKV + kvh) * kvrot::NSINK + k) * HD + lane + 32 * i]));
        yv[i] = __half2float(__ushort_as_half(sink[((size_t)(1 * NKV + kvh) * kvrot::NSINK + k) * HD + lane + 32 * i]));
      }
      step(yk, yv);
    }
  }
  for (int pos = kvrot::NSINK; pos < ctx; pos++) {
    const int blk = pos >> 6;
    if (tokm) { if (!tokm[(size_t)kvh * tokstride + pos]) continue; }
    else if (!((rowm[(size_t)kvh * nbt + blk] >> t) & 1)) { pos = ((blk + 1) << 6) - 1; continue; }
    kvrot::dequant_row(kc + (size_t)pos * (HD / 2), km[pos], lane, kvrot::LOG2_KGAIN, yk);
    kvrot::dequant_row(vc + (size_t)pos * (HD / 2), vm[pos], lane, kvrot::LOG2_VGAIN, yv);
    step(yk, yv);
  }
  for (int pos = ctx; pos <= pos0 + t; pos++) {
    const int slot = pos & (kvrot::RING_R - 1);
#pragma unroll
    for (int i = 0; i < 8; i++) {
      yk[i] = __half2float(__ushort_as_half(ring[((size_t)(0 * NKV + kvh) * kvrot::RING_R + slot) * HD + lane + 32 * i]));
      yv[i] = __half2float(__ushort_as_half(ring[((size_t)(1 * NKV + kvh) * kvrot::RING_R + slot) * HD + lane + 32 * i]));
    }
    step(yk, yv);
  }
  if (dbgml && lane == 0) { dbgml[(t * NH + h) * 2] = m; dbgml[(t * NH + h) * 2 + 1] = l; }
  float o[8];
#pragma unroll
  for (int i = 0; i < 8; i++) o[i] = acc[i] / l;
  kvrot::wht256(o, lane);
#pragma unroll
  for (int i = 0; i < 8; i++) {
    const int d = lane + 32 * i;
    const float gt = __uint_as_float((uint32_t)qproj[(size_t)t * QG_STRIDE + h * 2 * HD + HD + d] << 16);
    out[((size_t)t * NH + h) * HD + d] = f2bf(o[i] / (1.f + expf(-gt)));
  }
}

// ---------------------------------------------------------------------------------------------------------------- launchers
static int g_attn_dbg = 0;
static int g_attn_bps = 0;  // 0 = engine split geometry
template <int NTG, int NG, int DS, int KS>
static cudaError_t launch_split_i4s(int T, const void* kv, int ctx_max, const int* pos0, const uint16_t* qbuf, float2* ml, float* part, const uint32_t* lst, int lst_stride, cudaStream_t st) {
  constexpr int NTT = NTG * NG;
  constexpr size_t merge_b = ((size_t)KS * NTT * 8 * 128 + (size_t)KS * NTT * 16) * sizeof(float);
  constexpr size_t loop_b = ((size_t)NTT * 16 * 64 + (size_t)KS * NG * 2 * DS * NTG * 4 * 32) * sizeof(float);
  constexpr size_t smem = merge_b > loop_b ? merge_b : loop_b;
  static bool attr_set = false;
  if (!attr_set) {
    cudaError_t e = cudaFuncSetAttribute(attn_split_i4s_kernel<NTG, NG, DS, KS>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    if (e != cudaSuccess) return e;
    attr_set = true;
  }
  attn_split_i4s_kernel<NTG, NG, DS, KS><<<dim3(S_MAX, NKV), 32 * DS * KS * NG, smem, st>>>(T, kv, ctx_max, pos0, qbuf, ml, part, lst, lst_stride, g_attn_dbg, g_attn_bps);
  return cudaGetLastError();
}


template <int NTG, int NG, int DS, int KS>
static cudaError_t launch_split_i4t(int T, const void* kv, int ctx_max, const int* pos0, const uint16_t* qbuf, float2* ml, float* part, const uint32_t* lst, int lst_stride, int bps, cudaStream_t st) {
  constexpr int NTT = NTG * NG;
  constexpr size_t merge_b = ((size_t)KS * NTT * 8 * 128 + (size_t)KS * NTT * 16) * sizeof(float);
  constexpr size_t loop_b = ((size_t)NTT * 16 * 64 + (size_t)KS * NG * 2 * DS * NTG * 4 * 32) * sizeof(float);
  constexpr size_t smem = merge_b > loop_b ? merge_b : loop_b;
  static bool attr_set = false;
  if (!attr_set) {
    cudaError_t e = cudaFuncSetAttribute(attn_split_i4t_kernel<NTG, NG, DS, KS>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    if (e != cudaSuccess) return e;
    attr_set = true;
  }
  attn_split_i4t_kernel<NTG, NG, DS, KS><<<dim3(S_MAX, NKV), 32 * DS * KS * NG, smem, st>>>(T, kv, ctx_max, pos0, qbuf, ml, part, lst, lst_stride, 0, bps);
  return cudaGetLastError();
}
struct AttnBufs {
  int ctx_max = 0, nctx = 0, nc = 0;
  size_t kvbytes = 0;
  std::vector<uint8_t*> kv;
  void* ws = nullptr;
  int* pos0 = nullptr;
  uint16_t *qproj = nullptr, *kk = nullptr, *vv = nullptr, *qn = nullptr, *kn = nullptr;
  uint16_t *out_dense = nullptr, *out_sp = nullptr, *out_ref = nullptr;
  uint8_t* rowm = nullptr;
  uint32_t* lst = nullptr;  // [nc][4][stride]
  int stride = 0, nbt = 0;
};

// dense: the engine's own split launch (launch_multi, the config table of `run`) + reduce, WITHOUT the prep kernel; engine = swiftmk_attn_step (prep + split + reduce)
static cudaError_t dense_split_reduce(const AttnBufs& A, int T, int c, cudaStream_t st) {
  char* w = reinterpret_cast<char*>(A.ws);
  uint16_t* qbuf = reinterpret_cast<uint16_t*>(w + WS_Q);
  float2* ml = reinterpret_cast<float2*>(w + WS_ML);
  float* part = reinterpret_cast<float*>(w + WS_PART);
  cudaError_t e;
  if (T == 1) e = launch_multi<false, true, 1, ATTN_KS1, ATTN_VC1>(T, A.kv[c], A.ctx_max, 0.f, 0.f, A.pos0, qbuf, ml, part, st);
  else e = launch_multi<false, true, 6, ATTN_KS8, ATTN_VC8>(T, A.kv[c], A.ctx_max, 0.f, 0.f, A.pos0, qbuf, ml, part, st);
  if (e != cudaSuccess) return e;
  attn_reduce_kernel<true><<<dim3(GRP * T, NKV), HD, 0, st>>>(T, (const __nv_bfloat16*)A.qproj, 1.f, A.pos0, ml, part, (__nv_bfloat16*)A.out_dense, A.kv[c], A.ctx_max, qbuf);
  return cudaGetLastError();
}
static cudaError_t sparse_split_only(const AttnBufs& A, int T, int c, cudaStream_t st) {
  char* w = reinterpret_cast<char*>(A.ws);
  uint16_t* qbuf = reinterpret_cast<uint16_t*>(w + WS_Q);
  float2* ml = reinterpret_cast<float2*>(w + WS_ML);
  float* part = reinterpret_cast<float*>(w + WS_PART);
  const uint32_t* lst = A.lst + (size_t)c * NKV * A.stride;
  if (T == 1) return launch_split_i4s<1, 1, 2, 4>(T, A.kv[c], A.ctx_max, A.pos0, qbuf, ml, part, lst, A.stride, st);
  return launch_split_i4s<3, 2, 2, 2>(T, A.kv[c], A.ctx_max, A.pos0, qbuf, ml, part, lst, A.stride, st);
}
static cudaError_t sparse_reduce_only(const AttnBufs& A, int T, int c, cudaStream_t st, uint16_t* out) {
  char* w = reinterpret_cast<char*>(A.ws);
  uint16_t* qbuf = reinterpret_cast<uint16_t*>(w + WS_Q);
  float2* ml = reinterpret_cast<float2*>(w + WS_ML);
  float* part = reinterpret_cast<float*>(w + WS_PART);
  const uint32_t* lst = A.lst + (size_t)c * NKV * A.stride;
  attn_reduce_s_kernel<true><<<dim3(GRP * T, NKV), HD, 0, st>>>(T, (const __nv_bfloat16*)A.qproj, 1.f, A.pos0, ml, part, (__nv_bfloat16*)out, A.kv[c], A.ctx_max, qbuf, lst, A.stride, g_attn_bps);
  return cudaGetLastError();
}
static cudaError_t sparse_split_reduce(const AttnBufs& A, int T, int c, cudaStream_t st, uint16_t* out) {
  cudaError_t e = sparse_split_only(A, T, c, st);
  if (e != cudaSuccess) return e;
  return sparse_reduce_only(A, T, c, st, out);
}


struct Sel {
  std::vector<uint32_t> lst;  // [4][stride]
  std::vector<uint8_t> rowm;  // [4][nbt]
  double union_frac = 0, row_frac = 0;
  int ub[4] = {0, 0, 0, 0};
};
static void dbg_ml(const AttnBufs& A, int T, const char* label, const float* refml) {
  const int ncols = GRP * T;
  std::vector<float2> h((size_t)NKV * S_MAX * ncols);
  CK(cudaStreamSynchronize(g_st));
  CK(cudaMemcpy(h.data(), (char*)A.ws + WS_ML, h.size() * sizeof(float2), cudaMemcpyDeviceToHost));
  printf("   dbg_ml %-14s", label);
  for (int col = 0; col < 3; col++) {  // kv head 0, columns 0..2 = q heads 0..2 of row 0
    double M = -1e30; for (int sp = 0; sp < S_MAX; sp++) M = std::max<double>(M, h[((size_t)0 * S_MAX + sp) * ncols + col].x);
    double L = 0; for (int sp = 0; sp < S_MAX; sp++) { float2 v = h[((size_t)0 * S_MAX + sp) * ncols + col]; if (v.x > -1e30) L += std::exp2(v.x - M) * v.y; }
    printf(" | col%d M(nat)=%.4f L=%.4f", col, M / 1.4426950408889634, L);
  }
  if (refml) printf(" || ref:   L = %.4f %.4f %.4f  M = %.4f", refml[1], refml[3], refml[5], refml[0]);
  printf("\n");
}
// KERN_DBG=1: sparse kernel vs the fp32 reference on a few hand-made selections + row-mask / L(mass) diagnostics (T = 1 or 8)
static void attn_debug(AttnBufs& A, int T, int p0, uint16_t* qbuf, const std::vector<uint16_t>& h_engine) {
  float* dml = (float*)dmalloc((size_t)T * NH * 2 * 4);
  std::vector<float> hml((size_t)T * NH * 2);
  const size_t nout = (size_t)T * NH * HD;
  auto fetch = [&](uint16_t* d) { std::vector<uint16_t> h(nout); CK(cudaStreamSynchronize(g_st)); CK(cudaMemcpy(h.data(), d, nout * 2, cudaMemcpyDeviceToHost)); return h; };
  struct P { const char* name; std::vector<int> blocks; int dbg; };
  std::vector<int> all(A.nbt); std::iota(all.begin(), all.end(), 0);
  std::vector<P> ps = {{"blk0 only", {0}, 0}, {"blk0+1", {0, 1}, 0}, {"blk0+400", {0, 400}, 0}, {"blk0..24", {}, 0}, {"all", all, 0}, {"all, rowmask off", all, 1}};
  for (int i = 0; i < 25; i++) ps[3].blocks.push_back(i);
  for (auto& pp : ps) {
    Sel sel; sel.lst.assign((size_t)NKV * A.stride, 0); sel.rowm.assign((size_t)NKV * A.nbt, 0);
    for (int h = 0; h < NKV; h++) {
      int cnt = 0;
      for (int b : pp.blocks) { sel.rowm[(size_t)h * A.nbt + b] = (uint8_t)((1u << T) - 1); sel.lst[(size_t)h * A.stride + 1 + cnt++] = (uint32_t)b | ((uint32_t)((1u << T) - 1) << 16); }
      sel.lst[(size_t)h * A.stride] = cnt;
    }
    for (int c = 0; c < A.nc; c++) CK(cudaMemcpy(A.lst + (size_t)c * NKV * A.stride, sel.lst.data(), sel.lst.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(A.rowm, sel.rowm.data(), sel.rowm.size(), cudaMemcpyHostToDevice));
    g_attn_dbg = pp.dbg;
    CK(sparse_split_reduce(A, T, 0, g_st, A.out_sp));
    g_attn_dbg = 0;
    k_ref_attn<<<T * NH, 32, 0, g_st>>>(A.kv[0], A.ctx_max, p0, T, qbuf, A.qproj, A.rowm, A.nbt, A.out_ref, dml);
    CK(cudaGetLastError());
    const std::vector<uint16_t> a = fetch(A.out_sp), r = fetch(A.out_ref);
    CK(cudaMemcpy(hml.data(), dml, hml.size() * 4, cudaMemcpyDeviceToHost));
    const Err e = compare_bf16(a, r);
    printf("  [dbg T=%d] %-18s out: max/rms %.3e, %.1f%% differ | sp[0..3] %.4f %.4f %.4f %.4f  ref %.4f %.4f %.4f %.4f\n", T, pp.name, e.max_over_rms, e.frac_diff * 100, bf2f_h(a[0]), bf2f_h(a[1]), bf2f_h(a[2]), bf2f_h(a[3]),
           bf2f_h(r[0]), bf2f_h(r[1]), bf2f_h(r[2]), bf2f_h(r[3]));
    dbg_ml(A, T, "sparse", hml.data());
  }
  CK(dense_split_reduce(A, T, 0, g_st));
  const Err e = compare_bf16(fetch(A.out_dense), h_engine);
  printf("  [dbg T=%d] dense kernel re-run vs first engine output: max/rms %.2e\n", T, e.max_over_rms);
  dbg_ml(A, T, "dense", nullptr);
  cudaFree(dml);
}

// f = fraction of the nbt 64-token blocks per row (block 0 = sink block always kept); indep: every query row draws its own subset (the kernel reads the union, masks the other rows)
static Sel make_sel(int T, int nbt, int stride, double f, bool indep, std::mt19937_64& rng) {
  Sel s;
  s.lst.assign((size_t)NKV * stride, 0);
  s.rowm.assign((size_t)NKV * nbt, 0);
  const int nsel = std::min(nbt, std::max(1, (int)std::lround(f * nbt)));
  double rowsum = 0, unionsum = 0;
  for (int h = 0; h < NKV; h++) {
    std::vector<int> perm(nbt - 1); std::iota(perm.begin(), perm.end(), 1);
    std::vector<int> pick;
    for (int r = 0; r < T; r++) {
      if (r == 0 || indep) { std::shuffle(perm.begin(), perm.end(), rng); pick.assign(perm.begin(), perm.begin() + (nsel - 1)); }
      s.rowm[(size_t)h * nbt + 0] |= 1u << r;
      for (int b : pick) s.rowm[(size_t)h * nbt + b] |= 1u << r;
    }
    int cnt = 0;
    for (int b = 0; b < nbt; b++) if (s.rowm[(size_t)h * nbt + b]) s.lst[(size_t)h * stride + 1 + cnt++] = (uint32_t)b | ((uint32_t)s.rowm[(size_t)h * nbt + b] << 16);
    s.lst[(size_t)h * stride] = cnt;
    s.ub[h] = cnt;
    unionsum += cnt; rowsum += nsel;
  }
  s.union_frac = unionsum / ((double)NKV * nbt);
  s.row_frac = rowsum / ((double)NKV * nbt);
  return s;
}


// ---------------------------------------------------------------------------------------------------------------- token-granular gather (--tok)
// One selection per KV head shared by all rows and the 6 q heads: sorted token list (0..3 = exact sink keys first).  random: aligned groups of g tokens at random; clustered: runs of g at random
// unaligned starts.  Bytes needed per token and KV head: 128 B code + 4 B meta for K and V = 264 B; sector bytes count the 32 B sectors the 4 B meta words really touch.
static void run_tok(AttnBufs& A, int T, int p0, uint16_t* qbuf, std::mt19937_64& rng, int nctx, bool full) {
  const int ctxq = nctx - kvrot::RING_W;
  const int TS = 16384 + 4 + 16;  // list stride (exactly ntok selected tokens + 4 sink keys)
  uint32_t* tl = (uint32_t*)dmalloc((size_t)A.nc * NKV * TS * 4);
  uint8_t* tokmd = (uint8_t*)dmalloc((size_t)NKV * ctxq);
  const size_t nout = (size_t)T * NH * HD;
  auto fetch = [&](uint16_t* d) { std::vector<uint16_t> h(nout); CK(cudaStreamSynchronize(g_st)); CK(cudaMemcpy(h.data(), d, nout * 2, cudaMemcpyDeviceToHost)); return h; };
  const double FIXED_HEAD = 2.0 * kvrot::NSINK * HD * 2 + 2.0 * kvrot::RING_W * HD * 2;
  const double dense_bytes = NKV * ((double)A.nbt * 2.0 * (64 * 128 + 64 * 4) + FIXED_HEAD) + (double)T * NH * HD * 2;
  std::vector<int> ntoks = nctx >= 100000 ? std::vector<int>{4096, 8192, 16384} : std::vector<int>{1024, 2048, 4096};
  if (!full) ntoks.resize(1);
  const std::vector<int> gs = full ? std::vector<int>{1, 2, 4, 8, 16, 32, 64} : std::vector<int>{1, 4, 16, 64};
  const int bps = 64;
  for (int ntok : ntoks) for (int mode = 0; mode < 2; mode++) for (int g : gs) {
    std::vector<uint8_t> m((size_t)NKV * ctxq, 0);
    std::vector<uint32_t> lst((size_t)NKV * TS, 0);
    double bytes_need = 0, bytes_sec = 0;
    for (int h = 0; h < NKV; h++) {
      uint8_t* mh = m.data() + (size_t)h * ctxq;
      int cnt = 0;
      if (mode == 0) {  // aligned groups, first group start >= 32 (clear of the sink)
        const int g0 = (32 + g - 1) / g, g1 = ctxq / g;
        std::vector<int> p; for (int q = g0; q < g1; q++) p.push_back(q);
        std::shuffle(p.begin(), p.end(), rng);
        for (int q = 0; q < ntok / g; q++) for (int i = 0; i < g; i++) { mh[p[q] * g + i] = 1; cnt++; }
      } else {
        while (cnt < ntok) {
          const int st = 4 + (int)(rng() % (uint64_t)(ctxq - g - 4));
          for (int i = 0; i < g && cnt < ntok; i++) if (!mh[st + i]) { mh[st + i] = 1; cnt++; }
        }
      }
      uint32_t* lh = lst.data() + (size_t)h * TS;
      int n = 0;
      for (int q = 0; q < 4; q++) lh[1 + n++] = q;
      for (int q = 4; q < ctxq; q++) if (mh[q]) lh[1 + n++] = q;
      lh[0] = n;
      // sectors: K / V rows 128 B each, meta words 4 B -> distinct 32 B sectors
      std::vector<uint8_t> sec((ctxq + 7) / 8 + 1, 0);
      for (int q = 0; q < n; q++) sec[lh[1 + q] >> 3] = 1;
      int nsec = 0; for (auto v : sec) nsec += v;
      bytes_need += (double)n * 264.0;
      bytes_sec += (double)n * 256.0 + 2.0 * nsec * 32.0;
    }
    for (int c = 0; c < A.nc; c++) CK(cudaMemcpy(tl + (size_t)c * NKV * TS, lst.data(), lst.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(tokmd, m.data(), m.size(), cudaMemcpyHostToDevice));
    auto run = [&](int c, cudaStream_t st, uint16_t* out) -> cudaError_t {
      char* w = reinterpret_cast<char*>(A.ws);
      uint16_t* qb = reinterpret_cast<uint16_t*>(w + WS_Q);
      float2* ml = reinterpret_cast<float2*>(w + WS_ML);
      float* part = reinterpret_cast<float*>(w + WS_PART);
      const uint32_t* l = tl + (size_t)c * NKV * TS;
      cudaError_t e = T == 1 ? launch_split_i4t<1, 1, 2, 4>(T, A.kv[c], A.ctx_max, A.pos0, qb, ml, part, l, TS, bps, st) : launch_split_i4t<3, 2, 2, 2>(T, A.kv[c], A.ctx_max, A.pos0, qb, ml, part, l, TS, bps, st);
      if (e != cudaSuccess) return e;
      attn_reduce_s_kernel<true><<<dim3(GRP * T, NKV), HD, 0, st>>>(T, (const __nv_bfloat16*)A.qproj, 1.f, A.pos0, ml, part, (__nv_bfloat16*)out, A.kv[c], A.ctx_max, qb, l, TS, bps);
      return cudaGetLastError();
    };
    CK(run(0, g_st, A.out_sp));
    k_ref_attn<<<T * NH, 32, 0, g_st>>>(A.kv[0], A.ctx_max, p0, T, qbuf, A.qproj, A.rowm, A.nbt, A.out_ref, nullptr, tokmd, ctxq);
    CK(cudaGetLastError());
    const Err er = compare_bf16(fetch(A.out_sp), fetch(A.out_ref));
    const double d0 = time_graph([&](int i, cudaStream_t s) { dense_split_reduce(A, T, i % A.nc, s); }, 30);
    const double ms = time_graph([&](int i, cudaStream_t s) { run(i % A.nc, s, A.out_sp); });
    const double d1 = time_graph([&](int i, cudaStream_t s) { dense_split_reduce(A, T, i % A.nc, s); }, 30);
    const double dl = 0.5 * (d0 + d1), dgbs = dense_bytes / dl * 1e-6;
    const double bn = bytes_need + NKV * FIXED_HEAD + (double)T * NH * HD * 2, bs = bytes_sec + NKV * FIXED_HEAD + (double)T * NH * HD * 2;
    const double g1 = bn / ms * 1e-6, g2 = bs / ms * 1e-6;
    printf("  ctx %6d T %d %-9s ntok %5d g %2d | dense %7.1f us (%4.0f GB/s) | gather %7.1f us | GB/s needed %5.0f sector %5.0f | frac of dense GB/s %.2f | speedup %5.2fx (ideal %.1fx) | 16L save %.2f ms | err %.1e\n", nctx, T,
           mode ? "clustered" : "random", ntok, g, dl * 1e3, dgbs, ms * 1e3, g1, g2, g1 / dgbs, dl / ms, dense_bytes / bn, 16 * (dl - ms), er.max_over_rms);
    ja.rec("{\"what\":\"tok\",\"ctx\":%d,\"T\":%d,\"mode\":\"%s\",\"ntok\":%d,\"g\":%d,\"dense_us\":%.2f,\"dense_GBps\":%.1f,\"gather_us\":%.2f,\"bytes_needed\":%.0f,\"bytes_sector\":%.0f,\"GBps_needed\":%.1f,\"GBps_sector\":%.1f,"
           "\"frac_dense_GBps\":%.4f,\"speedup\":%.4f,\"ideal_bytes_ratio\":%.3f,\"saving_16_layers_ms\":%.4f,\"err_max_over_rms\":%.3e}", nctx, T, mode ? "clustered" : "random", ntok, g, dl * 1e3, dgbs, ms * 1e3, bn, bs, g1, g2,
           g1 / dgbs, dl / ms, dense_bytes / bn, 16 * (dl - ms), er.max_over_rms);
  }
  cudaFree(tl); cudaFree(tokmd);
}

static void run_attn(int nctx, const std::vector<int>& Ts, const std::vector<double>& fs, bool full, bool tokmode = false) {
  AttnBufs A;
  A.nctx = nctx; A.ctx_max = nctx; A.nc = 4;
  A.kvbytes = swiftmk_attn_kv_bytes(A.ctx_max, 2);
  const int T_MAXQ = 8;
  for (int c = 0; c < A.nc; c++) A.kv.push_back((uint8_t*)dmalloc(A.kvbytes));
  A.ws = dmalloc(swiftmk_attn_ws_bytes());
  A.pos0 = (int*)dmalloc(4);
  A.qproj = (uint16_t*)dmalloc((size_t)T_MAXQ * QG_STRIDE * 2);
  A.kk = (uint16_t*)dmalloc((size_t)T_MAXQ * 1024 * 2); A.vv = (uint16_t*)dmalloc((size_t)T_MAXQ * 1024 * 2);
  A.qn = (uint16_t*)dmalloc(256 * 2); A.kn = (uint16_t*)dmalloc(256 * 2);
  A.out_dense = (uint16_t*)dmalloc((size_t)T_MAXQ * NH * HD * 2); A.out_sp = (uint16_t*)dmalloc((size_t)T_MAXQ * NH * HD * 2); A.out_ref = (uint16_t*)dmalloc((size_t)T_MAXQ * NH * HD * 2);
  const int ctxq = nctx - kvrot::RING_W;  // pos0 + T - RING_W for pos0 = nctx - T
  A.nbt = (ctxq + 63) / 64;
  A.stride = (A.nbt + 1 + 7) & ~7;
  A.rowm = (uint8_t*)dmalloc((size_t)NKV * A.nbt);
  A.lst = (uint32_t*)dmalloc((size_t)A.nc * NKV * A.stride * 4);
  std::mt19937_64 rng(777);
  std::normal_distribution<float> nd(0.f, 1.f);
  {
    std::vector<uint16_t> h((size_t)T_MAXQ * QG_STRIDE); for (auto& v : h) v = f2bf_h(nd(rng)); CK(cudaMemcpy(A.qproj, h.data(), h.size() * 2, cudaMemcpyHostToDevice));
    h.resize((size_t)T_MAXQ * 1024); for (auto& v : h) v = f2bf_h(nd(rng)); CK(cudaMemcpy(A.kk, h.data(), h.size() * 2, cudaMemcpyHostToDevice));
    for (auto& v : h) v = f2bf_h(nd(rng)); CK(cudaMemcpy(A.vv, h.data(), h.size() * 2, cudaMemcpyHostToDevice));
    h.resize(256); for (auto& v : h) v = f2bf_h(0.1f * nd(rng)); CK(cudaMemcpy(A.qn, h.data(), 512, cudaMemcpyHostToDevice));
    for (auto& v : h) v = f2bf_h(0.1f * nd(rng)); CK(cudaMemcpy(A.kn, h.data(), 512, cudaMemcpyHostToDevice));
  }
  k_kvfill<<<dim3((nctx + 7) / 8, 2 * NKV), 256, 0, g_st>>>(A.kv[0], A.ctx_max, nctx);
  CK(cudaGetLastError());
  CK(cudaStreamSynchronize(g_st));
  const double KVB_BLOCK = 2.0 * (64 * 128 + 64 * 4);          // K + V codes + meta of one 64-token block of one kv head
  const double FIXED_HEAD = 2.0 * kvrot::NSINK * HD * 2 + 2.0 * kvrot::RING_W * HD * 2;  // exact sink + ring rows read per kv head
  printf("\n[attn ctx=%d] ctx_max=%d, 4-bit region %d tokens = %d blocks of 64, KV copy %.1f MB x %d copies, dense per layer = %.1f MB (+ %.2f MB sink/ring)\n", nctx, A.ctx_max, ctxq, A.nbt, A.kvbytes * 1e-6, A.nc,
         NKV * A.nbt * KVB_BLOCK * 1e-6, NKV * FIXED_HEAD * 1e-6);
  printf("  %-3s %-8s %5s %6s %7s | %8s %7s %9s | %8s %7s %8s %7s %7s %8s | %s\n", "T", "mode", "f%", "row%", "union%", "dense_us", "GB/s", "(engine)", "sparse_us", "GB/s*", "frac", "speedup", "ideal", "16L save", "err vs ref (max/rms; %diff)");

  for (int T : Ts) {
    const int p0 = nctx - T;
    CK(cudaMemcpy(A.pos0, &p0, 4, cudaMemcpyHostToDevice));
    // engine entry once: writes rows p0..p0+T-1 into copy 0, leaves the rotated fp16 q in the workspace
    CK(swiftmk_attn_step(T, A.qproj, A.kk, A.vv, A.qn, A.kn, A.kv[0], A.ctx_max, 2, 0.f, 0.f, A.pos0, A.out_dense, A.ws, g_st));
    for (int c = 1; c < A.nc; c++) CK(cudaMemcpyAsync(A.kv[c], A.kv[0], A.kvbytes, cudaMemcpyDeviceToDevice, g_st));
    CK(cudaStreamSynchronize(g_st));
    uint16_t* qbuf = reinterpret_cast<uint16_t*>(A.ws);
    const size_t nout = (size_t)T * NH * HD;
    auto fetch = [&](uint16_t* d) { std::vector<uint16_t> h(nout); CK(cudaStreamSynchronize(g_st)); CK(cudaMemcpy(h.data(), d, nout * 2, cudaMemcpyDeviceToHost)); return h; };
    std::vector<uint16_t> h_engine = fetch(A.out_dense);
    // reference with every block selected vs the engine's dense output: validates the fill + reference
    {
      std::vector<uint8_t> all((size_t)NKV * A.nbt, 0xFF);
      CK(cudaMemcpy(A.rowm, all.data(), all.size(), cudaMemcpyHostToDevice));
      k_ref_attn<<<T * NH, 32, 0, g_st>>>(A.kv[0], A.ctx_max, p0, T, qbuf, A.qproj, A.rowm, A.nbt, A.out_ref, nullptr);
      CK(cudaGetLastError());
      const Err e = compare_bf16(fetch(A.out_ref), h_engine);
      printf("  [T=%d] fp32 reference (all blocks) vs engine dense int4 path: max|d|/rms = %.2e, %.1f%% elements differ (bf16 rounding of a different accumulation order)\n", T, e.max_over_rms, e.frac_diff * 100);
      ja.rec("{\"what\":\"ref_vs_engine\",\"ctx\":%d,\"T\":%d,\"max_over_rms\":%.3e,\"frac_diff\":%.4f}", nctx, T, e.max_over_rms, e.frac_diff);
    }
    // dense timings: engine entry (prep + split + reduce) and split + reduce only
    const double ms_engine = time_graph([&](int i, cudaStream_t s) {
      swiftmk_attn_step(T, A.qproj, A.kk, A.vv, A.qn, A.kn, A.kv[i % A.nc], A.ctx_max, 2, 0.f, 0.f, A.pos0, A.out_dense, A.ws, s);
    });
    const double ms_dense = time_graph([&](int i, cudaStream_t s) { dense_split_reduce(A, T, i % A.nc, s); });
    const double dense_bytes = NKV * (A.nbt * KVB_BLOCK + FIXED_HEAD) + (double)T * NH * HD * 2;
    const double dense_gbs = dense_bytes / ms_dense * 1e-6;
    printf("  [T=%d] dense: engine swiftmk_attn_step (prep+split+reduce) %.1f us | split+reduce %.1f us = %.0f GB/s on %.1f MB | prep ~%.1f us\n", T, ms_engine * 1e3, ms_dense * 1e3, dense_gbs, dense_bytes * 1e-6, (ms_engine - ms_dense) * 1e3);
    ja.rec("{\"what\":\"dense\",\"ctx\":%d,\"T\":%d,\"engine_step_us\":%.2f,\"split_reduce_us\":%.2f,\"bytes\":%.0f,\"GBps\":%.1f}", nctx, T, ms_engine * 1e3, ms_dense * 1e3, dense_bytes, dense_gbs);

    if (getenv("KERN_DBG")) attn_debug(A, T, p0, qbuf, h_engine);
    if (tokmode) { run_tok(A, T, p0, qbuf, rng, nctx, full); continue; }
    // sparse
    struct Case { double f; bool indep; };
    std::vector<Case> cases;
    for (double f : fs) { cases.push_back({f, false}); if (T > 1 && f < 1.0 && f > 0.0) cases.push_back({f, true}); }
    for (const Case& cs_ : cases) {
      Sel sel = make_sel(T, A.nbt, A.stride, cs_.f, cs_.indep, rng);
      for (int c = 0; c < A.nc; c++) CK(cudaMemcpy(A.lst + (size_t)c * NKV * A.stride, sel.lst.data(), sel.lst.size() * 4, cudaMemcpyHostToDevice));
      CK(cudaMemcpy(A.rowm, sel.rowm.data(), sel.rowm.size(), cudaMemcpyHostToDevice));
      CK(sparse_split_reduce(A, T, 0, g_st, A.out_sp));
      k_ref_attn<<<T * NH, 32, 0, g_st>>>(A.kv[0], A.ctx_max, p0, T, qbuf, A.qproj, A.rowm, A.nbt, A.out_ref, nullptr);
      CK(cudaGetLastError());
      const std::vector<uint16_t> h_sp = fetch(A.out_sp), h_ref = fetch(A.out_ref);
      const Err er = compare_bf16(h_sp, h_ref);
      Err ed; if (cs_.f >= 1.0) ed = compare_bf16(h_sp, h_engine);
      const double d0 = time_graph([&](int i, cudaStream_t s) { dense_split_reduce(A, T, i % A.nc, s); }, 40);  // local dense baseline (drift)
      g_attn_bps = 0;
      const double ms = time_graph([&](int i, cudaStream_t s) { sparse_split_reduce(A, T, i % A.nc, s, A.out_sp); });
      // tuned split count: ceil(kept blocks / bps) splits + the list-driven reduce; best of a few bps, each checked against the reference
      double ms_t = 1e9; int bps_t = 0; Err er_t;
      for (int bps : (full ? std::vector<int>{1, 2, 4, 8} : std::vector<int>{2, 4})) {
        g_attn_bps = bps;
        CK(sparse_split_reduce(A, T, 0, g_st, A.out_sp));
        const Err e2 = compare_bf16(fetch(A.out_sp), h_ref);
        const double m2 = time_graph([&](int i, cudaStream_t s) { sparse_split_reduce(A, T, i % A.nc, s, A.out_sp); });
        if (m2 < ms_t) { ms_t = m2; bps_t = bps; er_t = e2; }
      }
      g_attn_bps = 0;
      const double d1 = time_graph([&](int i, cudaStream_t s) { dense_split_reduce(A, T, i % A.nc, s); }, 40);
      const double ms_dense_l = 0.5 * (d0 + d1), dense_gbs_l = dense_bytes / ms_dense_l * 1e-6;
      double ubsum = 0; for (int h = 0; h < NKV; h++) ubsum += sel.ub[h];
      const double bytes = KVB_BLOCK * ubsum + NKV * FIXED_HEAD + (double)T * NH * HD * 2;
      const double gbs = bytes / ms * 1e-6, gbs_t = bytes / ms_t * 1e-6;
      const double ideal = dense_bytes / bytes;
      printf("  %-3d %-8s %5.0f %6.1f %7.1f | %8.1f %7.0f %9.1f | %8.1f %7.0f %8.2f %7.2f %7.2f %7.3fms | %.1e; %.1f%%%s\n", T, cs_.indep ? "indep" : "shared", cs_.f * 100, sel.row_frac * 100, sel.union_frac * 100, ms_dense_l * 1e3, dense_gbs_l,
             ms_engine * 1e3, ms * 1e3, gbs, gbs / dense_gbs_l, ms_dense_l / ms, ideal, 16 * (ms_dense_l - ms), er.max_over_rms, er.frac_diff * 100,
             cs_.f >= 1.0 ? (std::string("  [vs engine dense: ") + std::to_string(ed.max_over_rms) + "]").c_str() : "");
      printf("  %-3s %-8s %5s %6s %7s | %8s %7s %9s | %8.1f %7.0f %8.2f %7.2f %7s %7.3fms | %.1e; %.1f%%   <- tuned splits (bps=%d)\n", "", "", "", "", "", "", "", "", ms_t * 1e3, gbs_t, gbs_t / dense_gbs_l, ms_dense_l / ms_t, "", 16 * (ms_dense_l - ms_t),
             er_t.max_over_rms, er_t.frac_diff * 100, bps_t);
      ja.rec("{\"what\":\"sparse\",\"ctx\":%d,\"T\":%d,\"mode\":\"%s\",\"f\":%.2f,\"row_frac\":%.4f,\"union_frac\":%.4f,\"dense_us\":%.2f,\"engine_step_us\":%.2f,\"sparse_us\":%.2f,\"bytes_needed\":%.0f,\"GBps_needed\":%.1f,\"dense_GBps\":%.1f,"
             "\"frac_dense_GBps\":%.4f,\"speedup\":%.4f,\"ideal_bytes_ratio\":%.4f,\"saving_16_layers_ms\":%.4f,\"tuned_us\":%.2f,\"tuned_bps\":%d,\"tuned_GBps\":%.1f,\"tuned_speedup\":%.4f,\"tuned_saving_16_layers_ms\":%.4f,\"tuned_err_max_over_rms\":%.3e,\"err_max_over_rms\":%.3e,\"frac_elems_differ\":%.4f,\"err_vs_engine_dense\":%.3e}",
             nctx, T, cs_.indep ? "indep" : "shared", cs_.f, sel.row_frac, sel.union_frac, ms_dense_l * 1e3, ms_engine * 1e3, ms * 1e3, bytes, gbs, dense_gbs_l, gbs / dense_gbs_l, ms_dense_l / ms, ideal, 16 * (ms_dense_l - ms),
             ms_t * 1e3, bps_t, gbs_t, ms_dense_l / ms_t, 16 * (ms_dense_l - ms_t), er_t.max_over_rms, er.max_over_rms, er.frac_diff, cs_.f >= 1.0 ? ed.max_over_rms : -1.0);
    }
  }
  // where does the fixed cost go?  (timing only: dbg bits skip the exact sink path / the exact ring, numerics are then wrong on purpose)
  for (int T : Ts) {
    if (tokmode) break;
    const int p0 = nctx - T;
    CK(cudaMemcpy(A.pos0, &p0, 4, cudaMemcpyHostToDevice));
    Sel sel = make_sel(T, A.nbt, A.stride, 0.0, false, rng);  // block 0 only
    for (int c = 0; c < A.nc; c++) CK(cudaMemcpy(A.lst + (size_t)c * NKV * A.stride, sel.lst.data(), sel.lst.size() * 4, cudaMemcpyHostToDevice));
    double t_split[4]; int dbgs[4] = {0, 2, 4, 6};
    for (int q = 0; q < 4; q++) { g_attn_dbg = dbgs[q]; t_split[q] = time_graph([&](int i, cudaStream_t s) { sparse_split_only(A, T, i % A.nc, s); }) * 1e3; }
    g_attn_dbg = 0;
    const double t_red = time_graph([&](int i, cudaStream_t s) { sparse_reduce_only(A, T, i % A.nc, s, A.out_sp); }) * 1e3;
    const double t_empty = time_graph([&](int, cudaStream_t s) { k_empty<<<80, 256, 0, s>>>(); }) * 1e3;
    printf("  [floor T=%d, block 0 only] split kernel %.1f us (no exact sink %.1f, no ring %.1f, neither %.1f) | reduce kernel %.1f us | empty 80x256 launch %.1f us\n", T, t_split[0], t_split[1], t_split[2], t_split[3], t_red, t_empty);
    ja.rec("{\"what\":\"floor\",\"ctx\":%d,\"T\":%d,\"split_us\":%.2f,\"split_no_sink_us\":%.2f,\"split_no_ring_us\":%.2f,\"split_no_sink_no_ring_us\":%.2f,\"reduce_us\":%.2f,\"empty_launch_us\":%.2f}", nctx, T, t_split[0], t_split[1], t_split[2], t_split[3], t_red, t_empty);
  }
  for (auto p : A.kv) cudaFree(p);
  cudaFree(A.ws); cudaFree(A.pos0); cudaFree(A.qproj); cudaFree(A.kk); cudaFree(A.vv); cudaFree(A.qn); cudaFree(A.kn); cudaFree(A.out_dense); cudaFree(A.out_sp); cudaFree(A.out_ref); cudaFree(A.rowm); cudaFree(A.lst);
}

// ================================================================================================================================ main
int main(int argc, char** argv) {
  std::string part = "all";
  bool launches_given = false;
  std::string outdir = "/media/fatman/data/workspace/research/experiments/2026-11-06-sparsity-probes/results/kern";
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--quick")) g_quick = true;
    else if (!strncmp(argv[i], "--part=", 7)) part = argv[i] + 7;
    else if (!strcmp(argv[i], "--tok")) part = "tok";
    else if (!strncmp(argv[i], "--out=", 6)) outdir = argv[i] + 6;
    else if (!strncmp(argv[i], "--launches=", 11)) { g_n_launch = atoi(argv[i] + 11); launches_given = true; }
  }
  if (g_quick && !launches_given) g_n_launch = 60;
  CK(cudaSetDevice(0));
  CK(cudaStreamCreate(&g_st));
  cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0));
  printf("device %s, %d SMs; mode %s; %d launches/graph x %d reps (median), distinct weight/KV copies cycled\n", pr.name, pr.multiProcessorCount, g_quick ? "QUICK" : "FULL", g_n_launch, g_reps);
  jg.open((outdir + "/kern_gemv.json").c_str()); ja.open((outdir + "/kern_attn.json").c_str()); jm.open((outdir + "/kern_misc.json").c_str());
  measure_floors();
  const bool full = !g_quick;
  if (part == "all" || part == "gemv") {
    struct S { const char* n; int kind; };
    const std::vector<S> shapes = full ? std::vector<S>{{"l31_qkv", 0}, {"l31_o", 0}, {"l31_gu", 1}, {"l31_dn", 0}, {"l20_qkvz", 0}} : std::vector<S>{{"l31_qkv", 0}, {"l31_dn", 0}};
    for (const S& s : shapes) run_gemv(s.n, s.kind, full, {1, 8});
  }
  if (part == "tok") {
    for (int ctx : {131072, 32768}) run_attn(ctx, {1, 8}, {}, full, true);
  }
  if (part == "all" || part == "attn") {
    const std::vector<double> fs = full ? std::vector<double>{1.0, 0.5, 0.25, 0.10, 0.0} : std::vector<double>{1.0, 0.25, 0.10, 0.0};
    for (int ctx : {32768, 131072}) run_attn(ctx, {1, 8}, fs, full);
  }
  jg.close(); ja.close(); jm.close();
  return 0;
}
