// Host driver for the W4 stream persistence-vs-graph experiment.  Modes: unit | barrier | gemv | full | check
// Build: nvcc -O3 -std=c++17 -arch=sm_86 bench.cu -o bench      (see run.py)
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <cooperative_groups.h>
#include "kernels.cuh"
#include "df.cuh"

using namespace e1;
namespace cg = cooperative_groups;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorName(e_), __FILE__, __LINE__, cudaGetErrorString(e_)); exit(2); } } while (0)

static double now_s() {
  return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

// ===================================================================== model
struct Bufs { float *x0, *x, *y1, *t, *logits; uint16_t *c, *h; };

struct Model {
  std::vector<Phase> ph;
  Phase* d_ph = nullptr;
  Bufs b{};
  size_t wbytes = 0;  // weights + scales bytes streamed per step
  std::vector<std::pair<const void*, size_t>> spans;
  std::vector<void*> allocs;
  uint64_t seed = 12345;
  int nsm = 0;
  int layers = 64;
  std::map<int, std::pair<void*, void*>> dn_rep;  // down-proj phase index -> repacked (weights, scales) for the fused MLP
  std::vector<CtaTab> tab_h; CtaTab* tab_d = nullptr; int tab_ncta = 0;

  void* alloc(size_t n) { void* p; CK(cudaMalloc(&p, n)); allocs.push_back(p); return p; }
  void alloc_bufs() {
    b.x0 = (float*)alloc(HID * 4); b.x = (float*)alloc(HID * 4); b.y1 = (float*)alloc(16384 * 4);
    b.t = (float*)alloc(6144 * 4); b.logits = (float*)alloc(248320 * 4);
    b.c = (uint16_t*)alloc(6144 * 2); b.h = (uint16_t*)alloc(INTER * 2);
    k_fill_f32<<<80, 256>>>(b.x0, HID, 777);
    CK(cudaMemset(b.x, 0, HID * 4));
  }
  Phase& add_w4(int N, int K, int pro, int epi, const void* in, void* out) {
    Phase p{};
    p.type = PH_W4; p.N = N; p.K = K; p.pro = pro; p.epi = epi; p.tiles = K / 1024;
    size_t wb = (size_t)N * K / 2, sb = (size_t)N * K / 64;
    uint32_t* w = (uint32_t*)alloc(wb);
    uint16_t* s = (uint16_t*)alloc(sb);
    k_fill_w4<<<2048, 256>>>(w, wb / 4, seed++ * 0x9E3779B97F4A7C15ull);
    k_fill_scale<<<256, 256>>>(s, sb / 2, seed++ * 0x9E3779B97F4A7C15ull, 0.10f / sqrtf((float)K));
    p.w = w; p.s = s; p.in = in; p.out = out;
    wbytes += wb + sb; spans.push_back({w, wb}); spans.push_back({s, sb});
    ph.push_back(p);
    return ph.back();
  }
  Phase& add_bf16(int N, int K, int pro, int epi, const void* in, void* out) {
    Phase p{};
    p.type = PH_BF16; p.N = N; p.K = K; p.pro = pro; p.epi = epi; p.tiles = K / 256;
    size_t wb = (size_t)N * K * 2;
    uint16_t* w = (uint16_t*)alloc(wb);
    k_fill_bf16<<<2048, 256>>>(w, wb / 2, seed++ * 0x9E3779B97F4A7C15ull, 1.7f / sqrtf((float)K));
    p.w = w; p.s = nullptr; p.in = in; p.out = out;
    wbytes += wb; spans.push_back({w, wb});
    ph.push_back(p);
    return ph.back();
  }
  void add_core(int type, int N, const void* in, void* out) {
    Phase p{}; p.type = type; p.N = N; p.in = in; p.out = out; p.next_gemv = -1;
    ph.push_back(p);
  }
  void finalize() {
    int nxt = -1;
    for (int i = (int)ph.size() - 1; i >= 0; i--) { ph[i].next_gemv = nxt; if (is_gemv(ph[i].type)) nxt = i; }
    if (d_ph) cudaFree(d_ph);
    CK(cudaMalloc(&d_ph, ph.size() * sizeof(Phase)));
    CK(cudaMemcpy(d_ph, ph.data(), ph.size() * sizeof(Phase), cudaMemcpyHostToDevice));
    CK(cudaDeviceSynchronize());
  }
  int first_gemv() const { for (size_t i = 0; i < ph.size(); i++) if (is_gemv(ph[i].type)) return (int)i; return -1; }
  int ngemv() const { int n = 0; for (auto& p : ph) n += is_gemv(p.type); return n; }
  void build_full() {
    alloc_bufs();
    add_core(PH_EMBED, HID, b.x0, b.x);
    for (int L = 0; L < layers; L++) {
      const bool attn = (L % 4) == 3;
      if (!attn) {
        add_w4(16384, HID, PRO_RMS, EPI_STORE, b.x, b.y1);               // [qkv|z]
        add_core(PH_CORE_GDN, 6144, b.y1, b.c);                          // DeltaNet core stand-in
        add_w4(HID, 6144, PRO_VEC, EPI_RESID, b.c, b.x);                 // out_proj
      } else {
        add_w4(14336, HID, PRO_RMS, EPI_STORE, b.x, b.y1);               // [q+gate|k|v]
        add_core(PH_CORE_A1, 6144, b.y1, b.t);                           // qk-norm / rope stand-in
        add_core(PH_CORE_A2, 6144, b.t, b.c);                            // softmax*V stand-in
        add_w4(HID, 6144, PRO_VEC, EPI_RESID, b.c, b.x);                 // o_proj
      }
      add_w4(34816, HID, PRO_RMS, EPI_GATEUP, b.x, b.h);                 // gate|up (rows interleaved)
      add_w4(HID, INTER, PRO_VEC, EPI_RESID, b.h, b.x);                  // down
    }
    add_bf16(248320, HID, PRO_RMS, EPI_STORE, b.x, b.logits);            // lm_head
    finalize();
  }
};

// ===================================================================== runners
struct Runner {
  std::string name, info;
  std::function<void(cudaStream_t)> step;
  double bytes = 0;  // bytes moved per step for GB/s
};

struct E2Opts { bool df = false, fuse = false, bal = false, cheap = false; int pf = 0, abl = 0; };
static int g_smem_optin = 0;
static int g_grid_override = 0;  // >0: CTAs per kernel instead of #SM*blocks/SM (imbalance study)
static unsigned long long* g_stamps_for_runner = nullptr;  // set before make_persist to enable per-phase profiling
static int g_abl = 0;  // ablation bits for k_persist (see kernels.cuh)
static int g_pfd = 0;  // L2 prefetch distance in tiles beyond the smem ring (0 = off)

template <class C>
static std::string cfg_name() {
  return "w" + std::to_string(C::NW) + "r" + std::to_string(C::R) + "d" + std::to_string(C::D);
}

static std::function<void(cudaStream_t)> graph_step(cudaGraphExec_t ex) {
  return [ex](cudaStream_t st) { CK(cudaGraphLaunch(ex, st)); };
}

// V1: one kernel per phase in a CUDA graph.  grid = #SM * blocks/SM (fully resident, same static work split as V3/V4)
template <class C, bool PREF>
static bool make_v1(Model& m, Runner& r) {
  const int smem = ACT_MAX_BYTES + C::RING_BYTES;
  if (smem > g_smem_optin) return false;
  CK(cudaFuncSetAttribute(k_gemv<C, PREF>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  int bps = 0;
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, k_gemv<C, PREF>, C::NT, smem));
  if (bps < 1) return false;
  const int grid = g_grid_override ? g_grid_override : m.nsm * bps;
  cudaStream_t cs; CK(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking));
  cudaGraph_t g; cudaGraphExec_t ex;
  CK(cudaStreamBeginCapture(cs, cudaStreamCaptureModeGlobal));
  for (int p = 0; p < (int)m.ph.size(); p++) {
    if (is_gemv(m.ph[p].type)) k_gemv<C, PREF><<<grid, C::NT, smem, cs>>>(m.d_ph, p, ACT_MAX_BYTES, g_pfd, (int)m.ph.size());
    else k_core<<<(m.ph[p].N + 255) / 256, 256, 0, cs>>>(m.d_ph, p);
  }
  CK(cudaStreamEndCapture(cs, &g));
  CK(cudaGraphInstantiate(&ex, g, 0));
  r.step = graph_step(ex);
  r.bytes = (double)m.wbytes;
  r.info = "grid=" + std::to_string(grid) + " blocks/SM=" + std::to_string(bps) + " smem=" + std::to_string(smem);
  return true;
}

// V1c: like V1b but "classic" launch: one task per warp, grid = ceil(tasks/NW) CTAs, dynamic smem sized to the phase's activation
// vector so several CTAs fit per SM and the hardware scheduler load-balances; the next CTA's prologue overlaps streaming of others.
template <class C>
static bool make_v1c(Model& m, Runner& r) {
  const int smem_max = ACT_MAX_BYTES + C::RING_BYTES;
  if (smem_max > g_smem_optin) return false;
  CK(cudaFuncSetAttribute(k_gemv<C, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max));
  cudaStream_t cs; CK(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking));
  cudaGraph_t g; cudaGraphExec_t ex;
  CK(cudaStreamBeginCapture(cs, cudaStreamCaptureModeGlobal));
  int minb = 99;
  for (int p = 0; p < (int)m.ph.size(); p++) {
    if (is_gemv(m.ph[p].type)) {
      const int ring_off = (m.ph[p].K * 2 + 127) / 128 * 128;
      const int smem = ring_off + C::RING_BYTES;
      int bps = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, k_gemv<C, true>, C::NT, smem)); minb = std::min(minb, bps);
      const int grid = (m.ph[p].N / C::R + C::NW - 1) / C::NW;
      k_gemv<C, true><<<grid, C::NT, smem, cs>>>(m.d_ph, p, ring_off, 0, (int)m.ph.size());
    } else k_core<<<(m.ph[p].N + 255) / 256, 256, 0, cs>>>(m.d_ph, p);
  }
  CK(cudaStreamEndCapture(cs, &g));
  CK(cudaGraphInstantiate(&ex, g, 0));
  r.step = graph_step(ex);
  r.bytes = (double)m.wbytes;
  r.info = "grid=ceil(tasks/NW) CTAs per phase, min blocks/SM over phases=" + std::to_string(minb);
  return true;
}

// V3 (CROSS=false) / V4 (CROSS=true): persistent cooperative kernel
template <class C, bool CROSS>
static bool make_persist(Model& m, Runner& r) {
  const int smem = ACT_MAX_BYTES + C::RING_BYTES;
  if (smem > g_smem_optin) return false;
  CK(cudaFuncSetAttribute(k_persist<C, CROSS>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  int bps = 0;
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, k_persist<C, CROSS>, C::NT, smem));
  if (bps < 1) return false;
  const int grid = g_grid_override ? g_grid_override : m.nsm * bps;
  unsigned* cnt; CK(cudaMalloc(&cnt, 256)); CK(cudaMemset(cnt, 0, 256));
  auto epoch = std::make_shared<unsigned>(0);
  const int nph = (int)m.ph.size(), fg = m.first_gemv();
  Phase* dph = m.d_ph;
  const unsigned adv = (unsigned)(nph - 1) * (unsigned)grid;
  const int ablv = g_abl, pfdv = g_pfd;
  unsigned long long* const g_stamps_for_runner = ::g_stamps_for_runner;
  r.step = [=](cudaStream_t st) {
    int ring_off = ACT_MAX_BYTES; unsigned ep = *epoch;
    int abl = ablv; unsigned long long* stp = g_stamps_for_runner; int pfd = pfdv; void* args[] = {(void*)&dph, (void*)&nph, (void*)&cnt, (void*)&ep, (void*)&ring_off, (void*)&fg, (void*)&abl, (void*)&stp, (void*)&pfd};
    CK(cudaLaunchCooperativeKernel((void*)k_persist<C, CROSS>, dim3(grid), dim3(C::NT), args, smem, st));
    *epoch += adv;
  };
  r.bytes = (double)m.wbytes;
  r.info = "grid=" + std::to_string(grid) + " blocks/SM=" + std::to_string(bps) + " smem=" + std::to_string(smem);
  return true;
}


// ===================================================================== E2: fused / dataflow / balanced persistent kernel
static void make_tab(Model& m, int ncta) {
  if (m.tab_d && m.tab_ncta == ncta) return;
  m.tab_h.assign(ncta, CtaTab{});
  unsigned long long off = 0;
  for (int c = 0; c < ncta; c++) {
    CtaTab t{};
    t.wlo = (int)((long long)MLP_CHUNKS * c / ncta);
    const int whi = (int)((long long)MLP_CHUNKS * (c + 1) / ncta);
    t.nwc = whi - t.wlo;
    t.nseg = ((whi - 1) >> 4) - (t.wlo >> 4) + 1;
    t.sc_off = off;
    off += (unsigned long long)HID * t.nseg * 2;
    m.tab_h[c] = t;
  }
  if (m.tab_d) cudaFree(m.tab_d);
  CK(cudaMalloc(&m.tab_d, ncta * sizeof(CtaTab)));
  CK(cudaMemcpy(m.tab_d, m.tab_h.data(), ncta * sizeof(CtaTab), cudaMemcpyHostToDevice));
  m.tab_ncta = ncta;
}
static unsigned long long tab_sc_bytes(Model& m) { auto& t = m.tab_h.back(); return t.sc_off + (unsigned long long)HID * t.nseg * 2; }

static void ensure_repack(Model& m, int down_idx, int ncta) {
  if (m.dn_rep.count(down_idx)) return;
  const Phase& p = m.ph[down_idx];
  void *w, *sc;
  CK(cudaMalloc(&w, (size_t)HID * INTER / 2));
  CK(cudaMalloc(&sc, tab_sc_bytes(m)));
  m.allocs.push_back(w); m.allocs.push_back(sc);
  k_repack_down<<<dim3(64, ncta), 256>>>((const uint32_t*)p.w, (uint32_t*)w, m.tab_d, ncta);
  k_repack_down_scales<<<dim3(16, ncta), 256>>>((const uint16_t*)p.s, (uint16_t*)sc, m.tab_d);
  CK(cudaGetLastError());
  m.dn_rep[down_idx] = {w, sc};
}

struct E2Plan {
  std::vector<Ph2> ph;
  int n_counters = 0, grid_syncs = 0, ordinals = 0, phases_after = 0;
  size_t actb = 0;
  int first_gemv = 0;
};

template <class C>
static bool build_e2_plan(Model& m, const E2Opts& o, int ncta, E2Plan& pl) {
  const bool fuse = o.fuse;
  if (fuse && C::R != 2) return false;
  if (fuse) make_tab(m, ncta); else make_tab(m, ncta);  // table always needed by the kernel
  std::vector<Ph2> v;
  for (size_t i = 0; i < m.ph.size(); i++) {
    const Phase& b = m.ph[i];
    if (fuse && b.type == PH_W4 && b.epi == EPI_GATEUP && i + 1 < m.ph.size() && m.ph[i + 1].type == PH_W4 && m.ph[i + 1].K == INTER) {
      ensure_repack(m, (int)i + 1, ncta);
      Ph2 a{}; a.b = b; a.b.epi = EPI_MLPA; a.b.out = nullptr; a.sync = 0;
      Ph2 d{}; d.b = m.ph[i + 1]; d.b.type = PH_MLPB; d.b.tiles = 0; d.b.pro = PRO_VEC; d.b.epi = EPI_RESID;
      d.b.w = m.dn_rep[(int)i + 1].first; d.b.s = m.dn_rep[(int)i + 1].second; d.b.in = nullptr; d.sync = 1;
      v.push_back(a); v.push_back(d);
      i++;
    } else {
      Ph2 a{}; a.b = b; a.sync = 1; v.push_back(a);
    }
  }
  const int n = (int)v.size();
  // GEMV-like links
  int nxt = -1;
  for (int i = n - 1; i >= 0; i--) { v[i].b.next_gemv = nxt; if (is_gemv2(v[i].b.type)) nxt = i; }
  pl.first_gemv = nxt;
  int ord = 0, cntr = 64, xr = 0;
  // activation buffers (DF): by K class
  const int szX = 10240, szY = 12288, szBIG = 34816;
  int offX = 0, offY = 0, offBIG = 0;
  if (o.df) {
    offX = 0; offY = szX; offBIG = szX + szY;
    bool anyBig = false; for (auto& d : v) if (is_staged(d) && d.b.K == INTER) anyBig = true;
    pl.actb = szX + szY + (anyBig ? szBIG : 0);
  } else {
    pl.actb = fuse ? szY : szBIG;
  }
  int last_ord[3] = {-1, -1, -1};
  for (int i = 0; i < n; i++) {
    Ph2& d = v[i];
    d.src = -1; d.sig = -1; d.nout = 0; d.need_mult = 1; d.abuf = 0; d.reuse_after = -1; d.gord = 0; d.xread = -1; d.ticks = 0;
    d.asg = (o.bal && d.b.epi != EPI_MLPA) ? 1 : 0;
    if (d.b.in) for (int j = i - 1; j >= 0; j--) if (v[j].b.out == d.b.in) { d.src = j; break; }
    if (is_gemv2(d.b.type)) d.gord = ord++;
    // output counters
    int nout = 0;
    switch (d.b.type) {
      case PH_EMBED: case PH_CORE_GDN: case PH_CORE_A1: case PH_CORE_A2: nout = d.b.N; break;
      case PH_MLPB: nout = HID; d.need_mult = ncta; break;
      default: nout = d.b.epi == EPI_GATEUP ? d.b.N / 2 : (d.b.epi == EPI_MLPA ? 0 : (d.b.type == PH_BF16 ? 0 : d.b.N)); break;
    }
    d.nout = nout;
    if (nout > 0) { d.sig = cntr; cntr += (nout + SL - 1) / SL; }
    if (d.b.epi == EPI_MLPA) { d.xread = xr; }
    if (d.b.type == PH_MLPB) { d.xread = xr++; }  // the MLPA just before shares the index (set below)
    if (d.b.epi == EPI_MLPA) { /* xr incremented by the matching MLPB */ }
    if (is_staged(d)) {
      const int cls = d.b.K == INTER ? 2 : (d.b.K == 6144 ? 1 : 0);
      d.abuf = o.df ? (cls == 2 ? offBIG : cls == 1 ? offY : offX) : 0;
      d.reuse_after = o.df ? last_ord[cls] : -1;
      last_ord[cls] = d.gord;
      d.ticks = d.b.K / SL + (d.b.pro == PRO_RMS ? 1 : 0);
    }
  }
  pl.n_counters = cntr + xr + 8;
  // xread counters live after the slice counters
  for (auto& d : v) if (d.xread >= 0) d.xread += cntr;
  pl.grid_syncs = 0;
  for (int i = 0; i + 1 < n; i++) if (v[i].sync) pl.grid_syncs++;
  pl.phases_after = pl.grid_syncs + 1;
  pl.ordinals = ord;
  pl.ph = v;
  return true;
}

template <class C, bool DF, bool CHEAP>
static bool make_e2(Model& m, Runner& r, const E2Opts& o) {
  if (o.df != DF || o.cheap != CHEAP) return false;
  E2Plan pl;
  if (!build_e2_plan<C>(m, o, m.nsm, pl)) return false;
  const size_t ring_off = (pl.actb + 1024 + 127) / 128 * 128;
  const size_t smem = ring_off + C::RING_BYTES;
  if (smem > (size_t)g_smem_optin - 512) return false;
  CK(cudaFuncSetAttribute(k_main<C, DF, CHEAP>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  int bps = 0;
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, k_main<C, DF, CHEAP>, C::NT + (DF ? 32 : 0), smem));
  if (bps < 1) return false;
  const int grid = m.nsm;
  Ph2* d_ph; CK(cudaMalloc(&d_ph, pl.ph.size() * sizeof(Ph2)));
  CK(cudaMemcpy(d_ph, pl.ph.data(), pl.ph.size() * sizeof(Ph2), cudaMemcpyHostToDevice));
  unsigned* cnt; CK(cudaMalloc(&cnt, (size_t)pl.n_counters * 4)); CK(cudaMemset(cnt, 0, (size_t)pl.n_counters * 4));
  CK(cudaDeviceSynchronize());
  auto step_no = std::make_shared<unsigned>(0);
  const int nph = (int)pl.ph.size(), fg = pl.first_gemv, ring = (int)ring_off, ablv = o.abl, pfdv = o.pf;
  const unsigned adv = (unsigned)pl.grid_syncs * (unsigned)grid;
  CtaTab* tab = m.tab_d;
  r.step = [=](cudaStream_t st) {
    unsigned bt = *step_no * adv, ep = *step_no + 1;
    int fgv = fg, ringv = ring, ablx = ablv, pfx = pfdv, nphv = nph; unsigned long long* stp = nullptr;
    const Ph2* pph = d_ph; const CtaTab* ptab = tab; unsigned* pc = cnt;
    void* args[] = {(void*)&pph, (void*)&nphv, (void*)&ptab, (void*)&pc, (void*)&bt, (void*)&ep, (void*)&ringv, (void*)&fgv, (void*)&ablx, (void*)&pfx, (void*)&stp};
    CK(cudaFuncSetAttribute(k_main<C, DF, CHEAP>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));  // several runners share one instantiation
    CK(cudaLaunchCooperativeKernel((void*)k_main<C, DF, CHEAP>, dim3(grid), dim3(C::NT + (DF ? 32 : 0)), args, smem, st));
    (*step_no)++;
  };
  r.bytes = (double)m.wbytes;
  r.info = "e2 grid=" + std::to_string(grid) + " smem=" + std::to_string(smem) + " dependent_phases/step=" + std::to_string(pl.phases_after) +
           " counters=" + std::to_string(pl.n_counters) + (o.df ? " DF" : " BAR") + (o.fuse ? " fuse" : "") + (o.bal ? " bal" : "") + (o.pf ? " pf" + std::to_string(o.pf) : "");
  return true;
}

// V0g: independent read-only kernel per GEMV matrix, graph.  V0: one kernel over all buffers.
static void make_v0(Model& m, Runner& r) {
  std::vector<Span> sp;
  for (auto& s : m.spans) sp.push_back({(const uint4*)s.first, s.second / 16});
  Span* dsp; CK(cudaMalloc(&dsp, sp.size() * sizeof(Span)));
  CK(cudaMemcpy(dsp, sp.data(), sp.size() * sizeof(Span), cudaMemcpyHostToDevice));
  uint32_t* o; CK(cudaMalloc(&o, 4));
  int ns = (int)sp.size(), grid = m.nsm * 3;
  r.step = [=](cudaStream_t st) { k_sumread<<<grid, 512, 0, st>>>(dsp, ns, o); };
  r.bytes = (double)m.wbytes;
  r.info = "single kernel, grid=" + std::to_string(grid) + "x512, " + std::to_string(ns) + " spans";
}
static void make_v0g(Model& m, Runner& r) {
  uint32_t* o; CK(cudaMalloc(&o, 4));
  cudaStream_t cs; CK(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking));
  cudaGraph_t g; cudaGraphExec_t ex;
  int grid = m.nsm * 3, n = 0;
  CK(cudaStreamBeginCapture(cs, cudaStreamCaptureModeGlobal));
  for (auto& p : m.ph) {
    if (!is_gemv(p.type)) continue;
    size_t wb = p.type == PH_W4 ? (size_t)p.N * p.K / 2 : (size_t)p.N * p.K * 2;
    size_t sb = p.type == PH_W4 ? (size_t)p.N * p.K / 64 : 0;
    k_readone<<<grid, 512, 0, cs>>>((const uint4*)p.w, wb / 16, (const uint4*)p.s, sb / 16, o);
    n++;
  }
  CK(cudaStreamEndCapture(cs, &g));
  CK(cudaGraphInstantiate(&ex, g, 0));
  r.step = graph_step(ex);
  r.bytes = (double)m.wbytes;
  r.info = std::to_string(n) + " independent read kernels in a graph, grid=" + std::to_string(grid) + "x512";
}

// ===================================================================== timing
static std::vector<float> time_steps(const std::function<void(cudaStream_t)>& step, int warm, int steps, cudaStream_t st) {
  for (int i = 0; i < warm; i++) step(st);
  CK(cudaStreamSynchronize(st));
  std::vector<cudaEvent_t> ev(steps + 1);
  for (auto& e : ev) CK(cudaEventCreate(&e));
  CK(cudaEventRecord(ev[0], st));
  for (int i = 0; i < steps; i++) { step(st); CK(cudaEventRecord(ev[i + 1], st)); }
  CK(cudaEventSynchronize(ev[steps]));
  std::vector<float> t(steps);
  for (int i = 0; i < steps; i++) CK(cudaEventElapsedTime(&t[i], ev[i], ev[i + 1]));
  for (auto& e : ev) CK(cudaEventDestroy(e));
  return t;
}
static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); size_t n = v.size(); return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]); }
static double medianf(std::vector<float> v) { std::vector<double> d(v.begin(), v.end()); return median(d); }

// ===================================================================== config registry
struct CfgEntry {
  std::string name;
  std::function<bool(Model&, Runner&)> v1a, v1b, v1c, v3, v4;
  std::function<bool(Model&, Runner&, const E2Opts&)> e2[4];  // [DF*2 + CHEAP]
};
static std::vector<CfgEntry>& registry() { static std::vector<CfgEntry> r; return r; }
template <int NW, int R, int D, int E2_ENABLE = 0>
static void reg() {
  using C = Cfg<NW, R, D>;
  CfgEntry e;
  e.name = cfg_name<C>();
  e.v1a = [](Model& m, Runner& r) { return make_v1<C, false>(m, r); };
  e.v1b = [](Model& m, Runner& r) { return make_v1<C, true>(m, r); };
  e.v1c = [](Model& m, Runner& r) { return make_v1c<C>(m, r); };
  e.v3 = [](Model& m, Runner& r) { return make_persist<C, false>(m, r); };
  e.v4 = [](Model& m, Runner& r) { return make_persist<C, true>(m, r); };
  if constexpr (E2_ENABLE >= 1) {
    e.e2[0] = [](Model& m, Runner& r, const E2Opts& o) { return make_e2<C, false, false>(m, r, o); };
    e.e2[2] = [](Model& m, Runner& r, const E2Opts& o) { return make_e2<C, true, false>(m, r, o); };
  }
  if constexpr (E2_ENABLE >= 2) {
    e.e2[1] = [](Model& m, Runner& r, const E2Opts& o) { return make_e2<C, false, true>(m, r, o); };
    e.e2[3] = [](Model& m, Runner& r, const E2Opts& o) { return make_e2<C, true, true>(m, r, o); };
  }
  registry().push_back(e);
}
static void register_all() {
#include "cfgs.inc"
}
static CfgEntry* find_cfg(const std::string& n) { for (auto& e : registry()) if (e.name == n) return &e; return nullptr; }

// ===================================================================== reference
static void run_reference(Model& m, const Bufs& rb, float* actf, float* raw, int rev = 0, bool late = false) {
  static float* d_inv = nullptr; if (!d_inv) CK(cudaMalloc(&d_inv, 16));
  auto remap = [&](const void* p) -> const void* {
    if (p == m.b.x0) return rb.x0; if (p == m.b.x) return rb.x; if (p == m.b.y1) return rb.y1; if (p == m.b.t) return rb.t;
    if (p == m.b.c) return rb.c; if (p == m.b.h) return rb.h; if (p == m.b.logits) return rb.logits; return p;
  };
  for (auto p : m.ph) {
    p.in = remap(p.in); p.out = (void*)remap(p.out);
    if (!is_gemv(p.type)) { k_ref_core<<<(p.N + 255) / 256, 256>>>(p); continue; }
    k_ref_prologue<<<1, 1024>>>(p, actf, d_inv, late ? 1 : 0);
    if (p.epi == EPI_GATEUP) {
      k_ref_gemv_raw<<<(p.N * 32 + 255) / 256, 256>>>(p, actf, raw, rev, late ? d_inv : nullptr);
      k_ref_gateup<<<(p.N / 2 + 255) / 256, 256>>>(p, raw);
    } else {
      k_ref_gemv<<<(p.N * 32 + 255) / 256, 256>>>(p, actf, rev, late ? d_inv : nullptr);
    }
  }
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
}

static std::vector<float> d2h_f(const float* p, size_t n) { std::vector<float> v(n); CK(cudaMemcpy(v.data(), p, n * 4, cudaMemcpyDeviceToHost)); return v; }
struct Err { double rel_l2, max_abs_over_rms; bool finite; double rms; };
static Err compare(const std::vector<float>& a, const std::vector<float>& ref) {
  double num = 0, den = 0, mx = 0; bool fin = true;
  for (size_t i = 0; i < a.size(); i++) {
    if (!std::isfinite(a[i])) fin = false;
    double d = (double)a[i] - ref[i]; num += d * d; den += (double)ref[i] * ref[i]; mx = std::max(mx, std::fabs(d));
  }
  double rms = std::sqrt(den / a.size());
  return {std::sqrt(num / den), mx / rms, fin, rms};
}

// ===================================================================== modes
static float bf16_to_f(uint16_t u) { uint32_t x = (uint32_t)u << 16; float f; memcpy(&f, &x, 4); return f; }
static uint16_t f_to_bf16(float f) { uint32_t x; memcpy(&x, &f, 4); uint32_t r = x + 0x7fff + ((x >> 16) & 1); return (uint16_t)(r >> 16); }

template <class C>
static void unit_cfg(const char* tag, bool& all_ok, std::string& json) {
  // tiny table: W4 store (PRO_VEC), W4 gateup (PRO_VEC), BF16 store (PRO_VEC), W4 store (PRO_RMS)
  Model m;
  int nsm; CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0)); m.nsm = nsm;
  const int K1 = 2048, N1 = 64, K3 = 1024, K4 = HID, N4 = 64;
  uint16_t* v1; float* xr; float *o1, *o3, *o4; uint16_t* o2;
  v1 = (uint16_t*)m.alloc(K1 * 2); xr = (float*)m.alloc(HID * 4);
  o1 = (float*)m.alloc(N1 * 4); o2 = (uint16_t*)m.alloc(N1 * 2); o3 = (float*)m.alloc(N1 * 4); o4 = (float*)m.alloc(N4 * 4);
  uint16_t* v3 = (uint16_t*)m.alloc(K3 * 2);
  k_fill_bf16_vec<<<8, 256>>>(v1, K1, 5); k_fill_bf16_vec<<<8, 256>>>(v3, K3, 6); k_fill_f32<<<8, 256>>>(xr, HID, 7);
  m.add_w4(N1, K1, PRO_VEC, EPI_STORE, v1, o1);
  m.add_w4(N1, K1, PRO_VEC, EPI_GATEUP, v1, o2);
  m.add_bf16(N1, K3, PRO_VEC, EPI_STORE, v3, o3);
  m.add_w4(N4, K4, PRO_RMS, EPI_STORE, xr, o4);
  m.finalize();
  const int nph = (int)m.ph.size();

  auto run_kernel_path = [&](int mode) {  // 0 = per-phase kernels (V1b), 1 = V3, 2 = V4
    CK(cudaMemset(o1, 0, N1 * 4)); CK(cudaMemset(o2, 0, N1 * 2)); CK(cudaMemset(o3, 0, N1 * 4)); CK(cudaMemset(o4, 0, N4 * 4));
    const int smem = ACT_MAX_BYTES + C::RING_BYTES;
    if (mode == 0) {
      CK(cudaFuncSetAttribute(k_gemv<C, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
      for (int p = 0; p < nph; p++) k_gemv<C, true><<<nsm, C::NT, smem>>>(m.d_ph, p, ACT_MAX_BYTES, 0, nph);
    } else {
      unsigned* cnt; CK(cudaMalloc(&cnt, 64)); CK(cudaMemset(cnt, 0, 64));
      int ring_off = ACT_MAX_BYTES; unsigned ep = 0; int fg = m.first_gemv(); Phase* dph = m.d_ph; int abl = 0; unsigned long long* stp = nullptr; int pfd = 0;
      void* args[] = {(void*)&dph, (void*)&nph, (void*)&cnt, (void*)&ep, (void*)&ring_off, (void*)&fg, (void*)&abl, (void*)&stp, (void*)&pfd};
      if (mode == 1) { CK(cudaFuncSetAttribute(k_persist<C, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        CK(cudaLaunchCooperativeKernel((void*)k_persist<C, false>, dim3(nsm), dim3(C::NT), args, smem)); }
      else { CK(cudaFuncSetAttribute(k_persist<C, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        CK(cudaLaunchCooperativeKernel((void*)k_persist<C, true>, dim3(nsm), dim3(C::NT), args, smem)); }
      CK(cudaDeviceSynchronize()); cudaFree(cnt);
    }
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  };

  // CPU reference (double) from the raw device data
  auto h_bf = [&](const uint16_t* d, size_t n) { std::vector<uint16_t> v(n); CK(cudaMemcpy(v.data(), d, n * 2, cudaMemcpyDeviceToHost)); return v; };
  auto cpu_w4 = [&](const Phase& p, const std::vector<double>& act) {
    std::vector<uint32_t> w((size_t)p.N * p.K / 8); std::vector<uint16_t> s((size_t)p.N * p.K / 128);
    CK(cudaMemcpy(w.data(), p.w, w.size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(s.data(), p.s, s.size() * 2, cudaMemcpyDeviceToHost));
    std::vector<double> out(p.N, 0.0);
    for (int n = 0; n < p.N; n++) for (int k = 0; k < p.K; k++) {
      int nib = (w[(size_t)n * p.K / 8 + k / 8] >> (4 * (k % 8))) & 15;
      out[n] += (double)(nib - 8) * bf16_to_f(s[(size_t)n * (p.K / 128) + k / 128]) * act[k];
    }
    return out;
  };
  auto act_of = [&](const uint16_t* d, int K) { auto v = h_bf(d, K); std::vector<double> a(K); for (int i = 0; i < K; i++) a[i] = bf16_to_f(v[i]); return a; };
  auto errs = [&](const std::vector<double>& ref, const std::vector<double>& got) {
    double num = 0, den = 0; for (size_t i = 0; i < ref.size(); i++) { num += (ref[i] - got[i]) * (ref[i] - got[i]); den += ref[i] * ref[i]; }
    return std::sqrt(num / den);
  };
  std::vector<double> ref1 = cpu_w4(m.ph[0], act_of(v1, K1));
  std::vector<double> ref2raw = cpu_w4(m.ph[1], act_of(v1, K1));
  std::vector<double> ref2(N1 / 2);
  for (int j = 0; j < N1 / 2; j++) { double g = ref2raw[2 * j], u = ref2raw[2 * j + 1]; ref2[j] = g / (1.0 + std::exp(-g)) * u; }
  std::vector<double> ref3(N1, 0.0);
  { std::vector<uint16_t> w((size_t)N1 * K3); CK(cudaMemcpy(w.data(), m.ph[2].w, w.size() * 2, cudaMemcpyDeviceToHost));
    auto a = act_of(v3, K3); for (int n = 0; n < N1; n++) for (int k = 0; k < K3; k++) ref3[n] += (double)bf16_to_f(w[(size_t)n * K3 + k]) * a[k]; }
  std::vector<double> ref4;
  { auto xv = d2h_f(xr, HID); double ss = 0; for (float f : xv) ss += (double)f * f; float inv = 1.0f / sqrtf((float)(ss / HID) + 1e-6f);
    std::vector<double> a(HID); for (int i = 0; i < HID; i++) a[i] = bf16_to_f(f_to_bf16(xv[i] * inv)); ref4 = cpu_w4(m.ph[3], a); }

  static const char* modes[] = {"V1b", "V3", "V4"};
  for (int mode = 0; mode < 3; mode++) {
    run_kernel_path(mode);
    std::vector<float> g1 = d2h_f(o1, N1), g3 = d2h_f(o3, N1), g4 = d2h_f(o4, N4);
    auto g2b = h_bf(o2, N1 / 2);
    std::vector<double> d1(g1.begin(), g1.end()), d3(g3.begin(), g3.end()), d4(g4.begin(), g4.end()), d2(N1 / 2);
    for (int j = 0; j < N1 / 2; j++) d2[j] = bf16_to_f(g2b[j]);
    // gateup ref is rounded to bf16 by the kernel: compare against bf16-rounded ref
    std::vector<double> ref2r(N1 / 2); for (int j = 0; j < N1 / 2; j++) ref2r[j] = bf16_to_f(f_to_bf16((float)ref2[j]));
    double e1 = errs(ref1, d1), e2 = errs(ref2r, d2), e3 = errs(ref3, d3), e4 = errs(ref4, d4);
    bool ok = e1 < 1e-5 && e2 < 8e-3 && e3 < 1e-5 && e4 < 2e-2;
    all_ok &= ok;
    char buf[512];
    snprintf(buf, sizeof buf, "%s{\"cfg\":\"%s\",\"path\":\"%s\",\"w4_store_rel_l2\":%.3e,\"w4_gateup_bf16out_rel_l2\":%.3e,\"bf16_store_rel_l2\":%.3e,\"w4_rms_prologue_rel_l2\":%.3e,\"ok\":%s}",
             json.empty() ? "" : ",", tag, modes[mode], e1, e2, e3, e4, ok ? "true" : "false");
    json += buf;
  }
}

__global__ void k_barrier_bench(unsigned* cnt, unsigned* flag, unsigned epoch, int iters, int mode) {
  unsigned target = epoch, local = 0;
  for (int i = 0; i < iters; i++) {
    if (mode == 0) { target += gridDim.x; grid_barrier<false>(cnt, target); }
    else if (mode == 1) { target += gridDim.x; grid_barrier<true>(cnt, target); }
    else if (mode == 2) grid_barrier_sense(cnt, flag, gridDim.x, local);
    else cg::this_grid().sync();
  }
}

static int run_barrier(Model& m, FILE* out) {
  const char* names[] = {"monotone_counter_spin", "monotone_counter_nanosleep32", "sense_reversing", "cg_grid_sync"};
  unsigned *cnt, *flag; CK(cudaMalloc(&cnt, 256)); CK(cudaMalloc(&flag, 256));
  fprintf(out, "{\"barrier_us\":[");
  bool first = true;
  int blocks_list[] = {m.nsm, 2 * m.nsm};
  int threads_list[] = {128, 256, 512, 1024};
  for (int bl : blocks_list) for (int th : threads_list) {
    if (bl * th > m.nsm * 1536 * 1) {}  // occupancy check via launch result below
    for (int mode = 0; mode < 4; mode++) {
      std::vector<double> reps;
      bool ok = true;
      for (int rep = 0; rep < 5 && ok; rep++) {
        CK(cudaMemset(cnt, 0, 256)); CK(cudaMemset(flag, 0, 256));
        unsigned ep = 0; int iters = 20000;
        void* args[] = {(void*)&cnt, (void*)&flag, (void*)&ep, (void*)&iters, (void*)&mode};
        cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
        // warmup
        int w = 1000; void* wargs[] = {(void*)&cnt, (void*)&flag, (void*)&ep, (void*)&w, (void*)&mode};
        cudaError_t e = cudaLaunchCooperativeKernel((void*)k_barrier_bench, dim3(bl), dim3(th), wargs, 0, 0);
        if (e != cudaSuccess) { cudaGetLastError(); ok = false; break; }
        CK(cudaDeviceSynchronize());
        CK(cudaMemset(cnt, 0, 256)); CK(cudaMemset(flag, 0, 256));
        CK(cudaEventRecord(a));
        CK(cudaLaunchCooperativeKernel((void*)k_barrier_bench, dim3(bl), dim3(th), args, 0, 0));
        CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
        float ms; CK(cudaEventElapsedTime(&ms, a, b)); reps.push_back(ms * 1000.0 / iters);
      }
      if (!ok) continue;
      fprintf(out, "%s{\"mode\":\"%s\",\"blocks\":%d,\"threads\":%d,\"us_per_barrier_median\":%.4f,\"us_min\":%.4f}", first ? "" : ",", names[mode], bl, th, median(reps), *std::min_element(reps.begin(), reps.end()));
      first = false;
    }
  }
  fprintf(out, "]}\n");
  return 0;
}

static E2Opts parse_e2(const std::string& c) {
  E2Opts o; size_t pos = c.find(':');
  while (pos != std::string::npos) {
    size_t nx = c.find(':', pos + 1);
    std::string tok = c.substr(pos + 1, nx == std::string::npos ? std::string::npos : nx - pos - 1);
    size_t eq = tok.find('='); std::string k = tok.substr(0, eq); int val = eq == std::string::npos ? 1 : atoi(tok.c_str() + eq + 1);
    if (k == "df") o.df = val; else if (k == "fuse") o.fuse = val; else if (k == "bal") o.bal = val; else if (k == "cheap") o.cheap = val;
    else if (k == "pf") o.pf = val; else if (k == "abl") o.abl = val;
    pos = nx;
  }
  return o;
}

static bool make_runner(Model& m, const std::string& sp, Runner& r, const std::function<void(Runner&)>& make_memcpy) {
  std::string v = sp, c;
  size_t col = sp.find(':'); if (col != std::string::npos) { v = sp.substr(0, col); c = sp.substr(col + 1); }
  g_abl = 0; g_pfd = 0;
  if (v != "e2") {
    { size_t c2 = c.find(":abl="); if (c2 != std::string::npos) { g_abl = atoi(c.c_str() + c2 + 5); c = c.substr(0, c2); } }
    { size_t c2 = c.find(":pf="); if (c2 != std::string::npos) { g_pfd = atoi(c.c_str() + c2 + 4); c = c.substr(0, c2); } }
  }
  if (v == "v0") { make_v0(m, r); return true; }
  if (v == "v0g") { make_v0g(m, r); return true; }
  if (v == "memcpy") { make_memcpy(r); return true; }
  if (v == "e2") {
    E2Opts o = parse_e2(c);
    CfgEntry* ce = find_cfg(c.substr(0, c.find(':')));
    if (!ce) { fprintf(stderr, "unknown cfg in spec %s\n", sp.c_str()); exit(3); }
    auto& f = ce->e2[(o.df ? 2 : 0) + (o.cheap ? 1 : 0)];
    return f ? f(m, r, o) : false;
  }
  CfgEntry* ce = find_cfg(c);
  if (!ce) { fprintf(stderr, "unknown cfg in spec %s\n", sp.c_str()); exit(3); }
  return v == "v1a" ? ce->v1a(m, r) : v == "v1b" ? ce->v1b(m, r) : v == "v1c" ? ce->v1c(m, r) : v == "v3" ? ce->v3(m, r) : ce->v4(m, r);
}

int main(int argc, char** argv) {
  std::string mode = argc > 1 ? argv[1] : "full";
  int dev = 0; CK(cudaSetDevice(dev));
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
  CK(cudaDeviceGetAttribute(&g_smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
  register_all();
  auto arg = [&](const char* k, const char* def) -> std::string {
    for (int i = 2; i + 1 < argc; i++) if (!strcmp(argv[i], k)) return argv[i + 1];
    return def;
  };
  int steps = atoi(arg("--steps", "200").c_str()), warm = atoi(arg("--warmup", "20").c_str()), reps = atoi(arg("--repeats", "5").c_str());
  std::string outp = arg("--out", "");
  g_grid_override = atoi(arg("--grid", "0").c_str());
  FILE* out = outp.empty() ? stdout : fopen(outp.c_str(), "w");
  cudaStream_t st; CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));

  if (mode == "unit") {
    bool ok = true; std::string js;
    int nsm; CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
#define UNIT(NW, R, D) unit_cfg<Cfg<NW, R, D>>(#NW "x" #R "x" #D, ok, js);
    UNIT(16, 4, 1) UNIT(8, 4, 3) UNIT(16, 2, 2) UNIT(12, 4, 2)
    fprintf(out, "{\"unit\":[%s],\"all_ok\":%s}\n", js.c_str(), ok ? "true" : "false");
    return ok ? 0 : 1;
  }
  if (mode == "barrier") {
    Model m; m.nsm = prop.multiProcessorCount;
    return run_barrier(m, out);
  }

  // ---- full model path
  Model m; m.nsm = prop.multiProcessorCount; m.layers = atoi(arg("--layers", "64").c_str());
  size_t fr, tot; CK(cudaMemGetInfo(&fr, &tot));
  fprintf(stderr, "free %.2f GB of %.2f GB\n", fr / 1e9, tot / 1e9);
  double t0 = now_s();
  if (mode != "gemv") m.build_full();
  fprintf(stderr, "built model: %zu phases, %d GEMV, %.3f GB streamed per step, %.1fs\n", m.ph.size(), m.ngemv(), m.wbytes / 1e9, now_s() - t0);

  if (mode == "check") {
    Bufs rb{}; rb.x0 = m.b.x0;
    rb.x = (float*)m.alloc(HID * 4); rb.y1 = (float*)m.alloc(16384 * 4); rb.t = (float*)m.alloc(6144 * 4); rb.logits = (float*)m.alloc(248320 * 4);
    rb.c = (uint16_t*)m.alloc(6144 * 2); rb.h = (uint16_t*)m.alloc(INTER * 2);
    float* actf = (float*)m.alloc(INTER * 4); float* raw = (float*)m.alloc(34816 * 4);
    CK(cudaMemset(rb.x, 0, HID * 4));
    run_reference(m, rb, actf, raw);
    auto ref_log = d2h_f(rb.logits, 248320); auto ref_x = d2h_f(rb.x, HID);
    // noise floor: same naive reference with the reduction order reversed (reassociation only)
    CK(cudaMemset(rb.x, 0, HID * 4));
    run_reference(m, rb, actf, raw, 1);
    auto rev_log = d2h_f(rb.logits, 248320); auto rev_x = d2h_f(rb.x, HID);
    Err nl = compare(rev_log, ref_log), nx = compare(rev_x, ref_x);
    fprintf(out, "{\"reference\":{\"logits_rms\":%.4f,\"x_rms\":%.4f,\"reassoc_noise_floor_logits_rel_l2\":%.3e,\"reassoc_noise_floor_x_rel_l2\":%.3e}", compare(ref_log, ref_log).rms, compare(ref_x, ref_x).rms, nl.rel_l2, nx.rel_l2);
    // late-scale reference (RMSNorm scale applied to the dot product, as the dataflow variants do): same naive fp32 kernels
    CK(cudaMemset(rb.x, 0, HID * 4));
    run_reference(m, rb, actf, raw, 0, true);
    auto late_log = d2h_f(rb.logits, 248320); auto late_x = d2h_f(rb.x, HID);
    Err dl = compare(late_log, ref_log);
    fprintf(out, ",\"late_scale_reference_vs_ref\":{\"logits_rel_l2\":%.3e}", dl.rel_l2);
    fprintf(out, ",\"variants\":[");
    bool first = true; std::vector<float> base_log;
    std::string cname = arg("--cfg", "w8r2d5");
    CfgEntry* ce = find_cfg(cname);
    if (!ce) { fprintf(stderr, "unknown cfg %s\n", cname.c_str()); return 3; }
    struct V { std::string n; std::function<bool(Model&, Runner&)> f; };
    std::vector<V> vs;
    if (arg("--e1", "1") == "1") vs = {{"V1a", ce->v1a}, {"V1b", ce->v1b}, {"V3", ce->v3}, {"V4", ce->v4}};
    for (int i = 2; i < argc; i++) { if (argv[i][0] == '-' && argv[i][1] == '-') { i++; continue; } std::string sp = argv[i]; vs.push_back({sp, [sp](Model& mm, Runner& rr) { return make_runner(mm, sp, rr, [](Runner&) {}); }}); }
    for (auto& v : vs) {
      Runner r; if (!v.f(m, r)) { fprintf(stderr, "cfg infeasible for %s\n", v.n.c_str()); continue; }
      CK(cudaMemset(m.b.logits, 0, 248320 * 4)); CK(cudaMemset(m.b.x, 0, HID * 4));
      r.step(st); CK(cudaStreamSynchronize(st)); CK(cudaGetLastError());
      r.step(st); CK(cudaStreamSynchronize(st));  // second step: also checks epoch/counter reuse
      auto lg = d2h_f(m.b.logits, 248320); auto xx = d2h_f(m.b.x, HID);
      Err el = compare(lg, ref_log), ex = compare(xx, ref_x), ell = compare(lg, late_log), exl = compare(xx, late_x);
      double bit = base_log.empty() ? 0.0 : compare(lg, base_log).rel_l2;
      if (base_log.empty()) base_log = lg;
      fprintf(out, "%s{\"variant\":\"%s\",\"cfg\":\"%s\",\"logits_rel_l2_vs_ref\":%.3e,\"logits_max_abs_over_rms\":%.3e,\"x_rel_l2_vs_ref\":%.3e,\"logits_rel_l2_vs_late_ref\":%.3e,\"x_rel_l2_vs_late_ref\":%.3e,\"finite\":%s,\"logits_rel_l2_vs_first_variant\":%.3e}",
              first ? "" : ",", v.n.c_str(), cname.c_str(), el.rel_l2, el.max_abs_over_rms, ex.rel_l2, ell.rel_l2, exl.rel_l2, (el.finite && ex.finite) ? "true" : "false", bit);
      first = false;
    }
    fprintf(out, "]}\n");
    return 0;
  }

  if (mode == "profile") {
    // per-phase timeline of the persistent kernel (V3/V4): where does the time between phases go?
    std::string cname = arg("--cfg", "w8r4d1"); bool cross = arg("--cross", "1") == "1";
    CfgEntry* ce = find_cfg(cname);
    if (!ce) return 3;
    int nph = (int)m.ph.size(); int grid = 0;
    { Runner tmp; (cross ? ce->v4 : ce->v3)(m, tmp); size_t p0 = tmp.info.find("grid=") + 5; grid = atoi(tmp.info.c_str() + p0); }
    unsigned long long* dst; CK(cudaMalloc(&dst, (size_t)nph * grid * 4 * 8));
    g_stamps_for_runner = dst;
    Runner r; (cross ? ce->v4 : ce->v3)(m, r);
    g_stamps_for_runner = nullptr;
    for (int i = 0; i < 5; i++) { r.step(st); }
    CK(cudaStreamSynchronize(st));
    std::vector<double> acc; const int nrep = 20;
    std::vector<unsigned long long> h((size_t)nph * grid * 4);
    struct Agg { double n = 0, wall = 0, skew = 0, pro = 0, stream_med = 0, tail = 0, bar = 0; };
    std::map<std::string, Agg> agg; Agg tot;
    for (int rep = 0; rep < nrep; rep++) {
      r.step(st); CK(cudaStreamSynchronize(st));
      CK(cudaMemcpy(h.data(), dst, h.size() * 8, cudaMemcpyDeviceToHost));
      for (int p = 0; p < nph; p++) {
        std::vector<double> t0(grid), t1(grid), t2(grid), t3(grid);
        for (int b = 0; b < grid; b++) { auto* q = &h[((size_t)p * grid + b) * 4]; t0[b] = q[0]; t1[b] = q[1]; t2[b] = q[2]; t3[b] = q[3]; }
        double mn0 = *std::min_element(t0.begin(), t0.end()), mx0 = *std::max_element(t0.begin(), t0.end());
        double mx2 = *std::max_element(t2.begin(), t2.end()), mn3 = *std::min_element(t3.begin(), t3.end());
        std::vector<double> pro(grid), strm(grid), t2c = t2;
        for (int b = 0; b < grid; b++) { pro[b] = t1[b] - t0[b]; strm[b] = t2[b] - t1[b]; }
        double prom = 0; for (double v : pro) prom += v; prom /= grid;
        std::sort(t2c.begin(), t2c.end());
        double medarr = t2c[grid / 2];
        double smed = median(strm);
        // phase wall = from first block starting this phase to first block leaving the barrier (next phase start)
        double wall = (p + 1 < nph ? (double)h[((size_t)(p + 1) * grid) * 4] : mx2) ; (void)wall;
        double nxt = 1e300; if (p + 1 < nph) for (int b = 0; b < grid; b++) nxt = std::min(nxt, (double)h[((size_t)(p + 1) * grid + b) * 4]); else nxt = mx2;
        const Phase& d = m.ph[p];
        std::string key = !is_gemv(d.type) ? "core" : d.type == PH_BF16 ? "lm_head" : d.epi == EPI_GATEUP ? "gateup" : d.K == INTER ? "down" : d.K == 6144 ? "oproj" : d.N == 16384 ? "qkvz_gdn" : "qkv_attn";
        for (Agg* a : {&agg[key], &tot}) {
          a->n += 1; a->wall += nxt - mn0; a->skew += mx0 - mn0; a->pro += prom; a->stream_med += smed; a->tail += mx2 - medarr; a->bar += (p + 1 < nph ? mn3 - mx2 : 0);
        }
      }
    }
    fprintf(out, "{\"profile\":{\"cfg\":\"%s\",\"cross\":%d,\"grid\":%d,\"unit\":\"us per phase (mean over phases of that kind and %d steps)\",\"groups\":[", cname.c_str(), (int)cross, grid, nrep);
    bool first = true;
    auto emit = [&](const std::string& k, const Agg& a, double steps_n) {
      fprintf(out, "%s{\"kind\":\"%s\",\"phases_per_step\":%.0f,\"phase_wall_us\":%.2f,\"start_skew_us\":%.2f,\"prologue_us\":%.2f,\"stream_median_block_us\":%.2f,\"tail_us(last_arrival-median_arrival)\":%.2f,\"barrier_us(last_arrival->first_release)\":%.2f,\"ms_per_step_total\":%.3f}",
              first ? "" : ",", k.c_str(), a.n / steps_n, a.wall / a.n / 1e3, a.skew / a.n / 1e3, a.pro / a.n / 1e3, a.stream_med / a.n / 1e3, a.tail / a.n / 1e3, a.bar / a.n / 1e3, a.wall / steps_n / 1e6);
      first = false; };
    for (auto& kv : agg) emit(kv.first, kv.second, nrep);
    emit("ALL", tot, nrep);
    fprintf(out, "]}}\n");
    return 0;
  }

  // peak buffers for memcpy
  size_t cb = (size_t)1 << 30;
  void *csrc = nullptr, *cdst = nullptr;
  auto make_memcpy = [&](Runner& r) {
    if (!csrc) { CK(cudaMalloc(&csrc, cb)); CK(cudaMalloc(&cdst, cb)); CK(cudaMemset(csrc, 1, cb)); }
    r.step = [=](cudaStream_t s) { CK(cudaMemcpyAsync(cdst, csrc, cb, cudaMemcpyDeviceToDevice, s)); };
    r.bytes = 2.0 * cb;  // read + write
    r.info = "cudaMemcpyAsync D2D 1 GiB (bytes = read+write)";
  };

  if (mode == "gemv") {
    // single-matrix GEMV bandwidth per config: G distinct matrices of one shape, launched back-to-back in a graph
    struct Shape { const char* n; int N, K; bool bf16; } shapes[] = {
        {"gateup_34816x5120", 34816, 5120, false}, {"down_5120x17408", 5120, 17408, false}, {"qkvz_16384x5120", 16384, 5120, false},
        {"oproj_5120x6144", 5120, 6144, false}, {"lm_head_bf16_248320x5120", 248320, 5120, true}};
    fprintf(out, "{\"gemv\":[");
    bool first = true;
    std::string only = arg("--cfgs", "");
    for (auto& sh : shapes) {
      Model g; g.nsm = prop.multiProcessorCount;
      float* o = (float*)g.alloc((size_t)sh.N * 4);
      uint16_t* v = (uint16_t*)g.alloc((size_t)sh.K * 2);
      k_fill_bf16_vec<<<8, 256>>>(v, sh.K, 3);
      size_t one = sh.bf16 ? (size_t)sh.N * sh.K * 2 : (size_t)sh.N * sh.K * 33 / 64;
      int cnt = (int)std::max<size_t>(8, std::min<size_t>(64, ((size_t)3 << 29) / one));  // >= ~1.5 GB, distinct allocs
      for (int i = 0; i < cnt; i++) { if (sh.bf16) g.add_bf16(sh.N, sh.K, PRO_VEC, EPI_STORE, v, o); else g.add_w4(sh.N, sh.K, PRO_VEC, EPI_STORE, v, o); }
      g.finalize();
      Runner r0; make_v0g(g, r0);
      for (int pass = 0; pass < 2; pass++) {
        for (auto& ce : registry()) {
          if (!only.empty() && ("," + only + ",").find("," + ce.name + ",") == std::string::npos) continue;
          Runner r; bool ok = (pass == 0) ? ce.v1a(g, r) : ce.v1b(g, r);
          if (!ok) continue;
          std::vector<double> med;
          for (int k = 0; k < 3; k++) { auto t = time_steps(r.step, 3, 10, st); med.push_back(medianf(t)); }
          double ms = median(med);
          fprintf(out, "%s{\"shape\":\"%s\",\"kernel\":\"%s\",\"cfg\":\"%s\",\"ms_per_pass\":%.4f,\"GBps\":%.1f,\"matrices\":%d,\"bytes\":%.0f,\"info\":\"%s\"}", first ? "" : ",", sh.n, pass ? "V1b" : "V1a", ce.name.c_str(), ms, g.wbytes / (ms * 1e6), cnt, (double)g.wbytes, r.info.c_str());
          first = false;
        }
      }
      { std::vector<double> med; for (int k = 0; k < 3; k++) { auto t = time_steps(r0.step, 3, 10, st); med.push_back(medianf(t)); }
        double ms = median(med);
        fprintf(out, ",{\"shape\":\"%s\",\"kernel\":\"read_only_independent\",\"cfg\":\"-\",\"ms_per_pass\":%.4f,\"GBps\":%.1f,\"matrices\":%d,\"bytes\":%.0f}", sh.n, ms, g.wbytes / (ms * 1e6), cnt, (double)g.wbytes); }
      fflush(out);
      for (auto p : g.allocs) cudaFree(p);
    }
    fprintf(out, "]}\n");
    return 0;
  }

  // ---- full: interleaved repeats over a list of variant specs
  std::vector<std::string> specs;
  for (int i = 2; i < argc; i++) { if (argv[i][0] == '-' && argv[i][1] == '-') { i++; continue; } specs.push_back(argv[i]); }
  std::vector<Runner> rs;
  for (auto& sp : specs) {
    Runner r; r.name = sp;
    bool ok = make_runner(m, sp, r, make_memcpy);
    if (!ok) { fprintf(stderr, "spec %s infeasible, skipped\n", sp.c_str()); continue; }
    r.name = sp; rs.push_back(r);
  }
  std::vector<std::vector<double>> rep_med(rs.size()), rep_mean(rs.size());
  std::vector<std::vector<float>> all_steps(rs.size());
  for (int rep = 0; rep < reps; rep++) {
    for (size_t i = 0; i < rs.size(); i++) {
      fprintf(stderr, "MARK %s rep=%d start %.3f\n", rs[i].name.c_str(), rep, now_s());
      auto t = time_steps(rs[i].step, warm, steps, st);
      fprintf(stderr, "MARK %s rep=%d end %.3f\n", rs[i].name.c_str(), rep, now_s());
      rep_med[i].push_back(medianf(t));
      double s = 0; for (float f : t) s += f; rep_mean[i].push_back(s / t.size());
      all_steps[i].insert(all_steps[i].end(), t.begin(), t.end());
    }
  }
  fprintf(out, "{\"steps\":%d,\"warmup\":%d,\"repeats\":%d,\"bytes_per_step_weights\":%.0f,\"phases\":%zu,\"gemv_phases\":%d,\"results\":[", steps, warm, reps, (double)m.wbytes, m.ph.size(), m.ngemv());
  for (size_t i = 0; i < rs.size(); i++) {
    double med = median(rep_med[i]);
    double mn = *std::min_element(rep_med[i].begin(), rep_med[i].end()), mx = *std::max_element(rep_med[i].begin(), rep_med[i].end());
    double p10 = 0, p90 = 0; { auto v = all_steps[i]; std::sort(v.begin(), v.end()); p10 = v[v.size() / 10]; p90 = v[v.size() * 9 / 10]; }
    fprintf(out, "%s{\"name\":\"%s\",\"info\":\"%s\",\"median_ms\":%.4f,\"min_rep_median_ms\":%.4f,\"max_rep_median_ms\":%.4f,\"step_p10_ms\":%.4f,\"step_p90_ms\":%.4f,\"mean_ms_median_of_reps\":%.4f,\"rep_medians_ms\":[", i ? "," : "", rs[i].name.c_str(), rs[i].info.c_str(), med, mn, mx, p10, p90, median(rep_mean[i]));
    for (size_t k = 0; k < rep_med[i].size(); k++) fprintf(out, "%s%.4f", k ? "," : "", rep_med[i][k]);
    fprintf(out, "],\"bytes\":%.0f,\"GBps\":%.1f}", rs[i].bytes, rs[i].bytes / (med * 1e6));
  }
  fprintf(out, "]}\n");
  return 0;
}
