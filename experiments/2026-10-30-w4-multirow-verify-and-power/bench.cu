// Host driver: modes  check | q1 | q2 | probe.   Build: nvcc -O3 -std=c++17 -arch=sm_86 -Xptxas -v bench.cu -o bench   (see run.py)
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdarg>
#include <functional>
#include <map>
#include <string>
#include <vector>
#include "kernels.cuh"

using namespace e3;
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorName(e_), __FILE__, __LINE__, cudaGetErrorString(e_)); exit(2); } } while (0)

static double now_s() { return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count(); }
static std::string fmt(const char* f, ...) {
  char b[2048]; va_list a; va_start(a, f); vsnprintf(b, sizeof b, f, a); va_end(a); return b;
}

// ===================================================================== data generation / helpers
__device__ __forceinline__ uint32_t hash32(uint64_t x) {
  x ^= x >> 33; x *= 0xff51afd7ed558ccdull; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ull; x ^= x >> 33; return (uint32_t)x;
}
__global__ void k_fill_u32(uint32_t* p, size_t n, uint64_t seed) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) p[i] = hash32(seed + i);
}
__global__ void k_fill_bf16(uint16_t* p, size_t n, uint64_t seed, float amp) {  // uniform [-amp, amp)
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    float f = ((hash32(seed + i) & 0xffff) / 32768.f - 1.f) * amp;
    p[i] = __bfloat16_as_ushort(__float2bfloat16_rn(f));
  }
}
__global__ void k_fill_scale(uint16_t* p, size_t n, uint64_t seed, float base) {  // bf16 scale in [0.5, 1.5) * base
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    float f = (0.5f + (hash32(seed + i) & 0xffff) / 65536.f) * base;
    p[i] = __bfloat16_as_ushort(__float2bfloat16_rn(f));
  }
}
// x natural [16][K] bf16 -> XP (W4 mma order [x0,x4,x1,x5,x2,x6,x3,x7] per 8 k), XS (sum per 128-k group), XCC (lane-permuted, plain / prescaled)
__global__ void k_build_x(const uint16_t* x, uint4* xp, float* xs, uint4* xcc, uint4* xcc_ps, int K) {
  const int tk = blockIdx.y;
  const int nch = K / 8;
  for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < nch; j += gridDim.x * blockDim.x) {
    const uint16_t* p = x + (size_t)tk * K + 8 * j;
    uint32_t v[8];
    for (int e = 0; e < 8; e++) v[e] = p[e];
    xp[(size_t)tk * nch + j] = make_uint4(v[0] | (v[4] << 16), v[1] | (v[5] << 16), v[2] | (v[6] << 16), v[3] | (v[7] << 16));
    if (tk == 0) {  // CC layouts for token 0 (CUDA-core path is T=1)
      uint4 plain = make_uint4(v[0] | (v[1] << 16), v[2] | (v[3] << 16), v[4] | (v[5] << 16), v[6] | (v[7] << 16));
      uint32_t q[8];
      for (int e = 0; e < 8; e++) {
        const float t[8] = {8388608.f, 524288.f, 32768.f, 2048.f, 128.f, 8388608.f, 524288.f, 32768.f};
        float f = __uint_as_float(v[e] << 16) * t[e];
        q[e] = (uint32_t)__bfloat16_as_ushort(__float2bfloat16_rn(f));
      }
      uint4 ps = make_uint4(q[0] | (q[1] << 16), q[2] | (q[3] << 16), q[4] | (q[5] << 16), q[6] | (q[7] << 16));
      const int tile = j >> 7, l = (j & 127) >> 2, c = j & 3;
      const size_t slot = (size_t)tile * 128 + c * 32 + l;
      xcc[slot] = plain; xcc_ps[slot] = ps;
    }
  }
  // group sums
  const int nkg = K / 128;
  for (int kg = blockIdx.x * blockDim.x + threadIdx.x; kg < nkg; kg += gridDim.x * blockDim.x) {
    float s = 0.f;
    for (int e = 0; e < 128; e++) s += __uint_as_float((uint32_t)x[(size_t)tk * K + kg * 128 + e] << 16);
    xs[(size_t)tk * nkg + kg] = s;
  }
}
// bf16 lm_head: CC lane-contiguous natural layout = x itself; mma B natural layout = x itself.

// fp32 references (warp per row), all 16 tokens: ref[row*16+tk]
__global__ void k_ref_w4(const uint4* W, const uint16_t* S, const uint16_t* X, float* ref, int N, int K) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  if (warp >= N) return;
  float acc[16];
  for (int i = 0; i < 16; i++) acc[i] = 0.f;
  for (int j = lane; j < K / 32; j += 32) {
    const uint4 w = W[(size_t)warp * (K / 32) + j];
    const float s = __uint_as_float((uint32_t)S[(size_t)warp * (K / 128) + j / 4] << 16);
    const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
    for (int c = 0; c < 4; c++)
      for (int e = 0; e < 8; e++) {
        const float wv = (float)((int)((ww[c] >> (4 * e)) & 15u) - 8) * s;
        const int k = j * 32 + c * 8 + e;
        for (int tk = 0; tk < 16; tk++) acc[tk] = fmaf(wv, __uint_as_float((uint32_t)X[(size_t)tk * K + k] << 16), acc[tk]);
      }
  }
  for (int tk = 0; tk < 16; tk++) { float v = warp_sum(acc[tk]); if (lane == 0) ref[(size_t)warp * 16 + tk] = v; }
}
__global__ void k_ref_bf16(const uint4* W, const uint16_t* X, float* ref, int N, int K) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  if (warp >= N) return;
  float acc[16];
  for (int i = 0; i < 16; i++) acc[i] = 0.f;
  for (int j = lane; j < K / 8; j += 32) {
    const uint4 w = W[(size_t)warp * (K / 8) + j];
    const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
    for (int c = 0; c < 4; c++)
      for (int e = 0; e < 2; e++) {
        const float wv = __uint_as_float((e ? (ww[c] & 0xffff0000u) : (ww[c] << 16)));
        const int k = j * 8 + c * 2 + e;
        for (int tk = 0; tk < 16; tk++) acc[tk] = fmaf(wv, __uint_as_float((uint32_t)X[(size_t)tk * K + k] << 16), acc[tk]);
      }
  }
  for (int tk = 0; tk < 16; tk++) { float v = warp_sum(acc[tk]); if (lane == 0) ref[(size_t)warp * 16 + tk] = v; }
}

// ===================================================================== shapes and buffers
struct Shape { std::string name; int N, K; bool w4; int count_gdn, count_attn; };
static const Shape SHAPES[] = {
    {"gateup", 34816, 5120, true, 48, 16}, {"down", 5120, 17408, true, 48, 16}, {"qkvz", 16384, 5120, true, 48, 0},
    {"oproj", 5120, 6144, true, 48, 16}, {"lmhead", 248320, 5120, false, 0, 0}};

struct Ctx {
  Shape sh;
  int nb = 0;
  std::vector<uint4*> W; std::vector<uint16_t*> S;
  uint16_t* x = nullptr; uint4 *xp = nullptr, *xcc = nullptr, *xcc_ps = nullptr; float* xs = nullptr;
  void* out = nullptr; float* ref = nullptr;
  size_t wbytes = 0, sbytes = 0;
  size_t bytes() const { return wbytes + sbytes; }
};
static int g_nsm = 82;

static void build_ctx(Ctx& c, const Shape& sh, size_t target_bytes = 400ull << 20) {
  c.sh = sh;
  c.wbytes = sh.w4 ? (size_t)sh.N * sh.K / 2 : (size_t)sh.N * sh.K * 2;
  c.sbytes = sh.w4 ? (size_t)sh.N * sh.K / 64 : 0;
  c.nb = (int)std::min<size_t>(16, std::max<size_t>(2, (target_bytes + c.bytes() - 1) / c.bytes()));
  for (int i = 0; i < c.nb; i++) {
    uint4* w; CK(cudaMalloc(&w, c.wbytes));
    uint64_t seed = 0x9E3779B97F4A7C15ull * (i + 1) + sh.N;
    if (sh.w4) k_fill_u32<<<2048, 256>>>((uint32_t*)w, c.wbytes / 4, seed);
    else k_fill_bf16<<<2048, 256>>>((uint16_t*)w, c.wbytes / 2, seed, 1.0f);
    c.W.push_back(w);
    uint16_t* s = nullptr;
    if (sh.w4) { CK(cudaMalloc(&s, c.sbytes)); k_fill_scale<<<256, 256>>>(s, c.sbytes / 2, seed + 77, 0.10f / sqrtf((float)sh.K)); }
    c.S.push_back(s);
  }
  const int K = sh.K;
  CK(cudaMalloc(&c.x, (size_t)16 * K * 2));
  k_fill_bf16<<<256, 256>>>(c.x, (size_t)16 * K, 4242, 1.0f);
  CK(cudaMalloc(&c.xp, (size_t)16 * K * 2)); CK(cudaMalloc(&c.xs, (size_t)16 * (K / 128) * 4));
  CK(cudaMalloc(&c.xcc, (size_t)K * 2)); CK(cudaMalloc(&c.xcc_ps, (size_t)K * 2));
  k_build_x<<<dim3(8, 16), 256>>>(c.x, c.xp, c.xs, c.xcc, c.xcc_ps, K);
  CK(cudaMalloc(&c.out, (size_t)sh.N * 16 * 4));
  CK(cudaMalloc(&c.ref, (size_t)sh.N * 16 * 4));
  CK(cudaDeviceSynchronize());
}
static void free_ctx(Ctx& c) {
  for (auto p : c.W) cudaFree(p);
  for (auto p : c.S) cudaFree(p);
  cudaFree(c.x); cudaFree(c.xp); cudaFree(c.xs); cudaFree(c.xcc); cudaFree(c.xcc_ps); cudaFree(c.out); cudaFree(c.ref);
  c = Ctx();
}

// ===================================================================== variants
struct Var {
  std::string kind;  // read | cc | mma
  std::string mode;  // sub|ones|abl|e1|naive|ablfma|ablload|bf16|read
  std::string cfg;
  bool valid = true;       // numerically meaningful output
  int outkind = 0;         // 0: fp32 out[N]  (T=1),  1: bf16 out[N][TP]
  int TP = 0;
  std::function<void(int, cudaStream_t)> run;  // run(bufidx, stream)
};

// Persistent grids: every warp must do the same amount of work (DRAM-bound => all resident warps progress at the same rate), so use the
// largest divisor of `groups` that fits in the resident capacity (occupancy * SMs); falls back to the capacity if that divisor is < 60% of it.
template <class K>
static int grid_for(K kern, int threads, int groups) {
  int occ = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kern, threads, 0));
  if (occ < 1) { fprintf(stderr, "occupancy 0\n"); exit(2); }
  const int cap = std::min(groups, occ * g_nsm);
  int d = cap;
  while (d > 1 && groups % d) --d;
  return d * 10 >= cap * 6 ? d : cap;
}

template <int MODE, int NT, int NW, int KS, int U, int MINB>
static void add_mma_w4(std::vector<Var>& v, Ctx& c, int T, const char* mode) {
  const int N = c.sh.N, K = c.sh.K;
  if ((K >> 7) % KS != 0 || ((N >> 4) % (NW / KS)) != 0) return;
  auto kern = k_mma_w4<MODE, NT, NW, KS, U, MINB>;
  const int groups = (N >> 4) / (NW / KS);
  const int grid = grid_for(kern, NW * 32, groups);
  Var x; x.kind = "mma"; x.mode = mode; x.cfg = fmt("w%dk%du%d", NW, KS, U); x.valid = MODE < MM_ABL; x.outkind = 1; x.TP = 8 * NT;
  Ctx* cp = &c;
  x.run = [=](int i, cudaStream_t st) { kern<<<grid, NW * 32, 0, st>>>(cp->W[i], cp->S[i], cp->xp, cp->xs, cp->out, N, K, T); };
  v.push_back(x);
}
template <int NT, int NW, int KS, int U, int MINB>
static void add_mma_bf16(std::vector<Var>& v, Ctx& c, int T) {
  const int N = c.sh.N, K = c.sh.K;
  if ((K >> 5) % KS != 0 || ((N >> 4) % (NW / KS)) != 0) return;
  auto kern = k_mma_bf16<NT, NW, KS, U, MINB>;
  const int groups = (N >> 4) / (NW / KS);
  const int grid = grid_for(kern, NW * 32, groups);
  Var x; x.kind = "mma"; x.mode = "bf16"; x.cfg = fmt("w%dk%du%d", NW, KS, U); x.outkind = 1; x.TP = 8 * NT;
  Ctx* cp = &c;
  x.run = [=](int i, cudaStream_t st) { kern<<<grid, NW * 32, 0, st>>>((const uint4*)cp->W[i], (const uint4*)cp->x, cp->out, N, K, T); };
  v.push_back(x);
}
template <bool W4, int MODE, int NW, int R, int U, int MINB>
static void add_cc(std::vector<Var>& v, Ctx& c, const char* mode) {
  const int N = c.sh.N, K = c.sh.K;
  if (N % R) return;
  auto kern = k_cc<W4, MODE, NW, R, U, MINB>;
  const int grid = grid_for(kern, NW * 32, (N / R + NW - 1) / NW);  // tasks divisible by grid*NW when possible
  Var x; x.kind = "cc"; x.mode = mode; x.cfg = fmt("w%dr%du%d", NW, R, U);
  x.valid = !(W4 && (MODE == CC_ABL_FMA || MODE == CC_ABL_LOAD)); x.outkind = 0; x.TP = 1;
  Ctx* cp = &c;
  x.run = [=](int i, cudaStream_t st) {
    const uint4* xa = W4 ? (MODE == CC_E1 ? cp->xcc_ps : cp->xcc) : (const uint4*)cp->x;
    kern<<<grid, NW * 32, 0, st>>>((const uint4*)cp->W[i], cp->S[i], xa, (float*)cp->out, N, K);
  };
  v.push_back(x);
}
static void add_read(std::vector<Var>& v, Ctx& c) {
  Var x; x.kind = "read"; x.mode = "read"; x.cfg = "g246x512"; x.valid = false;
  Ctx* cp = &c;
  static uint32_t* sink = nullptr;
  if (!sink) CK(cudaMalloc(&sink, 16));
  x.run = [=](int i, cudaStream_t st) {
    k_read<<<g_nsm * 3, 512, 0, st>>>((const uint4*)cp->W[i], cp->wbytes / 16, (const uint4*)cp->S[i], cp->sbytes / 16, sink);
  };
  v.push_back(x);
}

#define MMA_CFGS(MODE, NT, NAME)                       \
  add_mma_w4<MODE, NT, 8, 8, 2, 2>(v, c, T, NAME);     \
  add_mma_w4<MODE, NT, 8, 4, 2, 2>(v, c, T, NAME);     \
  add_mma_w4<MODE, NT, 8, 2, 3, 2>(v, c, T, NAME);     \
  add_mma_w4<MODE, NT, 4, 4, 3, 4>(v, c, T, NAME);     \
  add_mma_w4<MODE, NT, 16, 8, 2, 1>(v, c, T, NAME);    \
  add_mma_w4<MODE, NT, 8, 1, 3, 2>(v, c, T, NAME);

static std::vector<Var> make_variants(Ctx& c, int T, bool include_cc, bool include_abl_cc) {
  std::vector<Var> v;
  if (c.sh.w4) {
    if (T <= 8) { MMA_CFGS(MM_SUB, 1, "sub") MMA_CFGS(MM_ONES, 1, "ones") MMA_CFGS(MM_ABL, 1, "abl") MMA_CFGS(MM_ABLB, 1, "ablb") }
    else { MMA_CFGS(MM_SUB, 2, "sub") MMA_CFGS(MM_ONES, 2, "ones") MMA_CFGS(MM_ABL, 2, "abl") MMA_CFGS(MM_ABLB, 2, "ablb") }
    if (include_cc && T == 1) {
      add_cc<true, CC_E1, 8, 2, 2, 2>(v, c, "e1"); add_cc<true, CC_E1, 8, 2, 4, 2>(v, c, "e1");
      add_cc<true, CC_E1, 8, 4, 2, 2>(v, c, "e1"); add_cc<true, CC_E1, 8, 4, 4, 2>(v, c, "e1");
      add_cc<true, CC_NAIVE, 8, 2, 2, 2>(v, c, "naive"); add_cc<true, CC_NAIVE, 8, 2, 4, 2>(v, c, "naive");
      add_cc<true, CC_NAIVE, 8, 4, 2, 2>(v, c, "naive"); add_cc<true, CC_NAIVE, 8, 4, 4, 2>(v, c, "naive");
      if (include_abl_cc) {
        add_cc<true, CC_ABL_FMA, 8, 2, 2, 2>(v, c, "ablfma"); add_cc<true, CC_ABL_FMA, 8, 2, 4, 2>(v, c, "ablfma");
        add_cc<true, CC_ABL_FMA, 8, 4, 2, 2>(v, c, "ablfma"); add_cc<true, CC_ABL_FMA, 8, 4, 4, 2>(v, c, "ablfma");
        add_cc<true, CC_ABL_LOAD, 8, 2, 2, 2>(v, c, "ablload"); add_cc<true, CC_ABL_LOAD, 8, 2, 4, 2>(v, c, "ablload");
        add_cc<true, CC_ABL_LOAD, 8, 4, 2, 2>(v, c, "ablload"); add_cc<true, CC_ABL_LOAD, 8, 4, 4, 2>(v, c, "ablload");
      }
    }
  } else {
    if (T <= 8) {
      add_mma_bf16<1, 8, 8, 4, 2>(v, c, T); add_mma_bf16<1, 8, 4, 4, 2>(v, c, T); add_mma_bf16<1, 16, 8, 3, 1>(v, c, T); add_mma_bf16<1, 8, 1, 6, 2>(v, c, T);
    } else {
      add_mma_bf16<2, 8, 8, 4, 2>(v, c, T); add_mma_bf16<2, 8, 4, 4, 2>(v, c, T); add_mma_bf16<2, 16, 8, 3, 1>(v, c, T); add_mma_bf16<2, 8, 1, 6, 2>(v, c, T);
    }
    if (include_cc && T == 1) {
      add_cc<false, 0, 8, 2, 4, 2>(v, c, "bf16"); add_cc<false, 0, 8, 4, 4, 2>(v, c, "bf16"); add_cc<false, 0, 8, 2, 8, 2>(v, c, "bf16");
    }
  }
  if (T == 1) add_read(v, c);
  return v;
}

// ===================================================================== timing (CUDA graph of nb launches over distinct buffers)
struct Timing { double ms_med = 0, ms_min = 0, t0 = 0, t1 = 0; int launches = 0; };

static Timing time_var(Var& v, Ctx& c, double warm_s, double win_s) {
  cudaStream_t st; CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  cudaGraph_t g; cudaGraphExec_t ge;
  CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
  for (int i = 0; i < c.nb; i++) v.run(i, st);
  CK(cudaStreamEndCapture(st, &g));
  CK(cudaGraphInstantiate(&ge, g, 0));
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  double t = now_s();
  do { CK(cudaGraphLaunch(ge, st)); CK(cudaStreamSynchronize(st)); } while (now_s() - t < warm_s);
  std::vector<double> ms;
  Timing r; r.t0 = now_s();
  do {
    CK(cudaEventRecord(e0, st)); CK(cudaGraphLaunch(ge, st)); CK(cudaEventRecord(e1, st)); CK(cudaEventSynchronize(e1));
    float m; CK(cudaEventElapsedTime(&m, e0, e1)); ms.push_back(m / c.nb);
  } while (now_s() - r.t0 < win_s || ms.size() < 5);
  r.t1 = now_s();
  std::sort(ms.begin(), ms.end());
  r.ms_med = ms[ms.size() / 2]; r.ms_min = ms[0]; r.launches = (int)ms.size() * c.nb;
  CK(cudaGetLastError());
  CK(cudaGraphExecDestroy(ge)); CK(cudaGraphDestroy(g)); CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1)); CK(cudaStreamDestroy(st));
  return r;
}

static void mark(const std::string& name, const Timing& t) {
  fprintf(stderr, "MARK %s start %.3f\nMARK %s end %.3f\n", name.c_str(), t.t0, name.c_str(), t.t1);
}

// ===================================================================== correctness
static float bf2f(uint16_t u) { uint32_t b = (uint32_t)u << 16; float f; memcpy(&f, &b, 4); return f; }

static void compute_ref(Ctx& c) {
  const int N = c.sh.N, K = c.sh.K;
  const int thr = 256, blocks = (N * 32 + thr - 1) / thr;
  if (c.sh.w4) k_ref_w4<<<blocks, thr>>>(c.W[0], c.S[0], c.x, c.ref, N, K);
  else k_ref_bf16<<<blocks, thr>>>(c.W[0], c.x, c.ref, N, K);
  CK(cudaDeviceSynchronize());
}
// rel L2 of variant output (T tokens) vs fp32 reference; also the bf16-rounding floor of the reference itself
static double g_rel_bfref = 0;  // rel L2 of the bf16 output vs bf16(ref): isolates the kernel's own error from output rounding
static void check_var(Var& v, Ctx& c, int T, double& rel, double& floor_rel, std::vector<float>& href) {
  const int N = c.sh.N;
  CK(cudaMemset(c.out, 0xFF, (size_t)N * 16 * 4));  // NaN fill, so unwritten outputs are caught
  v.run(0, 0);
  CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
  double num = 0, den = 0, fnum = 0, n2 = 0;
  g_rel_bfref = 0;
  if (v.outkind == 0) {
    std::vector<float> o(N); CK(cudaMemcpy(o.data(), c.out, (size_t)N * 4, cudaMemcpyDeviceToHost));
    for (int n = 0; n < N; n++) { double r = href[(size_t)n * 16], d = (double)o[n] - r; num += d * d; den += r * r; }
    fnum = 0;
  } else {
    std::vector<uint16_t> o((size_t)N * v.TP); CK(cudaMemcpy(o.data(), c.out, o.size() * 2, cudaMemcpyDeviceToHost));
    for (int n = 0; n < N; n++)
      for (int t = 0; t < T; t++) {
        double r = href[(size_t)n * 16 + t], d = (double)bf2f(o[(size_t)n * v.TP + t]) - r;
        float rb = bf2f(__bfloat16_as_ushort(__float2bfloat16_rn((float)r)));
        num += d * d; den += r * r; fnum += ((double)rb - r) * ((double)rb - r);
        n2 += (double)(bf2f(o[(size_t)n * v.TP + t]) - rb) * (bf2f(o[(size_t)n * v.TP + t]) - rb);
      }
  }
  rel = std::sqrt(num / den); floor_rel = std::sqrt(fnum / den); g_rel_bfref = std::sqrt(n2 / den);
}

// ===================================================================== JSON helpers
static std::string jstr(const std::string& s) { return "\"" + s + "\""; }

// ===================================================================== modes
static std::vector<Shape> pick_shapes(const std::string& list) {
  std::vector<Shape> r;
  for (auto& s : SHAPES) if (list.empty() || list.find(s.name) != std::string::npos) r.push_back(s);
  return r;
}
static std::vector<int> parse_ints(const std::string& s) {
  std::vector<int> r; size_t p = 0;
  while (p < s.size()) { r.push_back(atoi(s.c_str() + p)); p = s.find(',', p); if (p == std::string::npos) break; ++p; }
  return r;
}

static int mode_check(const std::string& shapes, const std::string& out) {
  std::string js = "{\"results\":[\n"; bool first = true;
  for (auto& sh : pick_shapes(shapes)) {
    Ctx c; build_ctx(c, sh, 1);  // 2 buffers (min) is enough for correctness
    compute_ref(c);
    std::vector<float> href((size_t)sh.N * 16);
    CK(cudaMemcpy(href.data(), c.ref, href.size() * 4, cudaMemcpyDeviceToHost));
    for (int T : {1, 3, 8, 12, 16}) {
      auto vars = make_variants(c, T, true, false);
      for (auto& v : vars) {
        if (!v.valid || v.kind == "read") continue;
        if (v.kind == "cc" && T != 1) continue;
        double rel, fl; check_var(v, c, T, rel, fl, href);
        fprintf(stderr, "check %s T=%d %s/%s/%s rel=%.3e floor=%.3e\n", sh.name.c_str(), T, v.kind.c_str(), v.mode.c_str(), v.cfg.c_str(), rel, fl);
        js += (first ? "" : ",\n") + fmt("{\"shape\":\"%s\",\"T\":%d,\"kind\":\"%s\",\"mode\":\"%s\",\"cfg\":\"%s\",\"rel_l2\":%.4e,\"bf16_round_floor\":%.4e,\"rel_vs_bf16_ref\":%.4e}",
                                       sh.name.c_str(), T, v.kind.c_str(), v.mode.c_str(), v.cfg.c_str(), rel, fl, g_rel_bfref);
        first = false;
      }
    }
    free_ctx(c);
  }
  js += "\n]}\n";
  FILE* f = fopen(out.c_str(), "w"); fputs(js.c_str(), f); fclose(f);
  return 0;
}

static const int TLIST[] = {1, 2, 3, 4, 5, 6, 8, 12, 16};

// q1: for every shape and T, time every config of every mode (short window), then re-time the best config per (shape, kind/mode, T) 3x interleaved.
static int mode_q1(const std::string& shapes, const std::string& out, const std::string& tl) {
  std::vector<int> Ts = tl.empty() ? std::vector<int>(TLIST, TLIST + 9) : parse_ints(tl);
  std::string js_sweep, js_final; bool f1 = true, f2 = true;
  for (auto& sh : pick_shapes(shapes)) {
    Ctx c; build_ctx(c, sh);
    fprintf(stderr, "== shape %s N=%d K=%d bytes=%.1f MB nb=%d\n", sh.name.c_str(), sh.N, sh.K, c.bytes() / 1e6, c.nb);
    // key: mode|T -> (best ms, cfg index)
    std::map<std::string, std::pair<double, std::string>> best;
    for (int T : Ts) {
      auto vars = make_variants(c, T, true, true);
      for (auto& v : vars) {
        Timing t = time_var(v, c, 0.10, 0.25);
        double gbs = c.bytes() / (t.ms_med * 1e-3) / 1e9;
        std::string key = v.kind + "/" + v.mode + "|" + std::to_string(T);
        if (!best.count(key) || t.ms_med < best[key].first) best[key] = {t.ms_med, v.cfg};
        js_sweep += (f1 ? "" : ",\n") + fmt("{\"shape\":\"%s\",\"T\":%d,\"kind\":\"%s\",\"mode\":\"%s\",\"cfg\":\"%s\",\"ms\":%.5f,\"gbs\":%.1f}",
                                          sh.name.c_str(), T, v.kind.c_str(), v.mode.c_str(), v.cfg.c_str(), t.ms_med, gbs);
        f1 = false;
      }
      fprintf(stderr, "  swept T=%d (%zu variants)\n", T, vars.size());
    }
    // final: 3 interleaved repeats at the best config, 0.6 s windows
    struct Item { int T; Var v; std::string key; std::vector<double> ms; double t0 = 0, t1 = 0; };
    std::vector<Item> items;
    for (int T : Ts) {
      auto vars = make_variants(c, T, true, true);
      for (auto& v : vars) {
        std::string key = v.kind + "/" + v.mode + "|" + std::to_string(T);
        if (best.count(key) && best[key].second == v.cfg) items.push_back({T, v, key, {}, 0, 0});
      }
    }
    for (int rep = 0; rep < 3; rep++)
      for (auto& it : items) {
        Timing t = time_var(it.v, c, 0.30, 0.60);
        it.ms.push_back(t.ms_med);
        if (rep == 1) { it.t0 = t.t0; it.t1 = t.t1; mark(sh.name + "/" + it.key, t); }
      }
    for (auto& it : items) {
      std::vector<double> m = it.ms; std::sort(m.begin(), m.end());
      js_final += (f2 ? "" : ",\n") + fmt("{\"shape\":\"%s\",\"T\":%d,\"kind\":\"%s\",\"mode\":\"%s\",\"cfg\":\"%s\",\"ms\":%.5f,\"ms_reps\":[%.5f,%.5f,%.5f],\"gbs\":%.1f,\"t0\":%.3f,\"t1\":%.3f}",
                                        sh.name.c_str(), it.T, it.v.kind.c_str(), it.v.mode.c_str(), it.v.cfg.c_str(), m[1],
                                        it.ms[0], it.ms[1], it.ms[2], c.bytes() / (m[1] * 1e-3) / 1e9, it.t0, it.t1);
      f2 = false;
    }
    free_ctx(c);
  }
  std::string js = "{\"sweep\":[\n" + js_sweep + "\n],\n\"final\":[\n" + js_final + "\n]}\n";
  FILE* f = fopen(out.c_str(), "w"); fputs(js.c_str(), f); fclose(f);
  return 0;
}

// q2: sustained power/clock windows.  For each shape: best config per (kind, mode) at T=1 from a short sweep, then each variant is run
// back-to-back (interleaved order, 2 rounds) for warm 2 s + window 3 s with MARK lines for nvidia-smi 10 Hz matching.
static int mode_q2(const std::string& shapes, const std::string& out) {
  std::string js; bool first = true;
  for (auto& sh : pick_shapes(shapes)) {
    Ctx c; build_ctx(c, sh);
    auto vars = make_variants(c, 1, true, true);
    std::map<std::string, std::pair<double, int>> best;
    for (size_t i = 0; i < vars.size(); i++) {
      Timing t = time_var(vars[i], c, 0.1, 0.25);
      std::string key = vars[i].kind + "/" + vars[i].mode;
      if (!best.count(key) || t.ms_med < best[key].first) best[key] = {t.ms_med, (int)i};
    }
    std::vector<int> sel;
    for (auto& kv : best) sel.push_back(kv.second.second);
    fprintf(stderr, "== q2 shape %s: %zu variants selected\n", sh.name.c_str(), sel.size());
    for (int round = 0; round < 2; round++)
      for (int i : sel) {
        Var& v = vars[i];
        Timing t = time_var(v, c, 2.0, 3.0);
        double gbs = c.bytes() / (t.ms_med * 1e-3) / 1e9;
        std::string key = sh.name + "/" + v.kind + "/" + v.mode + "/r" + std::to_string(round);
        mark(key, t);
        fprintf(stderr, "q2 %s cfg=%s %.4f ms %.1f GB/s\n", key.c_str(), v.cfg.c_str(), t.ms_med, gbs);
        js += (first ? "" : ",\n") + fmt("{\"shape\":\"%s\",\"kind\":\"%s\",\"mode\":\"%s\",\"cfg\":\"%s\",\"round\":%d,\"valid\":%s,\"ms\":%.5f,\"ms_min\":%.5f,\"gbs\":%.1f,\"t0\":%.3f,\"t1\":%.3f,\"launches\":%d}",
                                       sh.name.c_str(), v.kind.c_str(), v.mode.c_str(), v.cfg.c_str(), round, v.valid ? "true" : "false", t.ms_med, t.ms_min, gbs, t.t0, t.t1, t.launches);
        first = false;
      }
    free_ctx(c);
  }
  FILE* f = fopen(out.c_str(), "w"); fprintf(f, "{\"results\":[\n%s\n]}\n", js.c_str()); fclose(f);
  return 0;
}

// one: run a single variant a few times (for ncu): bench one --shapes gateup --T 8 --var mma/sub/w8k4u2
static int mode_one(const std::string& shapes, const std::string& tl, const std::string& var) {
  Ctx c; build_ctx(c, pick_shapes(shapes)[0]);
  const int T = atoi(tl.c_str());
  auto vars = make_variants(c, T, true, true);
  for (auto& v : vars)
    if (v.kind + "/" + v.mode + "/" + v.cfg == var) {
      for (int i = 0; i < 6; i++) v.run(i % c.nb, 0);
      CK(cudaDeviceSynchronize()); fprintf(stderr, "ran %s\n", var.c_str()); return 0;
    }
  fprintf(stderr, "variant %s not found\n", var.c_str()); return 1;
}

int main(int argc, char** argv) {
  std::string mode = argc > 1 ? argv[1] : "", shapes, out = "out.json", tl, var;
  for (int i = 2; i + 1 < argc; i += 2) {
    std::string k = argv[i];
    if (k == "--shapes") shapes = argv[i + 1]; else if (k == "--out") out = argv[i + 1]; else if (k == "--T") tl = argv[i + 1]; else if (k == "--var") var = argv[i + 1];
  }
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0)); g_nsm = p.multiProcessorCount;
  fprintf(stderr, "%s sm_%d%d SMs=%d L2=%d KB\n", p.name, p.major, p.minor, g_nsm, p.l2CacheSize / 1024);
  if (mode == "check") return mode_check(shapes, out);
  if (mode == "q1") return mode_q1(shapes, out, tl);
  if (mode == "q2") return mode_q2(shapes, out);
  if (mode == "one") return mode_one(shapes, tl, var);
  fprintf(stderr, "usage: bench check|q1|q2 [--shapes a,b] [--T 1,2] [--out f]\n");
  return 1;
}
