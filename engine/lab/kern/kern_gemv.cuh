// Part (i)/(ii) driver: dense baseline (the engine's smk_w4r_gemv) vs sparse W4 GEMV sweeps.  Included at the end of kern.cu.
#pragma once

struct Wt {
  std::string name;
  int N = 0, K = 0, kind = 0;  // kind 0 = SEGS (bf16 [T][N]), 1 = SILU (bf16 [T][N/2])
  int nc = 0;
  std::vector<void*> w, s;
  std::vector<char> hw, hs;  // host copy (reference check)
};

static void load_weights(Wt& W, const char* name, int kind) {
  W.name = name; W.kind = kind;
  std::string base = std::string(WDIR) + "/" + name;
  FILE* f = fopen((base + ".meta").c_str(), "r");
  if (!f) { fprintf(stderr, "missing %s.meta: run export_weights.py once\n", base.c_str()); exit(3); }
  if (fscanf(f, "%d %d", &W.N, &W.K) != 2) exit(3);
  fclose(f);
  W.hw = read_file(base + ".w.bin");
  W.hs = read_file(base + ".s.bin");
  const size_t bytes = W.hw.size() + W.hs.size();
  W.nc = (int)std::min<size_t>(8, std::max<size_t>(4, (size_t)400e6 / bytes));
  for (int c = 0; c < W.nc; c++) {
    void* w = dmalloc(W.hw.size()); void* s = dmalloc(W.hs.size());
    if (c == 0) { CK(cudaMemcpy(w, W.hw.data(), W.hw.size(), cudaMemcpyHostToDevice)); CK(cudaMemcpy(s, W.hs.data(), W.hs.size(), cudaMemcpyHostToDevice)); }
    else { CK(cudaMemcpy(w, W.w[0], W.hw.size(), cudaMemcpyDeviceToDevice)); CK(cudaMemcpy(s, W.s[0], W.hs.size(), cudaMemcpyDeviceToDevice)); }
    W.w.push_back(w); W.s.push_back(s);
  }
}
static void free_weights(Wt& W) { for (auto p : W.w) cudaFree(p); for (auto p : W.s) cudaFree(p); W.w.clear(); W.s.clear(); }

// value of the repacked weight at (row, col): gemv.cu / pack.py repack_w4 layout
static float w_at(const Wt& W, int row, int col) {
  const int nkg = W.K >> 7;
  const uint32_t* w = (const uint32_t*)W.hw.data();
  const uint16_t* s = (const uint16_t*)W.hs.data();
  const int tile = row >> 4, rr = row & 15, half = rr >> 3, g = rr & 7, kg = col >> 7, c = col & 127, t = c >> 5, u = (c >> 3) & 3, e = c & 7;
  const uint32_t word = w[(((size_t)tile * nkg + kg) * 64 + half * 32 + g * 4 + t) * 4 + u];
  const int nib = (e & 1) * 4 + (e >> 1);
  const int n = (word >> (4 * nib)) & 15;
  const float sc = bf2f_h(s[(((size_t)tile * nkg + kg) * 8 + g) * 2 + half]);
  return (float)(n - 8) * sc;
}

struct Err { double max_over_rms = 0, max_rel = 0, frac_diff = 0; };
static Err compare_bf16(const std::vector<uint16_t>& a, const std::vector<uint16_t>& r) {
  double ss = 0; for (auto v : r) { double f = bf2f_h(v); ss += f * f; }
  const double rms = std::sqrt(ss / r.size());
  Err e; size_t nd = 0;
  for (size_t i = 0; i < r.size(); i++) {
    const double x = bf2f_h(a[i]), y = bf2f_h(r[i]);
    const double d = std::fabs(x - y);
    if (a[i] != r[i]) nd++;
    e.max_over_rms = std::max(e.max_over_rms, d / rms);
    e.max_rel = std::max(e.max_rel, d / (std::fabs(y) + 0.05 * rms));
  }
  e.frac_diff = (double)nd / r.size();
  return e;
}

enum { MK_SHARED, MK_STRUCT, MK_INDEP, MK_TOPK, MK_TOPKAGG };
static const char* mk_name[] = {"shared", "struct", "indep", "topk", "topkagg"};

// chunk-level keep sets (chunk = 32 columns): keep[r][c], c = 4 * group + t.  B = block size in columns (128 / 64 / 32).
static std::vector<std::vector<uint8_t>> make_keep(int mode, int B, double f, int M, int nb, std::mt19937_64& rng) {
  const int nch = nb * 4, gu = B / 32, nunit = nch / gu;
  const int ku = std::max(1, (int)std::lround(f * nunit));
  std::vector<std::vector<uint8_t>> keep(M, std::vector<uint8_t>(nch, 0));
  auto subset = [&](std::vector<uint8_t>& row) {
    std::vector<int> p(nunit); std::iota(p.begin(), p.end(), 0);
    std::shuffle(p.begin(), p.end(), rng);
    for (int i = 0; i < ku; i++) for (int q = 0; q < gu; q++) row[p[i] * gu + q] = 1;
  };
  if (mode == MK_SHARED) { subset(keep[0]); for (int r = 1; r < M; r++) keep[r] = keep[0]; }
  else if (mode == MK_INDEP) { for (int r = 0; r < M; r++) subset(keep[r]); }
  else if (mode == MK_STRUCT) {  // R contiguous runs, one per slot at a random offset, total ~ku units
    const int R = 4, slot = nunit / R, len = std::max(1, std::min(slot, (int)std::lround((double)ku / R)));
    for (int i = 0; i < R; i++) {
      const int off = slot > len ? (int)(rng() % (slot - len + 1)) : 0;
      for (int q = 0; q < len; q++) for (int c = 0; c < gu; c++) keep[0][(i * slot + off + q) * gu + c] = 1;
    }
    for (int r = 1; r < M; r++) keep[r] = keep[0];
  }
  return keep;
}

struct ListStat { int nk = 0; double union_frac = 0, row_frac = 0; double wneed = 0, wsector = 0, xneed = 0; };
// list + x zeroing from chunk keep sets; returns stats (bytes per tile counted x tiles)
static ListStat build_list(const std::vector<std::vector<uint8_t>>& keep, int nb, int M, int tiles, bool cm, std::vector<uint32_t>& lst) {
  lst.assign(1, 0);
  ListStat st;
  size_t rowk = 0;
  for (int r = 0; r < M; r++) for (int c = 0; c < nb * 4; c++) rowk += keep[r][c];
  st.row_frac = (double)rowk / ((double)M * nb * 4);
  double need = 0, sec = 0, ch_union = 0;
  for (int g = 0; g < nb; g++) {
    uint32_t m = 0;
    for (int r = 0; r < M; r++) for (int t = 0; t < 4; t++) if (keep[r][g * 4 + t]) m |= 1u << t;
    if (!m) continue;
    lst.push_back((uint32_t)g | ((cm ? m : 0xFu) << 16));
    const int pc = __builtin_popcount(m), pairs = ((m & 3) ? 1 : 0) + ((m & 12) ? 1 : 0);
    const int pcu = cm ? pc : 4, pairu = cm ? pairs : 2;
    need += pcu * 256.0 + 32; sec += pairu * 512.0 + 32; ch_union += pcu;
  }
  lst[0] = (uint32_t)(lst.size() - 1);
  st.nk = (int)lst[0];
  st.union_frac = ch_union / (nb * 4.0);
  st.wneed = need * tiles; st.wsector = sec * tiles;
  st.xneed = (double)M * ch_union * 32 * 2;
  return st;
}

static void zero_x(const std::vector<uint16_t>& hx, const std::vector<std::vector<uint8_t>>& keep, int M, int K, std::vector<uint16_t>& hxz) {
  hxz = hx;
  for (int r = 0; r < M; r++) for (int c = 0; c < K / 32; c++) if (!keep[r][c]) for (int i = 0; i < 32; i++) hxz[(size_t)r * K + c * 32 + i] = 0;
}

static void run_gemv(const char* name, int kind, bool full, const std::vector<int>& Ms) {
  Wt W; load_weights(W, name, kind);
  const int N = W.N, K = W.K, nb = K >> 7, tiles = N >> 4, NC = W.nc;
  const int outw = kind == 1 ? N / 2 : N;
  std::mt19937_64 rng(12345);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::uniform_real_distribution<float> ud(0.f, 1.f);
  // heavy-tailed activation: static per-channel lognormal scale (outlier channels persist across rows) x per-token gaussian x rare spikes
  std::vector<float> cs(K);
  for (int k = 0; k < K; k++) cs[k] = std::exp(1.0f * nd(rng));
  std::vector<uint16_t> hx((size_t)8 * K), hxz;
  for (int r = 0; r < 8; r++) for (int k = 0; k < K; k++) hx[(size_t)r * K + k] = f2bf_h(cs[k] * nd(rng) * (ud(rng) < 0.003f ? 8.f : 1.f));
  uint16_t *xd = (uint16_t*)dmalloc(hx.size() * 2), *xzd = (uint16_t*)dmalloc(hx.size() * 2), *xz2 = (uint16_t*)dmalloc(hx.size() * 2);
  CK(cudaMemcpy(xd, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
  uint16_t *od = (uint16_t*)dmalloc((size_t)8 * outw * 2), *orf = (uint16_t*)dmalloc((size_t)8 * outw * 2);
  const int LS = 144;
  uint32_t* lstd = (uint32_t*)dmalloc((size_t)NC * LS * 4);
  uint32_t* selws = (uint32_t*)dmalloc(SEL_WS_BYTES);
  CK(cudaMemset(selws, 0, SEL_WS_BYTES));

  smk_epi epi = {};
  if (kind == 1) { epi.kind = SMK_EPI_SILU; epi.out[0] = od; }
  else { epi.kind = SMK_EPI_SEGS; epi.end[0] = epi.end[1] = epi.end[2] = N; epi.out[0] = od; }

  auto dense = [&](int i, int T, const void* x, int cfg, cudaStream_t st) -> cudaError_t {
    const int c = i % NC;
    return cfg < 0 ? smk_w4r_gemv(W.w[c], W.s[c], x, N, K, T, &epi, st) : smk_w4r_gemv_cfg(W.w[c], W.s[c], x, N, K, T, &epi, cfg, st);
  };
  auto fetch = [&](int T, std::vector<uint16_t>& h) { h.resize((size_t)T * outw); CK(cudaStreamSynchronize(g_st)); CK(cudaMemcpy(h.data(), od, h.size() * 2, cudaMemcpyDeviceToHost)); };

  // host reference of the dense kernel on a few rows (layout/semantics check of the harness itself): rows 0..31 and the last 16 (SEGS only), engine default config + each engine config
  if (kind == 0) {
    std::vector<int> rows; for (int i = 0; i < 32; i++) rows.push_back(i); for (int i = N - 16; i < N; i++) rows.push_back(i);
    std::vector<double> ref(rows.size() * 8);
    double rms = 0;
    for (size_t ri = 0; ri < rows.size(); ri++) for (int t = 0; t < 8; t++) {
      double a = 0; for (int c = 0; c < K; c++) a += (double)w_at(W, rows[ri], c) * bf2f_h(hx[(size_t)t * K + c]);
      ref[ri * 8 + t] = a; rms += a * a;
    }
    rms = std::sqrt(rms / ref.size());
    for (int cfg = -1; cfg < 6; cfg++) {
      if (dense(0, 8, xd, cfg, g_st) != cudaSuccess) { cudaGetLastError(); continue; }
      std::vector<uint16_t> h; fetch(8, h);
      double maxe = 0; int wr = -1, wt = -1; double wv = 0;
      for (size_t ri = 0; ri < rows.size(); ri++) for (int t = 0; t < 8; t++) {
        const double d = std::fabs(ref[ri * 8 + t] - bf2f_h(h[(size_t)t * N + rows[ri]]));
        if (d > maxe) { maxe = d; wr = rows[ri]; wt = t; wv = bf2f_h(h[(size_t)t * N + rows[ri]]); }
      }
      printf("[%s] harness check cfg %-8s: engine dense kernel vs fp64 host decode of the repacked layout, %d rows x 8 tokens: max|d|/rms = %.2e (worst row %d t %d: got %.3f want %.3f)\n", name, cfg < 0 ? "default" : cfg_name[cfg], (int)rows.size(), maxe / rms, wr, wt, wv, ref[0] * 0 + (wr >= 0 ? ref[(std::find(rows.begin(), rows.end(), wr) - rows.begin()) * 8 + wt] : 0));
      if (cfg < 0) jg.rec("{\"what\":\"host_check\",\"shape\":\"%s\",\"max_over_rms\":%.3e}", name, maxe / rms);
    }
  }

  const std::vector<int> dense_cfgs = {0, 1, 2, 3, 4, 5};  // the engine's own config table
  const std::vector<int> cfgs_full = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, cfgs_quick = {0, 1, 2, 5, 6, 8};
  const std::vector<int>& cfgs = full ? cfgs_full : cfgs_quick;
  const double dense_w = (double)tiles * nb * (1024 + 32);

  for (int M : Ms) {
    // ---- dense baselines
    const double dms = time_graph([&](int i, cudaStream_t s) { dense(i, M, xd, -1, s); });
    double bms = 1e9; int bcfg = -1;
    for (int c : dense_cfgs) {
      if (dense(0, M, xd, c, g_st) != cudaSuccess) { cudaGetLastError(); continue; }
      const double ms = time_graph([&](int i, cudaStream_t s) { dense(i, M, xd, c, s); });
      if (ms < bms) { bms = ms; bcfg = c; }
    }
    const double dbytes = dense_w + (double)M * K * 2;
    const double dgbs = dbytes / dms * 1e-6, bgbs = dbytes / bms * 1e-6;
    printf("\n[%s N=%d K=%d M=%d] dense bytes %.2f MB | repo default smk_w4r_gemv: %.1f us = %.0f GB/s | best of its 6 cfgs: %s %.1f us = %.0f GB/s\n", name, N, K, M, dbytes * 1e-6, dms * 1e3, dgbs, cfg_name[bcfg], bms * 1e3, bgbs);
    printf("  %-8s %4s %5s %6s %6s %6s %9s %8s %8s %8s %7s %7s %6s %s\n", "mode", "B", "f%", "nk/nb", "union%", "row%", "cfg", "us", "GB/s*", "GB/s^", "vsDense", "speedup", "ideal", "err(max/rms; %diff)");
    jg.rec("{\"what\":\"dense\",\"shape\":\"%s\",\"N\":%d,\"K\":%d,\"M\":%d,\"bytes\":%.0f,\"default_us\":%.3f,\"default_GBps\":%.1f,\"best_cfg\":\"%s\",\"best_us\":%.3f,\"best_GBps\":%.1f}", name, N, K, M, dbytes, dms * 1e3, dgbs, cfg_name[bcfg], bms * 1e3, bgbs);

    struct Case { int mode, B; double f; };
    std::vector<Case> cases;
    const std::vector<double> fs = full ? std::vector<double>{1.0, 0.75, 0.5, 0.35, 0.25} : std::vector<double>{1.0, 0.5, 0.25};
    const std::vector<int> modes128 = full ? (M == 1 ? std::vector<int>{MK_SHARED, MK_STRUCT, MK_TOPK} : std::vector<int>{MK_SHARED, MK_STRUCT, MK_INDEP, MK_TOPK, MK_TOPKAGG})
                                           : (M == 1 ? std::vector<int>{MK_SHARED, MK_TOPK} : std::vector<int>{MK_SHARED, MK_INDEP, MK_TOPK});
    for (int md : modes128) for (double f : fs) cases.push_back({md, 128, f});
    cases.push_back({MK_SHARED, 128, 0.0});  // floor: one 128-column group kept (launch + ramp + epilogue)
    if (full || strcmp(name, "l31_qkv") == 0) for (int B : {64, 32}) for (double f : {0.5, 0.25}) cases.push_back({MK_SHARED, B, f});

    std::vector<std::pair<double, double>> fitpts;  // (needed bytes, us) of the shared-128 points
    for (const Case& cs_ : cases) {
      const int B = cs_.B; const bool cm = B != 128;
      std::vector<uint32_t> lst; ListStat ls;
      std::vector<std::vector<uint8_t>> keep;
      const bool dev_sel = cs_.mode == MK_TOPK || cs_.mode == MK_TOPKAGG;
      const int kk = std::max(1, (int)std::lround(cs_.f * nb));
      if (dev_sel) {
        k_select<<<M, 128, 0, g_st>>>((const uint16_t*)xd, M, K, kk, cs_.mode == MK_TOPK ? 0 : 1, lstd, xzd, selws);
        CK(cudaGetLastError());
        lst.resize(LS); CK(cudaStreamSynchronize(g_st)); CK(cudaMemcpy(lst.data(), lstd, LS * 4, cudaMemcpyDeviceToHost));
        lst.resize(1 + lst[0]);
        keep.assign(M, std::vector<uint8_t>(nb * 4, 0));
        if (cs_.mode == MK_TOPK) {  // per-row keep sets from the device xz (zero slice == dropped) for the byte accounting
          std::vector<uint16_t> hz(hx.size()); CK(cudaMemcpy(hz.data(), xzd, hz.size() * 2, cudaMemcpyDeviceToHost));
          for (int r = 0; r < M; r++) for (int c = 0; c < nb * 4; c++) { bool any = false; for (int i = 0; i < 32; i++) any |= hz[(size_t)r * K + c * 32 + i] != 0; keep[r][c] = any; }
        } else {  // shared mask: the kernel writes no xz (the GEMV reads x itself); the reference input is zeroed here from the list
          for (size_t e = 1; e < lst.size(); e++) for (int r = 0; r < M; r++) for (int t = 0; t < 4; t++) keep[r][(lst[e] & 0xFFFF) * 4 + t] = 1;
          zero_x(hx, keep, M, K, hxz);
          CK(cudaMemcpy(xzd, hxz.data(), hxz.size() * 2, cudaMemcpyHostToDevice));
        }
        ls = build_list(keep, nb, M, tiles, false, lst);  // recomputes the same list (sorted ascending): used for stats only
      } else {
        keep = make_keep(cs_.mode, B, cs_.f, M, nb, rng);
        if (cs_.f >= 1.0) for (auto& rw : keep) std::fill(rw.begin(), rw.end(), 1);
        ls = build_list(keep, nb, M, tiles, cm, lst);
        zero_x(hx, keep, M, K, hxz);
        CK(cudaMemcpy(xzd, hxz.data(), hxz.size() * 2, cudaMemcpyHostToDevice));
        for (int c = 0; c < NC; c++) CK(cudaMemcpy(lstd + (size_t)c * LS, lst.data(), lst.size() * 4, cudaMemcpyHostToDevice));
      }
      if (dev_sel) for (int c = 1; c < NC; c++) CK(cudaMemcpy(lstd + (size_t)c * LS, lstd, LS * 4, cudaMemcpyDeviceToDevice));
      // reference: the engine's dense kernel on the zeroed activation
      CK(dense(0, M, xzd, -1, g_st));
      std::vector<uint16_t> href; fetch(M, href);
      const double sbytes_need = ls.wneed + ls.xneed, sbytes_sec = ls.wsector + ls.xneed;
      const double d0 = time_graph([&](int i, cudaStream_t s) { dense(i, M, xd, -1, s); });  // dense baseline measured right before / after this case (clock / thermal drift)
      const uint16_t* xin = cs_.mode == MK_TOPKAGG ? xd : xzd;  // GEMV input of the sparse kernels
      double best = 1e9; int bestc = -1; Err beste;
      for (int c : cfgs) {
        cudaError_t e = cm ? sp_cfg<true>(c, W.w[0], W.s[0], xin, N, K, M, &epi, lstd, g_st) : sp_cfg<false>(c, W.w[0], W.s[0], xin, N, K, M, &epi, lstd, g_st);
        if (e != cudaSuccess) { cudaGetLastError(); continue; }
        std::vector<uint16_t> h; fetch(M, h);
        const Err er = compare_bf16(h, href);
        const double ms = time_graph([&](int i, cudaStream_t s) {
          const int cc = i % NC;
          if (cm) sp_cfg<true>(c, W.w[cc], W.s[cc], xin, N, K, M, &epi, lstd + (size_t)cc * LS, s);
          else sp_cfg<false>(c, W.w[cc], W.s[cc], xin, N, K, M, &epi, lstd + (size_t)cc * LS, s);
        });
        if (getenv("KERN_CFGDBG")) printf("      cfg %-8s %.1f us (err %.1e)\n", cfg_name[c], ms * 1e3, er.max_over_rms);
        if (ms < best) { best = ms; bestc = c; beste = er; }
      }
      const double d1 = time_graph([&](int i, cudaStream_t s) { dense(i, M, xd, -1, s); });
      const double dms_l = 0.5 * (d0 + d1), dgbs_l = dbytes / dms_l * 1e-6;
      const double g1 = sbytes_need / best * 1e-6, g2 = sbytes_sec / best * 1e-6;
      const double ideal = dbytes / (sbytes_sec);  // bytes-ratio bound
      printf("  %-8s %4d %5.0f %3d/%-3d %6.1f %6.1f %9s %8.1f %8.0f %8.0f %7.2f %7.2f %6.2f %.1e; %.2f%%\n", mk_name[cs_.mode], B, cs_.f * 100, ls.nk, nb, ls.union_frac * 100, ls.row_frac * 100,
             cfg_name[bestc], best * 1e3, g1, g2, g2 / dgbs_l, dms_l / best, ideal, beste.max_over_rms, beste.frac_diff * 100);
      if (cs_.mode == MK_SHARED && B == 128 && cs_.f > 0.0) fitpts.push_back({sbytes_need, best * 1e3});
      double tot_sel_us = -1;
      if (dev_sel) {  // full pipeline: score+select kernel then the sparse GEMV, each iteration with its own list
        const double ms = time_graph([&](int i, cudaStream_t s) {
          const int cc = i % NC;
          k_select<<<M, 128, 0, s>>>((const uint16_t*)xd, M, K, kk, cs_.mode == MK_TOPK ? 0 : 1, lstd + (size_t)cc * LS, xz2, selws);
          sp_cfg<false>(bestc, W.w[cc], W.s[cc], cs_.mode == MK_TOPKAGG ? xd : xz2, N, K, M, &epi, lstd + (size_t)cc * LS, s);
        });
        const double sel_ms = time_graph([&](int i, cudaStream_t s) {
          const int cc = i % NC;
          k_select<<<M, 128, 0, s>>>((const uint16_t*)xd, M, K, kk, cs_.mode == MK_TOPK ? 0 : 1, lstd + (size_t)cc * LS, xz2, selws);
        });
        tot_sel_us = ms * 1e3;
        printf("           -> + k_select in the same graph: %.1f us total (select alone %.1f us) ; speedup vs dense incl. mask build %.2f\n", ms * 1e3, sel_ms * 1e3, dms_l / ms);
        jm.rec("{\"what\":\"select\",\"shape\":\"%s\",\"M\":%d,\"mode\":\"%s\",\"f\":%.2f,\"select_us\":%.2f,\"select_plus_gemv_us\":%.2f,\"gemv_us\":%.2f,\"dense_us\":%.2f,\"speedup_incl_select\":%.3f,\"union_frac\":%.3f}",
               name, M, mk_name[cs_.mode], cs_.f, sel_ms * 1e3, ms * 1e3, best * 1e3, dms_l * 1e3, dms_l / ms, ls.union_frac);
      }
      jg.rec("{\"what\":\"sparse\",\"shape\":\"%s\",\"N\":%d,\"K\":%d,\"M\":%d,\"mode\":\"%s\",\"B\":%d,\"f\":%.2f,\"nk\":%d,\"nb\":%d,\"union_frac\":%.4f,\"row_frac\":%.4f,\"cfg\":\"%s\",\"us\":%.3f,"
             "\"bytes_needed\":%.0f,\"bytes_sector\":%.0f,\"GBps_needed\":%.1f,\"GBps_sector\":%.1f,\"dense_GBps\":%.1f,\"frac_dense_GBps_sector\":%.4f,\"frac_dense_GBps_needed\":%.4f,"
             "\"speedup\":%.4f,\"ideal_bytes_ratio\":%.4f,\"err_max_over_rms\":%.3e,\"err_max_rel\":%.3e,\"frac_elems_differ\":%.5f,\"dense_us\":%.3f,\"best_dense_us\":%.3f,\"select_plus_gemv_us\":%.3f}",
             name, N, K, M, mk_name[cs_.mode], B, cs_.f, ls.nk, nb, ls.union_frac, ls.row_frac, cfg_name[bestc], best * 1e3, sbytes_need, sbytes_sec, g1, g2, dgbs_l, g2 / dgbs_l, g1 / dgbs_l, dms_l / best, ideal,
             beste.max_over_rms, beste.max_rel, beste.frac_diff, dms_l * 1e3, bms * 1e3, tot_sel_us);
    }
    // time = a + bytes / b over the shared-128 points (f = 1, .5, .25 [, .75, .35]): a = fixed cost per launch, b = marginal streaming rate; dense floor = engine kernel with a 1-group K
    if (fitpts.size() >= 2) {
      double sx = 0, sy = 0, sxx = 0, sxy = 0; const int n = (int)fitpts.size();
      for (auto& pt : fitpts) { sx += pt.first; sy += pt.second; sxx += pt.first * pt.first; sxy += pt.first * pt.second; }
      const double slope = (n * sxy - sx * sy) / (n * sxx - sx * sx), icpt = (sy - slope * sx) / n;  // us per byte, us
      const double dfl = time_graph([&](int i, cudaStream_t s) { smk_w4r_gemv(W.w[i % NC], W.s[i % NC], xd, N, 128, M, &epi, s); }) * 1e3;
      printf("  fit (sparse, shared 128-col blocks): time = %.2f us + bytes / %.0f GB/s   |   dense engine kernel with K=128 (launch + ramp floor): %.2f us\n", icpt, 1e-3 / slope, dfl);
      jg.rec("{\"what\":\"fit\",\"shape\":\"%s\",\"M\":%d,\"fixed_us\":%.3f,\"marginal_GBps\":%.1f,\"dense_K128_floor_us\":%.3f}", name, M, icpt, 1e-3 / slope, dfl);
    }
  }
  cudaFree(xd); cudaFree(xzd); cudaFree(xz2); cudaFree(od); cudaFree(orf); cudaFree(lstd); cudaFree(selws);
  free_weights(W);
}
