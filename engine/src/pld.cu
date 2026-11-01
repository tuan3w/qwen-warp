// swift-mk E6: host-side PLD tail proposer (see include/pld.h).  Plain C++17, no CUDA calls.
#include "pld.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace {

typedef std::array<int, 4> Key;  // exact n-gram key (unused slots 0): no hash-collision false candidates

struct KeyHash {
  size_t operator()(const Key& k) const noexcept {
    uint64_t h = 0x9E3779B97F4A7C15ull;
    for (int i = 0; i < 4; i++) {
      h ^= (uint64_t)(uint32_t)k[i] + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
      h *= 0xBF58476D1CE4E5B9ull;
    }
    return (size_t)(h ^ (h >> 31));
  }
};

}  // namespace

struct smk_pld {
  smk_pld_cfg c;
  std::vector<int> S;                                              // history; S.back() is the anchor
  std::unordered_map<Key, std::vector<int>, KeyHash> idx;          // n-gram -> ascending end positions (all <= anchor)

  void index_end(int e) {  // n-gram ending at position e
    if (e < c.n - 1) return;
    Key k{0, 0, 0, 0};
    for (int t = 0; t < c.n; t++) k[t] = S[e - c.n + 1 + t];
    idx[k].push_back(e);
  }
};

extern "C" {

void smk_pld_default_cfg(smk_pld_cfg* c) {
  c->n = 2;
  c->min_len = 3;
  c->max_tail = 4;
  c->cap = 64;
  c->back = 64;
  c->longest = 1;
}

smk_pld* smk_pld_create(const smk_pld_cfg* c) {
  smk_pld* p = new smk_pld();
  if (c) p->c = *c; else smk_pld_default_cfg(&p->c);
  p->c.n = std::min(4, std::max(1, p->c.n));
  p->c.max_tail = std::min(7, std::max(0, p->c.max_tail));
  p->c.cap = std::max(1, p->c.cap);
  p->c.back = std::max(1, p->c.back);
  p->c.longest = p->c.longest ? 1 : 0;
  return p;
}

void smk_pld_destroy(smk_pld* p) { delete p; }

void smk_pld_reset(smk_pld* p, const int* ctx, int n) {
  p->S.clear();
  p->idx.clear();
  p->S.reserve((size_t)n + 4096);
  smk_pld_append(p, ctx, n);
}

void smk_pld_append(smk_pld* p, const int* toks, int n) {
  for (int i = 0; i < n; i++) {
    p->S.push_back(toks[i]);
    p->index_end((int)p->S.size() - 1);
  }
}

int smk_pld_len(const smk_pld* p) { return (int)p->S.size(); }

int smk_pld_propose(smk_pld* p, const int* drafts, int nd, int* out, int* l_out) {
  const smk_pld_cfg& c = p->c;
  if (l_out) *l_out = 0;
  const int pa = (int)p->S.size() - 1;  // anchor position
  if (pa < 0) return 0;
  if (nd < 0) nd = 0;
  const int end = pa + nd;  // last index of K
  const int n = c.n;
  const int* S = p->S.data();
  auto K = [&](int i) -> int { return i <= pa ? S[i] : drafts[i - pa - 1]; };
  if (end + 1 < n) return 0;
  Key key{0, 0, 0, 0};
  for (int t = 0; t < n; t++) key[t] = K(end - n + 1 + t);
  auto it = p->idx.find(key);
  if (it == p->idx.end() || it->second.empty()) return 0;
  const std::vector<int>& occ = it->second;  // ascending, all <= pa
  const int hi = (int)occ.size();
  const int ncand = c.longest ? std::min(hi, c.cap) : 1;
  int best_j = -1, best_l = -1;
  for (int ci = 0; ci < ncand; ci++) {
    const int j = occ[hi - 1 - ci];
    int l = n;
    while (l < c.back && j - l >= 0 && K(j - l) == K(end - l)) l++;
    if (l > best_l) {
      best_j = j;
      best_l = l;
      if (l >= c.back) break;  // cannot be exceeded; ties keep the most recent
    }
  }
  if (l_out) *l_out = best_l;
  if (best_l < c.min_len) return 0;
  const int hiK = std::min(best_j + 1 + c.max_tail, end + 1);
  int ln = 0;
  for (int i = best_j + 1; i < hiK; i++) out[ln++] = K(i);
  return ln;
}

}  // extern "C"
