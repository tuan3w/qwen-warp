static std::vector<Variant> lab_variants() {
  std::vector<Variant> v;
  v.push_back(mkv<8, 4, 2, 2, 0, 0, 0, 0, 0, 0>("w8k4u2 base"));
  v.push_back(mkv<8, 8, 2, 2, 0, 0, 0, 0, 0, 0>("w8k8u2 base"));
  v.push_back(mkv2<8, 4, 2, 2, 0>("v2 repack w8k4u2"));
  v.push_back(mkv2<8, 8, 2, 2, 0>("v2 repack w8k8u2"));
  v.push_back(mkv2<8, 4, 2, 2, 1>("v2 NAT w8k4u2"));
  v.push_back(mkv2<8, 8, 2, 2, 1>("v2 NAT w8k8u2"));
  v.push_back(mkv2<8, 4, 2, 2, 2>("v2 NATPERM w8k4u2"));
  v.push_back(mkv2<8, 8, 2, 2, 2>("v2 NATPERM w8k8u2"));
  v.push_back(mkv2<8, 8, 2, 3, 2>("v2 NATPERM w8k8u2 minb3"));
  return v;
}
