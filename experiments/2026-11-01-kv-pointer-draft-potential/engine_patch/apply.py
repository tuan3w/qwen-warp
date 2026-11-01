"""Apply the E7 probe to a PRIVATE copy of the engine tree (git archive HEAD 2d16e21): python3 apply.py <engine dir> ; then make -C <engine dir>.
Touches three files of the copy (engine_impl.h: probe fields + prototype; engine.cu: one hook after the K2 attention step; Makefile: probe.cu) and copies probe.cu in."""
import shutil
import sys
from pathlib import Path

eng = Path(sys.argv[1])
here = Path(__file__).resolve().parent
shutil.copy(here / "probe.cu", eng / "src" / "probe.cu")

p = eng / "src" / "engine_impl.h"
s = p.read_text()
if "probe_on" not in s:
    s = s.replace("  int dn_idx[smk::NLAYER], at_idx[smk::NLAYER];\n",
                  "  int dn_idx[smk::NLAYER], at_idx[smk::NLAYER];\n"
                  "  // E7 probe (src/probe.cu)\n"
                  "  int probe_on = 0, probe_snap = 0, probe_pos0 = 0, probe_row_off = 0, probe_rows = 0;\n"
                  "  float* probe_sc = nullptr;\n  int16_t* probe_idx = nullptr;\n  __half* probe_mass = nullptr;\n  uint16_t *probe_q = nullptr, *probe_gate = nullptr, *probe_out = nullptr;\n", 1)
    s = s.replace("#include <vector>\n", "#include <vector>\n#include <cuda_fp16.h>\n", 1)
    s += "\ncudaError_t smk_probe_attn(smk_engine* e, int l, int T, cudaStream_t st);\n"
    p.write_text(s)

p = eng / "src" / "engine.cu"
s = p.read_text()
if "smk_probe_attn" not in s:
    key = "                           L.k_scale, L.v_scale, ds + DS_POS, e->y6, e->ws_attn, st));\n"
    assert key in s
    s = s.replace(key, key + "      if (e->probe_on) CK(smk_probe_attn(e, l, T, st));\n", 1)
    p.write_text(s)

p = eng / "Makefile"
s = p.read_text()
if "probe" not in s:
    s = s.replace("SRCS := $(SRCS_CORE)", "SRCS_PROBE := $(wildcard src/probe*.cu)\nSRCS := $(SRCS_PROBE) $(SRCS_CORE)", 1)
    p.write_text(s)
