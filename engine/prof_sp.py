"""Per-stage cycle time of the DSpine / DFlash2 drafters (CUDA graphs, steady state): python prof_sp.py dspine|dflash"""
import os
import sys, time, torch
sys.path.insert(0, ".")
from transformers import AutoTokenizer
mode = sys.argv[1]
tok = AutoTokenizer.from_pretrained(os.environ.get("SWIFT_MODEL", os.path.expanduser("~/models/swift-1.5-int4")))
ids = list(tok.apply_chat_template([{"role": "user", "content": "Write a Python LRU cache with tests."}], add_generation_prompt=True, tokenize=True)["input_ids"])
if mode == "dspine":
    from dspine import DspineEngine
    e = DspineEngine(ctx_max=2048, kv_fp8=0, verbose=False); e.sp_prefill(ids); e.sp_cycles(4); launch = e.sp_launch
else:
    from dflash import DflashEngine
    e = DflashEngine(ctx_max=2048, kv_fp8=0, verbose=False, tmax=8, host_tables=True); e.df_prefill(ids); e.df_cycles(4); launch = e.df_launch
def tm(w, n=30):
    launch(w, 2); torch.cuda.synchronize(); t = time.time(); launch(w, n); torch.cuda.synchronize(); return (time.time() - t) / n * 1e3
for nm, w in (("full cycle", 0), ("verify+accept", 1), ("stage", 2), ("ctx", 3), ("noise", 4), ("head", 5)):
    print(f"{mode} {nm:14s} {tm(w):7.2f} ms")
