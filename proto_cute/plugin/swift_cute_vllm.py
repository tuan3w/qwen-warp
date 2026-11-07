"""vLLM general plugin (spike): VLLM_CUTE_W4=1 puts the CuTe DSL W4A16 GEMV first in vLLM's mixed-precision linear kernel list.
   PYTHONPATH=proto_cute/plugin VLLM_CUTE_W4=1 vllm serve ...   (the dist-info next to this file carries the entry point)"""
import os, sys
from pathlib import Path

def register():
    if os.environ.get("VLLM_CUTE_W4", "0") not in ("1", "true"): return
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.path.insert(0, "/media/fatman/data/workspace/qwen-warp/engine")
    import vllm_mp_kernel
    vllm_mp_kernel.register()
