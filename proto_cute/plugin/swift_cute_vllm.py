"""vLLM general plugin (spike): VLLM_CUTE_W4=1 puts the CuTe DSL W4A16 GEMV first in vLLM's mixed-precision linear kernel list.
   PYTHONPATH=proto_cute/plugin VLLM_CUTE_W4=1 vllm serve ...   (the dist-info next to this file carries the entry point)"""
import os, sys
from pathlib import Path

def register():
    on = lambda k: os.environ.get(k, "0") in ("1", "true")
    if not (on("VLLM_CUTE_W4") or on("VLLM_CUTE_HEAD") or on("VLLM_CUTE_MISC")): return
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.path.insert(0, "/media/fatman/data/workspace/qwen-warp/engine")
    if on("VLLM_CUTE_W4"):
        import vllm_mp_kernel
        vllm_mp_kernel.register()
    if on("VLLM_CUTE_MISC"):
        import cute_misc
        cute_misc.install()
    if on("VLLM_CUTE_HEAD"):
        import cute_head
        cute_head.register()
