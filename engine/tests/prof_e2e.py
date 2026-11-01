"""Profiling target for nsys:  nsys profile --cuda-graph-trace=node -o /tmp/smk_prof ~/llm-lab/venv/bin/python tests/prof_e2e.py [ctx]"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import Engine
ctx = int(sys.argv[1]) if len(sys.argv) > 1 else 64
eng = Engine(ctx_max=ctx + 128)
ids = torch.randint(1000, 100000, (ctx,), generator=torch.Generator().manual_seed(0)).tolist()
eng.reset(); eng.prefill(ids); eng.decode(8)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
eng.decode(20)
torch.cuda.cudart().cudaProfilerStop()
