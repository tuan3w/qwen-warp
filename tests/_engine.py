"""The one place tier-1 tests build an engine (so it can be switched to weights resident in another process without touching the tests).

  from _engine import make_engine
  r = make_engine(ctx_max=8192, vision="device")     # smk_runner.Runner; keyword arguments override DEFAULTS
  r = make_engine(max_num_seqs=4)                    # smk_batch.BatchRunner (continuous batching over 4 sequence slots)

SMK_RESIDENT=1 (with $QWRES_DIR = the directory of the holder's resident.py): the packed weights come from the resident holder (CUDA IPC, ~1 s instead of a 20-30 s load).  Run such tests WITHOUT /tmp/qw/gpu.sh (the holder owns the GPU lock).  The bf16 tables are host memory there already, so host_tables defaults to False.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "server"))
sys.path.insert(0, str(ROOT / "engine"))

DEFAULTS = dict(ctx_max=8192 + 32, k=3, group=2, max_snaps=16, pld=True, prefix_cache_gb=1.0, warmup=False, log=lambda *a: None)


def attach():
    """Use the resident weights when $SMK_RESIDENT=1 (no-op otherwise); call before any engine is built."""
    if os.environ.get("SMK_RESIDENT") == "1":
        sys.path.insert(0, os.environ["QWRES_DIR"])  # the directory of resident.py
        import resident

        resident.attach()


def make_engine(**kw):
    attach()
    a = {**DEFAULTS, **kw}
    if os.environ.get("SMK_RESIDENT") == "1":
        a["host_tables"] = False
    if a.get("max_num_seqs", 1) > 1:
        from smk_batch import BatchRunner

        return BatchRunner(**a)
    from smk_runner import Runner

    return Runner(**a)
