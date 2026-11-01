"""Shared helpers of the swift-mk campaign harness (numpy only: check.py / compare.py import this without torch).

Layout of a campaign root (~/llm-lab/swift-campaign):
  campaign.json, adapter/, oracle/                 harness (never inside an agent's tree)
  private/{dev,held,oracle}_prompts.json           frozen prompt token ids (held-out text lives only here)
  private/tf_tokens.json                           pristine greedy continuation of the 3 oracle prompts (teacher-forcing reference)
  base/                                            pristine engine copy (control); <lineage>/work = agent trees
"""
import json
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("CAMPAIGN_ROOT") or Path(__file__).resolve().parent.parent)
PRIV = ROOT / "private"
PY = os.path.expanduser("~/llm-lab/venv/bin/python")
V = 248320
TOPK_SAVED = 32  # top-k logits saved per teacher-forced position
ORACLE_NAMES = ("code", "prose", "math")
LONG = "long8k"  # 4th teacher-forced prompt: the 8K-context dev prompt (attention / DeltaNet state at long context is invisible to 130-token contexts)
COPY = "edit"  # 5th: the copy-heavy code-edit dev prompt (3.7K tokens, its output copies the prompt): the prompt-lookup tail is active on it, so T = 5..8 verify rows and ring wraps are exercised
LONGS = {LONG: 33, COPY: 65}  # teacher-forced long-context prompts -> number of pristine continuation tokens
TF_NAMES = ORACLE_NAMES + tuple(LONGS)
TF_MODES = (("t8", 8), ("t4", 4), ("t1", 1))  # teacher-forcing chunk widths: T=8 = the widest verify step of the hybrid draft, T=4 = chain verify, T=1 = plain decode
SAMP = dict(temperature=1.0, top_k=20, top_p=0.95)


def jload(p):
    return json.loads(Path(p).read_text())


def jdump(obj, p):
    Path(p).write_text(json.dumps(obj))


def limits():
    return jload(ROOT / "adapter" / "limits.json")


def geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def gpu_clocks():
    """(sm MHz, mem MHz, power W, temp C, mem used MiB) of GPU 0 or None."""
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
        return [float(x) for x in o.strip().split(",")]
    except Exception:
        return None


def foreign_gpu_procs(min_mib=400):
    """compute processes on the GPU other than ours that hold real memory (desktop clients such as the browser's GPU process hold < 400 MiB and are ignored; an engine, vLLM
    or a training job holds GBs)."""
    o = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
    out = []
    for l in o.splitlines():
        f = [x.strip() for x in l.split(",")]
        if len(f) >= 3 and f[0] != str(os.getpid()) and f[-1].isdigit() and int(f[-1]) >= min_mib:
            out.append(f"{f[0]} {f[1][:60]} {f[-1]} MiB")
    return out


def wait_gpu_idle(max_wait=300):
    """block until no other compute process is on the GPU (a sibling job would corrupt the timing); fail after max_wait."""
    t0 = time.time()
    while True:
        procs = foreign_gpu_procs()
        if not procs:
            return
        if time.time() - t0 > max_wait:
            raise SystemExit(f"GPU not exclusive after {max_wait}s: {procs}")
        time.sleep(5)


# ---------------------------------------------------------------------------------------------------------------- sampling reference (numpy)
def processed_dist(logits, temperature=1.0, top_k=20, top_p=0.95):
    """(ids, p) of the target distribution after temperature -> top-k -> top-p (keep token i iff the mass before it < top_p) -> softmax over the survivors; logits: 1-D fp32."""
    l = np.asarray(logits, dtype=np.float64) / temperature
    idx = np.argpartition(-l, top_k)[:top_k] if len(l) > top_k else np.arange(len(l))
    idx = idx[np.argsort(-l[idx], kind="stable")]
    s = l[idx]
    p = np.exp(s - s.max())
    p /= p.sum()
    before = np.cumsum(p) - p
    keep = before < top_p
    keep[0] = True
    idx, p = idx[keep], p[keep]
    return idx, p / p.sum()


def tv(p, q):
    return 0.5 * float(np.abs(np.asarray(p) - np.asarray(q)).sum())


# ---------------------------------------------------------------------------------------------------------------- acceptance reference
def ref_tpc(kind, tree, current, create=True):
    """tokens/cycle of the pristine control per cell (private/ref_tpc.json, written by the first run on base/).  Numerics-level changes (any reassociation of a kernel) flip near-ties of
    the *generated text*, which moves the mean accepted length by a few % (prose +2..4% in calibration) without changing the engine's speed: the score therefore uses the cycle
    rate x the control's tokens/cycle, with the candidate's measured tokens/cycle credited / charged only beyond a per-cell noise band (limits.json)."""
    p = PRIV / "ref_tpc.json"
    refs = jload(p) if p.exists() else {}
    if kind not in refs:
        if not create or Path(tree).resolve() != (ROOT / "base").resolve():
            raise SystemExit(f"private/ref_tpc.json has no '{kind}' reference: run the pristine base/ through this metric first (campaign baseline)")
        refs[kind] = current
        jdump(refs, p)
    return refs[kind]


def band_adjust(m, band):
    """m = E_cand / E_ref: 1 inside +-band, else the distance beyond the band (real acceptance changes are credited / charged, luck is not)."""
    if abs(m - 1) <= band:
        return 1.0
    return m - band if m > 1 else m + band


# ---------------------------------------------------------------------------------------------------------------- result files
def tf_path(result):
    """sidecar npz of a metric result (teacher-forced top-k logits + sampling probes)."""
    r = Path(result)
    return r.with_name(r.stem + ".tf.npz")


def control_results(result):
    """the pristine control result(s) a candidate result must be checked against: the gate's bracketing controls (g-ctrlA/B next to g-dev*.json) or, for `try`, the baseline launch."""
    r = Path(result)
    if r.name.startswith("g-"):
        return [r.with_name("g-ctrlA.json"), r.with_name("g-ctrlB.json")]
    return [ROOT / "baseline" / "main-dev1.json"]
