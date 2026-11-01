"""MTP draft-path ablation on Swift-1.5-Qwen3.8-27b-INT4 (vLLM 0.30, in-process engine). See README.md.

reproduce: LAB_MIN_FREE_GB=2 projects/llm-lab/bin/lab run --detach experiments/2026-10-31-mtp-draft-head-ablation/run.py [arm ...]
           (SMOKE=1 for a short test run -> results/smoke.json)

One engine, loaded once with VLLM_ENABLE_V1_MULTIPROCESSING=0 (in-process). `Qwen3_5MTP.compute_logits` (the single function every
draft step -- prefill step 0 and fused decode steps 1..k-1, MRV2 speculator -- goes through; vllm/v1/worker/gpu/spec_decode/
speculator.py:sample_draft) is replaced at class level before the engine is built by
    logits = linear(h, head)[:, :V] + BIAS          (head = the target lm_head, or a chunk-dequantised int8/int4 copy)
BIAS is a bf16 vector (0 / -inf allowed-token mask). The function is outside the CUDA graphs (vLLM falls back to PIECEWISE
cudagraph mode with FlashInfer + spec decode), so arms are switched by plain python state: BIAS in place, STATE["HQ"] (an
int8-stored fake-quantised lm_head, 1.27 GB + scales, built per head arm) and in-place copy_ of the MTP layer weights
(restored from a CPU copy of the originals). The verification path (target model, rejection sampler) is never touched, so
target outputs are lossless and only acceptance changes.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

sys.path.insert(0, "projects/llm-lab")
EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
from lab.proc import isolated  # noqa: E402
from lab.record import manifest, write_result  # noqa: E402

MODEL = Path.home() / "models" / "swift-1.5-int4"
FREQ = Path.home() / "llm-lab" / "mtp-ablation" / "freq.npy"
K = 3
SMOKE = bool(os.environ.get("SMOKE"))
N_NEW = 64 if SMOKE else 384
V_FULL = 248320

# arm name -> spec. vocab: draft argmax restricted to top-V frequency tokens (+forced); ctx: + every token of the prompt;
# mtp: fake-quant of all MTP linears; head: fake-quant of the draft lm_head rows.
ARMS = {
    "base": {},
    "base2": {},
    "v64k": {"vocab": 65536},
    "v32k": {"vocab": 32768},
    "v16k": {"vocab": 16384},
    "v8k": {"vocab": 8192},
    "v32k+ctx": {"vocab": 32768, "ctx": True},
    "mtp-int8": {"mtp": "int8"},
    "mtp-int4": {"mtp": "int4"},
    "mtp-int4clip": {"mtp": "int4clip"},
    "head-int8": {"head": "int8"},
    "head-int4": {"head": "int4"},
    "comb-head8": {"vocab": 32768, "ctx": True, "mtp": "int4", "head": "int8"},
    "comb-head4": {"vocab": 32768, "ctx": True, "mtp": "int4", "head": "int4"},
    "base3": {},
    "v32k+gen": {"vocab": 32768, "ctx": True, "gen": True},
    "v16k+gen": {"vocab": 16384, "ctx": True, "gen": True},
    "v48k": {"vocab": 49152},
    "v48k+gen": {"vocab": 49152, "ctx": True, "gen": True},
}
SMOKE_ARMS = ["base", "v32k+ctx", "mtp-int4", "head-int4"]
# name -> (sampler family, sampling kwargs incl. seed). T=1 is run with two seeds per prompt as replicates (pooled in report.py).
SAMPLERS = {"greedy": ("greedy", dict(temperature=0.0, seed=0)),
            "t1s0": ("t1", dict(temperature=1.0, top_p=0.95, top_k=20, seed=0)),
            "t1s1": ("t1", dict(temperature=1.0, top_p=0.95, top_k=20, seed=1))}

STATE: dict = {}
GROUP = 128
HEAD_CHUNK = 4096  # rows dequantised at a time in the draft head GEMM


def q_int8(w):
    """int8 symmetric per-output-channel (row) round-to-nearest -> (q int8 [r,c], scale fp32 [r,1])."""
    import torch
    wf = w.float()
    s = (wf.abs().amax(1, keepdim=True) / 127).clamp_min(1e-12)
    return torch.round(wf / s).clamp(-127, 127).to(torch.int8), s


def q_int4(w):
    """int4 symmetric g128 along the input dim, compressed-tensors style (scale = amax/7.5, q in [-8,7]), RTN
    -> (q int8 holding int4 values [r,c], scale fp32 [r,c/128])."""
    import torch
    r, c = w.shape
    assert c % GROUP == 0, (r, c)
    g = w.float().view(r, c // GROUP, GROUP)
    s = (g.abs().amax(-1, keepdim=True) / 7.5).clamp_min(1e-12)
    return torch.round(g / s).clamp(-8, 7).to(torch.int8).view(r, c), s.squeeze(-1)


def dq(q, s):
    """Dequantise to bf16 (scale [r,1] per-row or [r,G] per-group)."""
    r, c = q.shape
    return (q.float().view(r, s.shape[1], c // s.shape[1]) * s.unsqueeze(-1)).view(r, c).to(__import__("torch").bfloat16)


def fq_int8(w):
    return dq(*q_int8(w))


def fq_int4(w, clip=False):
    """int4 g128 fake-quant (RTN, or per-group MSE-optimal clip ratio in {1.0, 0.95, ..., 0.5} when clip=True) -> bf16."""
    import torch
    if not clip:
        return dq(*q_int4(w))
    r, c = w.shape
    g = w.float().view(r, c // GROUP, GROUP)
    amax = g.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    best_err, best_q = None, None
    for ratio in [1.0 - 0.05 * i for i in range(11)]:
        s = amax * ratio / 7.5
        q = torch.round(g / s).clamp(-8, 7) * s
        err = (q - g).pow(2).sum(-1, keepdim=True)
        if best_err is None:
            best_err, best_q = err, q
        else:
            m = err < best_err
            best_err = torch.where(m, err, best_err)
            best_q = torch.where(m, q, best_q)
    return best_q.view(r, c).to(torch.bfloat16)


def fq_into(dst, src, kind, max_elems=8_000_000):
    """dst <- fake-quant(src) (plain copy if kind is None), in row chunks of <= max_elems elements so fp32 temporaries stay
    ~30 MB each (the 3090 has <2 GB spare once the engine and KV cache are resident)."""
    chunk = max(16, max_elems // src.shape[1])
    for i in range(0, src.shape[0], chunk):
        piece = src[i:i + chunk].to("cuda")
        if kind == "int8":
            piece = fq_int8(piece)
        elif kind == "int4":
            piece = fq_int4(piece)
        elif kind == "int4clip":
            piece = fq_int4(piece, clip=True)
        dst[i:i + chunk].copy_(piece)


def build_quant_head(head, kind, max_elems=8_000_000):
    """Quantised copy of the lm_head kept as int8 storage (1.27 GB) + fp32 scales; dequantised chunk-wise in the draft GEMM."""
    import torch
    V, H = head.shape
    qf = q_int8 if kind == "int8" else q_int4
    G = 1 if kind == "int8" else H // GROUP
    q = torch.empty(V, H, dtype=torch.int8, device="cuda")
    s = torch.empty(V, G, dtype=torch.float32, device="cuda")
    chunk = max(16, max_elems // H)
    for i in range(0, V, chunk):
        qi, si = qf(head[i:i + chunk])
        q[i:i + chunk] = qi
        s[i:i + chunk] = si
    return {"kind": kind, "q": q, "s": s}


def install_patch():
    import torch
    import torch.nn.functional as F
    from vllm.model_executor.models import qwen3_5_mtp as m
    from vllm.v1.worker.gpu.spec_decode.mtp import speculator as mtp_speculator

    cls = m.Qwen3_5MTP
    orig_load_eagle = mtp_speculator.load_eagle_model

    def load_eagle_model(target_model, vllm_config):
        model = orig_load_eagle(target_model, vllm_config)
        import gc
        gc.collect()  # the MTP's private embed/lm_head (dropped by load_eagle_model) are in cycles; free their 4.7 GiB now
        torch.cuda.empty_cache()
        STATE["mtp"] = model
        return model

    def compute_logits(self, hidden_states, spec_step_idx=0):
        # THE draft-logits function of vLLM 0.30's MRV2 MTP speculator (v1/worker/gpu/spec_decode/speculator.py:sample_draft).
        # It runs eagerly (PIECEWISE cudagraph mode: compute_logits is outside the captured pieces, see log), so python state
        # (BIAS, STATE["HQ"]) is read on every call. Stock path = linear(h, target lm_head); verification does not go through here.
        if "BIAS" not in STATE:
            STATE["BIAS"] = torch.zeros(self.config.vocab_size, dtype=hidden_states.dtype, device=hidden_states.device)
        hq = STATE.get("HQ")
        if hq is None:
            logits = F.linear(hidden_states, self.lm_head.weight)
        else:
            q, s = hq["q"], hq["s"]
            logits = torch.cat([F.linear(hidden_states, dq(q[i:i + HEAD_CHUNK], s[i:i + HEAD_CHUNK]))
                                for i in range(0, q.shape[0], HEAD_CHUNK)], dim=-1)
        return logits + STATE["BIAS"]

    from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import AutoRegressiveSpeculator
    orig_run_model = AutoRegressiveSpeculator._run_model

    def _run_model(self, num_tokens, *args, **kwargs):
        # "gen" arms: tokens fed to the MTP layer during decode steps (<= 1+k tokens: target-accepted/bonus tokens and drafts; the
        # buffer is exactly num_tokens long at bs=1, no padding) are added to the allowed draft set, i.e. generated tokens join V and
        # the prompt tokens dynamically. Big calls (prompt prefill; may include stale padding) are skipped: set_ctx adds the prompt.
        # (Hooked here, not in Qwen3_5MTP.forward: a data-dependent branch there breaks the torch.compile dynamic-shape guards.)
        if STATE.get("gen") and num_tokens <= 8:
            STATE["BIAS"].index_fill_(0, self.input_buffers.input_ids[:num_tokens].long(), 0)
        return orig_run_model(self, num_tokens, *args, **kwargs)

    mtp_speculator.load_eagle_model = load_eagle_model
    cls.compute_logits = compute_logits
    AutoRegressiveSpeculator._run_model = _run_model


def mtp_linears(mtp):
    from vllm.model_executor.layers.linear import LinearBase
    mods = {}
    for name, mod in mtp.model.named_modules():
        if isinstance(mod, LinearBase):
            mods[name] = mod
    return mods


def forced_ids(tok):
    """Special tokens, the 256 byte-level base tokens, and every token in rendered chat templates."""
    ids = set(tok.all_special_ids) | set(tok.get_added_vocab().values()) | set(range(256))
    convos = [
        [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello"},
         {"role": "assistant", "content": "Hi! How can I help?"}, {"role": "user", "content": "Call a tool."}],
        [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a", "tool_calls": [
            {"type": "function", "function": {"name": "f", "arguments": {"x": 1}}}]}, {"role": "tool", "content": "r"}],
    ]
    for c in convos:
        for think in (False, True):
            try:
                txt = tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True, enable_thinking=think)
                ids |= set(tok(txt, add_special_tokens=False)["input_ids"])
            except Exception:  # noqa: BLE001 - template may reject a role layout; the other layouts still cover it
                pass
    tpl = (MODEL / "chat_template.jinja").read_text()
    ids |= set(tok(tpl, add_special_tokens=False)["input_ids"])
    return sorted(i for i in ids if i < V_FULL)


def allowed_mask(freq, forced, V):
    import numpy as np
    keep = set(forced)
    for i in np.argsort(-freq, kind="stable"):
        if len(keep) >= V:
            break
        keep.add(int(i))
    mask = np.zeros(V_FULL, dtype=bool)
    mask[list(keep)] = True
    return mask, len(keep)


def _run(arm_names):
    import numpy as np
    import torch

    install_patch()
    from vllm import LLM, SamplingParams
    from prompts import AGENT, CODE, MATH, PROSE, long_prompts

    llm = LLM(model=str(MODEL), served_model_name="swift", dtype="bfloat16", language_model_only=True,
              max_model_len=9728, gpu_memory_utilization=0.86, max_num_seqs=4, max_num_batched_tokens=4096,
              enable_prefix_caching=False, kv_cache_dtype="auto",
              speculative_config={"method": "mtp", "num_speculative_tokens": K}, seed=0, disable_log_stats=False)
    tok = llm.get_tokenizer()
    mtp = STATE["mtp"]
    torch.cuda.synchronize()
    info = {"mtp_class": type(mtp).__name__, "mtp_lm_head_is_target_head": True}

    # attention backends actually chosen (target + MTP layer)
    try:
        ctx = llm.llm_engine.vllm_config.compilation_config.static_forward_context
        backs = {}
        for name, layer in ctx.items():
            if hasattr(layer, "get_attn_backend"):
                backs.setdefault(layer.get_attn_backend().get_name(), []).append(name)
        info["attention_backends"] = {k: {"n_layers": len(v), "example": v[0]} for k, v in backs.items()}
    except Exception as e:  # noqa: BLE001
        info["attention_backends_error"] = repr(e)

    head = mtp.lm_head.weight
    assert head.shape == (V_FULL, mtp.config.hidden_size) and head.dtype == torch.bfloat16, (head.shape, head.dtype)
    lin = mtp_linears(mtp)
    info["mtp_linears"] = {n: [list(l.weight.shape), str(l.weight.dtype)] for n, l in lin.items()}
    orig_mtp = {n: l.weight.data.detach().cpu().clone() for n, l in lin.items()}
    for n, l in lin.items():
        assert l.weight.dtype == torch.bfloat16, (n, l.weight.dtype)
    with torch.inference_mode():
        STATE.setdefault("BIAS", torch.zeros(V_FULL, dtype=torch.bfloat16, device="cuda"))
    print("engine ready", json.dumps(info), flush=True)

    freq = np.load(FREQ)
    forced = forced_ids(tok)
    info["forced_tokens"] = len(forced)
    masks = {}

    def mask_for(V):
        if V not in masks:
            masks[V] = allowed_mask(freq, forced, V)
        return masks[V]

    def chat(text):
        return tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False,
                                       enable_thinking=False)

    cats = {"code": CODE, "math": MATH, "prose": PROSE, "agent": AGENT, "long": long_prompts(tok)}
    prompts = {}
    for cat, d in cats.items():
        for n, t in d.items():
            prompts[n] = (cat, chat(t))
    if SMOKE:
        keep = {"code-lru", "math-sqrt2", "prose-story", "agent-call", "long-4k-papers"}
        prompts = {n: v for n, v in prompts.items() if n in keep}
    ctx_ids = {n: torch.tensor(sorted(set(tok(p, add_special_tokens=False)["input_ids"])), device="cuda")
               for n, (_, p) in prompts.items()}
    info["prompt_tokens"] = {n: len(tok(p, add_special_tokens=False)["input_ids"]) for n, (_, p) in prompts.items()}

    def metrics():
        out = {}
        for m_ in llm.get_metrics():
            if m_.name.startswith("vllm:spec_decode"):
                out[m_.name.split(":")[1]] = list(m_.values) if getattr(m_, "values", None) is not None else m_.value
        return out

    def set_arm(spec):
        with torch.inference_mode():
            # MTP layers: restore, then optionally fake-quantise
            for n, l in lin.items():
                fq_into(l.weight.data, orig_mtp[n], spec.get("mtp"))
            STATE["HQ"] = None
            STATE["gen"] = bool(spec.get("gen"))
            torch.cuda.empty_cache()
            if spec.get("head"):
                STATE["HQ"] = build_quant_head(head.data, spec["head"])
            bias = STATE["BIAS"]
            if "vocab" in spec:
                mask, n_allowed = mask_for(spec["vocab"])
                STATE["base_mask"] = torch.from_numpy(mask).to("cuda")
            else:
                STATE["base_mask"] = torch.ones(V_FULL, dtype=torch.bool, device="cuda")
            bias.zero_()
            bias.masked_fill_(~STATE["base_mask"], float("-inf"))
        torch.cuda.synchronize()

    def set_ctx(spec, pname):
        if not spec.get("ctx"):
            return
        with torch.inference_mode():
            allowed = STATE["base_mask"].clone()
            allowed[ctx_ids[pname]] = True
            STATE["BIAS"].zero_()
            STATE["BIAS"].masked_fill_(~allowed, float("-inf"))
        torch.cuda.synchronize()

    # warm-up (graphs already captured by LLM(); first real run primes allocator/JIT)
    set_arm({})
    llm.generate([next(iter(prompts.values()))[1]], SamplingParams(max_tokens=32, temperature=0.0), use_tqdm=False)

    path = EXP / "results" / ("smoke.json" if SMOKE else "arms.json")
    done = json.loads(path.read_text())["arms"] if path.exists() else {}
    for arm in arm_names:
        spec = ARMS[arm]
        if arm in done and done[arm].get("n_new") == N_NEW and set(done[arm]["rows"]) == {
                f"{p}/{s}" for p in prompts for s in SAMPLERS}:
            print("skip (done)", arm, flush=True)
            continue
        t_arm = time.time()
        set_arm(spec)
        extra = {}
        if "vocab" in spec:
            extra["allowed_tokens"] = mask_for(spec["vocab"])[1]
        rows, outs = {}, {}
        for pname, (cat, ptxt) in prompts.items():
            for sname, (family, kw) in SAMPLERS.items():
                set_ctx(spec, pname)  # reset BIAS to base | prompt tokens before every request (gen arms add tokens during it)
                sp = SamplingParams(max_tokens=N_NEW, **kw)
                m0 = metrics()
                t = time.perf_counter()
                r = llm.generate([ptxt], sp, use_tqdm=False)
                dt = time.perf_counter() - t
                m1 = metrics()
                toks = list(r[0].outputs[0].token_ids)
                outs[f"{pname}/{sname}"] = toks
                nd = m1["spec_decode_num_drafts"] - m0["spec_decode_num_drafts"]
                na = m1["spec_decode_num_accepted_tokens"] - m0["spec_decode_num_accepted_tokens"]
                pos = [b - a for a, b in zip(m0["spec_decode_num_accepted_tokens_per_pos"],
                                              m1["spec_decode_num_accepted_tokens_per_pos"])]
                rows[f"{pname}/{sname}"] = {"cat": cat, "prompt": pname, "sampler": family, "drafts": nd, "accepted": na, "accepted_per_pos": pos,
                                            "n_out": len(toks), "sha": hashlib.sha1(json.dumps(toks).encode()).hexdigest()[:12],
                                            "wall_s": round(dt, 2)}
        tot_d = sum(r["drafts"] for r in rows.values())
        tot_a = sum(r["accepted"] for r in rows.values())
        print(f"ARM {arm} {spec} drafts={tot_d} mean_accepted_len={1 + tot_a / tot_d:.3f} "
              f"time={time.time() - t_arm:.0f}s", flush=True)
        out_dir = Path.home() / "llm-lab" / "runs" / "2026-10-31-mtp-draft-head-ablation" / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{arm}.json").write_text(json.dumps(outs))  # raw generations stay out of git (results/ is tracked)
        done[arm] = {"spec": spec, "n_new": N_NEW, "rows": rows, **extra}
        write_result(EXP, "smoke" if SMOKE else "arms", {"arms": done, "info": info, "manifest": manifest(), "k": K})
    set_arm({})
    return info


def main() -> None:
    names = sys.argv[1:] or (SMOKE_ARMS if SMOKE else list(ARMS))
    isolated(_run, names)


if __name__ == "__main__":
    main()
