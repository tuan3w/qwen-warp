#!/usr/bin/env python
"""One visit of the swift-mk-server vs stock-vLLM A/B: start ONE server, measure boot time + E0 decode (greedy and T=1) + TTFT (20 tokens, 8K), stop it, check the GPU is empty.

  ~/llm-lab/venv/bin/python server/ab_visit.py ours|vllm --round N --out DIR [--reps 3] [--ours-args ...]
Run from the repo root with the GPU lease held by the caller; alternate the two servers (A,B,A,B) from a driver loop: both servers are never up at the same time.
ours = server/serve.py --port 8000 (defaults: prompt-lookup auto, bf16 KV, k=3);
vllm = the experiments/2026-10-30-vllm-swift-int4-bs1-baseline stock configuration (MTP k=3, --max-model-len 16384, --max-num-seqs 4, prefix caching off, --language-model-only).
Client (same for both): httpx streaming /v1/completions with chat-rendered TOKEN IDS (bench_server.build_prompts: thinking off), ignore_eos, return_token_ids, 385 tokens;
decode tok/s = (tokens after the first chunk) / (t_last - t_first) (E0 / bench_server.py); greedy = temperature 0, T1 = temperature 1 top_p .95 top_k 20 (model-card defaults);
TTFT20 = request sent -> first chunk for plain 20-token corpus prompts (unique slice per rep), TTFT8K = the 8K E0 prompt with a unique 1-token prefix per rep (no prefix cache / snapshot can hit).
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
import bench_server  # noqa: E402

PY = os.path.expanduser("~/llm-lab/venv/bin/python")
MODEL = bench_server.MODEL
N_DEC = 385
SAMPLING = {"greedy": dict(temperature=0), "T1": dict(temperature=1.0, top_p=0.95, top_k=20)}
VLLM = [PY, "-m", "vllm.entrypoints.cli.main", "serve", str(MODEL), "--served-model-name", "swift", "--dtype", "bfloat16", "--language-model-only", "--max-model-len", "16384",
        "--gpu-memory-utilization", "0.93", "--max-num-seqs", "4", "--no-enable-prefix-caching", "--speculative-config", '{"method":"mtp","num_speculative_tokens":3}', "--port", "8000"]


def gpu_used():
    return int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.split()[0])


def stream(cl, ids, n, sampling):
    body = dict(model="swift", prompt=ids, max_tokens=n, ignore_eos=True, stream=True, stream_options={"include_usage": True}, return_token_ids=True, **sampling)
    t0 = time.perf_counter()
    times, counts = [], []
    with cl.stream("POST", "http://127.0.0.1:8000/v1/completions", json=body, timeout=900) as r:
        if r.status_code != 200:
            r.read()
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        for line in r.iter_lines():
            if not line.startswith("data:") or line.strip() == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("choices"):
                tid = d["choices"][0].get("token_ids") or []
                if tid:
                    times.append(time.perf_counter())
                    counts.append(len(tid))
    total, first = sum(counts), counts[0]
    return dict(ttft=times[0] - t0, decode=(total - first) / (times[-1] - times[0]), total=total, chunks=len(times))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("server", choices=["ours", "vllm"])
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ours-args", nargs="*", default=[])
    a = ap.parse_args()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok, P = bench_server.build_prompts(["code", "prose", "math", "long8k"])

    corpus = tok("\n\n".join(p.read_text(errors="ignore") for p in sorted((REPO / "topics").glob("*.md"))), add_special_tokens=False)["input_ids"]
    ttft20 = [corpus[500 + 997 * r: 500 + 997 * r + 20] for r in range(8)]  # E11's slices
    cmd = [PY, str(HERE / "serve.py"), "--port", "8000", *a.ours_args] if a.server == "ours" else VLLM
    log = open(out_dir / f"{a.server}_r{a.round}.server.log", "w")
    res = dict(server=a.server, round=a.round, cmd=cmd, started=time.strftime("%Y-%m-%dT%H:%M:%S"))
    if gpu_used() > 1500:
        raise SystemExit(f"GPU not empty: {gpu_used()} MiB")
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=REPO, env={**os.environ, "VLLM_LOGGING_LEVEL": "INFO"})
    try:
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"{a.server} exited {proc.returncode}; see {log.name}")
            try:
                if httpx.get("http://127.0.0.1:8000/health", timeout=2).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if time.time() - t0 > 900:
                raise RuntimeError("boot timeout")
            time.sleep(0.5)
        res["boot_s"] = time.time() - t0
        print(f"[{a.server} r{a.round}] healthy after {res['boot_s']:.1f} s", flush=True)
        cl = httpx.Client()
        for ids in list(P.values()) + [P["code"], P["prose"]]:  # warm-up (graphs, first-batch allocations)
            stream(cl, ids, 128, SAMPLING["greedy"])
        for mode in ("greedy", "T1"):
            res[mode] = {}
            for name, ids in P.items():
                rows = [stream(cl, ids, N_DEC, SAMPLING[mode]) for _ in range(a.reps if name != "long8k" else max(2, a.reps - 1))]
                res[mode][name] = dict(decode=[round(r["decode"], 2) for r in rows], median=statistics.median(r["decode"] for r in rows), chunks=statistics.median(r["chunks"] for r in rows), prompt=len(ids))
                print(f"  {mode:6s} {name:7s} {res[mode][name]['median']:7.1f} tok/s {res[mode][name]['decode']}", flush=True)
        t20 = [stream(cl, ids, 4, SAMPLING["greedy"])["ttft"] for ids in ttft20[:5]]
        t8k = [stream(cl, tok(f"#{i}\n", add_special_tokens=False)["input_ids"] + P["long8k"], 4, SAMPLING["greedy"])["ttft"] for i in range(3)]
        res["ttft"] = dict(t20_s=t20, t20_ms=statistics.median(t20) * 1e3, t8k_s=t8k, t8k_ms=statistics.median(t8k) * 1e3, prompt8k=len(P["long8k"]) + len(tok("#0\n", add_special_tokens=False)["input_ids"]))
        print(f"  TTFT 20 tok {res['ttft']['t20_ms']:.0f} ms {[round(x * 1e3) for x in t20]}   8K {res['ttft']['t8k_ms']:.0f} ms {[round(x * 1e3) for x in t8k]}", flush=True)
        res["vram_used_mib"] = gpu_used()  # after the run, server still up
    except Exception as e:  # noqa: BLE001
        res["error"] = repr(e)[:500]
        print("ERROR", res["error"], flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        for _ in range(60):
            if gpu_used() < 1500:
                break
            time.sleep(1)
        res["vram_after_stop_mib"] = gpu_used()
        json.dump(res, open(out_dir / f"{a.server}_r{a.round}.json", "w"), indent=1)
        print(f"[{a.server} r{a.round}] done; GPU after stop {res['vram_after_stop_mib']} MiB", flush=True)


if __name__ == "__main__":
    main()
