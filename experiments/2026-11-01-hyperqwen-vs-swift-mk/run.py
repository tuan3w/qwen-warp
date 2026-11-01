"""hyperqwen-vs-swift-mk: ONE client, ONE prompt set, two serving stacks on the same Swift-1.5 INT4 weights and the same 3090.

  ~/llm-lab/venv/bin/python experiments/2026-11-01-hyperqwen-vs-swift-mk/run.py prompts            CPU: freeze every prompt as token ids (runs/hq/prompts.json + results/prompts_manifest.json)
  ~/llm-lab/venv/bin/python experiments/2026-11-01-hyperqwen-vs-swift-mk/run.py visit <server> [--suite full|conc|long|ttft] [--round N]
        <server> = ours | ours-fp8 | mtp-fast | mtp-long | dflash2 | dflash15 | dflash2-long   (ours = server/serve.py --prompt-lookup auto; others = hq.sh configs)
  ~/llm-lab/venv/bin/python experiments/2026-11-01-hyperqwen-vs-swift-mk/run.py tables              CPU: results/*.json -> results/tables.md

GPU lease: the caller holds `projects/llm-lab/bin/lab gpu acquire` (this script never touches the lease).  `visit` starts the server itself, warms it up, runs the suite, stops it and
checks nvidia-smi is empty.  Client: httpx streaming /v1/completions with TOKEN IDS (identical ids to both servers; chat template applied client-side, thinking off), ignore_eos,
return_token_ids; decode tok/s = (tokens after the first chunk) / (t_last - t_first) (E0's definition, bench_server.py); TTFT = request sent -> first token chunk.
Greedy = temperature 0; "T1" = temperature 1, top_p .95, top_k 20 (the model card defaults).  Per-visit GPU telemetry (1 Hz nvidia-smi: SM clock, power, temperature) is stored with every phase.
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RES = HERE / "results"
RAW = REPO / "runs" / "hq"
MODEL = Path.home() / "models" / "swift-1.5-int4"
PY = os.path.expanduser("~/llm-lab/venv/bin/python")
E6_PROMPTS = REPO / "experiments/2026-10-31-ngram-draft-potential/runs/prompts.json"
E8_DIR = REPO / "experiments/e8-task-set-not-released"
N_DEC = 385   # E0 prompts (code / prose / math / long8k) like bench_server.py
N_SET = 256   # E6 / E8 tasks
SAMPLING = {"greedy": dict(temperature=0), "T1": dict(temperature=1.0, top_p=0.95, top_k=20)}
NEEDLE = "copper-lantern-4912"


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# prompts (CPU)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def build_prompts():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(MODEL))
    enc = lambda text: tok(text, add_special_tokens=False)["input_ids"]  # noqa: E731
    chat = lambda msgs, **kw: tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False, **kw)  # noqa: E731
    P = {}
    # E0 (bench_server.py): 3 short + 8K summarisation from topics/*.md
    short = {
        "code": "Write a Python function that returns the n-th Fibonacci number using memoisation, then explain its complexity.",
        "prose": "Write a short story about a lighthouse keeper who discovers that the light has been guiding something other than ships.",
        "math": "Prove that the sum of the first n odd numbers equals n squared, then give two different proofs.",
    }
    P["e0"] = {k: enc(chat([{"role": "user", "content": v}])) for k, v in short.items()}
    topics = "\n\n".join(p.read_text(errors="ignore") for p in sorted((REPO / "topics").glob("*.md")))
    ids = enc(topics)[:8000]
    P["e0"]["long8k"] = enc(chat([{"role": "user", "content": tok.decode(ids) + "\n\nSummarise the main claims above in a numbered list, citing the section names."}]))
    # E6: the 32 pre-rendered tasks (a code edits, b log extraction, c agent tool calls, d prose, e math)
    P["e6"] = [dict(id=t["id"], cls=t["cls"], ids=enc(t["text"])) for t in json.load(open(E6_PROMPTS))]
    # E8 English side (48 paired tasks: qa ex sm tr em mt cd ag); non-agent tasks are rendered as plain user messages, agent turns with the tools list
    sys.path.insert(0, str(E8_DIR))
    import prompts_ab as pa

    passages = json.load(open(E8_DIR / "runs/passages.json"))
    e8 = []
    for t in pa.build(passages):
        kw = dict(tools=t["tools"]) if t["tools"] else {}
        e8.append(dict(id=t["id"], cat=t["cat"], ids=enc(chat(t["en"], **kw))))
    P["e8"] = e8
    # TTFT: plain corpus ids of exactly 20 tokens, one distinct slice per rep (bench_ttft.py convention for L < 24)
    corpus = enc(topics)
    P["ttft20"] = [corpus[500 + 997 * r: 500 + 997 * r + 20] for r in range(8)]
    # long context: distinct repo markdown (topics, papers, experiment READMEs, project docs) cut to N ids, a needle paragraph at 40 % depth, the question at the end
    files = sorted((REPO / "topics").glob("*.md")) + sorted((REPO / "papers").glob("*.md")) + sorted((REPO / "experiments").glob("*/README.md")) \
        + sorted((REPO / "docs").glob("*.md")) + sorted((REPO / "sources").glob("*.md"))
    text = "\n\n".join(f"## FILE {p.relative_to(REPO)}\n{p.read_text(errors='ignore')}" for p in files)
    big = enc(text)
    P["long_corpus_tokens_available"] = len(big)
    if len(big) < 105000:
        raise SystemExit(f"need 105k filler tokens, have {len(big)}")
    question = "\n\nTwo tasks. First, state the maintenance passphrase for vault 7 mentioned somewhere above (answer the passphrase on the first line). Second, summarise the document in a numbered list of ten items."
    needle = f"\n\nNOTE FOR THE OPERATOR: the maintenance passphrase for vault 7 is {NEEDLE}. Remember it.\n\n"
    P["long"] = {}
    for n in (28000, 56000, 100000):
        body = big[: n - 200]
        cut = int(len(body) * 0.4)
        doc = tok.decode(body[:cut]) + needle + tok.decode(body[cut:])
        P["long"][str(n)] = enc(chat([{"role": "user", "content": doc + question}]))
    P["needle"] = NEEDLE
    json.dump(P, open(RAW / "prompts.json", "w"))
    man = dict(
        e0={k: dict(n=len(v), sha=hashlib.sha256(json.dumps(v).encode()).hexdigest()[:12]) for k, v in P["e0"].items()},
        e6=[dict(id=t["id"], cls=t["cls"], n=len(t["ids"])) for t in P["e6"]],
        e8=[dict(id=t["id"], cat=t["cat"], n=len(t["ids"])) for t in P["e8"]],
        long={k: len(v) for k, v in P["long"].items()}, ttft20=[len(x) for x in P["ttft20"]], needle=NEEDLE,
        sha_all=hashlib.sha256(json.dumps(P, sort_keys=True).encode()).hexdigest()[:16])
    json.dump(man, open(RES / "prompts_manifest.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in man.items() if k in ("e0", "long", "sha_all")}))


def load_prompts():
    return json.load(open(RAW / "prompts.json"))


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# client
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def stream(cl, url, model, ids, n, sampling, ignore_eos=True, timeout=1800):
    body = dict(model=model, prompt=ids, max_tokens=n, ignore_eos=ignore_eos, stream=True, stream_options={"include_usage": True}, return_token_ids=True, **sampling)
    t0 = time.perf_counter()
    times, toks = [], []
    with cl.stream("POST", url + "/v1/completions", json=body, timeout=timeout) as r:
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
                    toks.append(tid)
    if not toks:
        raise RuntimeError("no tokens")
    total = sum(len(t) for t in toks)
    first = len(toks[0])
    dec = (total - first) / (times[-1] - times[0]) if times[-1] > times[0] else float("nan")
    return dict(t0=t0, t_first=times[0], t_last=times[-1], ttft=times[0] - t0, decode=dec, total=total, first=first, chunks=len(toks), ids=[x for t in toks for x in t],
                maxchunk=max(len(t) for t in toks))


class Gpu:
    """1 Hz nvidia-smi sampler; `with gpu.phase(name)` stores the samples that fall in the phase."""

    def __init__(self):
        self.samples = []
        self.stop = False
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()
        self.phases = {}

    def _run(self):
        p = subprocess.Popen(["nvidia-smi", "--query-gpu=clocks.sm,power.draw,temperature.gpu,utilization.gpu,memory.used,power.limit", "--format=csv,noheader,nounits", "-l", "1"],
                             stdout=subprocess.PIPE, text=True)
        for line in p.stdout:
            if self.stop:
                p.kill()
                return
            try:
                v = [float(x) for x in line.split(",")]
            except ValueError:
                continue
            self.samples.append((time.time(), *v))

    def phase(self, name):
        g = self

        class C:
            def __enter__(s):
                s.t0 = time.time()

            def __exit__(s, *a):
                sel = [x for x in g.samples if s.t0 <= x[0] <= time.time()]
                if sel:
                    act = [x for x in sel if x[4] > 50] or sel
                    g.phases[name] = dict(n=len(sel), sm_mhz_med=statistics.median(x[1] for x in act), power_w_med=statistics.median(x[2] for x in act), power_w_max=max(x[2] for x in sel),
                                          temp_c_max=max(x[3] for x in sel), power_limit_w=sel[-1][6])
        return C()


def med(xs):
    xs = [x for x in xs if x == x]
    return statistics.median(xs) if xs else float("nan")


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# servers
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
class Server:
    def __init__(self, name):
        self.name = name
        self.proc = None
        if name.startswith("ours"):
            self.url, self.model = "http://127.0.0.1:8000", "swift"
        else:
            self.url, self.model = "http://127.0.0.1:18020", "qwen3.8-27b"

    def start(self, extra=()):
        t0 = time.time()
        if self.name.startswith("ours"):
            args = ["--port", "8000", "--prompt-lookup", "auto"]
            if self.name == "ours-fp8":
                args = ["--port", "8000", "--prompt-lookup", "0", "--kv-cache-dtype", "fp8", "--max-model-len", "65536"]
            args += list(extra)
            log = open(RAW / "logs" / f"ours-{int(t0)}.log", "w")
            self.proc = subprocess.Popen([PY, str(REPO / "server/serve.py"), *args], stdout=log, stderr=subprocess.STDOUT, cwd=REPO)
            while True:
                if self.proc.poll() is not None:
                    raise RuntimeError(f"ours exited {self.proc.returncode}; see {log.name}")
                try:
                    if httpx.get(self.url + "/health", timeout=2).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                if time.time() - t0 > 300:
                    raise RuntimeError("ours boot timeout")
                time.sleep(1)
        else:
            r = subprocess.run([str(HERE / "hq.sh"), "start", self.name, *extra], capture_output=True, text=True, cwd=REPO)
            print(r.stdout[-1500:], r.stderr[-500:], flush=True)
            if r.returncode:
                subprocess.run([str(HERE / "hq.sh"), "stop"], capture_output=True)
                raise RuntimeError("hq start failed")
        self.boot_s = time.time() - t0
        return self.boot_s

    def stop(self):
        if self.name.startswith("ours"):
            if self.proc:
                self.proc.terminate()
                try:
                    self.proc.wait(30)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
        else:
            subprocess.run([str(HERE / "hq.sh"), "stop"], capture_output=True, cwd=REPO)
        for _ in range(30):
            used = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.split()[0])
            if used < 1500:
                return used
            time.sleep(1)
        return used


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# suites
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def suite_full(srv, P, gpu, out):
    cl = httpx.Client()
    url, model = srv.url, srv.model
    S = lambda ids, n, mode="greedy": stream(cl, url, model, ids, n, SAMPLING[mode])  # noqa: E731
    # warm-up (graphs, Triton variants, first-batch allocations: HQ doc says a first run after start reads 30-50 % low)
    for ids in list(P["e0"].values()) + [P["e0"]["code"], P["e0"]["prose"]]:
        S(ids, 128)
    with gpu.phase("e0"):
        out["e0"] = {}
        for name, ids in P["e0"].items():
            r = S(ids, N_DEC)
            out["e0"][name] = dict(prompt=len(ids), decode=r["decode"], ttft_warm=r["ttft"], total=r["total"], chunks=r["chunks"], maxchunk=r["maxchunk"], ids=r["ids"])
            print(f"  e0 {name:7s} {r['decode']:7.1f} tok/s  (chunks {r['chunks']}, max {r['maxchunk']}/chunk)", flush=True)
        out["e0_T1"] = {}
        for name, ids in P["e0"].items():
            r = S(ids, N_DEC, "T1")
            out["e0_T1"][name] = dict(prompt=len(ids), decode=r["decode"], total=r["total"])
            print(f"  e0-T1 {name:7s} {r['decode']:7.1f} tok/s", flush=True)
    with gpu.phase("ttft"):
        t20 = [S(ids, 4)["ttft"] for ids in P["ttft20"]]
        # 8K: unique 1-token prefix per rep so no prefix cache / snapshot can hit
        base = P["e0"]["long8k"]
        t8k = [S([151643 - 7 * (i + 1)] + base, 4)["ttft"] for i in range(3)]
        out["ttft"] = dict(t20_s=t20, t20_med=med(t20), t8k_s=t8k, t8k_med=med(t8k), prompt8k=len(base) + 1)
        print(f"  ttft 20tok {med(t20) * 1e3:6.0f} ms  8K {med(t8k) * 1e3:7.0f} ms", flush=True)
    for setname, tasks in (("e6", P["e6"]), ("e8", P["e8"])):
        for mode in ("greedy", "T1"):
            with gpu.phase(f"{setname}_{mode}"):
                rows = []
                for t in tasks:
                    r = S(t["ids"], N_SET, mode)
                    rows.append(dict(id=t["id"], cls=t.get("cls") or t.get("cat"), prompt=len(t["ids"]), decode=r["decode"], total=r["total"], ttft=r["ttft"], maxchunk=r["maxchunk"], ids=r["ids"] if mode == "greedy" else None))
                out[f"{setname}_{mode}"] = rows
                by = {}
                for x in rows:
                    by.setdefault(x["cls"], []).append(x["decode"])
                tot_tok = sum(x["total"] - 1 for x in rows)
                print(f"  {setname} {mode:6s} " + " ".join(f"{k}:{med(v):.0f}" for k, v in sorted(by.items())) + f"  pooled-median {med([x['decode'] for x in rows]):.1f}", flush=True)


def suite_ttft(srv, P, gpu, out):
    cl = httpx.Client()
    for ids in list(P["e0"].values())[:2]:
        stream(cl, srv.url, srv.model, ids, 64, SAMPLING["greedy"])
    res = {}
    base = P["e0"]["long8k"]
    # plain corpus prompts of L tokens (a 20-token prompt is slice 0..19 of the shifted corpus); unique prefix token per rep
    for L in (20, 40, 128, 512, 2048, 4096):
        pts = []
        for r in range(4):
            ids = ([151000 + 13 * r + L % 7] + base[: L - 1]) if L > 20 else P["ttft20"][r]
            pts.append(stream(cl, srv.url, srv.model, ids, 4, SAMPLING["greedy"])["ttft"] * 1e3)
        res[L] = pts
        print(f"  TTFT {L:5d}: {med(pts):7.0f} ms", flush=True)
    out["ttft_sweep_ms"] = res


def suite_conc(srv, P, gpu, out):
    cl = httpx.Client(limits=httpx.Limits(max_connections=32))
    S = lambda ids, n: stream(cl, srv.url, srv.model, ids, n, SAMPLING["greedy"])  # noqa: E731
    for ids in list(P["e0"].values())[:3]:
        S(ids, 128)
    short = [t["ids"] for t in P["e8"] if t["cat"] in ("qa", "ex", "em", "mt", "cd")]
    big = P["e0"]["long8k"]
    # 4k-token distinct prompts (HQ's own concurrency ladder uses distinct 4k prompts): slices of the 8K document with a unique leading token + a question
    k4 = [[151000 + 11 * i] + big[300 * i: 300 * i + 3900] + big[-30:] for i in range(8)]
    out["conc"] = {}
    for kind, pool, n in (("short", short, 192), ("4k", k4, 192)):
        for C in (1, 2, 4, 8):
            rounds = []
            for rd in range(2):
                prompts = [pool[(rd * C + i) % len(pool)] for i in range(C)]
                if kind == "4k":
                    prompts = [[152000 + 100 * rd + i] + p[1:] for i, p in enumerate(prompts)]  # unique prefix per round
                with gpu.phase(f"conc_{kind}_{C}_{rd}"):
                    t0 = time.perf_counter()
                    with cf.ThreadPoolExecutor(C) as ex:
                        rs = list(ex.map(lambda p: S(p, n), prompts))
                    wall = time.perf_counter() - t0
                tot = sum(r["total"] for r in rs)
                win = max(r["t_last"] for r in rs) - min(r["t_first"] for r in rs)
                rounds.append(dict(C=C, wall=wall, agg_e2e=tot / wall, agg_decode=sum(r["total"] - r["first"] for r in rs) / win if win > 0 else float("nan"),
                                   per_stream_decode=med([r["decode"] for r in rs]), ttft_med=med([r["ttft"] for r in rs]), ttft_max=max(r["ttft"] for r in rs), tokens=tot))
            out["conc"][f"{kind}_{C}"] = rounds
            print(f"  conc {kind:5s} C={C}: aggregate e2e {med([x['agg_e2e'] for x in rounds]):6.1f}  decode-window {med([x['agg_decode'] for x in rounds]):6.1f} tok/s,  per-stream {med([x['per_stream_decode'] for x in rounds]):6.1f},  "
                  f"TTFT med {med([x['ttft_med'] for x in rounds]) * 1e3:6.0f} ms max {max(x['ttft_max'] for x in rounds) * 1e3:6.0f} ms", flush=True)


def suite_long(srv, P, gpu, out, lens):
    cl = httpx.Client()
    S = lambda ids, n: stream(cl, srv.url, srv.model, ids, n, SAMPLING["greedy"], ignore_eos=False, timeout=3600)  # noqa: E731
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(MODEL))
    stream(cl, srv.url, srv.model, P["e0"]["code"], 64, SAMPLING["greedy"])
    out["long"] = {}
    for n in lens:
        ids = P["long"][str(n)]
        with gpu.phase(f"long_{n}"):
            try:
                r = S(ids, 160)
            except Exception as e:  # noqa: BLE001
                out["long"][str(n)] = dict(prompt=len(ids), error=str(e)[:400])
                print(f"  long {n}: ERROR {str(e)[:200]}", flush=True)
                continue
        text = tok.decode(r["ids"][:60])
        out["long"][str(n)] = dict(prompt=len(ids), ttft=r["ttft"], prefill_tok_s=len(ids) / r["ttft"], decode=r["decode"], total=r["total"], needle_found=P["needle"] in text, head=text[:120])
        print(f"  long {n}: prompt {len(ids)}  TTFT {r['ttft']:.1f} s ({len(ids) / r['ttft']:.0f} tok/s prefill)  decode {r['decode']:.1f} tok/s  needle {'OK' if P['needle'] in text else 'MISS'}", flush=True)


def visit(a):
    RES.mkdir(exist_ok=True)
    (RAW / "logs").mkdir(parents=True, exist_ok=True)
    P = load_prompts()
    srv = Server(a.server)
    gpu = Gpu()
    out = dict(server=a.server, suite=a.suite, round=a.round, started=time.strftime("%Y-%m-%dT%H:%M:%S"), extra=a.extra)
    try:
        out["boot_s"] = srv.start(a.extra)
        print(f"[{a.server}] healthy after {out['boot_s']:.0f} s", flush=True)
        with gpu.phase("boot_idle"):
            time.sleep(2)
        if a.suite == "full":
            suite_full(srv, P, gpu, out)
        elif a.suite == "ttft":
            suite_ttft(srv, P, gpu, out)
        elif a.suite == "conc":
            suite_conc(srv, P, gpu, out)
        elif a.suite == "long":
            suite_long(srv, P, gpu, out, [int(x) for x in a.lens.split(",")])
    except Exception as e:  # noqa: BLE001
        out["error"] = repr(e)[:500]
        print("ERROR", repr(e)[:500], flush=True)
    finally:
        out["gpu_phases"] = gpu.phases
        out["vram_peak_mib"] = max((x[5] for x in gpu.samples), default=None)
        out["vram_after_stop_mib"] = srv.stop()
        gpu.stop = True
        tag = f"{a.server}_{a.suite}_r{a.round}"
        raw = dict(out)
        json.dump(raw, open(RAW / f"visit_{tag}.json", "w"))
        slim = json.loads(json.dumps(raw))
        for k in ("e0", "e6_greedy", "e8_greedy"):  # token ids stay in runs/ (ignored); results/ keeps summaries + a hash
            v = slim.get(k)
            if isinstance(v, dict):
                for x in v.values():
                    x["ids_sha"] = hashlib.sha256(json.dumps(x.pop("ids", [])).encode()).hexdigest()[:12]
            elif isinstance(v, list):
                for x in v:
                    x["ids_sha"] = hashlib.sha256(json.dumps(x.pop("ids", None)).encode()).hexdigest()[:12]
        json.dump(slim, open(RES / f"visit_{tag}.json", "w"), indent=1)
        print(f"[{a.server}] done; vram after stop {out['vram_after_stop_mib']} MiB; wrote results/visit_{tag}.json", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prompts")
    v = sub.add_parser("visit")
    v.add_argument("server")
    v.add_argument("--suite", default="full")
    v.add_argument("--round", type=int, default=1)
    v.add_argument("--lens", default="28000,56000,100000")
    v.add_argument("--extra", nargs="*", default=[], help="extra server args (ours: CLI flags; HQ: -e KEY=VAL pairs passed to docker run)")
    sub.add_parser("tables")
    a = ap.parse_args()
    if a.cmd == "prompts":
        RES.mkdir(exist_ok=True)
        (RAW).mkdir(parents=True, exist_ok=True)
        build_prompts()
    elif a.cmd == "visit":
        visit(a)
    else:
        sys.path.insert(0, str(HERE))
        import tables

        tables.main()
