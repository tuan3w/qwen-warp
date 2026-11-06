"""One-time real-trace dump (SparseTraces).  GPU job: run under `flock /tmp/gpu-engine.lock`.

  SMK_TRACE=1 SWIFT_MODEL=... python dump.py DOC [DOC ...] [--ctx 32768] [--kvlayers 7,23,39,63] [--win 128] [--ndoc 64]

For each doc (token ids .npy under /media/fatman/data/sparselab/traces/docs or a path):
  1. fast prefill of ids[:-1] (rows 0..P-2) with the Tracer sampling `ndoc` random document rows (linear inputs only);
  2. greedy decode of the model's own continuation (engine decode graph, EOS-aware) -> true decode tokens;
  3. DeltaNet state restored; the window [last prompt token] + generated[:win-1] is run as ONE teacher-forced segment (W4 mma path, same maths as decode up to
     accumulation order) with the Tracer capturing, for all window rows: linear inputs (qkv, o, gu, down) of the trace layers, rope'd q of the attention layers, the
     final-norm hidden + dense logits (top-256 + logsumexp), and for the KV layers the PACKED int4 cache bytes of the history (positions < P-1, engine layout) + the exact
     bf16 rows of the window itself + sink rows;
  4. check: argmax of window row i == window token i+1 (teacher forcing reproduces the engine's greedy decode).
Output: /media/fatman/data/sparselab/traces/<doc>/  (bf16 tensors stored as int16 bit patterns in .npy; kvdeq.py / load.py read them).
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("SMK_TRACE", "1")
sys.path.insert(0, "/media/fatman/data/workspace/qwen-warp-traces/engine")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine import Engine, pack  # noqa: E402
from prefill import FastPrefill  # noqa: E402

DOCS = Path("/media/fatman/data/sparselab/traces/docs")
OUT = Path("/media/fatman/data/sparselab/traces")
EOS = {248046, 248044}
LIN_LAYERS = [2, 7, 14, 23, 30, 39, 46, 63]  # DeltaNet 2,14,30,46 + attention 7,23,39,63 (budget: trace dir capped at 1.5 GB)
ATTN_ALL = [7, 23, 39, 63]


def bf16_np(t):
    return t.contiguous().view(torch.int16).numpy()


def logits_chunked(xn, lm_head, chunk=24832):
    outs = []
    x = xn.float()
    for c0 in range(0, lm_head.shape[0], chunk):
        outs.append(x @ lm_head[c0 : c0 + chunk].float().t())
    return torch.cat(outs, 1)


def main():
    args = sys.argv[1:]
    opt = dict(ctx=32768, kvlayers=ATTN_ALL, win=128, ndoc=32, gen=200)
    docs = []
    i = 0
    while i < len(args):
        if args[i] in ("--ctx", "--win", "--ndoc", "--gen"):
            opt[args[i][2:]] = int(args[i + 1])
            i += 2
        elif args[i] == "--kvlayers":
            opt["kvlayers"] = [int(x) for x in args[i + 1].split(",") if x]
            i += 2
        else:
            docs.append(args[i])
            i += 1
    t0 = time.time()
    e = Engine(ctx_max=opt["ctx"], kv_fp8=2, verbose=False)
    fp = FastPrefill(e)
    print(f"engine loaded {time.time() - t0:.1f}s", flush=True)
    tr = fp.trace
    tr.layers = set(LIN_LAYERS)
    tr.attn_layers = set(ATTN_ALL)
    CM = e.ctx_max
    at_idx = fp.at_idx
    for name in docs:
        ids = np.load(name if name.endswith(".npy") else DOCS / f"{name}.npy").astype(np.int32).tolist()
        base = Path(name).stem
        P = len(ids)
        assert P + 256 < CM, (P, CM)
        d = OUT / base
        d.mkdir(parents=True, exist_ok=True)
        t1 = time.time()
        e.reset()
        smk = __import__("smk")
        smk.check(fp.lib.smkp_materialize(e.h, e.st), "materialize")
        torch.cuda.synchronize()
        cur = int(fp.ds[1])
        rng = np.random.default_rng(1234)
        want_doc = sorted(rng.choice(np.arange(64, P - 2), size=opt["ndoc"], replace=False).tolist())
        tr.want, tr.keep_kv = want_doc, False
        n = P - 1
        i = 0
        while i < n:
            T = min(4096, n - i)
            fp._segment(ids[i : i + T], i, cur)
            i += T
        torch.cuda.synchronize()
        rows_doc, _ = tr.fetch()
        fp.ds[0] = n
        e.pos = n
        t_pre = time.time() - t1
        # --- true decode tokens
        ds_save = fp.ds.clone()
        st_save = fp.dn_state[:, cur].clone()
        cv_save = fp.dn_conv[:, cur].clone()
        e._step(ids[-1:], 1)
        first = e.read_out_tok()
        toks = e.decode(opt["gen"])
        seq = [first] + toks
        cut = next((j for j, t in enumerate(seq) if t in EOS), len(seq))
        eos_hit = cut < len(seq)
        seq = seq[:cut]
        fp.ds.copy_(ds_save)
        fp.dn_state[:, cur].copy_(st_save)
        fp.dn_conv[:, cur].copy_(cv_save)
        e.pos = n
        torch.cuda.synchronize()
        W = min(opt["win"], len(seq) + 1)
        win = [ids[-1]] + seq[: W - 1]
        # --- window pass
        kvl = [l for l in opt["kvlayers"] if l in ATTN_ALL]
        tr.attn_layers = set(ATTN_ALL)
        tr.want = [n + j for j in range(W)]
        tr.keep_kv = True
        tr.kv = {}
        fp._segment(win, n, cur)
        torch.cuda.synchronize()
        a = e._fp_alloc
        xn_final = a["xn"][:W].clone()
        rows_win, q_win = tr.fetch()
        kv_ex = {l: tr.kv[l] for l in kvl}
        tr.keep_kv = False
        # --- logits reference + teacher-forcing check
        lg = logits_chunked(xn_final, e.G.lm_head.cuda() if not e.G.lm_head.is_cuda else e.G.lm_head)
        lse = torch.logsumexp(lg, 1)
        topv, topi = lg.topk(256, dim=1)
        am = lg.argmax(1).cpu().tolist()
        match = sum(am[j] == win[j + 1] for j in range(W - 1))
        # --- save
        for (li, kind), t in rows_win.items():
            np.save(d / f"lin_L{li}_{kind}_dec.npy", bf16_np(t))
        for (li, kind), t in rows_doc.items():
            np.save(d / f"lin_L{li}_{kind}_doc.npy", bf16_np(t))
        for li, t in q_win.items():
            np.save(d / f"q_L{li}.npy", bf16_np(t))
        np.save(d / "final_hidden.npy", bf16_np(xn_final.cpu()))
        np.savez(d / "logits_ref.npz", topv=topv.cpu().numpy(), topi=topi.cpu().numpy().astype(np.int32), lse=lse.cpu().numpy())
        for li in kvl:
            ai = at_idx[li]
            kvp = fp.kv_base + ai * fp.kv_layer_bytes
            codes = torch.as_tensor(__import__("prefill")._Buf(kvp, 2 * 4 * CM * 128), device="cuda").view(torch.uint8).view(2, 4, CM, 128)[:, :, :n].cpu().numpy()
            meta = torch.as_tensor(__import__("prefill")._Buf(kvp + 2 * 4 * CM * 128, 2 * 4 * CM * 4), device="cuda").view(torch.uint8).view(torch.int32).view(2, 4, CM)[:, :, :n].cpu().numpy()
            sink_off = 2 * 4 * CM * 132
            sink = torch.as_tensor(__import__("prefill")._Buf(kvp + sink_off, 2 * 4 * 4 * 256 * 2), device="cuda").view(torch.int16).view(2, 4, 4, 256).cpu().numpy()
            np.savez(d / f"kv_L{li}.npz", codes=codes, meta=meta, sink=sink, wk=bf16_np(kv_ex[li]["k"][n : n + W]), wv=bf16_np(kv_ex[li]["v"][n : n + W]))
            np.save(d / f"_check_kdeq_L{li}.npy", bf16_np(kv_ex[li]["k"][1000:1016].clone()))  # tiny: lets kvdeq.py self-check against the engine gather
        info = dict(doc=base, P=P, n_hist=n, W=W, eos_hit=bool(eos_hit), n_gen=len(seq), tf_match=f"{match}/{W - 1}", win_tokens=win, doc_pos=want_doc, kv_layers=kvl,
                    lin_layers=LIN_LAYERS, prefill_s=round(t_pre, 1), ctx_max=CM)
        (d / "meta.json").write_text(json.dumps(info))
        tok = __import__("transformers").AutoTokenizer.from_pretrained(os.environ.get("SWIFT_MODEL", "/media/fatman/data/models/Swift-1.5-Qwen3.8-27b-INT4"))
        print(f"{base}: P={P} prefill {t_pre:.0f}s  W={W} eos={eos_hit} teacher-forcing argmax match {match}/{W - 1}  total {time.time() - t1:.0f}s\n   gen: {tok.decode(win[1:60])!r}", flush=True)
        del kv_ex, rows_win, rows_doc, q_win, lg
        tr.kv = {}


if __name__ == "__main__":
    main()
