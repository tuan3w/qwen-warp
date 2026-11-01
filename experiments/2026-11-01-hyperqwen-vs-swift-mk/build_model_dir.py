"""Build the HyperQwen "fast-variant" layout for OUR Swift-1.5 INT4 checkpoint without touching either source (both are mounted read-only).

Run inside the HyperQwen image as root (HQ's -fast shard 6 is mode 0600 root), CPU only, no network:
  docker run --rm --network none --entrypoint /app/venv/bin/python \
    -v ~/models/swift-1.5-int4:~/models/swift-1.5-int4:ro \
    -v ~/HyperQwen/models:~/HyperQwen/models:ro \
    -v $OUT:/out -v $PWD:/w:ro ghcr.io/syv-ai/hyperqwen:latest /w/build_model_dir.py /out

Evidence that this is legitimate (runs/hq/work/cmp_heads2.py): Swift's embed_tokens / lm_head / MTP equal the base Qwen3.8-27B ones up to quantisation error
(embed int8 rel. error 0.0067 = pure int8 error; lm_head / mtp.fc / mtp.q int4 rel. error 0.145-0.154 = int4 g128 error), so HQ's fast-variant heads
(int4-GPTQ lm_head, int8 embed, int4-GPTQ MTP, 40,960-row draft head + id list, all calibrated on BASE hidden states / outputs) are reused unchanged.
Layout: Swift body shards 2-4 symlinked, shards 1 and 5 rewritten WITHOUT the bf16 embed / lm_head (vLLM loads every key of a file it opens), HQ int8 embed written to its own file
(not HQ's shard 6: it carries base's final norm.weight, which would shadow Swift's), HQ lm_head shard + model_extra_tensors symlinked, model_mtp.safetensors dropped.
config.json = Swift's with HQ's group_1..3 (lm_head int4, embed int8, mtp int4) appended and 'lm_head' / 're:^mtp.*' removed from `ignore`; Swift's kv_cache_scheme (fp8 k/v scales) kept.
"""
import json
import os
import shutil
import sys

from safetensors import safe_open
from safetensors.torch import save_file

OUT = sys.argv[1]
SW = "~/models/swift-1.5-int4"
HQ = "~/HyperQwen/models/Qwen3.8-27B-W4A16-AutoRound-fast"
EMB = "model.language_model.embed_tokens"

idx = json.load(open(f"{SW}/model.safetensors.index.json"))
hq_idx = json.load(open(f"{HQ}/model.safetensors.index.json"))
wm = {}
shard1 = idx["weight_map"][EMB + ".weight"]
for k, f in idx["weight_map"].items():
    if k == "lm_head.weight" or k == EMB + ".weight" or f == "model_mtp.safetensors":
        continue
    wm[k] = f
# rewrite the shards that hold a dropped bf16 tensor (embed in shard 1, lm_head in shard 5) without it
rewritten = {}
for drop in (EMB + ".weight", "lm_head.weight"):
    sh = idx["weight_map"][drop]
    rewritten.setdefault(sh, set()).add(drop)
for sh, drops in rewritten.items():
    out_t = {}
    with safe_open(f"{SW}/{sh}", "pt") as f:
        meta = f.metadata()
        for k in f.keys():
            if k not in drops:
                out_t[k] = f.get_tensor(k)
    assert all(wm.get(k) == sh for k in out_t), "keys of rewritten shard not all in index"
    save_file(out_t, f"{OUT}/{sh}", metadata=meta or {"format": "pt"})
    print("rewrote", sh, len(out_t), "tensors, dropped", sorted(drops))
# int8 embed from HQ shard 6 (skip norm.weight: Swift's own stays in Swift's shard)
e = {}
with safe_open(f"{HQ}/model-00006-of-00007.safetensors", "pt") as f:
    for k in f.keys():
        if k.startswith(EMB + "."):
            e[k] = f.get_tensor(k)
assert len(e) == 3
save_file(e, f"{OUT}/embed_int8.safetensors", metadata={"format": "pt"})
for k in e:
    wm[k] = "embed_int8.safetensors"
# symlinks (targets resolve inside the container because the sources are mounted at the same absolute paths)
def link(dst, src):
    p = f"{OUT}/{dst}"
    if os.path.lexists(p):
        os.remove(p)
    os.symlink(src, p)
for f in sorted(set(wm.values())):
    if f not in rewritten and f != "embed_int8.safetensors":
        link(f, f"{SW}/{f}")
for k, f in hq_idx["weight_map"].items():
    if k.startswith("lm_head.") or k.startswith("mtp."):
        wm[k] = f
        link(f, f"{HQ}/{f}")
link("mtp_draft_vocab_ids.pt", f"{HQ}/mtp_draft_vocab_ids.pt")
json.dump({"metadata": idx.get("metadata", {}), "weight_map": wm}, open(f"{OUT}/model.safetensors.index.json", "w"), indent=1)
for n in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json", "processor_config.json"]:
    shutil.copy(f"{SW}/{n}", f"{OUT}/{n}")
# config
c = json.load(open(f"{SW}/config.json"))
h = json.load(open(f"{HQ}/config.json"))
qc, hq_qc = c["quantization_config"], h["quantization_config"]
for g in ("group_1", "group_2", "group_3"):
    qc["config_groups"][g] = hq_qc["config_groups"][g]
qc["ignore"] = [x for x in qc["ignore"] if x not in ("lm_head", "re:^mtp.*")]
assert set(qc["ignore"]) == set(hq_qc["ignore"]), (set(qc["ignore"]) ^ set(hq_qc["ignore"]))
json.dump(c, open(f"{OUT}/config.json", "w"), indent=2)
print("ok;", len(wm), "index entries;", sorted(set(wm.values())))
