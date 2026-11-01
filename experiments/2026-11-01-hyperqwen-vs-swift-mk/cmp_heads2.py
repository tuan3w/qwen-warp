import torch
from safetensors import safe_open
def deq(packed, scale, bits, rows):
    p = packed[rows].to(torch.int64) & 0xFFFFFFFF
    n = 32//bits
    vals = torch.stack([(p >> (bits*i)) & ((1<<bits)-1) for i in range(n)], -1).reshape(p.shape[0], -1).to(torch.float32) - (1<<(bits-1))
    s = scale[rows].float().repeat_interleave(128, dim=1)
    return vals*s
sw = "/sw/"
idx = __import__('json').load(open(sw+"model.safetensors.index.json"))["weight_map"]
H="/hq/Qwen3.8-27B-W4A16-AutoRound-fast/"
g = torch.Generator().manual_seed(0)
rows = torch.randperm(248320, generator=g)[:4000]
def cos(a,b): return torch.nn.functional.cosine_similarity(a,b,dim=1).mean().item()
with safe_open(sw+idx["lm_head.weight"],"pt") as f: W=f.get_tensor("lm_head.weight")[rows].float()
with safe_open(H+"model-00007-of-00007.safetensors","pt") as f: D=deq(f.get_tensor("lm_head.weight_packed"), f.get_tensor("lm_head.weight_scale"),4,rows)
print("lm_head cos(swift bf16, hq int4)", cos(W,D), "rel err", ((W-D).norm()/W.norm()).item())
with safe_open(sw+idx["model.language_model.embed_tokens.weight"],"pt") as f: W=f.get_tensor("model.language_model.embed_tokens.weight")[rows].float()
with safe_open(H+"model-00006-of-00007.safetensors","pt") as f: D=deq(f.get_tensor("model.language_model.embed_tokens.weight_packed"), f.get_tensor("model.language_model.embed_tokens.weight_scale"),8,rows)
print("embed cos", cos(W,D), "rel err", ((W-D).norm()/W.norm()).item())
with safe_open(sw+"model_mtp.safetensors","pt") as f:
    ks=list(f.keys()); print(ks)
    fc=f.get_tensor("mtp.fc.weight").float(); 
    q=f.get_tensor("mtp.layers.0.self_attn.q_proj.weight").float()
with safe_open(H+"model_extra_tensors.safetensors","pt") as f:
    r=torch.arange(fc.shape[0]); D=deq(f.get_tensor("mtp.fc.weight_packed"),f.get_tensor("mtp.fc.weight_scale"),4,r)
    print("mtp.fc cos",cos(fc,D), ((fc-D).norm()/fc.norm()).item())
    r=torch.arange(q.shape[0]); D=deq(f.get_tensor("mtp.layers.0.self_attn.q_proj.weight_packed"),f.get_tensor("mtp.layers.0.self_attn.q_proj.weight_scale"),4,r)
    print("mtp.q cos",cos(q,D), ((q-D).norm()/q.norm()).item())
