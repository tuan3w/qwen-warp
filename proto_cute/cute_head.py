"""P2 spike: int4 lm_head with exact rescoring (target) and a restricted draft vocabulary (MTP drafts) for stock vLLM, no vLLM files modified.
 - `ParallelLMHead` is replaced through PluggableLayer.register_oot; its quant method quantises the bf16 head to int4 g128 at load (the engine's `quant_head_i4`).
 - Target logits (<= 8 rows): CuTe W4 GEMV over the whole vocabulary -> top-R candidates -> exact bf16 rescoring -> a full-width logits tensor that is -inf outside the R candidates
   (greedy ids equal the bf16 head's unless the true top-1 falls outside the int4 top-R; sampling is truncated to R candidates).
 - Draft logits (MTP draft steps, flagged by the registered Qwen3_5MTP subclass): CuTe GEMV over the 68,864-token draft vocabulary only.
 - More than 8 rows: the original bf16 path."""
import sys, numpy as np, torch
sys.path.insert(0, "/media/fatman/data/workspace/qwen-warp/engine")
from vllm_mp_kernel import PRECOMPILE_T
import pack, cute_w4
import vllm.model_executor.layers.vocab_parallel_embedding as vpe
from vllm.model_executor.custom_op import PluggableLayer, op_registry_oot
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MTP

R_CAND = 32
MAX_T = 16
DRAFT_VOCAB = "/media/fatman/data/workspace/qwen-warp/engine/draft_vocab_ext_68864.npy"
_state = {"draft": False, "building_draft": False}

def _gemv(w, s, x):
    N, K = w.shape[0], w.shape[1] * 8
    out = torch.empty(x.shape[0], N, dtype=torch.bfloat16, device=x.device)
    fn = cute_w4.get(N, K, x.shape[0], w, s, x, out)
    fn(*cute_w4.wrap(w, s, x, out), cute_w4.cur())
    return out

def _head_impl(x: torch.Tensor, w: torch.Tensor, s: torch.Tensor, wb: torch.Tensor, ws: torch.Tensor, ss: torch.Tensor, ids: torch.Tensor,
               draft: bool, org: int) -> torch.Tensor:
    T = x.shape[0]
    if T > MAX_T:
        return torch.nn.functional.linear(x, wb)
    x = x.contiguous()
    out = torch.full((T, wb.shape[0]), float("-inf"), dtype=torch.bfloat16, device=x.device)
    if draft:
        out.index_copy_(1, ids, _gemv(ws, ss, x))
    else:
        q = _gemv(w, s, x)[:, :org]
        cand = torch.topk(q, R_CAND, dim=1).indices                      # [T, R]
        exact = torch.bmm(wb[cand], x.unsqueeze(-1)).squeeze(-1)         # [T, R] bf16 head rows of the candidates only
        out.scatter_(1, cand, exact)
    return out

def _head_fake(x, w, s, wb, ws, ss, ids, draft, org):
    return torch.empty(x.shape[0], wb.shape[0], dtype=torch.bfloat16, device=x.device)

class CuteHeadMethod(QuantizeMethodBase):
    def __init__(self, inner):
        super().__init__(); self.inner = inner

    def create_weights(self, *a, **k): return self.inner.create_weights(*a, **k)
    def embedding(self, layer, input_): return self.inner.embedding(layer, input_)
    def tie_weights(self, layer, embed): return self.inner.tie_weights(layer, embed)

    def process_weights_after_loading(self, layer):
        if hasattr(self.inner, "process_weights_after_loading"): self.inner.process_weights_after_loading(layer)
        wb = layer.weight.data; Vp, K = wb.shape
        if K % 512 or Vp % 32 or wb.dtype != torch.bfloat16: self.ok = False; return
        if not hasattr(torch.ops.vllm, "cute_head_logits"):
            from vllm.utils.torch_utils import direct_register_custom_op
            direct_register_custom_op("cute_head_logits", _head_impl, fake_impl=_head_fake)
        w, s = pack.quant_head_i4(wb, rows=512); pack.repack_w4(w, s, rows=128)
        ids = torch.from_numpy(np.load(DRAFT_VOCAB)).to(wb.device)
        nv = ids.numel(); ws = torch.empty(nv, K // 8, dtype=torch.int32, device=wb.device); ss = torch.empty(nv, K // 128, dtype=torch.bfloat16, device=wb.device)
        for c in range(0, nv, 4096):
            a, b = pack.quant_head_i4(wb[ids[c:c + 4096]], rows=512); ws[c:c + a.shape[0]] = a; ss[c:c + a.shape[0]] = b
        pack.repack_w4(ws, ss, rows=128)
        layer.cute_head = (w, s, ws, ss, ids); self.org = layer.org_vocab_size; self.ok = True
        for N, (ww, sw) in ((Vp, (w, s)), (nv, (ws, ss))):
            for t in PRECOMPILE_T:
                cute_w4.get(N, K, t, ww, sw, torch.zeros(t, K, dtype=torch.bfloat16, device=wb.device), torch.empty(t, N, dtype=torch.bfloat16, device=wb.device))

    def apply(self, layer, x, bias=None):
        if not getattr(self, "ok", False) or bias is not None or x.dim() != 2:
            return self.inner.apply(layer, x, bias)
        w, s, ws, ss, ids = layer.cute_head
        return torch.ops.vllm.cute_head_logits(x, w, s, layer.weight, ws, ss, ids, _state["draft"], self.org)

class CuteParallelLMHead(vpe.ParallelLMHead):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        if not _state["building_draft"]:       # the MTP model's own head is deleted later (the target's is shared in): do not quantise it
            self.quant_method = CuteHeadMethod(self.quant_method)

class CuteQwen3_5MTP(Qwen3_5MTP):
    def __init__(self, *a, **k):
        _state["building_draft"] = True
        try: super().__init__(*a, **k)
        finally: _state["building_draft"] = False

    def compute_logits(self, hidden_states):
        _state["draft"] = True
        try: return super().compute_logits(hidden_states)
        finally: _state["draft"] = False

def register():
    if "ParallelLMHead" not in op_registry_oot:
        PluggableLayer.register_oot(CuteParallelLMHead, name="ParallelLMHead")
    from vllm import ModelRegistry
    ModelRegistry.register_model("Qwen3_5MTP", "cute_head:CuteQwen3_5MTP")
