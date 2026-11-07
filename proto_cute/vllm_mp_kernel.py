"""Spike: the CuTe W4A16 GEMV as an out-of-tree vLLM `MPLinearKernel` (vLLM 0.30.0 API).
vLLM's compressed-tensors WNA16 layer stores weight_packed [N, K/8] int32 + weight_scale [N, K/128] = this repo's 'natural' layout, so
process_weights_after_loading() repacks in place (engine/pack.py) and apply_weights() calls the CuTe kernel for <=16 rows, dequant + GEMM above."""
import sys, torch
sys.path.insert(0, "/media/fatman/data/workspace/qwen-warp/engine")
import pack
import cute_w4
from vllm.model_executor.kernels.linear.mixed_precision import MPLinearKernel, MPLinearLayerConfig
from vllm.model_executor.kernels import linear as vlinear
from vllm.platforms import PlatformEnum
from vllm.scalar_type import scalar_types

_ORDER = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7])
_INV = {}

def _inv(dev):
    if dev not in _INV: _INV[dev] = torch.argsort(_ORDER).to(dev)       # created at load time: no host->device copy inside CUDA-graph capture
    return _INV[dev]

def dequant_repacked(w, s):
    """inverse of pack.repack_w4, then (nibble-8)*scale -> bf16 [N,K]; w [N,K/8] int32 repacked, s [N,K/128] bf16 repacked"""
    N, K8 = w.shape; nkg = K8 // 16; nt = N // 16
    sh = (4 * torch.arange(8, device=w.device)).to(torch.int32)
    out = torch.empty(N, K8 * 8, dtype=torch.bfloat16, device=w.device)
    inv = _inv(w.device)
    rows = 128
    for r0 in range(0, N, rows):
        n = min(rows, N - r0); t = n // 16
        wr = w[r0:r0 + n].view(t, nkg, 2, 8, 4, 4).permute(0, 2, 3, 1, 4, 5)           # [tile,half,g,kg,t,u]
        nib = ((wr.unsqueeze(-1) >> sh) & 0xF)[..., inv]                                  # natural nibble order
        nat = nib.permute(0, 1, 2, 3, 4, 5, 6).reshape(n, nkg, 4 * 4 * 8)                 # [N, kg, (t,u,e)] == 128 columns of the group
        sc = s[r0:r0 + n].view(t, nkg, 8, 2).permute(0, 3, 2, 1).reshape(n, nkg)
        out[r0:r0 + n] = ((nat.float() - 8.0) * sc.float().unsqueeze(-1)).reshape(n, K8 * 8).to(torch.bfloat16)
    return out

def _cute_linear_impl(x: torch.Tensor, w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    N, K = w.shape[0], w.shape[1] * 8
    x2 = x.reshape(-1, K).contiguous(); T = x2.shape[0]
    out = torch.empty(T, N, dtype=torch.bfloat16, device=x.device)
    if T <= CuteW4A16LinearKernel.MAX_GEMV_ROWS:
        for t0 in range(0, T, 8):                                           # <=8 rows per launch: a row's bits do not depend on the batch
            tn = min(8, T - t0); xs = x2[t0:t0 + tn]; os_ = out[t0:t0 + tn]
            fn = cute_w4.get(N, K, tn, w, s, xs, os_)
            fn(*cute_w4.wrap(w, s, xs, os_), cute_w4.cur())
    else:
        torch.matmul(x2, dequant_repacked(w, s).T, out=out)                 # spike: whole-matrix dequant; production = layer-major scratch + cuBLAS
    return out.view(*x.shape[:-1], N)

def _cute_linear_fake(x: torch.Tensor, w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    return torch.empty(*x.shape[:-1], w.shape[0], dtype=torch.bfloat16, device=x.device)

def _register_op():
    if not hasattr(torch.ops.vllm, "cute_w4a16_linear"):
        from vllm.utils.torch_utils import direct_register_custom_op
        direct_register_custom_op("cute_w4a16_linear", _cute_linear_impl, fake_impl=_cute_linear_fake)

class CuteW4A16LinearKernel(MPLinearKernel):
    MAX_GEMV_ROWS = 32

    @classmethod
    def get_min_capability(cls): return 80

    @classmethod
    def can_implement(cls, c):
        K, N = c.partition_weight_shape
        if c.weight_type != scalar_types.uint4b8 or c.zero_points: return False, "needs symmetric uint4b8"
        if c.act_type != torch.bfloat16: return False, "bf16 activations only"
        if c.group_size != 128: return False, "group size 128 only"
        if N % 16 or K % 512: return False, f"N%16 / K%512 ({N},{K})"
        return True, None

    def process_weights_after_loading(self, layer):
        w, s, _ = self._get_weight_params(layer)
        wd = w.data.contiguous().view(torch.int32); sd = s.data.contiguous().to(torch.bfloat16)
        pack.repack_w4(wd, sd, rows=128)
        layer.cute_w = torch.nn.Parameter(wd, requires_grad=False); layer.cute_s = torch.nn.Parameter(sd, requires_grad=False)
        _register_op(); _inv(wd.device)
        N, K = wd.shape[0], wd.shape[1] * 8
        for tn in range(1, 9):
            xs = torch.zeros(tn, K, dtype=torch.bfloat16, device=wd.device); os_ = torch.empty(tn, N, dtype=torch.bfloat16, device=wd.device)
            cute_w4.get(N, K, tn, wd, sd, xs, os_)
        for n in (self.w_q_name, self.w_s_name): setattr(layer, n, None)       # free the checkpoint copies

    def apply_weights(self, layer, x, bias=None):
        out = torch.ops.vllm.cute_w4a16_linear(x, layer.cute_w, layer.cute_s)
        return out + bias if bias is not None else out

def register(first=True):
    _register_op()
    lst = vlinear._POSSIBLE_KERNELS.setdefault(PlatformEnum.CUDA, [])
    if CuteW4A16LinearKernel not in lst:
        lst.insert(0, CuteW4A16LinearKernel) if first else lst.append(CuteW4A16LinearKernel)
