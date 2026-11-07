"""P2 spike, part 2: the two other bf16 weight reads in a vLLM decode cycle.
 1. small bf16 projections (GDN in_proj_ba, N <= 128): cuBLAS runs a 16x16-tile GEMM at 40 us per 1 MB weight; here a CuTe GEMV (one CTA per output row).
 2. the MTP layer (fc, qkv, o, gate_up, down), bf16 in the checkpoint and re-read by every draft step: quantised to int4 g128 at load (RTN) and run through the CuTe W4 GEMV.
Both are installed by patching vLLM's UnquantizedLinearMethod (process_weights_after_loading / apply), restricted by shape and layer prefix."""
import sys, torch
sys.path.insert(0, "/media/fatman/data/workspace/qwen-warp/engine")
import cutlass, cutlass.cute as cute
from cutlass import Int32, Uint32, Float32, BFloat16
import cuda.bindings.driver as cuda
import pack, cute_w4, ptx, vllm_mp_kernel

class Bf16Small:
    """out[T, N] = x[T, K] @ w[N, K]^T, bf16 in/out, fp32 accumulate; one 128-thread CTA per row."""
    def __init__(self, N, K, T):
        assert K % 1024 == 0 and 1 <= T <= 8
        self.N, self.K, self.T = N, K, T

    @cute.jit
    def __call__(self, mW: cute.Tensor, mX: cute.Tensor, mO: cute.Tensor, stream: cuda.CUstream):
        self.kernel(mW, mX, mO).launch(grid=[self.N, 1, 1], block=[128, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, mW: cute.Tensor, mX: cute.Tensor, mO: cute.Tensor):
        K, T = self.K, self.T
        tid, _, _ = cute.arch.thread_idx(); row, _, _ = cute.arch.block_idx()
        smem = cutlass.utils.SmemAllocator(); red = smem.allocate_tensor(Float32, cute.make_layout((4, 8)))
        acc = cute.make_rmem_tensor((T,), Float32)
        for t in cutlass.range_constexpr(T): acc[t] = Float32(0.0)
        for i in cutlass.range_constexpr(K // 8 // 128):
            c = tid + i * 128
            wv = ptx.ldw4(ptx.addr(mW, row * (K // 2) + c * 4))
            wf = []
            for u in cutlass.range_constexpr(4):
                wf.append(ptx.bits2f(wv[u] << 16)); wf.append(ptx.bits2f(wv[u] & Uint32(0xFFFF0000)))
            for t in cutlass.range_constexpr(T):
                xv = ptx.ldx4(ptx.addr(mX, t * (K // 2) + c * 4))
                s = acc[t]
                for u in cutlass.range_constexpr(4):
                    s = s + wf[2 * u] * ptx.bits2f(xv[u] << 16) + wf[2 * u + 1] * ptx.bits2f(xv[u] & Uint32(0xFFFF0000))
                acc[t] = s
        warp = tid // 32
        for t in cutlass.range_constexpr(T):
            v = cute.arch.warp_reduction_sum(acc[t])
            if tid % 32 == 0: red[warp, t] = v
        cute.arch.sync_threads()
        if tid < T:
            mO[tid, row] = (red[0, tid] + red[1, tid] + red[2, tid] + red[3, tid]).to(BFloat16)

_small = {}
def _small_fn(N, K, T, w, x, out):
    from cutlass.cute.runtime import from_dlpack
    key = (N, K, T)
    if key not in _small:
        args = (from_dlpack(w.view(torch.int32).view(-1), assumed_align=16), from_dlpack(x.view(torch.int32).view(-1), assumed_align=16), from_dlpack(out, assumed_align=16))
        _small[key] = cute.compile(Bf16Small(N, K, T), *args, cute_w4.cur())
    return _small[key]

def _bf16_small_impl(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    from cutlass.cute.runtime import from_dlpack
    T, K = x.shape; N = w.shape[0]
    x = x.contiguous(); out = torch.empty(T, N, dtype=torch.bfloat16, device=x.device)
    _small_fn(N, K, T, w, x, out)(from_dlpack(w.view(torch.int32).view(-1), assumed_align=16), from_dlpack(x.view(torch.int32).view(-1), assumed_align=16),
                                   from_dlpack(out, assumed_align=16), cute_w4.cur())
    return out
def _bf16_small_fake(x, w): return torch.empty(x.shape[0], w.shape[0], dtype=torch.bfloat16, device=x.device)

def _small_ok(layer, x=None):
    w = getattr(layer, "weight", None)
    return (w is not None and w.dim() == 2 and w.dtype == torch.bfloat16 and w.shape[0] <= 128 and w.shape[1] % 1024 == 0 and w.is_cuda
            and (x is None or (x.dim() == 2 and x.dtype == torch.bfloat16 and x.shape[0] <= 8)))

def _mtp_ok(layer):
    w = getattr(layer, "weight", None)
    return (w is not None and w.dim() == 2 and w.dtype == torch.bfloat16 and w.is_cuda and "mtp" in getattr(layer, "prefix", "") and w.shape[0] >= 1024
            and w.shape[0] % 32 == 0 and w.shape[1] % 512 == 0)

def install():
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    if getattr(UnquantizedLinearMethod, "_cute_patched", False): return
    from vllm.utils.torch_utils import direct_register_custom_op
    if not hasattr(torch.ops.vllm, "cute_bf16_small"):
        direct_register_custom_op("cute_bf16_small", _bf16_small_impl, fake_impl=_bf16_small_fake)
    vllm_mp_kernel._register_op()
    orig_post, orig_apply = UnquantizedLinearMethod.process_weights_after_loading, UnquantizedLinearMethod.apply

    def post(self, layer):
        orig_post(self, layer)
        if _mtp_ok(layer):
            wb = layer.weight.data
            w, s = pack.quant_head_i4(wb, rows=256); pack.repack_w4(w, s, rows=128)
            layer.cute_w, layer.cute_s = w, s
            N, K = w.shape[0], w.shape[1] * 8
            for t in range(1, 9):
                cute_w4.get(N, K, t, w, s, torch.zeros(t, K, dtype=torch.bfloat16, device=wb.device), torch.empty(t, N, dtype=torch.bfloat16, device=wb.device))
            layer.weight = torch.nn.Parameter(torch.empty(0, K, dtype=torch.bfloat16, device=wb.device), requires_grad=False)   # free the bf16 copy
            layer._cute_int4 = True
        elif _small_ok(layer):
            w = layer.weight.data
            for t in range(1, 9):
                _small_fn(w.shape[0], w.shape[1], t, w, torch.zeros(t, w.shape[1], dtype=torch.bfloat16, device=w.device), torch.empty(t, w.shape[0], dtype=torch.bfloat16, device=w.device))
    def apply(self, layer, x, bias=None):
        if getattr(layer, "_cute_int4", False):
            out = torch.ops.vllm.cute_w4a16_linear(x, layer.cute_w, layer.cute_s)
            return out + bias if bias is not None else out
        if bias is None and _small_ok(layer, x):
            return torch.ops.vllm.cute_bf16_small(x, layer.weight)
        return orig_apply(self, layer, x, bias)
    UnquantizedLinearMethod.process_weights_after_loading = post
    UnquantizedLinearMethod.apply = apply
    UnquantizedLinearMethod._cute_patched = True
