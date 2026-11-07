"""Host-resident bf16 tables for stock vLLM (the engine's `host_tables`): the embedding and the lm_head's bf16 rows stay in pinned host memory (4.7 GiB less VRAM),
and a CuTe kernel gathers the few rows a step needs straight over PCIe (UVA pointer).  Rows: R per call."""
import torch, cutlass, cutlass.cute as cute
from cutlass import Int32, Int64, Uint32
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda
import ptx

class HostGather:
    """out[r, :] = table[ids[r], :] ; table = bf16 [V, K] at host address `addr`; one 128-thread CTA per row, 16 B per thread per step."""
    def __init__(self, K):
        assert K % (8 * 128) == 0
        self.K = K

    @cute.jit
    def __call__(self, mIds: cute.Tensor, mO: cute.Tensor, addr: Int64, R: Int32, stream: cuda.CUstream):
        self.kernel(mIds, mO, addr).launch(grid=[R, 1, 1], block=[128, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, mIds: cute.Tensor, mO: cute.Tensor, addr: Int64):
        K = self.K
        tid, _, _ = cute.arch.thread_idx(); r, _, _ = cute.arch.block_idx()
        row = mIds[r].to(Int64)
        src = addr + row * (K * 2)
        for i in cutlass.range_constexpr(K // 8 // 128):
            c = tid + i * 128
            v = ptx.ldg4(src + c * 16)
            ptx.stg4(ptx.addr(mO, r * (K // 2) + c * 4), v[0], v[1], v[2], v[3])

_fns = {}
def host_gather(ids, addr, K):
    """ids int32 [R] (cuda) -> bf16 [R, K] (cuda)"""
    R = ids.numel()
    out = torch.empty(R, K, dtype=torch.bfloat16, device=ids.device)
    a = (from_dlpack(ids, assumed_align=4), from_dlpack(out.view(torch.int32).view(-1), assumed_align=16))
    st = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    if K not in _fns:
        _fns[K] = cute.compile(HostGather(K), *a, Int64(addr), Int32(R), st)
    _fns[K](*a, Int64(addr), Int32(R), st)
    return out

class HostTable:
    def __init__(self, w):
        self.K = w.shape[1]
        self.t = w.detach().to("cpu").pin_memory()           # keep alive
        self.addr = self.t.data_ptr()

def _gather_impl(ids: torch.Tensor, addr: int, K: int) -> torch.Tensor:
    return host_gather(ids.reshape(-1).to(torch.int32).contiguous(), addr, K)
def _gather_fake(ids, addr, K): return torch.empty(ids.numel(), K, dtype=torch.bfloat16, device=ids.device)

def register_op():
    if not hasattr(torch.ops.vllm, "cute_host_gather"):
        from vllm.utils.torch_utils import direct_register_custom_op
        direct_register_custom_op("cute_host_gather", _gather_impl, fake_impl=_gather_fake)
