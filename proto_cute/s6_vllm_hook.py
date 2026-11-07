import sys, torch
from w4_common import make_w4, dequant_nat
import vllm_mp_kernel as VK
from vllm.model_executor.kernels.linear import choose_mp_linear_kernel
from vllm.model_executor.kernels.linear.mixed_precision import MPLinearLayerConfig
from vllm.scalar_type import scalar_types

N, K = 5120, 6144
cfg = MPLinearLayerConfig(full_weight_shape=(K, N), partition_weight_shape=(K, N), weight_type=scalar_types.uint4b8, act_type=torch.bfloat16, group_size=128, zero_points=False)
print("selected without plugin:", choose_mp_linear_kernel(cfg).__name__)
VK.register()
cls = choose_mp_linear_kernel(cfg); print("selected with plugin:   ", cls.__name__)
w, s = make_w4(N, K, seed=3)                       # checkpoint layout == vLLM weight_packed [N,K/8], weight_scale [N,K/128]
Wref = dequant_nat(w, s).to(torch.bfloat16)
layer = torch.nn.Module()
layer.weight_packed = torch.nn.Parameter(w.clone(), requires_grad=False); layer.weight_scale = torch.nn.Parameter(s.clone(), requires_grad=False)
kern = cls(cfg, "weight_packed", "weight_scale", None); kern.process_weights_after_loading(layer)
print("inverse-repack dequant == natural dequant:", torch.equal(VK.dequant_repacked(layer.cute_w, layer.cute_s), Wref))
for T in (1, 5, 8, 16, 17, 64):
    x = torch.randn(T, K, device="cuda", dtype=torch.bfloat16)
    y = kern.apply_weights(layer, x); ref = (x.float() @ Wref.float().T)
    print(f"T={T:3d}  out {tuple(y.shape)}  max rel err vs fp32 ref {((y.float()-ref).abs().max()/ref.abs().max()).item():.1e}")
# 3-D input (vLLM passes [tokens, K] or [B, T, K])
y3 = kern.apply_weights(layer, torch.randn(2, 3, K, device="cuda", dtype=torch.bfloat16)); print("3-D input ->", tuple(y3.shape))
# host cost of the python wrapper path
import time
x = torch.randn(1, K, device="cuda", dtype=torch.bfloat16)
for _ in range(20): kern.apply_weights(layer, x)
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(500): kern.apply_weights(layer, x)
print(f"apply_weights host cost incl. dlpack wrapping: {(time.perf_counter()-t0)/500*1e6:.0f} us/call")
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g): yg = kern.apply_weights(layer, x)
g.replay(); torch.cuda.synchronize(); print("CUDA graph capture of apply_weights: ok, replay equal:", torch.equal(yg, kern.apply_weights(layer, x)))
