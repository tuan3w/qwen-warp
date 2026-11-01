"""ctypes wrapper around build/libswiftmk[_core].so for tests / the driver (lab venv: ~/llm-lab/venv/bin/python, torch cu130)."""
import ctypes as C
import os
import sys
from pathlib import Path

import torch

ENGINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ENGINE / "tests"))
sys.path.insert(0, str(ENGINE.parent / "oracle"))
sys.path.insert(0, str(ENGINE))

SMK_EPI_SEGS, SMK_EPI_SILU, SMK_EPI_F32 = 0, 1, 2


class Epi(C.Structure):
    _fields_ = [("kind", C.c_int), ("end", C.c_int * 3), ("out", C.c_void_p * 3)]


def load(name="libswiftmk.so"):
    bd = os.environ.get("SMK_BUILD", "build")  # private build dir override (make BUILD=...)
    p = ENGINE / bd / name
    if not p.exists():
        p = ENGINE / bd / "libswiftmk_core.so"
    lib = C.CDLL(str(p))
    vp, ci, cf = C.c_void_p, C.c_int, C.c_float
    lib.smk_w4_gemv.argtypes = [vp, vp, vp, ci, ci, ci, C.POINTER(Epi), vp]
    lib.smk_w4_gemv_cfg.argtypes = [vp, vp, vp, ci, ci, ci, C.POINTER(Epi), ci, vp]
    lib.smk_bf16_gemv_f32.argtypes = [vp, vp, ci, ci, ci, vp, vp]
    lib.smk_bf16_gemv_small.argtypes = [vp, vp, ci, ci, ci, ci, vp, vp, vp]
    lib.smk_rmsnorm.argtypes = [vp, vp, vp, vp, vp, ci, ci, cf, vp]
    lib.smk_embed.argtypes = [vp, vp, vp, ci, ci, vp]
    lib.smk_argmax_ws_bytes.restype = C.c_size_t
    lib.smk_argmax.argtypes = [vp, ci, ci, vp, vp, vp, vp]
    lib.smk_interleave_gateup.argtypes = [vp, vp, vp, ci, C.c_size_t, vp]
    for n in ("smk_w4_gemv", "smk_w4_gemv_cfg", "smk_bf16_gemv_f32", "smk_bf16_gemv_small", "smk_rmsnorm", "smk_embed", "smk_argmax", "smk_interleave_gateup"):
        getattr(lib, n).restype = ci
    return lib


def stream():
    return torch.cuda.current_stream().cuda_stream


def ptr(t):
    return None if t is None else t.data_ptr()


def check(err, what=""):
    if err != 0:
        raise RuntimeError(f"{what}: cuda/smk error {err}")


def epi_segs(N, outs):
    """outs: list of up to 3 tensors [T][width]; segments in order."""
    e = Epi()
    e.kind = SMK_EPI_SEGS
    end = 0
    for i in range(3):
        if i < len(outs):
            end += outs[i].shape[1]
            e.out[i] = outs[i].data_ptr()
        e.end[i] = end if i < len(outs) else N
    assert end == N, (end, N)
    return e


def epi_simple(kind, out):
    e = Epi()
    e.kind = kind
    e.out[0] = out.data_ptr()
    return e


def rel_l2(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


class LayerW(C.Structure):
    _fields_ = [("kind", C.c_int)] + [(n, C.c_void_p) for n in
                ("ln1", "ln2", "qkv_w", "qkv_s", "ab", "conv_w", "A_log", "dt_bias", "norm_w", "q_norm", "k_norm")] + \
               [("k_scale", C.c_float), ("v_scale", C.c_float)] + \
               [(n, C.c_void_p) for n in ("o_w", "o_s", "gu_w", "gu_s", "dn_w", "dn_s")]


class ModelW(C.Structure):
    _fields_ = [("layers", LayerW * 64), ("embed", C.c_void_p), ("final_norm", C.c_void_p), ("lm_head", C.c_void_p),
                ("head_q_kind", C.c_int), ("head_q", C.c_void_p), ("head_qs", C.c_void_p)]


def setup_engine_api(lib):
    vp, ci = C.c_void_p, C.c_int
    lib.smk_engine_create.argtypes = [C.POINTER(ModelW), ci, ci]
    lib.smk_engine_create.restype = vp
    lib.smk_engine_create_ex.argtypes = [C.POINTER(ModelW), ci, ci, ci]
    lib.smk_engine_create_ex.restype = vp
    lib.smk_engine_tmax.argtypes = [vp]
    lib.smk_engine_tmax.restype = ci
    lib.smk_engine_destroy.argtypes = [vp]
    for n in ("smk_engine_reset", "smk_engine_decode", "smk_engine_set_tokens", "smk_engine_step", "smk_engine_step_nograph"):
        getattr(lib, n).restype = ci
    lib.smk_engine_reset.argtypes = [vp, vp]
    lib.smk_engine_set_tokens.argtypes = [vp, C.POINTER(ci), ci, vp]
    lib.smk_engine_step.argtypes = [vp, ci, ci, vp]
    lib.smk_engine_step_nograph.argtypes = [vp, ci, ci, vp]
    lib.smk_engine_decode.argtypes = [vp, ci, C.POINTER(ci), vp]
    lib.smk_engine_read.argtypes = [vp, ci, vp, C.c_size_t]
    lib.smk_engine_read.restype = ci
    lib.smk_engine_set_head.argtypes = [vp, ci]
    lib.smk_engine_set_head.restype = ci
    lib.smk_engine_head.argtypes = [vp]
    lib.smk_engine_head.restype = ci
    lib.smk_engine_bytes.argtypes = [vp]
    lib.smk_engine_bytes.restype = C.c_size_t
