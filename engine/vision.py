"""Image input for Swift-1.5 (Qwen3.5 architecture): the checkpoint's bf16 vision tower on the GPU and the 3-D MRoPE positions of a prompt with images.

  from vision import Vision, mrope_positions
  vit = Vision()                                      # loads model.visual.* (bf16) from $SWIFT_MODEL onto the GPU
  rows = vit.encode(pixel_values, grid_thw)           # processor output -> [sum(t*h*w) / 4, 5120] bf16 embeddings, one row per image placeholder token
  pos, delta = mrope_positions(n_tokens, [(offset, (t, h, w)), ...])   # == vLLM Qwen3VLForConditionalGeneration._get_mrope_input_positions

The ViT (27 blocks, hidden 1152, 16 heads of 72, patch 16, temporal patch 2, 2x2 merger to 5120, no deepstack) follows transformers' Qwen3_5VisionModel:
  * patch embed: the Conv3d with kernel == stride is one GEMM over the flattened [3 * 2 * 16 * 16] patches;
  * learned 48 x 48 position table, bilinearly resampled to each image's patch grid (align_corners, fp32 weights), in 2x2 merge-block order;
  * 2-D rotary (axial: 18 h + 18 w frequencies, theta 1e4, rotate_half over the 72 dims), per-image (varlen) non-causal attention with FlashAttention-2 (vLLM's binding);
  * LayerNorm blocks with a gelu_pytorch_tanh MLP, then the merger (LayerNorm, 4 patches -> 4608, GELU, -> 5120).
Everything is batched over all patches of a request; the only per-image work is the tiny host-side index arithmetic (cached per grid).

MRoPE (text config: mrope_section [11, 11, 10], interleaved): rotary frequency l (0..31) of the 64 rotary dims takes its position from axis T if l % 3 == 0, H if l % 3 == 1 (l < 33),
W if l % 3 == 2 (l < 30) -- the kernels' `mrope_axis` (engine/src/prefill.cu).  Text rows have t = h = w; after the last image every row is at (KV position + delta).
"""
import hashlib
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

MODEL = os.environ.get("SWIFT_MODEL", os.path.expanduser("~/models/swift-1.5-int4"))
IMAGE_TOKEN, VISION_START, VISION_END = 248056, 248053, 248054
PREFIX = "model.visual."


def mrope_positions(n, images, merge=2):
    """3-D rope positions of a prompt of n tokens whose images are `images` = [(offset of the first placeholder token, (t, h, w) patch grid), ...] (vLLM's
    _get_mrope_input_positions for images: each image covers t * (h / merge) * (w / merge) placeholder tokens at its offset).  Returns (int64 [3][n], delta) with
    delta = max position + 1 - n, the offset of every later position (decode: rope position = KV position + delta)."""
    parts, st, nxt = [], 0, 0  # nxt = the next free position (max + 1 of the parts so far)
    for off, (t, h, w) in sorted(images, key=lambda x: x[0]):
        assert t == 1, f"an image has one temporal patch, got grid t={t}"
        gh, gw = h // merge, w // merge
        tl = off - st
        parts.append(np.broadcast_to(np.arange(tl), (3, tl)) + nxt)
        base = nxt + tl
        hh, ww = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
        parts.append(np.stack([np.zeros(gh * gw, dtype=np.int64), hh.ravel(), ww.ravel()]) + base)
        nxt = base + max(gh, gw)
        st = off + gh * gw
    if st < n:
        parts.append(np.broadcast_to(np.arange(n - st), (3, n - st)) + nxt)
        nxt += n - st
    pos = np.concatenate(parts, axis=1) if parts else np.zeros((3, 0), dtype=np.int64)
    return pos.astype(np.int64), int(nxt - n)


def n_tokens(grid, merge=2):
    t, h, w = (int(x) for x in grid)
    return t * (h // merge) * (w // merge)


@dataclass(eq=False)  # identity comparison: the pixels are a tensor
class ImageInput:
    """One image of a prompt: its placeholder tokens are [offset, offset + n), n = t * (h / 2) * (w / 2); key = the image's content hash (vLLM's mm hash);
    pixels = the processor's pixel_values of this image ([t * h * w][1536], host)."""
    offset: int
    grid: tuple
    key: str
    pixels: torch.Tensor | None = None

    @property
    def n(self):
        return n_tokens(self.grid)


def cache_key(ids, images):
    """The ids the prefix cache matches a prompt on (int32 array): the token ids, with every placeholder row of an image replaced by -1 - (31 bits of a hash of its mm hash).
    Two prompts that differ only in an image then diverge at that image's first placeholder row (no KV or state past it is shared), and the same image at the same place matches."""
    key = np.array(ids, dtype=np.int32)
    for im in images:
        h = int.from_bytes(hashlib.blake2b(str(im.key).encode(), digest_size=4).digest(), "little") & 0x7FFFFFFF
        key[im.offset : im.offset + im.n] = -1 - h
    return key


class Vision:
    """The vision tower on the GPU.  `mib` = device memory of its weights.  dtype=torch.float32, attn="sdpa": an fp32 copy for checking the implementation against an fp32 reference.

    weights="device": all 879 MiB of bf16 weights on the GPU.  weights="host": the 27 blocks' GEMM weights (790 MiB) stay in pinned host memory and are streamed block by block into two
    device buffers on a side stream (block i + 1 is copied while block i computes); the rest (patch embed, position table, merger, norms) stays on the device: 151 MiB in all.
    Encode is then PCIe-bound for small images (a block's 29 MiB take ~1.3 ms over PCIe 4.0 x16 against ~0.6 ms of compute at 448 x 448) and as fast as "device" for large ones.

    Precision: bf16 weights and GEMMs (fp32 accumulate), FlashAttention-2 in bf16, but the residual stream and the LayerNorms in fp32 (transformers keeps the residual in bf16).  27 bf16
    blocks amplify rounding on a few tokens: against an fp32 forward, transformers' bf16 model has tokens at a cosine of 0.96-0.99; the fp32 residual costs ~5% of the encode time
    (1920x1080: 219 vs 210 ms) and lifts, at 1920x1080, the mean cosine from 0.99969 to 0.99983 and the worst token from 0.970 to 0.982 (tests/test_vision.py)."""

    GEMM = ("attn.qkv.weight", "attn.qkv.bias", "attn.proj.weight", "attn.proj.bias", "mlp.linear_fc1.weight", "mlp.linear_fc1.bias", "mlp.linear_fc2.weight", "mlp.linear_fc2.bias")

    def __init__(self, model=MODEL, device="cuda", dtype=torch.bfloat16, attn="fa2", weights="device"):
        from safetensors import safe_open

        cfg = json.load(open(Path(model) / "config.json"))["vision_config"]
        self.cfg = cfg
        self.hid, self.nh, self.depth = cfg["hidden_size"], cfg["num_heads"], cfg["depth"]
        self.hd = self.hid // self.nh
        self.merge, self.patch, self.tp = cfg["spatial_merge_size"], cfg["patch_size"], cfg["temporal_patch_size"]
        self.side = int(cfg["num_position_embeddings"] ** 0.5)
        self.out_dim = cfg["out_hidden_size"]
        assert not cfg.get("deepstack_visual_indexes"), "deepstack mergers are not implemented (the Swift-1.5 tower has none)"
        assert cfg["hidden_act"] == "gelu_pytorch_tanh"
        idx = json.load(open(Path(model) / "model.safetensors.index.json"))["weight_map"]
        names = [k for k in idx if k.startswith(PREFIX)]
        W = {}
        for shard in sorted({idx[k] for k in names}):
            with safe_open(str(Path(model) / shard), framework="pt", device="cpu") as f:
                for k in names:
                    if idx[k] == shard:
                        W[k[len(PREFIX):]] = f.get_tensor(k).to(dtype=dtype)
        W["patch_embed.proj.weight"] = W["patch_embed.proj.weight"].reshape(self.hid, -1).contiguous()  # Conv3d(kernel == stride) == one GEMM over the flattened patch
        assert weights in ("device", "host"), weights
        self.weights = weights
        streamed = {f"blocks.{i}.{n}" for i in range(self.depth) for n in self.GEMM} if weights == "host" else set()
        for k in W:
            if k not in streamed:
                W[k] = W[k].to(device)
        self.W = W
        self.blocks = [{n: W[f"blocks.{i}.{n}"] for n in self.GEMM} for i in range(self.depth)]
        f32 = lambda n: W[n].float()  # noqa: E731
        self.n32 = [tuple(f32(f"blocks.{i}.{n}") for n in ("norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias")) for i in range(self.depth)]
        self.m32 = (f32("merger.norm.weight"), f32("merger.norm.bias"))
        self._host = None
        if weights == "host":  # one pinned flat buffer per block, two device buffers with a view per tensor
            sizes = [W[f"blocks.0.{n}"].numel() for n in self.GEMM]
            offs = np.concatenate([[0], np.cumsum(sizes)]).tolist()
            self._host = []
            for b in self.blocks:
                flat = torch.empty(offs[-1], dtype=dtype, pin_memory=True)
                for n, o, s in zip(self.GEMM, offs, sizes):
                    flat[o : o + s].copy_(b[n].reshape(-1))
                self._host.append(flat)
            self._dev = [torch.empty(offs[-1], dtype=dtype, device=device) for _ in range(2)]
            self._views = [{n: d[o : o + s].view(self.blocks[0][n].shape) for n, o, s in zip(self.GEMM, offs, sizes)} for d in self._dev]
            self._cs = torch.cuda.Stream(device)
            self._done = [None, None]  # per device buffer: event after the last block that read it (on the compute stream)
            self._ready = [None, None]  # per device buffer: event after its copy (on the side stream)
            self.blocks = None
            for k in streamed:
                W[k] = None
        self.mib = (sum(t.numel() * t.element_size() for t in W.values() if t is not None and t.is_cuda) + sum(t.numel() * t.element_size() for t in self.n32[0]) * self.depth
                    + (sum(d.numel() * d.element_size() for d in self._dev) if self._host else 0)) / 2**20
        sd = self.hd // 2  # axial rope: hd / 4 frequencies per axis
        self.inv_freq = (1.0 / (10000.0 ** (torch.arange(0, sd, 2, dtype=torch.float32) / sd))).to(device)
        self.device, self.dtype, self.attn = device, dtype, attn
        from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func

        self._fa = flash_attn_varlen_func

    def _attend(self, q, k, v, cu, mx, seq):
        if self.attn == "fa2":
            return self._fa(q, k, v.contiguous(), max_seqlen_q=mx, cu_seqlens_q=cu, max_seqlen_k=mx, cu_seqlens_k=cu, causal=False, softmax_scale=self.hd**-0.5, fa_version=2)
        outs, r = [], 0
        for n in seq:  # reference path: SDPA per image, in blocks of 1024 query rows (the [heads, rows, n] fp32 scores of a 1920x1080 image whole would be 4.3 GiB)
            kk, vv = (x[r : r + n].transpose(0, 1) for x in (k, v))
            outs += [F.scaled_dot_product_attention(q[a : min(a + 1024, r + n)].transpose(0, 1), kk, vv, scale=self.hd**-0.5).transpose(0, 1) for a in range(r, r + n, 1024)]
            r += n
        return torch.cat(outs)

    # ------------------------------------------------------------------------------------------------ per-grid index tables (host, cached)
    @lru_cache(maxsize=64)
    def _grid_tables(self, t, h, w):
        """Per patch, in 2x2 merge-block order: (h, w) coordinates, the 4 bilinear taps into the position table and their fp32 weights."""
        m, side = self.merge, self.side
        hh, ww = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        blk = (h // m, m, w // m, m)
        hh = hh.reshape(blk).transpose(0, 2, 1, 3).ravel()
        ww = ww.reshape(blk).transpose(0, 2, 1, 3).ravel()

        def axis(i, size):
            src = i.astype(np.float32) * np.float32(side - 1) / np.float32(max(size - 1, 1))
            fl = np.floor(src)
            taps = np.clip(fl.astype(np.int64)[:, None] + np.arange(2), 0, side - 1)
            wt = np.clip(1 - np.abs(src[:, None] - fl[:, None] - np.arange(2)), 0, None).astype(np.float32)
            return taps, wt

        ht, hw = axis(hh, h)
        wt_, wwt = axis(ww, w)
        taps = (ht[:, :, None] * side + wt_[:, None, :]).reshape(-1, 4)
        wts = (hw[:, :, None] * wwt[:, None, :]).reshape(-1, 4)
        coords = np.stack([hh, ww], axis=1)
        return np.tile(coords, (t, 1)), np.tile(taps, (t, 1)), np.tile(wts, (t, 1))

    def _tables(self, grids):
        co, ta, wt = zip(*(self._grid_tables(*g) for g in grids))
        d = self.device
        return (torch.from_numpy(np.concatenate(co)).to(d, non_blocking=True), torch.from_numpy(np.concatenate(ta)).to(d, non_blocking=True),
                torch.from_numpy(np.concatenate(wt)).to(d, non_blocking=True))

    # ------------------------------------------------------------------------------------------------ weights="host": block streaming
    def _fetch(self, i):
        """Copy block i's GEMM weights into device buffer i % 2 on the side stream, after the block that used that buffer last (i - 2) has run."""
        k = i % 2
        with torch.cuda.stream(self._cs):
            if self._done[k] is not None:
                self._cs.wait_event(self._done[k])
            self._dev[k].copy_(self._host[i], non_blocking=True)
            self._ready[k] = self._cs.record_event()

    def _block(self, i):
        """Block i's GEMM weights (device tensors); weights="host": wait for its copy and start the next block's."""
        if self._host is None:
            return self.blocks[i]
        torch.cuda.current_stream().wait_event(self._ready[i % 2])
        if i + 1 < self.depth:
            self._fetch(i + 1)
        return self._views[i % 2]

    # ------------------------------------------------------------------------------------------------ forward
    @torch.inference_mode()
    def encode(self, pixel_values, grid_thw):
        """pixel_values [N][3 * 2 * 16 * 16] (the processor's output, any float dtype, host or device), grid_thw [n_images][3] -> [N / 4][5120] bf16 on the GPU (images in order)."""
        grids = [tuple(int(x) for x in g) for g in (grid_thw.tolist() if torch.is_tensor(grid_thw) else grid_thw)]
        W, bf = self.W, self.dtype
        x = pixel_values.to(self.device, non_blocking=True).to(bf)
        N = x.shape[0]
        assert N == sum(t * h * w for t, h, w in grids), (N, grids)
        coords, taps, wts = self._tables(grids)
        x = F.linear(x, W["patch_embed.proj.weight"], W["patch_embed.proj.bias"]).float()
        x = x + (W["pos_embed.weight"][taps].float() * wts[:, :, None]).sum(1)  # the residual stream stays fp32 (see the class docstring)
        fr = coords.float()[:, :, None] * self.inv_freq  # [N][2][18]
        fr = fr.reshape(N, -1)
        emb = torch.cat([fr, fr], dim=-1)  # [N][72]: rotate_half layout
        cos, sin = emb.cos()[:, None, :], emb.sin()[:, None, :]
        seq = [h * w for t, h, w in grids for _ in range(t)]  # one attention sequence per frame
        cu = torch.tensor(np.concatenate([[0], np.cumsum(seq)]), dtype=torch.int32).to(self.device, non_blocking=True)
        mx = max(seq)
        if self._host is not None:  # prefetch the first block's weights (after the previous call's block that used its buffer: _done carries over)
            self._fetch(0)
        nh, hd = self.nh, self.hd
        for i, B32 in enumerate(self.n32):
            B = self._block(i)
            y = F.layer_norm(x, (self.hid,), B32[0], B32[1], 1e-6).to(bf)
            qkv = F.linear(y, B["attn.qkv.weight"], B["attn.qkv.bias"]).view(N, 3, nh, hd)
            q, k, v = qkv.unbind(1)
            q, k = _rope(q, cos, sin), _rope(k, cos, sin)
            o = self._attend(q, k, v, cu, mx, seq)
            x += F.linear(o.reshape(N, self.hid), B["attn.proj.weight"], B["attn.proj.bias"])
            y = F.layer_norm(x, (self.hid,), B32[2], B32[3], 1e-6).to(bf)
            y = F.gelu(F.linear(y, B["mlp.linear_fc1.weight"], B["mlp.linear_fc1.bias"]), approximate="tanh")
            x += F.linear(y, B["mlp.linear_fc2.weight"], B["mlp.linear_fc2.bias"])
            if self._host is not None:
                self._done[i % 2] = torch.cuda.current_stream().record_event()  # buffer i % 2 may be overwritten (by block i + 2) after this block's kernels
        y = F.layer_norm(x, (self.hid,), self.m32[0], self.m32[1], 1e-6).to(bf).view(-1, self.hid * self.merge**2)
        y = F.gelu(F.linear(y, W["merger.linear_fc1.weight"], W["merger.linear_fc1.bias"]))
        return F.linear(y, W["merger.linear_fc2.weight"], W["merger.linear_fc2.bias"])


def _rope(x, cos, sin):
    """transformers' apply_rotary_pos_emb_vision: fp32 rotate_half over the whole head, back to bf16."""
    xf = x.float()
    h = xf.shape[-1] // 2
    rot = torch.cat([-xf[..., h:], xf[..., :h]], dim=-1)
    return (xf * cos + rot * sin).to(x.dtype)
