"""Graph-building functions every family shares, over (graph, source, missing): `missing` collects the tensors the
checkpoint lacks, so a family's build reports all of them at once."""

from __future__ import annotations

from ..errors import UnsupportedModel
from ..ir import K_MULTIPLE, Graph, TType, WeightDecl
from ..sources import Source

# Source.quant() of a weight -> the scheme it keeps on the device: a block format with a fused kernel, or the W4G128
# codes; every other weight is dense (bf16). The schemes a weight may take depend on where it is read: a Q6_K matrix
# only as the lm_head, a Q4_0 / Q4_1 matrix only as a layer's linear (the head's GEMV is bf16 or Q6_K)
_SCHEMES = {"Q4_0": "q4_0", "Q4_1": "q4_1", "Q6_K": "q6_k", "W4G128": "w4g128"}
LINEAR_SCHEMES = ("q4_0", "q4_1", "w4g128")
HEAD_SCHEMES = ("q6_k", "w4g128")


def inputs(g: Graph) -> tuple[str, str]:
    """The graph inputs of one window: tokens and their positions, i32[T]."""
    return g.input("tokens", TType(("T",), "i32")), g.input("pos", TType(("T",), "i32"))


def _wide_enough(shape, multiple: int) -> bool:
    """A matrix whose K the fused kernel can take."""
    return len(shape) == 2 and shape[1] % multiple == 0


def weight(
    g: Graph,
    source: Source,
    missing: list[str],
    decl: str,
    tensor: str,
    shape,
    *,
    schemes: tuple[str, ...] = (),
    scheme: str | None = None,
) -> str:
    """Declare a weight: dense (bf16) unless `scheme` says otherwise or the source stores it packed in one of
    `schemes` (a linear's). A block format whose kernel needs K a multiple of K_MULTIPLE is widened to bf16 at any
    other width."""
    if tensor not in source:
        missing.append(tensor)
    if scheme is None:
        scheme = _SCHEMES.get(source.quant(tensor) or "") if schemes and tensor in source else None
        fused = scheme in schemes and (scheme not in K_MULTIPLE or _wide_enough(shape, K_MULTIPLE[scheme]))
        scheme = scheme if fused else "dense"
    return g.weight(WeightDecl(decl, tuple(shape), scheme, tensor))


def linear(
    g: Graph,
    source: Source,
    missing: list[str],
    x: str,
    decl: str,
    tensor: str,
    n: int,
    k: int,
    schemes: tuple[str, ...] = LINEAR_SCHEMES,
) -> str:
    return g.op("linear", [x], weight=weight(g, source, missing, decl, tensor, (n, k), schemes=schemes))


def rmsnorm(g: Graph, source: Source, missing: list[str], x: str, decl: str, tensor: str, size: int, **attrs) -> str:
    return g.op("rmsnorm", [x], weight=weight(g, source, missing, decl, tensor, (size,)), **attrs)


def text_prefix(source: Source) -> str:
    """The text model's tensor prefix: "model.language_model." in a vision-language checkpoint, "model." in a
    plain causal-LM one."""
    return "model.language_model." if "model.language_model.embed_tokens.weight" in source else "model."


def head(
    g: Graph, source: Source, missing: list[str], x: str, prefix: str, *, softcap: float | None = None, **norm
) -> None:
    """Final norm, lm_head (the embedding if tied), fp32 logits (softcapped if `softcap`) and their argmax; the graph
    outputs are those and the final-norm rows `hidden`; then the missing-tensor report and the verification every
    family ends with."""
    c = source.config
    x = rmsnorm(g, source, missing, x, "final_norm", f"{prefix}norm.weight", c.hidden_size, name="hidden", **norm)
    tensor = f"{prefix}embed_tokens.weight" if c.tie_embeddings else "lm_head.weight"
    lm = linear(g, source, missing, x, "lm_head", tensor, c.vocab_size, c.hidden_size, HEAD_SCHEMES)
    logits = g.op("cast", [lm], dtype="f32", name="logits")
    if softcap:
        logits = g.op("softcap", [logits], cap=float(softcap), name="logits")
    g.output(hidden=x, logits=logits, ids=g.op("argmax", [logits], name="ids"))
    finish(g, missing)


def finish(g: Graph, missing: list[str]) -> None:
    """The missing-tensor report and the verification every family ends with."""
    if missing:
        raise UnsupportedModel(f"checkpoint lacks {len(missing)} tensors the graph reads: {', '.join(missing[:4])}")
    g.verify()
