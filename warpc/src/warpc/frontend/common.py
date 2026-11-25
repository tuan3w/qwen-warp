"""Graph-building functions every family shares, over (graph, source, missing): `missing` collects the tensors the
checkpoint lacks, so a family's build reports all of them at once."""

from __future__ import annotations

from ..errors import UnsupportedModel
from ..ir import Graph, TType, WeightDecl
from ..sources import Source

# Source.quant() of a linear -> the scheme its weight keeps on the device; every other weight is dense (bf16)
_SCHEMES = {"Q4_0": "q4_0", "W4G128": "w4g128"}


def inputs(g: Graph) -> tuple[str, str]:
    """The graph inputs of one window: tokens and their positions, i32[T]."""
    return g.input("tokens", TType(("T",), "i32")), g.input("pos", TType(("T",), "i32"))


def weight(
    g: Graph,
    source: Source,
    missing: list[str],
    decl: str,
    tensor: str,
    shape,
    *,
    linear: bool = False,
    scheme: str | None = None,
) -> str:
    """Declare a weight: dense (bf16) unless `scheme` says otherwise or, for a linear, the source stores it packed."""
    if tensor not in source:
        missing.append(tensor)
    if scheme is None:
        scheme = _SCHEMES.get(source.quant(tensor), "dense") if linear and tensor in source else "dense"
    return g.weight(WeightDecl(decl, tuple(shape), scheme, tensor))


def linear(g: Graph, source: Source, missing: list[str], x: str, decl: str, tensor: str, n: int, k: int) -> str:
    return g.op("linear", [x], weight=weight(g, source, missing, decl, tensor, (n, k), linear=True))


def rmsnorm(g: Graph, source: Source, missing: list[str], x: str, decl: str, tensor: str, size: int, **attrs) -> str:
    return g.op("rmsnorm", [x], weight=weight(g, source, missing, decl, tensor, (size,)), **attrs)


def text_prefix(source: Source) -> str:
    """The text model's tensor prefix: "model.language_model." in a vision-language checkpoint, "model." in a
    plain causal-LM one."""
    return "model.language_model." if "model.language_model.embed_tokens.weight" in source else "model."


def head(
    g: Graph, source: Source, missing: list[str], x: str, prefix: str, *, softcap: float | None = None, **norm
) -> None:
    """Final norm, lm_head (the embedding if tied), fp32 logits (softcapped if `softcap`) and their argmax as the
    graph outputs; then the missing-tensor report and the verification every family ends with."""
    c = source.config
    x = rmsnorm(g, source, missing, x, "final_norm", f"{prefix}norm.weight", c.hidden_size, name="hidden", **norm)
    tensor = f"{prefix}embed_tokens.weight" if c.tie_embeddings else "lm_head.weight"
    lm = linear(g, source, missing, x, "lm_head", tensor, c.vocab_size, c.hidden_size)
    logits = g.op("cast", [lm], dtype="f32", name="logits")
    if softcap:
        logits = g.op("softcap", [logits], cap=float(softcap), name="logits")
    g.output(logits=logits, ids=g.op("argmax", [logits], name="ids"))
    finish(g, missing)


def finish(g: Graph, missing: list[str]) -> None:
    """The missing-tensor report and the verification every family ends with."""
    if missing:
        raise UnsupportedModel(f"checkpoint lacks {len(missing)} tensors the graph reads: {', '.join(missing[:4])}")
    g.verify()
