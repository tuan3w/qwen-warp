"""The bench workloads on the CPU: the corpus of the v0.12.0 tag, the agent trace, its probes, their scorers and the
loop that runs a trace against an engine.

  uv run --package warpc python warpc/tests/test_bench_workload.py   # WARPC_TEST_SWIFT27, WARPC_MODELS: see below

The structure tests run on a tiny fake tokenizer over a synthetic corpus. The corpus tests need the v0.12.0 tag (or
$WARPC_LEGACY_DIR); the real-tokenizer tests skip without $WARPC_TEST_SWIFT27 (the 27B's tokenizer.json) or
$WARPC_MODELS (Qwen3-0.6B's)."""

import dataclasses
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path
from typing import ClassVar
from unittest import mock

import numpy as np

from harness import Skip, run
from warpc.bench import workload as w
from warpc.bench.engines import Turn
from warpc.prefix import common_prefix

# ---- a fake tokenizer: ChatML specials, one token per digit, per newline, per run of spaces, per word


class FakeTokenizer:
    sha256 = "fake"
    SPECIAL: ClassVar[dict[str, int]] = {"<|im_start|>": 1, "<|im_end|>": 2, "<think>": 3, "</think>": 4}
    PIECE = re.compile(r"<\|im_start\|>|<\|im_end\|>|</?think>|\d|\n|[^\S\n]+|[A-Za-z_]+|[^\w\s]")

    def __init__(self):
        self.vocab = {i: t for t, i in self.SPECIAL.items()}

    def encode(self, text):
        ids, spans = [], []
        for m in self.PIECE.finditer(text):
            piece = m.group()
            i = self.SPECIAL.get(piece) or 1000 + zlib.crc32(piece.encode()) % 60000
            self.vocab[i] = piece
            ids.append(i)
            spans.append(m.span())
        return ids, spans

    def decode(self, ids):
        return "".join(self.vocab[i] for i in ids)

    def token_id(self, token):
        return self.SPECIAL.get(token)


def synthetic_text(lines: int = 4000) -> str:
    g = np.random.RandomState(7)
    out = []
    for n in range(lines):
        indent = " " * int(g.choice([0, 4, 8]))
        out.append(f"{indent}int v{int(g.randint(0, 999))} = f(a{n}, {int(g.randint(0, 99))}); // line {n}\n")
    return "".join(out)


def fake_trace(scale: int = 32_000, **kw):
    tok = FakeTokenizer()
    corpus = w.Corpus.encode(tok, synthetic_text())
    return tok, w.build_trace(tok, scale=scale, corpus=corpus, **kw)


FAKE_TRACE_SHA = "08da93007c9303882ebbc873fe12123d10a2a4bdf25a50b250f0cf6dac84c57a"

# the real tokenizers and the trace shas recorded on them
SWIFT27_TOKENIZER_SHA = "d73c2c5f7aa0ed522c8d96ef3524739eb61e3c78e74839a2ce4a1c56ea340a20"
QWEN3_TOKENIZER_SHA = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
SWIFT27_CORPUS_TOKENS = 161_644
QWEN3_CORPUS_TOKENS = 155_113
SWIFT27_TRACE_SHA = {
    32_000: "7f74656887a3c9e28ad12a8e226a730994efc9a03e6da50ee18bab63f786e69d",
    100_000: "805640632d7fe95dc9a0a4ac95c5701976f266c3f857ac837e32bc1ecb3396c6",
}
QWEN3_TRACE_SHA = {32_000: "4d4205184a051f36376b74d10c908cbe25ae9ace7e895dc04f7495c3194bed1d"}
SWIFT27_SLICE_SHA = {
    (4096, 0): "604df228ca04bda845cf392538a832520b79dd615a2f04bd1d1e651aa86733b5",
    (4096, 2): "4c72e95cebbd0130fc7316e44b9e48febeae37c3b1e63c3b4b857b5008c4dfa0",
    (385, 0): "86b46191ca805c7b41ea384f34a8c106423d3fb8302ff084e0e0c9049c80ce32",
}


def swift27():
    path = os.environ.get("WARPC_TEST_SWIFT27")
    if not path:
        raise Skip("WARPC_TEST_SWIFT27 unset")
    if not (Path(path) / "tokenizer.json").is_file():
        raise Skip("WARPC_TEST_SWIFT27 has no tokenizer.json")
    return w.FileTokenizer(path)


def qwen3():
    models = os.environ.get("WARPC_MODELS")
    if not models:
        raise Skip("WARPC_MODELS unset")
    path = Path(models) / "Qwen3-0.6B" / "hf"
    if not (path / "tokenizer.json").is_file():
        raise Skip("no Qwen3-0.6B tokenizer under WARPC_MODELS")
    return w.FileTokenizer(path)


def corpus_or_skip() -> str:
    try:
        return w.corpus_text()
    except w.CorpusUnavailable as e:
        raise Skip(str(e)) from e


# ---- the corpus


def test_corpus_is_the_tags_text():
    text = corpus_or_skip()
    assert len(text.encode()) == w.CORPUS_BYTES + 30  # 16 files, 15 blank-line joins
    assert hashlib.sha256(text.encode()).hexdigest() == w.CORPUS_SHA256
    files = w._tag_files()
    if files is None:
        raise Skip("the v0.12.0 tag is absent")
    assert len(files) == 16 and sum(map(len, files)) == w.CORPUS_BYTES


def test_without_the_tag_the_error_names_fetch_tags():
    with mock.patch.object(w, "_tag_files", return_value=None), mock.patch.dict(os.environ):
        os.environ.pop(w.LEGACY_ENV, None)
        try:
            w.corpus_text.__wrapped__()
        except w.CorpusUnavailable as e:
            assert "git fetch --tags" in str(e) and "v0.12.0" in str(e)
        else:
            raise AssertionError("no error")


def test_legacy_checkout_is_the_fallback():
    files = w._tag_files()
    if files is None:
        raise Skip("the v0.12.0 tag is absent")
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "engine" / "src"
        src.mkdir(parents=True)
        for i, blob in enumerate(files):
            (src / f"f{i:02d}.cu").write_bytes(blob)
        with mock.patch.object(w, "_tag_files", return_value=None), mock.patch.dict(os.environ, {w.LEGACY_ENV: tmp}):
            assert w.corpus_text.__wrapped__() == w.corpus_text()
        (src / "f00.cu").write_bytes(b"changed")
        with mock.patch.object(w, "_tag_files", return_value=None), mock.patch.dict(os.environ, {w.LEGACY_ENV: tmp}):
            try:
                w.corpus_text.__wrapped__()
            except w.CorpusMismatch:
                pass
            else:
                raise AssertionError("a different corpus was accepted")


def test_importing_the_workload_loads_no_tokenizer_library():
    code = "import sys, warpc.bench.workload; assert not {'tokenizers', 'transformers', 'vllm'} & set(sys.modules)"
    subprocess.run([sys.executable, "-c", code], check=True)


# ---- the trace on a fake tokenizer


def test_trace_structure():
    _, trace = fake_trace()
    assert len(trace.turns) == 12 and len(trace.probes) == 19
    kinds = [p.kind for p in trace.probes]
    assert [kinds.count(k) for k in ("needle", "tail", "vt", "json")] == [10, 3, 2, 4]
    assert max(len(w.probe_ids(trace, i)) for i in range(19)) == 32_000
    assert trace.ctx_max == 32_256
    assert [p.n_new for p in trace.probes] == [24] * 15 + [40] * 4
    assert len(trace.turns[-1].tool) >= 900
    assert all(500 <= len(t.tool) <= 2000 for t in trace.turns[:-1])
    assert all(100 <= len(t.assistant) < 300 and 100 <= t.decode < 300 for t in trace.turns)
    assert len(trace.prefix) > trace.haystack


def test_trace_is_deterministic_and_golden():
    _, first = fake_trace()
    _, second = fake_trace()
    assert first == second and first.trace_sha256 == second.trace_sha256
    _, other_seed = fake_trace(seed=1)
    assert other_seed.trace_sha256 != first.trace_sha256
    assert first.trace_sha256 == FAKE_TRACE_SHA, first.trace_sha256


def test_every_prompt_is_the_chat_format():
    tok, trace = fake_trace()
    chat = trace.chat
    assert (chat.im_start, chat.im_end, chat.think) == (1, 2, tuple(tok.encode("<think>\n")[0]))
    assert trace.prefix[:3] == [1, *chat.system]
    k = 5
    ids = w.render(trace, k)
    assert ids[: len(trace.prefix)] == trace.prefix
    assert ids[-len(chat.think) :] == list(chat.think)
    assert ids[-len(chat.think) - len(chat.assistant) - 1 : -len(chat.think)] == [1, *chat.assistant]
    # each prompt diverges from the next right after the assistant header: the template drops the think tail
    for j in range(11):
        a, b = w.render(trace, j), w.render(trace, j + 1)
        header = len(a) - len(chat.think)
        assert common_prefix(np.array(a), np.array(b)) == header, j
    history = w.history(trace)
    last = trace.turns[11]
    pair = [1, *chat.assistant, *last.assistant, 2, *chat.nl, 1, *chat.user, *last.tool, 2, *chat.nl]
    assert history == [*w.render(trace, 11)[: -len(chat.think) - len(chat.assistant) - 1], *pair]
    for i, probe in enumerate(trace.probes):
        ids = w.probe_ids(trace, i)
        assert ids[: len(history)] == history
        assert ids[len(history) :] == [
            1,
            *chat.user,
            *probe.question,
            2,
            *chat.nl,
            1,
            *chat.assistant,
            *chat.think_empty,
        ]


def test_linear_mode_extends_the_cached_tokens():
    _, trace = fake_trace()
    own = [[7000 + j] * (50 + j) for j in range(12)]
    for k in range(11):
        a, b = w.render(trace, k, "linear", own), w.render(trace, k + 1, "linear", own)
        assert b[: len(a)] == a


def test_facts_sit_at_line_starts_at_their_depths():
    tok, trace = fake_trace()
    prefix = trace.prefix
    start = 1 + len(trace.chat.system) + len(tok.encode(w.INSTR)[0])
    assert prefix[start + trace.haystack : start + trace.haystack + 1] == [trace.chat.im_end]
    for probe in trace.probes:
        if probe.kind != "tail":
            assert all(abs(d - t) < 0.011 for d, t in zip(probe.where["depth"], probe.where["target"], strict=True))
    for probe in trace.probes:
        if probe.kind not in ("needle", "json"):
            continue
        code = tok.encode(probe.answer["code"])[0]
        [at] = [i for i in range(start, start + trace.haystack) if prefix[i : i + len(code)] == code]
        line = at
        while tok.decode([prefix[line - 1]]) != "\n":
            line -= 1
        assert tok.decode(prefix[line:at]).startswith("// vault "), probe.label
        assert abs((at - start) / trace.haystack - probe.where["depth"][0]) < 0.001, probe.label
    text = tok.decode(prefix)
    for probe in trace.probes:
        if probe.kind == "vt":
            assert text.count(probe.answer["value"]) == 1, probe.label


def test_tail_needles_are_at_their_distances():
    tok, trace = fake_trace()
    tool = trace.turns[-1].tool
    for probe in (p for p in trace.probes if p.kind == "tail"):
        code = tok.encode(probe.answer["code"])[0]
        at = next(i for i in range(len(tool)) if tool[i : i + len(code)] == code)
        assert len(tool) + 2 - at == probe.where["distance"]
        assert abs(probe.where["distance"] - probe.where["target"]) <= w.TAIL_SNAP_TOLERANCE
    regimes = w.tail_regimes(trace)
    assert [r["target"] for r in regimes] == [128, 380, 700]
    from warpc.kvcache import FORMATS

    assert regimes[0]["last"] < FORMATS["k8v4w"].window < regimes[1]["first"]
    assert regimes[1]["last"] < FORMATS["int4_rot"].window < regimes[2]["first"]


def test_a_fork_prefills_only_its_question():
    _, trace = fake_trace()
    rows = w.fork_rows(trace)
    h = len(w.history(trace))
    assert {r["resume"] for r in rows} == {h + 2}
    assert all(30 <= r["prefill_rows"] < w.QUESTION_MAX for r in rows), rows


def test_only_a_plan_that_knows_the_turn_token_forks_cheaply():
    """The 8-row plan snapshots below the assistant header, and plan_windows with another tokenizer's turn token finds
    no header: either way a fork re-prefills everything above the system checkpoint."""
    from warpc.prefix import class_d_plan
    from warpc.windows import plan_windows

    _, trace = fake_trace()
    for plan in (class_d_plan, plan_windows):
        try:
            w.fork_rows(trace, plan=plan)
        except w.TraceError as e:
            assert "fork point" in str(e)
        else:
            raise AssertionError(f"{plan.__name__} forks cheaply")
    assert w.fork_rows(trace, plan=lambda ids: plan_windows(ids, turn_token=trace.chat.im_start)) == w.fork_rows(trace)


def test_the_builder_refuses_what_breaks_the_fork_claims():
    tok, trace = fake_trace()
    # one device snapshot cannot hold the system checkpoint and the fork point
    try:
        w.fork_rows(trace, device_snapshots=1)
    except w.TraceError as e:
        assert "device_snapshots" in str(e)
    else:
        raise AssertionError("device_snapshots=1 accepted")
    short = dataclasses.replace(trace, turns=[*trace.turns[:-1], dataclasses.replace(trace.turns[-1], tool=[5] * 899)])
    try:
        w.check_trace(short)
    except w.TraceError as e:
        assert "R_11" in str(e) or "899" in str(e)
    else:
        raise AssertionError("a short R_11 accepted")
    corpus = w.Corpus.encode(tok, synthetic_text())
    with mock.patch.object(w, "JSON_QUESTION", "word " * 250 + "{name}"):
        try:
            w.build_trace(tok, scale=32_000, corpus=corpus)
        except w.TraceError as e:
            assert "question" in str(e)
        else:
            raise AssertionError("a long question accepted")
    with mock.patch.object(w, "TAIL_DISTANCES", (220, 380, 700)):
        try:
            w.build_trace(tok, scale=32_000, corpus=corpus)
        except w.TraceError as e:
            assert "tail needle" in str(e)
        else:
            raise AssertionError("a tail needle inside the wrong window accepted")
    try:
        w.build_trace(tok, scale=32_000, corpus=w.Corpus.encode(tok, synthetic_text(500)))
    except ValueError as e:
        assert "too short" in str(e) or "corpus" in str(e)
    else:
        raise AssertionError("a short corpus accepted")


def test_a_tokenizer_without_chatml_is_unsupported():
    class Bare(FakeTokenizer):
        def token_id(self, token):
            return None if token == "<think>" else super().token_id(token)

    try:
        w.Chat.of(Bare())
    except w.Unsupported as e:
        assert "<think>" in str(e)
    else:
        raise AssertionError("no error")


def test_corpus_ids_slices():
    tok = FakeTokenizer()
    corpus = w.Corpus.encode(tok, synthetic_text(8000))
    assert w.slice_stride(385) == w.slice_stride(4096) == 4608 and w.slice_stride(16384) == 16384
    assert w.corpus_ids(tok, 4096, 2, corpus=corpus) == corpus.ids[2 * 4608 : 2 * 4608 + 4096]
    assert w.corpus_ids(tok, 16384, 1, corpus=corpus) == corpus.ids[16384 : 2 * 16384]
    try:
        w.corpus_ids(tok, 1_000_000, corpus=corpus)
    except ValueError as e:
        assert "too short" in str(e)
    else:
        raise AssertionError("no error")


def test_corpus_workload_and_stamp():
    tok = FakeTokenizer()
    corpus = w.Corpus.encode(tok, synthetic_text(8000))
    ids, workload = w.corpus_workload(tok, 385, 1, n_new=256, corpus=corpus)
    assert ids == corpus.ids[4608 : 4608 + 385]
    assert workload["ids_sha"] == w.ids_sha(ids) and workload["ctx_max"] == 1024 and workload["slice"] == 1
    assert workload["corpus_sha256"] == corpus.sha256 and workload["tokenizer_sha256"] == "fake"
    assert w.corpus_workload(tok, 4096, n_new=256, corpus=corpus)[1]["ctx_max"] == 4608  # the shipped 27B gate value
    assert w.stamp_fields(tok) == {"corpus_sha256": w.CORPUS_SHA256, "tokenizer_sha256": "fake"}


# ---- scoring


def probe_of(trace, label):
    return next(p for p in trace.probes if p.label == label)


def test_needle_and_vt_scores_are_exact_matches():
    _, trace = fake_trace()
    needle = probe_of(trace, "needle@0.04")
    code = needle.answer["code"]
    assert w.score_probe(needle, f"The code is {code}.")["ok"]
    assert not w.score_probe(needle, f"The code is {code[:-1]}0.")["ok"]
    assert not w.score_probe(needle, "00000-0000 or " + code)["ok"]  # the first code is the answer
    assert not w.score_probe(needle, "I cannot find it.")["ok"]
    assert w.score_probe(probe_of(trace, "tail@380"), "x " + probe_of(trace, "tail@380").answer["code"])["ok"]
    vt = probe_of(trace, "vt@0.10")
    assert w.score_probe(vt, f"It is {vt.answer['value']}")["ok"]
    assert not w.score_probe(vt, f"12345 then {vt.answer['value']}")["ok"]


def test_json_scores_strip_one_code_fence():
    _, trace = fake_trace()
    probe = probe_of(trace, "json@0.35")
    good = json.dumps(probe.answer)
    assert w.strip_fence("```json\n" + good + "\n```") == good
    assert w.strip_fence("```\n" + good) == good  # a fence the 40 tokens cut off
    assert w.strip_fence(good) == good
    for text in (good, f"```json\n{good}\n```", f"```\n{good}\n```", f"  {good}  ", f"```json\n{good}"):
        assert w.score_probe(probe, text) == {"ok": True, "format_ok": True, "value_ok": True, "found": None}, text
    wrong = json.dumps({**probe.answer, "code": "11111-2222"})
    assert w.score_probe(probe, wrong) == {"ok": False, "format_ok": True, "value_ok": False, "found": None}
    assert w.score_probe(probe, json.dumps({"vault": probe.answer["vault"]}))["format_ok"] is False
    assert w.score_probe(probe, json.dumps([1, 2]))["format_ok"] is False
    assert w.score_probe(probe, 'Sure: {"vault": ')["format_ok"] is False
    assert w.score_probe(probe, "")["ok"] is False


def test_score_counts_per_kind():
    tok, trace = fake_trace()
    outputs = [tok.encode(correct_text(p))[0] for p in trace.probes]
    scored = w.score(trace, tok.decode, outputs)
    assert scored["probes_ok"] == 19 and scored["json_format_ok"] == scored["json_value_ok"] == 4
    assert scored["by_kind"] == {
        "needle": {"ok": 10, "n": 10},
        "tail": {"ok": 3, "n": 3},
        "vt": {"ok": 2, "n": 2},
        "json": {"ok": 4, "n": 4},
    }
    outputs[3] = tok.encode("no idea")[0]
    assert w.score(trace, tok.decode, outputs)["by_kind"]["needle"] == {"ok": 9, "n": 10}


def test_the_reply_ends_at_the_end_of_turn_token():
    """The decode ignores the end of sequence, so a model that ends its turn goes on: text after `<|im_end|>` is not
    the answer (a JSON reply followed by more text must still parse, a wrong code after it must not match)."""
    tok, trace = fake_trace()
    after = tok.encode(" and also 99999-8888 {junk")[0]
    json_probe = probe_of(trace, "json@0.65")
    needle = probe_of(trace, "needle@0.24")
    cut = [*tok.encode(json.dumps(json_probe.answer))[0], trace.chat.im_end, *after]
    assert w.reply_ids(trace, cut) == cut[: cut.index(trace.chat.im_end)]
    assert w.reply_ids(trace, after) == after
    outputs = [tok.encode(correct_text(p))[0] for p in trace.probes]
    outputs[trace.probes.index(json_probe)] = cut
    outputs[trace.probes.index(needle)] = [*tok.encode("It is " + needle.answer["code"])[0], trace.chat.im_end, *after]
    assert w.score(trace, tok.decode, outputs)["probes_ok"] == 19
    outputs[trace.probes.index(needle)] = [*tok.encode("It is 99999-8888")[0], trace.chat.im_end, *after]
    scored = w.score(trace, tok.decode, outputs)
    assert scored["probes_ok"] == 18 and scored["probes"][trace.probes.index(needle)]["reply"] == "It is 99999-8888"


def correct_text(probe) -> str:
    if probe.kind == "vt":
        return f"The number is {probe.answer['value']}."
    if probe.kind == "json":
        return "```json\n" + json.dumps(probe.answer) + "\n```"
    return f"The code is {probe.answer['code']}."


# ---- the cannot table


def test_cannot_reasons():
    ok = {"engine": "warpc", "kv": "bf16", "ctx_max": 100_096, "hybrid": True, "backend": "swiftmk"}
    assert w.agent_cannot(**ok) is None
    assert w.agent_cannot(**ok | {"kv": "int4_rot"}) is None
    assert w.agent_cannot(**ok | {"kv": "k8v4w"})[0] == "kernel_missing"
    assert w.agent_cannot(**ok | {"kv": "fp8"})[0] == "env"
    assert w.agent_cannot(**ok | {"kv": "fp8", "swiftmk_lib": True}) is None
    assert w.agent_cannot(**ok | {"kv": "fp8", "backend": "cuda", "swiftmk_lib": True})[0] == "kernel_missing"
    assert w.agent_cannot(**ok | {"engine": "llama"}) == ("model_limit", "no GGUF of the 27B in models.toml")
    assert w.agent_cannot(**ok | {"engine": "vllm", "mtp": 3})[0] == "unsupported"
    assert w.agent_cannot(**ok | {"engine": "vllm"}) is None  # capacity is only known once loaded
    assert w.agent_cannot(**ok | {"engine": "hand", "kv": "k8v8"})[0] == "unsupported"
    assert w.agent_cannot(**ok | {"hybrid": False, "positions": 40_960})[0] == "model_limit"
    assert w.agent_cannot(**ok | {"hybrid": False, "ctx_max": 32_256, "positions": 40_960}) is None
    assert w.agent_cannot(**ok | {"chatml": False})[0] == "unsupported"


# ---- running a trace against an engine


class FakeEngine:
    """A prefix cache that is perfect on the longest shared prefix with the last prompt, answering every probe with the
    planted fact. `cold_differs`: a cold prefill decodes differently (a broken cache)."""

    def __init__(self, trace, tok, *, cold_differs=False, wrong=()):
        self.trace, self.tok, self.last, self.cold_differs, self.wrong = trace, tok, [], cold_differs, set(wrong)
        self.history = w.history(trace)
        self.pad = tok.encode(" ")[0][0]
        self.prompts = []

    def turn(self, ids, n_new, *, reuse):
        ids = list(ids)
        if not reuse:
            self.last = []
        cached = min(common_prefix(np.array(self.last, dtype=np.int64), np.array(ids, dtype=np.int64)), len(ids) - 1)
        self.last = ids
        self.prompts.append((len(ids), reuse, cached))
        out = self.answer(ids, n_new, reuse)
        return Turn(
            out,
            first_s=1e-3 * (len(ids) - cached) / 100,
            decode_s=1e-2 * (n_new - 1),
            cached=cached,
            extra={"source": "fake"},
        )

    def answer(self, ids, n_new, reuse):
        if len(ids) > len(self.history) and ids[: len(self.history)] == self.history:
            i = next(i for i in range(len(self.trace.probes)) if ids == w.probe_ids(self.trace, i, hist=self.history))
            text = "no idea" if i in self.wrong else correct_text(self.trace.probes[i])
            out = self.tok.encode(text)[0][:n_new]
            out = out + [self.pad] * (n_new - len(out))
        else:
            out = [(sum(ids[-5:]) + j) % 997 + 10 for j in range(n_new)]
        if not reuse and self.cold_differs:
            out[-1] += 1
        return out


def test_run_trace_with_a_fake_engine():
    tok, trace = fake_trace()
    engine = FakeEngine(trace, tok)
    [cell] = w.run_trace(engine, trace, tok, key="agent|fake", cold=True)
    assert cell.status == "ok" and cell.key == "agent|fake"
    assert cell.workload["n_prompt"] == 32_000 and cell.workload["ids_sha"] == trace.trace_sha256
    assert cell.workload["corpus_sha256"] == hashlib.sha256(synthetic_text().encode()).hexdigest()
    assert cell.metric("probes_ok") == 19 and cell.metric("json_format_ok") == 4 and cell.metric("json_value_ok") == 4
    assert cell.metric("resumed_equal") == 1 and cell.extra["resumed"] == {"equal": True, "first_diff": None}
    assert cell.metric("ttft_cold_s") > cell.metric("ttft_median_s")
    turns = cell.extra["turns"]
    assert [r["turn"] for r in turns] == list(range(12)) and turns[0]["cached"] == 0
    assert all(r["cached"] > 0 for r in turns[1:])
    assert [r["prompt"] for r in turns] == [len(w.render(trace, k)) for k in range(12)]
    assert cell.metric("cached_fraction") > 0.5
    assert cell.metric("fork_rows_median") == int(np.median([r["prompt"] - r["cached"] for r in cell.extra["forks"]]))
    assert len(cell.extra["forks"]) == 19 and len(cell.extra["probe_ids"]) == 19
    assert abs(cell.metric("decode_tok_s_median") - 100.0) < 1e-6
    assert [reuse for _, reuse, _ in engine.prompts[:2]] == [False, True] and engine.prompts[-1][1] is False


def test_run_trace_reports_a_broken_cache_and_a_lost_fact():
    tok, trace = fake_trace()
    [cell] = w.run_trace(FakeEngine(trace, tok, cold_differs=True, wrong=[2, 13]), trace, tok, key="k", cold=True)
    assert cell.metric("resumed_equal") == 0 and cell.extra["resumed"]["first_diff"] == 23
    assert cell.metric("probes_ok") == 17 and cell.metric("json_format_ok") == 4
    assert cell.extra["by_kind"]["needle"]["ok"] == 9 and cell.extra["by_kind"]["vt"]["ok"] == 1


def test_run_trace_in_two_units():
    tok, trace = fake_trace()
    [loop] = w.run_trace(FakeEngine(trace, tok), trace, tok, key="k")
    assert "resumed_equal" not in loop.metrics
    [cold] = w.run_cold(FakeEngine(trace, tok), trace, key="k")
    assert w.resumed_check(loop.extra["probe_ids"][0], cold.extra["probe_ids"][0])["equal"]
    assert cold.metric("ttft_cold_s") > 0
    assert w.resumed_check([1, 2, 3], [1, 2, 9]) == {"equal": False, "first_diff": 2}
    assert w.resumed_check([1, 2], [1, 2, 3]) == {"equal": False, "first_diff": 2}


def test_linear_mode_has_no_probes():
    tok, trace = fake_trace()
    [cell] = w.run_trace(FakeEngine(trace, tok), trace, tok, key="k", mode="linear")
    assert "probes_ok" not in cell.metrics and cell.extra["forks"] == [] and cell.extra["mode"] == "linear"


def test_a_reused_turn_zero_is_an_error():
    tok, trace = fake_trace()

    class Sticky(FakeEngine):
        def turn(self, ids, n_new, *, reuse):
            t = super().turn(ids, n_new, reuse=reuse)
            t.cached = 5
            return t

    try:
        w.run_trace(Sticky(trace, tok), trace, tok, key="k")
    except RuntimeError as e:
        assert "turn 0" in str(e)
    else:
        raise AssertionError("no error")


# ---- the real tokenizers


def test_the_27b_tokenizer_does_not_truncate_the_corpus():
    tok = swift27()
    assert tok.sha256 == SWIFT27_TOKENIZER_SHA
    corpus = w.load_corpus(tok)
    assert len(corpus.ids) == SWIFT27_CORPUS_TOKENS > w.MIN_CORPUS_TOKENS
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise Skip("transformers is not installed") from e
    auto = AutoTokenizer.from_pretrained(os.environ["WARPC_TEST_SWIFT27"])
    assert auto(corpus.text, add_special_tokens=False)["input_ids"] == corpus.ids


def test_the_27b_chat_ids():
    chat = w.Chat.of(swift27())
    assert (chat.im_start, chat.im_end) == (248045, 248046)
    assert (chat.system, chat.user, chat.assistant, chat.nl) == ((8678, 198), (846, 198), (74455, 198), (198,))
    assert chat.think == (248068, 198) and chat.think_empty == (248068, 271, 248069, 271)


def test_the_27b_slices_are_pinned():
    tok = swift27()
    corpus_or_skip()
    for (length, k), sha in SWIFT27_SLICE_SHA.items():
        ids = w.corpus_ids(tok, length, k)
        assert len(ids) == length and w.ids_sha(ids) == sha, (length, k)
    assert w.corpus_ids(tok, 16384, 9 - 1) and w.corpus_ids(tok, 32768, 3) and w.corpus_ids(tok, 65536, 1)


def test_the_27b_trace_at_100k():
    tok = swift27()
    corpus_or_skip()
    trace = w.build_trace(tok, scale=100_000)
    assert trace.tokenizer_sha256 == SWIFT27_TOKENIZER_SHA and trace.corpus_sha256 == w.CORPUS_SHA256
    assert trace.trace_sha256 == SWIFT27_TRACE_SHA[100_000], trace.trace_sha256
    assert trace.ctx_max == 100_096 and trace.corpus_tokens == SWIFT27_CORPUS_TOKENS
    assert max(len(w.probe_ids(trace, i)) for i in range(19)) == 100_000
    rows = w.fork_rows(trace)
    assert min(r["prefill_rows"] for r in rows) >= 34 and max(r["prefill_rows"] for r in rows) <= 48, rows
    assert [r["target"] for r in w.tail_regimes(trace)] == [128, 380, 700]


def test_the_27b_trace_is_deterministic_across_tokenizer_loads():
    corpus_or_skip()
    first = w.build_trace(swift27(), scale=32_000)
    w.load_corpus.cache_clear()
    second = w.build_trace(swift27(), scale=32_000)
    assert first == second and first.trace_sha256 == second.trace_sha256 == SWIFT27_TRACE_SHA[32_000]


def test_the_qwen3_proxy_trace():
    tok = qwen3()
    corpus_or_skip()
    assert tok.sha256 == QWEN3_TOKENIZER_SHA
    assert len(w.load_corpus(tok).ids) == QWEN3_CORPUS_TOKENS
    trace = w.build_trace(tok, scale=32_000)
    assert trace.chat.im_start == 151644 and trace.chat.think == (151667, 198)
    assert trace.trace_sha256 == QWEN3_TRACE_SHA[32_000], trace.trace_sha256
    assert trace.ctx_max == 32_256
    rows = w.fork_rows(trace)
    assert max(r["prefill_rows"] for r in rows) <= 48


run(globals())
