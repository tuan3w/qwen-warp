"""warpc against transformers on real weights (design gemma §6.1): teacher-forced logits and free-running greedy ids.

  uv run --package warpc python warpc/tools/compare_hf.py MODEL TOKENS WORK {it,rt,cmp,freerun} [--baselines DIR] [...]

Stages, each its own process, logits through files in WORK (GPU stages run under the lease):

  it        the reference interpreter on the first 768 ids of TOKENS["tf"], teacher forced: 8-row windows up to position
            704, then single-token steps (both window shapes, the sliding window crossed at 512 and the ring wrapped)
            -> WORK/it_tf.pt (fp32 [768, V])
  rt        the compiled model: one single-token head step per position (every position's logits), and a second pass
            with 8-row prompt windows to 704 and single-token head steps after (the last 64 positions' logits)
            -> WORK/rt_tf.pt, WORK/rt_win_tf.pt
  cmp       (CPU) WORK/{it,rt}_tf.pt against BASELINES/hf32_tf.pt (transformers in fp32) and hfbf_tf.pt (bf16). Gate:
            rel L2 vs fp32 <= that of bf16 vs fp32, and at EVERY position the argmax has an fp32 logit within 0.25 of
            the fp32 maximum; the windowed pass is gated alike on its 64 positions, against the floor of those positions
  freerun   greedy ids after the 2,000-id prefill of each TOKENS["freerun"] prompt (--engine rt | it), compared with
            BASELINES/hf32_freerun.json: every divergence must be an fp32 near-tie (top-2 margin <= 0.1)

BASELINES holds hf32_tf.pt, hfbf_tf.pt (float32 [768, V]) and hf32_freerun.json (per prompt {ids, top16: [[values,
ids]]}) as written by a transformers run on the same fixtures; TOKENS is a JSON with "tf" (768 ids) and "freerun" (id
lists).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

TF_WINDOWS_TO = 704
WINDOW = 8
GAP = 0.25
NEAR_TIE = 0.1


def teacher_forced_reference(model: Path, ids: list[int]) -> torch.Tensor:
    from warpc import frontend, sources
    from warpc.reference import Interpreter

    source = sources.open(model)
    it = Interpreter(frontend.build(source, ctx_max=len(ids) + WINDOW, rows_max=WINDOW), source, device="cuda")
    rows, at = [], 0
    while at < len(ids):
        w = WINDOW if at < TF_WINDOWS_TO else 1
        feed = {
            "tokens": torch.tensor(ids[at : at + w], dtype=torch.int32),
            "pos": torch.arange(at, at + w, dtype=torch.int32),
        }
        rows.append(it.run(feed)["logits"].cpu())
        at += w
    return torch.cat(rows)


def teacher_forced_runtime(model: Path, ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    import warpc

    m = warpc.compile(model, ctx_max=len(ids) + WINDOW, rows_max=WINDOW)
    single = []
    for t in ids:
        m.step(t)
        single.append(m.logits.cpu().clone())
    m.reset()
    # 8-row windows to 704 (no logits: the head graph has one row), then single-token head steps
    m.append(ids[: TF_WINDOWS_TO + 1])  # windows up to 704, the last id as a head step: logits of position 704
    windowed = [m.logits.cpu().clone()]
    for t in ids[TF_WINDOWS_TO + 1 :]:
        m.step(t)
        windowed.append(m.logits.cpu().clone())
    return torch.stack(single), torch.stack(windowed)


def table_row(name: str, x: torch.Tensor, y: torch.Tensor) -> tuple[float, float, float, float]:
    rel = ((x - y).norm() / y.norm()).item()
    ax = x.argmax(-1)
    top1 = (ax == y.argmax(-1)).float().mean().item()
    gap = y.max(-1).values - y.gather(-1, ax[:, None])[:, 0]
    within = (gap <= GAP).float().mean().item()
    print(f"| {name} | {rel:.3e} | {top1:.3f} | {within:.3f} | {(gap <= 1).float().mean():.3f} |")
    return rel, top1, within, gap.max().item()


def compare_teacher_forced(work: Path, baselines: Path) -> bool:
    hf32, hfbf = torch.load(baselines / "hf32_tf.pt"), torch.load(baselines / "hfbf_tf.pt")
    print("| teacher-forced (positions: 768) vs HF fp32 | rel L2 | top-1 | gap<=0.25 | gap<=1 |\n|---|---|---|---|---|")
    floor, *_ = table_row("HF bf16 (the floor)", hfbf, hf32)
    ok = True
    for name in ("it", "rt"):
        path = work / f"{name}_tf.pt"
        if not path.exists():
            continue
        x = torch.load(path)
        rel, _, within, worst = table_row({"it": "reference interpreter", "rt": "compiled"}[name], x, hf32)
        table_row(f"{name} vs HF bf16", x, hfbf)
        passed = rel <= floor and within == 1.0
        verdict = "PASS" if passed else "FAIL"
        print(f"  {name}: rel L2 {rel:.3e} vs floor {floor:.3e}, worst fp32 gap of the argmax {worst:.3f}: {verdict}")
        ok &= passed
        if (work / "rt_win_tf.pt").exists() and name == "rt":
            win = torch.load(work / "rt_win_tf.pt")
            n = len(win)
            rel_w, _, within_w, worst_w = table_row(
                f"compiled, 8-row windows to 704, last {n} positions", win, hf32[-n:]
            )
            floor_w = ((hfbf[-n:] - hf32[-n:]).norm() / hf32[-n:].norm()).item()
            passed_w = rel_w <= floor_w and within_w == 1.0
            print(
                f"  windows: rel L2 {rel_w:.3e} vs floor of those positions {floor_w:.3e}, worst gap {worst_w:.3f}: "
                f"{'PASS' if passed_w else 'FAIL'}"
            )
            ok &= passed_w
    if (work / "it_tf.pt").exists() and (work / "rt_tf.pt").exists():
        table_row("compiled vs reference interpreter", torch.load(work / "rt_tf.pt"), torch.load(work / "it_tf.pt"))
    return ok


def freerun(args) -> bool:
    prompts = json.loads(args.tokens.read_text())["freerun"]
    ref = json.loads((args.baselines / "hf32_freerun.json").read_text())
    outs = []
    if args.engine == "rt":
        import warpc

        m = warpc.compile(args.model, ctx_max=max(map(len, prompts)) + args.new + WINDOW, rows_max=WINDOW)
        for p in prompts:
            outs.append(m.generate(p, args.new))
    else:
        from warpc import frontend, sources
        from warpc.reference import Interpreter

        source = sources.open(args.model)
        graph = frontend.build(source, ctx_max=max(map(len, prompts)) + args.new + WINDOW, rows_max=WINDOW)
        it = Interpreter(graph, source, device="cuda")
        for p in prompts:
            it.reset()
            ids, at = [], 0
            while at < len(p):
                w = min(WINDOW, len(p) - at)
                feed = {
                    "tokens": torch.tensor(p[at : at + w], dtype=torch.int32),
                    "pos": torch.arange(at, at + w, dtype=torch.int32),
                }
                last = int(it.run(feed)["ids"][-1])
                at += w
            ids = [last]
            while len(ids) < args.new:
                feed = {
                    "tokens": torch.tensor(ids[-1:], dtype=torch.int32),
                    "pos": torch.tensor([at], dtype=torch.int32),
                }
                ids.append(int(it.run(feed)["ids"][-1]))
                at += 1
            outs.append(ids)
    (args.work / f"{args.engine}_freerun.json").write_text(json.dumps(outs))
    same, ok = 0, True
    print("| prompt | identical | first divergence: index, fp32 top-2 margin, fp32 logit lost |\n|---|---|---|")
    for i, (ids, r) in enumerate(zip(outs, ref, strict=True)):
        k = next((j for j, (a, b) in enumerate(zip(ids, r["ids"], strict=True)) if a != b), None)
        if k is None:
            same += 1
            print(f"| {i} | 64/64 | - |")
            continue
        values, top = r["top16"][k]
        margin = values[0] - values[1]
        lost = values[0] - values[top.index(ids[k])] if ids[k] in top else float("inf")
        tie = margin <= NEAR_TIE
        ok &= tie
        print(f"| {i} | {k}/{len(ids)} | {k}: {margin:.3f}, {lost:.3f} {'near-tie' if tie else 'NOT A NEAR-TIE'} |")
    print(f"free run ({args.engine}): {same}/{len(outs)} prompts identical; every divergence a near-tie: {ok}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model", type=Path)
    ap.add_argument("tokens", type=Path)
    ap.add_argument("work", type=Path)
    ap.add_argument("stage", choices=("it", "rt", "cmp", "freerun"))
    ap.add_argument("--baselines", type=Path, help="directory of the transformers dumps (cmp, freerun)")
    ap.add_argument("--engine", choices=("rt", "it"), default="rt", help="freerun: compiled model or interpreter")
    ap.add_argument("--new", type=int, default=64, help="freerun: tokens per prompt")
    args = ap.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    if args.stage in ("cmp", "freerun") and args.baselines is None:
        ap.error(f"{args.stage} needs --baselines")
    ids = json.loads(args.tokens.read_text())["tf"]
    if args.stage == "it":
        torch.save(teacher_forced_reference(args.model, ids), args.work / "it_tf.pt")
    elif args.stage == "rt":
        single, windowed = teacher_forced_runtime(args.model, ids)
        torch.save(single, args.work / "rt_tf.pt")
        torch.save(windowed, args.work / "rt_win_tf.pt")
    elif args.stage == "cmp":
        sys.exit(0 if compare_teacher_forced(args.work, args.baselines) else 1)
    else:
        sys.exit(0 if freerun(args) else 1)


if __name__ == "__main__":
    main()
