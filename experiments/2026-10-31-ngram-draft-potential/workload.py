"""Workload for ngram-draft-potential: 32 prompts (chat-template text, thinking off) built from files / logs / git state of this repo.

  ~/llm-lab/venv/bin/python experiments/2026-10-31-ngram-draft-potential/workload.py     (CPU only; writes runs/prompts.json + results/workload.json)

Classes: a = code edits (8), b = log / tool-output extraction (6), c = agent tool-call turns (6), d = free-form prose/QA (6), e = reasoning/math (6).
Everything comes from pinned repo content (vendor/ clones, experiments/, committed logs, `git show <hash>`), so the prompts are reproducible at this commit range.
"""
import hashlib
import json
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODEL = Path.home() / "models" / "swift-1.5-int4"
MAXTOK = 7000  # prompt budget: ctx_max 8192 - 768 new - slack


REV = "ba31ad5"  # projects/swift-mk/ is edited by other agents while this experiment runs: everything from there is read at this commit


def rd(p):
    return (ROOT / p).read_text(errors="ignore")


def rdrev(p):
    return subprocess.run(["git", "show", f"{REV}:{p}"], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def sh(cmd):
    return subprocess.run(cmd, shell=True, cwd=ROOT, capture_output=True, text=True, check=True).stdout


def fence(path, text, lang="python"):
    return f"`{path}`:\n```{lang}\n{text.rstrip()}\n```"


def mutate(text, old, new, count=1):
    assert text.count(old) >= 1, old
    return text.replace(old, new, count)


def cut_lines(text, tok, budget):
    """longest line-prefix of text within `budget` tokens"""
    lines = text.splitlines(keepends=True)
    out, n = [], 0
    for ln in lines:
        k = len(tok(ln, add_special_tokens=False)["input_ids"])
        if n + k > budget:
            break
        out.append(ln)
        n += k
    return "".join(out)


def build(tok):
    T = []  # (id, cls, kind, messages, tools)

    def user(i, cls, kind, text, system=None):
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": text}]
        T.append((i, cls, kind, msgs, None))

    # ---------------------------------------------------------------- (a) code edits
    f = "vendor/aider/aider/coders/editblock_coder.py"
    user("a1_rename_full", "a", "rename, full file", fence(f, rd(f)) + "\n\nRename the function `find_original_update_blocks` to `parse_edit_blocks` everywhere in this file (definition and every call site, docstrings and comments included). Output the complete modified file.")
    f = "vendor/modded-nanogpt/track_1_short/sampled_softmax.py"
    user("a2_types_full", "a", "add type hints, full file", fence(f, rd(f)) + "\n\nAdd complete type annotations (parameters and return types) to every function and method in this file that lacks them, keeping behaviour identical. Output the complete modified file.")
    f = "experiments/2026-10-31-lm-head-quantisation-quality/lmq.py"
    user("a3_logging_full", "a", "add logging, full file", fence(f, rd(f)) + "\n\nAdd the standard `logging` module to this file: a module-level `log = logging.getLogger(__name__)`, and a `log.info(...)` line at the start of `build_docs`, `quantise`, `weight_err` and `metrics` reporting their main arguments. Do not change anything else. Output the complete modified file.")
    f = "vendor/aider/scripts/issues.py"
    t = mutate(rd(f), "if created_at < oldest_date:", "if created_at > oldest_date:")
    user("a4_bug_diff", "a", "injected bug, unified diff", fence(f, t) + "\n\nBug report: `find_oldest_issue` returns `None` even when matching issues exist for the subject. Find the bug and reply with a unified diff (with 3 lines of context) that fixes it, plus a one-sentence explanation.")
    f = "vendor/aider/benchmark/problem_stats.py"
    t = mutate(rd(f), "lang = fname.parts[-5]  # Get language from path", "lang = fname.parts[-4]  # Get language from path")
    user("a5_bug_func", "a", "injected bug, corrected function", fence(f, t) + "\n\nBug report: `load_results` labels results with the wrong language (it reports the exercise name instead of the language directory), and the parse-error entries look inconsistent. Fix every instance of the problem and output the complete corrected `load_results` function, then list the lines you changed.")
    f = "vendor/aider/aider/coders/search_replace.py"
    user("a6_rename_diff", "a", "rename, unified diff", fence(f, rd(f)) + "\n\nRename the class `RelativeIndenter` to `IndentEncoder` (definition, every use, docstrings and comments) and rename its method `make_relative` to `encode` and `make_absolute` to `decode`. Reply with a unified diff (3 lines of context) against the file above, nothing else.")
    f = "vendor/modded-nanogpt/track_1_short/training.py"
    user("a7_validate_funcs", "a", "validation + docstrings, two methods", fence(f, rd(f)) + "\n\nRewrite the methods `TrainingManager.advance_schedule` and `TrainingManager.get_forward_args` so that they validate their arguments (raise `ValueError` with a clear message on invalid step / missing state), have complete type annotations and Google-style docstrings, and otherwise behave exactly as now. Output only the two rewritten methods in full.")
    f = "vendor/modded-nanogpt/track_1_short/ngram_table.py"
    user("a8_refactor_cfg", "a", "refactor into dataclass + call sites", fence(f, rd(f)) + "\n\nRefactor `is_update_step` and `adam_beta2_and_wd_mul` so that the constants they read come from a frozen dataclass `UpdateConfig` defined above them (one field per constant, same values). Output the new dataclass and the two rewritten functions in full, then a unified diff of every call site of the two functions in this file.")

    # ---------------------------------------------------------------- (b) log / tool-output extraction
    gl = sh("git log --format='commit %H%nAuthor: %an%nDate: %ad%n%n    %s%n' --date=iso --stat -n 70 ba31ad5")
    gl = cut_lines(gl, tok, 4200)
    user("b1_gitlog_table", "b", "git log --stat -> table", "Here is the output of `git log --stat`:\n\n```\n" + gl.rstrip() + "\n```\n\nList every commit in it that touched a file under `engine/src/` as a markdown table with columns: short hash (8 chars), date, subject, and the src files it touched (full paths).")
    bl = cut_lines(rd("experiments/2026-10-30-dataflow-fused-phases/runs/build.log"), tok, 4200)
    user("b2_ptxas_table", "b", "nvcc/ptxas log -> table", "Output of the nvcc build (ptxas -v):\n\n```\n" + bl.rstrip() + "\n```\n\nExtract, for every `Compiling entry function` block above, one row: the demangled-ish kernel name (strip the C++ mangling to the template arguments as far as you can read them), registers used, spill stores, spill loads, smem bytes. Present it as a markdown table and then state which kernel uses the most registers.")
    cl = cut_lines(rd("experiments/2026-10-30-w4-multirow-verify-and-power/runs/check.log"), tok, 4200)
    user("b3_check_filter", "b", "numerical check log -> filtered list", "This is the log of a kernel correctness check:\n\n```\n" + cl.rstrip() + "\n```\n\nCopy out, verbatim, every line whose `rel=` value is larger than 1e-3, grouped by T, and then give for each (shape, T) pair the worst `rel` and the config that produced it.")
    al = cut_lines(rd("projects/agent-harness-bench/results/arm_shrink2.log"), tok, 4200)
    user("b4_harness_log", "b", "traceback / run log -> errors", "Log of a benchmark run:\n\n```\n" + al.rstrip() + "\n```\n\nSummarise the failures: list each distinct exception type with the file:line where it was raised, the command line / arguments that triggered it if visible, and how many times it occurs. Quote the exact exception messages.")
    mp = cut_lines(rdrev("engine/MTP_PARITY.md"), tok, 4200)
    user("b5_parity_table", "b", "test output -> table", "Test output:\n\n```\n" + mp.rstrip() + "\n```\n\nBuild a markdown table of the 'speculative greedy == plain greedy' results: one row per (prompt, k) with the verdict (EQUAL or NEAR-TIE divergence), the mean tokens/cycle and the natural accepted-drafts histogram, copied exactly. Then list the near-tie divergences with their logit margins.")
    pp = rdrev("engine/PREFILL_PARITY.md")
    pp = cut_lines(pp[pp.index("### code (32 prompt tokens, kv_fp8=0)"):], tok, 4300)
    user("b6_prefill_parity", "b", "parity log -> table", "Output of the prefill parity run:\n\n```\n" + pp.rstrip() + "\n```\n\nFor each section (prompt) and for both `fast` and `fast_seg` rows, extract into a markdown table: prompt name, prompt token count, first-token verdict, max-over-layers hidden rel-L2, DeltaNet state max rel-L2, and the greedy / spec verdicts. Copy numbers exactly.")

    # ---------------------------------------------------------------- (c) agent tool-call turns
    tools = [
        {"type": "function", "function": {"name": "bash", "description": "Run a shell command in the repo root and return stdout/stderr.", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
        {"type": "function", "function": {"name": "read_file", "description": "Read a text file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
        {"type": "function", "function": {"name": "edit_file", "description": "Replace one exact occurrence of old_string with new_string in a file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"}}, "required": ["path", "old_string", "new_string"]}}},
        {"type": "function", "function": {"name": "write_file", "description": "Create or overwrite a file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    ]
    sysmsg = ("You are a coding agent working in the repository at <repo>. Use the tools to complete the task. When the user has already supplied or you have already retrieved the information a step needs, do NOT re-run read-only commands (git status, git log, grep, sed, cat, wc) to confirm it: act on it. "
              "Before the tool call(s) of each turn, write a short plan (3-6 sentences) saying what you are about to change and which exact paths / identifiers / lines you rely on; then issue all the tool calls the task needs in that same message.")

    def call(name, **kw):
        return {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": name, "arguments": kw}}]}

    def tool(text):
        return {"role": "tool", "content": text}

    def agent(i, kind, msgs):
        T.append((i, "c", kind, [{"role": "system", "content": sysmsg}] + msgs, tools))

    # v3 of the agent turns (v2 prompts + a system prompt that asks for a short plan and forbids re-confirming reads): the v1 prompts let the model take an exploratory first step (29-79 token tool calls, see README Dead ends); these force the long action turn.
    NOEXP = " The information above is complete: do not explore or inspect anything further, your next message must be the tool call(s) that perform the task."
    g = sh(f"git grep -n 'smk.check(' {REV} -- engine/prefill.py | sed 's/^{REV}://'")
    agent("c1_rename_edits", "grep hits -> one edit_file call per hit", [
        {"role": "user", "content": "In engine/prefill.py rename every call `smk.check(` to `smk.check_cuda(` (the helper itself is renamed elsewhere). Use exactly one edit_file call per matching line, in file order, with old_string = the complete original line (indentation included) and new_string = the complete updated line." + NOEXP},
        call("bash", command="grep -n 'smk.check(' engine/prefill.py"), tool(g)])
    ls = sh("ls -R experiments/2026-10-30-w4-multirow-verify-and-power experiments/2026-10-31-lm-head-quantisation-quality experiments/2026-10-31-mtp-draft-head-ablation experiments/2026-10-31-spec-sampling-exactness | head -150")
    ls = cut_lines(ls, tok, 2200)
    agent("c2_write_script", "ls -R -> write_file shell script", [
        {"role": "user", "content": "Create `scripts/collect_results.sh` with one write_file call. The script (bash, `set -euo pipefail`) must, for each of the four experiment directories in the listing below, `echo` a header with the directory name and then `head -n 20` every `results/*.md` file that appears in the listing, writing out each file path explicitly (no globs), and print a final line with the number of files shown. Use the exact directory and file names from the listing." + NOEXP},
        call("bash", command="ls -R experiments/2026-10-30-w4-multirow-verify-and-power experiments/2026-10-31-lm-head-quantisation-quality experiments/2026-10-31-mtp-draft-head-ablation experiments/2026-10-31-spec-sampling-exactness | head -150"), tool(ls)])
    src = rdrev("server/smk_core_client.py")
    src = cut_lines(src, tok, 3300)
    agent("c3_edit_anchor", "read_file -> edit_file with quoted anchors", [
        {"role": "user", "content": "In server/smk_core_client.py add an `uptime_s` property to the engine-core client class that returns seconds since construction (store `time.time()` in `__init__`), and add a `log.debug(...)` line at the start of every public method. Use one edit_file call per change with exact old_string anchors copied from the file (several lines of context each so they are unique)." + NOEXP},
        call("read_file", path="server/smk_core_client.py"), tool(src)])
    gd = sh(f"git show 8fcbc1e --stat --format='%H%n%s%n' ; git show 8fcbc1e -- engine/src/gemv.cu | head -400")
    gd = cut_lines(gd, tok, 3600)
    agent("c4_git_commit", "git diff -> git add + commit command", [
        {"role": "user", "content": "I changed the GEMV kernels as shown by the diff below. Stage exactly the files in the diff (by explicit path) and commit them with ONE bash tool call: a multi-line commit message in the repo's style (`[agent:claude-sonnet-5-5] swift-mk: ...` subject, then a body with one bullet per hunk of the diff quoting the changed identifiers, constants and line contents exactly)." + NOEXP},
        call("bash", command="git show 8fcbc1e --stat --format='%H%n%s%n'; git show 8fcbc1e -- engine/src/gemv.cu | head -400"), tool(gd)])
    td = sh(f"git grep -n 'TODO\\|FIXME\\|XXX' {REV} -- projects/swift-mk vendor/aider/aider 2>/dev/null | sed 's/^{REV}://' | grep -v __pycache__ | head -80")
    if not td.strip():
        td = sh("grep -rn 'TODO\\|FIXME\\|XXX' vendor/aider/aider --include=*.py --include=*.md | head -80")
    td = cut_lines(td, tok, 3000)
    agent("c5_write_todo", "grep TODO -> write_file report", [
        {"role": "user", "content": "Collect every TODO/FIXME/XXX hit from the grep output below into a file `notes/todo-report.md` with ONE write_file call: a title, then one heading per top-level directory, under each heading one bullet per hit in the form `- path:line - text` using the exact paths, line numbers and texts, then a final section counting hits per file." + NOEXP},
        call("bash", command="git grep -n 'TODO\\|FIXME\\|XXX' -- projects/swift-mk vendor/aider/aider | head -80"), tool(td)])
    fl = sh("find projects/swift-mk vendor/aider/aider -name '*.py' -size +6k | grep -v __pycache__ | sort | head -60")
    fl = cut_lines(fl, tok, 1800)
    big = cut_lines(rdrev("engine/tests/test_mtp.py"), tok, 2500)
    agent("c6_parallel_reads", "find + read -> parallel read_file / bash calls", [
        {"role": "user", "content": "I want to review the large Python files. From the listing below call read_file on every file under server and engine/tests, and run `wc -l` on every file under vendor/aider/aider/coders in a single bash call. Issue ALL calls in this one message (parallel tool calls, one per file, exact paths from the listing)." + NOEXP},
        call("bash", command="find projects/swift-mk vendor/aider/aider -name '*.py' -size +6k | grep -v __pycache__ | sort | head -60"), tool(fl)])

    # ---------------------------------------------------------------- (d) free-form prose / QA
    doc = cut_lines(rd("topics/llm-behavioral-properties.md"), tok, 3200)
    user("d1_doc_oped", "d", "long doc + opinion piece", "Background notes (do not quote them):\n\n" + doc + "\n\nUsing only the general themes of these notes as inspiration, write a 500-word opinion piece for a general audience arguing whether AI assistants should remember everything about their users. Use your own words and a lively style.")
    doc = cut_lines(rd("topics/posttraining.md"), tok, 3200)
    user("d2_doc_dialogue", "d", "long doc + dialogue", "Here are some research notes:\n\n" + doc + "\n\nWrite an imaginative 500-word dialogue between a sceptical senior engineer and an enthusiastic intern who has just read these notes, in which they argue about whether any of it matters in practice. Do not copy sentences from the notes.")
    user("d3_story", "d", "short story", "Write a short story (about 500 words) about a cartographer who is hired to map a city that rearranges itself every night.")
    user("d4_itinerary", "d", "travel advice", "I have three days in Lisbon in November, I like food, tiled buildings and long walks, and I get seasick. Plan the three days in detail, with reasons for each choice, and add practical tips on transport and weather.")
    user("d5_explain", "d", "explanation", "Explain to a curious ten-year-old how bread rises, why salt matters, and why sourdough tastes different from supermarket bread. Use concrete analogies and keep a friendly tone.")
    user("d6_advice", "d", "advice / QA", "My team of six has started missing deadlines since we went hybrid. Give me a thoughtful, practical analysis of the likely causes and a concrete 6-week plan to fix it, with the trade-offs of each measure.")

    # ---------------------------------------------------------------- (e) reasoning / math
    user("e1_odd_sum", "e", "proof", "Prove that the sum of the first n odd numbers equals n squared, then give two different proofs. Show every step.")
    user("e2_counting", "e", "combinatorics", "How many 8-digit numbers (no leading zero) have digits that are strictly increasing from left to right? Then how many have digits that are non-decreasing? Work it out step by step with the formulas you use, and verify the first answer by a second method.")
    user("e3_probability", "e", "probability", "Two players alternately roll a fair six-sided die; the first to roll a 6 wins. Player A rolls first. Derive the probability that A wins, the expected number of rolls in total, and the probability that the game lasts more than 10 rolls. Show the derivations step by step.")
    user("e4_recurrence", "e", "algorithm analysis", "Solve the recurrence T(n) = 3T(n/2) + n log n using the Master theorem and, separately, by expanding the recursion tree; then explain why Karatsuba multiplication satisfies T(n) = 3T(n/2) + O(n) and what its running time is. Be rigorous and show each step.")
    user("e5_physics", "e", "physics word problem", "A 2 kg block slides down a 30 degree incline of length 4 m with coefficient of kinetic friction 0.2, starting from rest, and then moves onto a horizontal surface with the same friction coefficient. Find its speed at the bottom, the distance it travels on the flat surface before stopping, and the total heat generated. Use g = 9.8 m/s^2 and show all steps.")
    user("e6_linear", "e", "linear algebra", "Let A be the 3x3 matrix [[2,1,0],[1,2,1],[0,1,2]]. Find its eigenvalues and eigenvectors, diagonalise it, compute A^10 using the diagonalisation, and check the result against the trace and determinant. Show every step.")

    out = []
    for i, cls, kind, msgs, tl in T:
        kw = dict(add_generation_prompt=True, tokenize=False, enable_thinking=False)
        if tl:
            kw["tools"] = tl
        text = tok.apply_chat_template(msgs, **kw)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        out.append(dict(id=i, cls=cls, kind=kind, text=text, n_prompt=len(ids), sha256=hashlib.sha256(text.encode()).hexdigest()))
    return out


def main():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(MODEL))
    out = build(tok)
    (HERE / "runs").mkdir(exist_ok=True)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "runs" / "prompts.json").write_text(json.dumps(out))
    meta = [{k: v for k, v in o.items() if k != "text"} for o in out]
    (HERE / "results" / "workload.json").write_text(json.dumps(meta, indent=1))
    for o in meta:
        print(f"{o['id']:20s} {o['cls']} {o['n_prompt']:5d}  {o['kind']}")
    assert all(o["n_prompt"] <= MAXTOK for o in meta), "prompt too long"
    print(len(meta), "tasks")


if __name__ == "__main__":
    main()
