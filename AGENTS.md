# AGENTS.md

Instructions for coding agents and contributors. Read [README.md](README.md) first for what this repo is and its status.

## Where to work

- `warpc/` is the active code: a compiler from model checkpoints to fused CUDA kernels. New work goes here. Its
  [README](warpc/README.md) has the architecture and usage.
- `engine/`, `server/`, top-level `tests/`, `Makefile`, `Dockerfile`, `docker/`, the compose files and every file in
  `scripts/` except `check_repo.sh` are the legacy 27B engine and server, frozen at release v0.12.0. Do not add features
  to them; fix only what blocks the build or the repo checks. Their docs are at the v0.12.0 tag, not in this tree.

## Commands

All from the repository root, through uv. Never call the venv's python directly and never `uv pip install`; change
dependencies in `warpc/pyproject.toml` and run `uv lock`.

```bash
uv sync --all-packages                                   # install
uv run --package warpc python warpc/tests/test_X.py      # one test file (plain script; prints ok / skip / FAIL)
uv run --package warpc ruff check warpc && uv run --package warpc ruff format --check warpc
scripts/check_repo.sh                                    # what CI runs: compile, lint, uv.lock, doc links, no machine paths or keys
```

Run `scripts/check_repo.sh` before every pull request. It needs no GPU. The index must contain your files (`git add` first),
because its path scan reads tracked files.

## Tests

- A test is a plain script: module-level `test_*` functions, ending in `run(globals())` from `warpc/tests/harness.py`.
  No pytest.
- A test whose prerequisite is missing (GPU, nvcc, a checkpoint, an environment variable) raises `Skip("reason")`; it never
  passes silently and never fails. Checkpoints and tools are reached only through environment variables documented in
  `warpc/README.md`.
- GPU tests need an NVIDIA GPU with nvcc; the 27B needs the whole 24 GB card, so run one GPU job at a time.
- The oracle decides, not the implementation. Correctness is bit-identity to a reference where the arithmetic allows it
  (the hand-written engine, ggml's C code, the CPU reference of a kernel), otherwise a stated tolerance against a
  trusted implementation. If a gate fails, find the cause. Do not loosen the gate to make it pass; if a gate is wrong,
  say why in the same change and show the evidence.
- Test behaviour through the public interface. Do not pin incidental text, buffer ids, or internal layouts.

## Code

- Python 3.12, ruff with line length 120 (config in `warpc/pyproject.toml`). If another formatter reflows a file and ruff
  keeps its exploded lists, run `uv run --package warpc ruff format --config 'format.skip-magic-trailing-comma = true' FILE`.
- Comments state an invariant, a reason something is not done another way, a non-obvious decision, or a measured number.
  No banners, no narration of what the next line does, no history ("was", "previously"), no references to files that do
  not exist.
- Names are full words; short names only for local math (`T`, `K`, `N`, `hd`).
- Deep modules: a small interface over real behaviour. Delete what nothing uses; do not keep shims, aliases or
  re-exports of old paths. No environment variable or flag that exists only to help development; the only inputs are
  the ones `warpc/README.md` lists.
- Never commit machine-specific paths (home directories, mount points, shared-memory scratch), user names, or keys. Take
  assets from environment variables or arguments. `scripts/check_repo.sh` enforces this.

## Documentation

Docs are a maintenance cost, so there are few and they describe only what stays true: what something is, how to run it,
what it supports, its limits. Do not add a markdown file for a feature, a design, or a result.

- Numbers go stale. A measurement belongs in the pull request that produced it and in a dated `CHANGELOG.md` entry:
  hardware, power limit, software versions, the command, how many runs, and what was not measured. The README keeps one
  small snapshot table of headline results, dated, with the tools that regenerate it. Change it only when the numbers
  change materially, in the same pull request.
- Do not list file names, flags or API surfaces in prose when `--help` or the code says it. Link to code only at the
  directory level.
- User-visible changes get a bullet under `Unreleased` in `CHANGELOG.md`.

## Commits

Semantic commits (Conventional Commits): `type(scope): subject`.

- Types: `feat`, `fix`, `perf`, `refactor`, `docs`, `test`, `build`, `ci`, `chore`. A `perf` commit puts its
  measurement in the body; a `fix` says what was wrong.
- Scope: the area changed, such as `cuda`, `compiler`, `sources`, `frontend`, `kernels`, `tools`; leave it out for
  repository-wide changes.
- Subject: imperative, lower case, no trailing period, at most 72 characters.
- Body, when the subject is not enough: why, not what the diff shows; measurements with their conditions; what was not
  verified. Wrap at 100 columns.
- A breaking change adds `!` after the type or scope and a `BREAKING CHANGE:` footer.
- One logical change per commit: do not mix a refactor with a behaviour change.
- A pull request title uses the same form. Merge commits keep GitHub's default message.

## Pull requests

- One topic per pull request; stack them when one builds on another, and retarget to `main` as the base merges. History uses
  merge commits.
- State what was verified and how (commands), what was not, and any gate that does not fully hold. Report measurements as
  measured: if another engine is ahead in a cell, the table says so.
- Baselines: vLLM at the version pinned in `uv.lock`, llama.cpp at the build named in the CHANGELOG entry. Compare like with
  like (same prompt, same tokens, decode only, same context, medians of several runs).
- New code derived from another project is credited in [NOTICE](NOTICE).
