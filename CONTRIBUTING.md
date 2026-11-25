# Contributing

Issues and pull requests are welcome. The code to work on is [`warpc/`](warpc/README.md); the legacy engine and server are
frozen at [v0.12.0](https://github.com/tuan3w/qwen-warp/releases/tag/v0.12.0) (see the [README](README.md)).

- **Report a problem:** open an issue with the template: GPU and driver (`nvidia-smi`), `nvcc --version`, the command, the
  model file, and the last 30 lines of the output. Only an RTX 3090 (sm_86) is tested; other cards are welcome but may
  not be fixable.
- **Ask whether something fits:** open an issue first. A new model family needs a checkpoint format warpc can read and an
  oracle to check it against (transformers, or the model's reference implementation).
- **Send a change:** the working rules, commands and checks are in [AGENTS.md](AGENTS.md). Run `scripts/check_repo.sh`
  (no GPU needed) and add a `CHANGELOG.md` bullet for user-visible changes.
- **Make a performance or quality claim:** give the command, the hardware and power limit, the software versions, how many
  runs, and what was not measured.
- **Add code derived from another project:** say so in [NOTICE](NOTICE).
