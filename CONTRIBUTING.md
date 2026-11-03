# Contributing

qwen-warp is a single-model, single-GPU project: Swift-1.5-Qwen3.8-27B INT4 on one RTX 3090. Issues and pull requests are welcome, within that scope.

- **Before a PR:** `scripts/check_repo.sh` (no GPU needed). If you touch `engine/` or `server/`, also run `make test` on an RTX 3090 with the checkpoint and paste the three test summaries into the PR.
- **Numbers:** a performance or quality claim needs the command, the hardware and power limit, and whether it is one run or several. See [docs/PERFORMANCE.md](docs/PERFORMANCE.md) for the format.
- **Out of scope for now:** other GPUs, other models, batch size above 1, vision input.
- **Credits:** if you add code derived from another project, say so in [NOTICE](NOTICE).
