# Contributing

qwen-warp is a single-model, single-GPU project: Swift-1.5-Qwen3.8-27B INT4 on one RTX 3090. Issues and pull requests are welcome, within that scope. Back to the [README](README.md).

| You want to... | Do this |
|---|---|
| Report a problem | open an issue with the bug template: `nvidia-smi`, `nvcc --version` (native), your command, the last 30 lines of the log |
| Ask whether something fits | open an issue first; [Ideas to explore](docs/IDEAS.md) lists open questions you can take |
| Send a change | run `scripts/check_repo.sh` (no GPU needed); if you touch `engine/` or `server/`, also run `make test` on an RTX 3090 with the checkpoint (it is long: the list is in [Getting started](docs/GETTING_STARTED.md#4-verify-it-works)) and paste the test summaries into the PR |
| Add a performance or quality claim | give the command, the hardware and power limit, and whether it is one run or several; follow the format of [docs/PERFORMANCE.md](docs/PERFORMANCE.md) and state what was not measured |
| Add code derived from another project | say so in [NOTICE](NOTICE) |

| In scope | Out of scope for now |
|---|---|
| Speed, memory and quality on one RTX 3090 with this model | other GPUs, other models, more than 8 concurrent sequences, vision input |
| Docs, tests, honest measurements, bug fixes | a general-purpose serving stack |

Where to read first: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) has a file map and a "where to start reading" table. What changed between versions: [CHANGELOG](CHANGELOG.md).
