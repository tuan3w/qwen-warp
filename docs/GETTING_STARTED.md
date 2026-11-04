# Getting started

From nothing to a working server in four steps: check the prerequisites, get the weights, pick Docker or native, verify. Back to the [README](../README.md).

Contents: [1. Prerequisites](#1-prerequisites) | [2. Get the weights](#2-get-the-weights) | [3A. Docker](#3a-docker) | [3B. Native](#3b-native-uv) | [4. Verify](#4-verify-it-works) | [Troubleshooting](#troubleshooting) | [Removing everything](#removing-everything)

## 1. Prerequisites

| Item | Requirement | Check | Needed for |
|---|---|---|---|
| GPU | RTX 3090 (24 GB). Nothing else was tested; the kernels are built for sm_86 | `nvidia-smi` | both |
| Driver | 580 or newer | `nvidia-smi` (top right) | both |
| OS | Linux | | both |
| Docker | Docker Engine with the [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/) | `docker info \| grep -i nvidia` | Docker |
| CUDA toolkit | 13.0 (only 13.0 was tested), to compile the engine | `nvcc --version` | native |
| uv | a recent version; it installs Python 3.12 and every locked dependency | `uv --version` | native, and the weights download |
| C compiler | a normal `gcc`/`build-essential`: Triton compiles small C stubs at run time | `gcc --version` | native |
| Disk | weights 19 GB, plus about 8 GB for the environment (native) or about 24 GB for the image (Docker) | `df -h` | both |
| Host RAM | about 5 GiB of **pinned** memory in the long-context profile (4.7 GiB of tables plus 151 MB per multi-turn snapshot, 4 by default) | `free -g` | both |
| Hugging Face account | to accept the checkpoint's licence | | both |

The server uses almost the whole card. If a desktop or another job holds VRAM, free it first: a desktop alone takes about 0.6 GiB, and the long-context profile is sized for a 24 GB card.

## 2. Get the weights

The checkpoint [`ukisai/Swift-1.5-Qwen3.8-27b-INT4`](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b-INT4) is gated and has its own licence (`swift-open-license-1.0`);
this repository does not include or relicense it.

1. Open the model page on Hugging Face while logged in and accept its licence.
2. Log in and download (about 19 GB). `uvx` runs the Hugging Face CLI in a throwaway environment, so this does not install the project's 8 GB environment first:

```bash
uvx --from huggingface_hub hf auth login
uvx --from huggingface_hub hf download ukisai/Swift-1.5-Qwen3.8-27b-INT4 --local-dir ~/models/swift-1.5-int4
```

The repository's `scripts/fetch_model.sh [directory]` does the same download after you are logged in, through the project environment (so it is a good choice on the native path, where you need that environment anyway).
If you download elsewhere, point the server at it with `export SWIFT_MODEL=/data/swift` (native) or `MODEL_DIR=/data/swift` (Docker).

The engine and the server look in `~/models/swift-1.5-int4` unless `$SWIFT_MODEL` (or `--model`) says otherwise.

## 3A. Docker

The image holds the code and the compiled engine, not the weights: the checkpoint is mounted read-only.

```bash
docker compose up --build                 # first build downloads about 8 GB of wheels; the image is about 24 GB
```

It serves on `http://localhost:8000` with the long-context profile. Change the defaults in a `.env` file next to `docker-compose.yml` (copy [.env.example](../.env.example)):

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_DIR` | `$HOME/models/swift-1.5-int4` | host directory with the checkpoint, mounted at `/models/swift-1.5-int4` |
| `PORT` | `8000` | host port |
| `PROFILE` | `long` | `long` (4-bit KV cache, 196,608 context) or `default` (bf16 KV cache, 28,672 context) |
| `EXTRA_ARGS` | empty | any [server flag](USAGE.md#server-flags), e.g. `--max-model-len 262144 --prompt-lookup 0` for the full 262,144-token context |

Good to know: a named volume keeps the Triton and vLLM compile caches, so the first start is slower than the next ones (about 80 s the first time in our test).
The container has a health check on `/health`, a restart policy of `unless-stopped`, and `memlock` unlimited because the engine pins host memory.

```bash
docker compose logs -f                    # watch it start; "Application startup complete" means ready
docker compose down                       # stop (the compile-cache volume stays; add -v to remove it)
```

## 3B. Native (uv)

```bash
make setup                                # uv sync from uv.lock, then builds engine/build/libswiftmk.so
uv run server/serve.py --profile long --port 8000
```

`make setup` took 27 minutes on a clean machine, almost all of it downloading packages (torch, vLLM and CUDA libraries are several GB). Re-runs are fast.
If `nvcc` is not on your `PATH`, run `make setup CUDA_HOME=/path/to/cuda-13.0`. Torch is locked to the `+cu130` builds from PyTorch's own index, and `uv.lock` pins everything else.

The server is ready 23 to 40 s after it starts (warm caches). Flags and modes: [Usage](USAGE.md#server-flags).

## 4. Verify it works

```bash
curl localhost:8000/health                          # HTTP 200 when ready
curl localhost:8000/v1/models                       # the served model name is "swift"
curl localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"swift","messages":[{"role":"user","content":"Say hello."}],"max_tokens":300}'
```

For a deeper check, `make test` (native path, about 6 minutes on an RTX 3090 with the weights; stop any running server first, because the tests start their own and the engine needs the VRAM):

| Test | What it checks |
|---|---|
| `tests/test_kv_dtypes.py` | perplexity with bf16, fp8 and 4-bit KV caches; fails if fp8 is worse than +1.5% or 4-bit worse than +3.0% (a regression guard: the real cost is about 0.03% and 0.26%) |
| `tests/test_host_tables.py` | host-pinned tables give bit-identical output and free 4.7 GiB of VRAM |
| `tests/test_server_smoke.py` | the server end to end, run with the default and the `--profile long` profile: greedy parity with the engine, streaming, stop strings, tool calls, multi-turn prefix reuse (23 checks each) |

## Troubleshooting

| Symptom | Likely cause | What to do |
|---|---|---|
| `nvcc: command not found` during `make setup` | CUDA toolkit missing or not on `PATH` | install CUDA 13.0, or `make setup CUDA_HOME=/path/to/cuda-13.0` |
| The weights download fails with 401 or 403 | the licence is not accepted, or you are not logged in | accept it on the model page, then `uv run hf auth login` |
| Server exits at start with an out-of-memory error | `--max-model-len` too large for the chosen KV cache, or something else holds VRAM | start with `--profile long`; see [memory and context](USAGE.md#memory-and-context); check `nvidia-smi` for other processes |
| "model not found" or the checkpoint is not found | the weights are not in `~/models/swift-1.5-int4` | set `SWIFT_MODEL` (native), `MODEL_DIR` (Docker), or pass `--model` |
| HTTP 400 `swift-mk engine does not support ...` | the request uses a sampling or structured-output parameter the engine does not implement | remove it; the full [list](USAGE.md#sampling-parameters) |
| Answers are cut off or empty | the model's thinking tokens count against `max_tokens` | raise `max_tokens`, or send `"chat_template_kwargs": {"enable_thinking": false}` ([details](USAGE.md#thinking-and-reasoning-output)) |
| The first request with a very long prompt takes minutes | prefill speed: about 1,830 tok/s at 16k, about 960 tok/s at 262k | expected; follow-up turns reuse the cached prefix |
| Port 8000 is taken | another server | `--port 8013`, or `PORT=8013` for Docker |
| First Docker start is slow | cold Triton and vLLM compile caches | wait; they persist in the `cache` volume |
| It does not build or run on another GPU | the kernels are written and tested for sm_86 only | not supported; see [Ideas](IDEAS.md) if you want to try |

Still stuck? Open an issue with the output of `nvidia-smi`, `nvcc --version`, your command and the last 30 lines of the log (there is a template).

## Removing everything

| What | How |
|---|---|
| Docker image, containers, cache volume | `docker compose down -v` then `docker image rm qwen-warp:local` |
| Native environment | delete the `.venv` directory and `engine/build` |
| Weights | delete `~/models/swift-1.5-int4` (or `$SWIFT_MODEL`) |
