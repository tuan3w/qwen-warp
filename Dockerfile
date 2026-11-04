# qwen-warp server image.  The weights are not baked in: mount the checkpoint at /models/swift-1.5-int4 (see docker-compose.yml).
# The engine is compiled for sm_86 (RTX 3090) only.  Needs the NVIDIA container toolkit and a driver >= 580 on the host.
FROM nvidia/cuda:13.0.2-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
# build-essential + python3.12-dev: Triton compiles small C stubs at run time (flash-linear-attention kernels).
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3.12 python3.12-dev build-essential ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.28 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_PYTHON=/usr/bin/python3.12 \
    UV_LINK_MODE=copy \
    SWIFT_MODEL=/models/swift-1.5-int4 \
    HF_HUB_OFFLINE=1 \
    PATH=/app/.venv/bin:$PATH

# Dependencies first so that source edits do not re-download ~8 GB of wheels.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY engine ./engine
COPY server ./server
COPY docker/entrypoint.sh /usr/local/bin/qwen-warp-entrypoint
RUN make -C engine -j"$(nproc)" && chmod +x /usr/local/bin/qwen-warp-entrypoint

EXPOSE 8000
ENTRYPOINT ["qwen-warp-entrypoint"]
