# make setup   -> uv environment (pyproject.toml / uv.lock) + the CUDA engine (engine/build/libswiftmk.so)
# make test    -> the checkpoint-only tests (needs the model; see README)
CUDA_HOME ?= /usr/local/cuda
export PATH := $(CUDA_HOME)/bin:$(PATH)

.PHONY: setup engine test
setup:
	uv sync --group dev
	$(MAKE) engine

engine:
	$(MAKE) -C engine -j$(shell nproc)

test:
	uv run tests/test_kv_dtypes.py
	uv run tests/test_host_tables.py
	uv run tests/test_tool_schema.py
	uv run tests/test_sampler_dist.py
	uv run tests/test_tool_call_rate.py
	uv run tests/test_server_smoke.py
	uv run tests/test_server_smoke.py --profile long --port 8012
	uv run tests/test_prefill_plan.py
	uv run tests/test_server_seed.py
	uv run tests/test_startup_configs.py
