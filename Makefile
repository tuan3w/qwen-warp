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
	uv run tests/test_logit_procs.py
	uv run tests/test_spec_procs.py
	uv run tests/test_tool_schema.py
	uv run tests/test_sampler_dist.py
	uv run tests/test_tool_call_rate.py
	uv run tests/test_server_smoke.py
	uv run tests/test_server_smoke.py --profile long --port 8012
	uv run tests/test_proc_model.py
	uv run tests/test_sampling_params.py
	uv run tests/test_sampling_params.py --profile long --port 8043
	uv run tests/test_prefill_plan.py
	uv run tests/test_group_prefill.py
	uv run tests/test_prefix_store.py
	uv run tests/test_lineage_cache.py
	uv run tests/test_server_seed.py
	uv run tests/test_startup_configs.py
	uv run tests/test_vision_logic.py
	uv run tests/test_vision.py
	uv run tests/test_vision_engine.py
	uv run tests/test_vision_server.py
	uv run tests/test_batch_sched.py
	uv run tests/test_batch_engine.py
	uv run tests/test_batch_vision_engine.py
	uv run tests/test_server_batch.py
