#!/bin/bash
# p4_run.sh <label> <vllm serve args...>   (env: VLLM_CUTE_W4 / HEAD / MISC / HOST). Starts a server, benchmarks it, stops it. Run under `campaign lease` with the container stopped.
LABEL=$1; shift
cd /media/fatman/data/workspace/qwen-warp
export VLLM_CACHE_ROOT=/tmp/vllm_cache_$LABEL PYTHONPATH=proto_cute/plugin PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
.venv/bin/vllm serve /home/fatman/models/swift-1.5-int4 --served-model-name swift --port 8100 "$@" > /tmp/p1/$LABEL.log 2>&1 &
PID=$!
for i in $(seq 1 360); do
  curl -s localhost:8100/health >/dev/null 2>&1 && break
  kill -0 $PID 2>/dev/null || { echo "server died"; grep -E "Error|error" /tmp/p1/$LABEL.log | grep -v "File \|\^\^" | tail -3 | cut -c1-300; exit 1; }
  sleep 5
done
grep -E "Model loading took|Available KV cache|GPU KV cache size" /tmp/p1/$LABEL.log | cut -c1-170
.venv/bin/python proto_cute/p1_stream.py 8100 swift
for c in 1 2 4; do .venv/bin/python proto_cute/p1_stream_mc.py 8100 swift $c | tail -1; done
nvidia-smi --query-gpu=memory.used --format=csv,noheader
kill $PID; wait $PID 2>/dev/null; sleep 5
