#!/bin/bash
# p4_nsys.sh <label> <vllm serve args...>: server under nsys (node-level graph trace), one profiled 60-token request, then sqlite for nsys_decode.py. Run under `campaign lease`, container stopped.
LABEL=$1; shift
cd /media/fatman/data/workspace/qwen-warp
export VLLM_CACHE_ROOT=/tmp/vllm_cache_$LABEL PYTHONPATH=proto_cute/plugin PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nsys profile -t cuda,nvtx --cuda-graph-trace=node -s none --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite true -o /tmp/p1/$LABEL .venv/bin/vllm serve /home/fatman/models/swift-1.5-int4 --served-model-name swift --port 8100 --profiler-config '{"profiler":"cuda"}' "$@" > /tmp/p1/$LABEL.log 2>&1 &
PID=$!
for i in $(seq 1 360); do curl -s localhost:8100/health >/dev/null 2>&1 && break; kill -0 $PID 2>/dev/null || { echo "server died"; exit 1; }; sleep 5; done
.venv/bin/python proto_cute/p1_profile_window.py
sleep 8; pkill -f "[v]llm serve"; sleep 12
nsys stats -r cuda_gpu_kern_sum --format csv --force-export=true /tmp/p1/$LABEL.nsys-rep > /dev/null 2>&1
ls -la /tmp/p1/$LABEL.sqlite
