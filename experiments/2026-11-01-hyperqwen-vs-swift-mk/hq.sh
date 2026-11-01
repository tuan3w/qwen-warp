#!/bin/bash
# Start / stop HyperQwen's single-user server (image ghcr.io/syv-ai/hyperqwen:latest) on OUR Swift-1.5 checkpoint.
#   hq.sh start <config> [extra -e KEY=VAL ...]   blocks until /health is OK (or the container died / 900 s)
#   hq.sh stop
#   hq.sh logs                                   tail of the container log
# Own explicit -e flags only: no env_file, no token, no `docker inspect` dump; HF_HUB_OFFLINE=1 so nothing can download.
# PREPARE=0 VERIFY=0: the entrypoint's prepare stage would download the base model (HF_REPO) into the bind: we built the dir offline (build_model_dir.py).
# Configs: mtp-fast  SPEC=mtp CTX=fast   (HQ default: k=4 chain, bf16 KV, FlashAttention + split-KV verify, 64k)
#          mtp-long  SPEC=mtp CTX=long   (fp8 KV via FlashInfer, k=3, MAX_LEN default 150000; override with MAX_LEN=..)
#          dflash2   SPEC=dflash2        (DFlash2 drafter, 7 drafts, lookup drafting on, 64k, 8 slots)
#          dflash15  SPEC=dflash2 DFLASH_TOKENS=15 (reproduction mode, 4 slots, 56k)
#          dflash2-long SPEC=dflash2 CTX=long (int8 KV, 128k); mtp-huge / dflash2-huge CTX=huge (KVarN 4/2-bit KV, 200k / 240k)
set -u
REPO=$(cd "$(dirname "$0")/../.." && pwd)
NAME=hq-e11
PORT=${HQ_PORT:-18020}
SW=~/models/swift-1.5-int4
HQM=~/HyperQwen/models
cmd=${1:-}; shift || true
case "$cmd" in
  start)
    cfg=$1; shift
    case $cfg in
      mtp-fast)  E=(-e SPEC=mtp -e CTX=fast) ;;
      mtp-long)  E=(-e SPEC=mtp -e CTX=long) ;;
      dflash2)   E=(-e SPEC=dflash2) ;;
      dflash15)  E=(-e SPEC=dflash2 -e DFLASH_TOKENS=15) ;;
      dflash2-long) E=(-e SPEC=dflash2 -e CTX=long) ;;
      mtp-huge)  E=(-e SPEC=mtp -e CTX=huge) ;;
      dflash2-huge) E=(-e SPEC=dflash2 -e CTX=huge) ;;
      *) echo "unknown config $cfg"; exit 2 ;;
    esac
    docker rm -f $NAME >/dev/null 2>&1
    docker run -d --name $NAME --gpus '"device=0"' --ipc=host -p 127.0.0.1:$PORT:$PORT \
      -e HOME=/cache -e PORT=$PORT -e PREPARE=0 -e VERIFY=0 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
      -e MODEL=/app/models/Swift-1.5-fast "${E[@]}" "$@" \
      -v $REPO/runs/hq/swift-fast:/app/models/Swift-1.5-fast:ro \
      -v $SW:$SW:ro -v $HQM:$HQM:ro \
      -v $HQM/Qwen3.8-27B-DFlash2-W4A16:/app/models/Qwen3.8-27B-DFlash2-W4A16:ro \
      -v hyperqwen_qwen-cache:/cache \
      ghcr.io/syv-ai/hyperqwen:latest single >/dev/null || exit 1
    t0=$(date +%s)
    until curl -sf -m 2 http://127.0.0.1:$PORT/health >/dev/null; do
      if [ "$(docker inspect -f '{{.State.Running}}' $NAME 2>/dev/null)" != true ]; then echo "container died"; docker logs $NAME 2>&1 | tail -40; exit 1; fi
      [ $(( $(date +%s) - t0 )) -gt ${HQ_BOOT_TIMEOUT:-1200} ] && { echo "boot timeout"; docker logs $NAME 2>&1 | tail -40; exit 1; }
      sleep 3
    done
    echo "healthy after $(( $(date +%s) - t0 )) s" ;;
  stop)
    docker logs $NAME > $REPO/runs/hq/logs/container-$(date +%H%M%S).log 2>&1
    docker rm -f $NAME >/dev/null 2>&1; echo stopped ;;
  logs) docker logs $NAME 2>&1 | tail -${1:-40} ;;
  *) sed -n 2,14p "$0"; exit 2 ;;
esac
