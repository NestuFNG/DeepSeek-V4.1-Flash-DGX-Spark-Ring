#!/bin/bash
set -euo pipefail
PROFILE_DIR="$(cd -- "$(dirname -- "$0")/.." && pwd)"
ENV_FILE="${1:?Usage: launch-node.sh /absolute/path/to/node.env}"
source "$ENV_FILE"
: "${STATE_DIR:?}" "${MODEL_DIR:?}" "${NCCL_DIR:?}" "${RANK:?}" "${HOST_IP:?}" "${MASTER_ADDR:?}" "${MGMT_IFNAME:?}" "${RDMA_HCAS:?}"
ROOT="$STATE_DIR"; MODEL="$MODEL_DIR"
IMAGE="${IMAGE:-local/dsv41-ring-ssd:fi07-20260911}"
NAME="${CONTAINER_NAME:-dsv41-ring-1m}"
rank="$RANK"; host_ip="$HOST_IP"
[[ "$rank" =~ ^[0-3]$ ]] || { echo 'rank must be 0..3'; exit 2; }
headless=(); if [ "$rank" != 0 ]; then headless=(--headless); fi
mkdir -p "$ROOT/logs" "$ROOT/cache/flashinfer"
python3 "$PROFILE_DIR/scripts/verify-model.py" "$MODEL" --check-stamp "$ROOT/checkpoint-verified.json"
python3 "$PROFILE_DIR/scripts/verify-profile.py"
test -f "$NCCL_DIR/libnccl.so.2"
test -f "$MODEL/model.safetensors.index.json"
if docker container inspect "$NAME" >/dev/null 2>&1; then
 echo "$NAME already exists; inspect its state before restarting" >&2; exit 3
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -Eq '[0-9]'; then
 echo 'GPU has an existing workload; inspect before starting' >&2; exit 4
fi
avail_kib=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
[ "$avail_kib" -ge 104857600 ] || { echo 'less than 100 GiB available'; exit 5; }
python3 - <<'PY'
import socket
for port in [8041,29557]:
 with socket.socket() as s:
  s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
  try:s.bind(('0.0.0.0',port))
  except OSError as e:raise RuntimeError(f'port {port} is not free: {e}') from e
PY
date -u +%FT%TZ > "$ROOT/logs/model-launch-at.txt"
docker run -d --name "$NAME" --restart no --pull never \
 --gpus all --network host --ipc host --memory 112g --memory-swap 112g \
 --ulimit memlock=-1:-1 --cap-add IPC_LOCK --device /dev/infiniband:/dev/infiniband \
 --oom-score-adj 500 \
 -v "$MODEL:/models/DeepSeek-V4.1-Flash:ro" -v "$ROOT/cache:/cache" \
 -v "$PROFILE_DIR/runtime/graph-patches/engram.py:/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/common/engram.py:ro" \
 -v "$PROFILE_DIR/runtime/graph-patches/model_state.py:/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/nvidia/model_state.py:ro" \
 -v "$PROFILE_DIR/runtime/memory-patches/indexer.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/indexer.py:ro" \
 -v "$ROOT/cache/flashinfer:/root/.cache/flashinfer" \
 -v "$NCCL_DIR:/opt/patched-nccl:ro" \
 -e LD_PRELOAD=/opt/patched-nccl/libnccl.so.2 -e VLLM_NCCL_SO_PATH=/opt/patched-nccl/libnccl.so.2 \
 -e NCCL_SKIP_TREE_CONNECT=1 -e NCCL_ALGO=Ring -e NCCL_PROTO=LL,LL128,Simple \
 -e NCCL_NET=IB -e NCCL_NET_PLUGIN=none -e NCCL_IB_DISABLE=0 \
 -e NCCL_IB_HCA="$RDMA_HCAS" \
 -e NCCL_IB_ADDR_FAMILY=AF_INET -e NCCL_IB_ROCE_VERSION_NUM=2 \
 -e NCCL_IB_SUBNET_AWARE_ROUTING=1 -e NCCL_IB_SUBNET_PREFIX_LEN=24 \
 -e NCCL_IB_MERGE_NICS=1 -e NCCL_CROSS_NIC=1 -e NCCL_SOCKET_IFNAME="$MGMT_IFNAME" \
 -e GLOO_SOCKET_IFNAME="$MGMT_IFNAME" -e TP_SOCKET_IFNAME="$MGMT_IFNAME" -e MN_IF_NAME="$MGMT_IFNAME" \
 -e NCCL_CUMEM_ENABLE=0 -e NCCL_NVLS_ENABLE=0 \
 -e NCCL_MIN_NCHANNELS=4 -e NCCL_MAX_NCHANNELS=4 -e NCCL_DEBUG=INFO \
 -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 -e VLLM_HOST_IP="$host_ip" \
 -e VLLM_ALLREDUCE_USE_SYMM_MEM=0 -e VLLM_ALLREDUCE_USE_FLASHINFER=0 \
 -e VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=0 \
 -e HF_HOME=/cache/huggingface -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
 -e VLLM_USE_BREAKABLE_CUDAGRAPH=1 -e VLLM_CACHE_ROOT=/cache/vllm-ring-1m -e VLLM_USE_RUST_FRONTEND=0 \
 -e VLLM_ENGINE_READY_TIMEOUT_S=7200 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
 -e DSV41_ENGRAM_DISK=1 -e DSV41_ENGRAM_DISK_THREADS=32 -e DSV41_ENGRAM_DISK_CHUNK=16 \
 -e MAX_JOBS=2 -e FLASHINFER_NVCC_THREADS=1 \
 -e TRITON_CACHE_DIR=/cache/triton -e TILELANG_CACHE_DIR=/cache/tilelang \
 -e VLLM_USE_FLASHINFER_SAMPLER=0 \
 -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
 "$IMAGE" /models/DeepSeek-V4.1-Flash \
 --served-model-name deepseek-v4.1-flash --host 0.0.0.0 --port 8041 \
 --tokenizer-mode deepseek_v41 --reasoning-parser deepseek_v41 --tool-call-parser deepseek_v41 --enable-auto-tool-choice \
 --limit-mm-per-prompt '{"image":4}' --mm-processor-cache-gb 1 \
 --tensor-parallel-size 4 --gpu-memory-utilization 0.80 --max-model-len 1048576 --kv-cache-memory-bytes 17179869184 \
 --max-num-seqs 6 --max-num-batched-tokens 4096 --enable-chunked-prefill \
 --block-size 128 --kv-cache-dtype fp8 --engram-config '{"cpu_offload":false}' \
 --default-chat-template-kwargs '{"thinking":true,"reasoning_effort":"max"}' \
 --load-format safetensors --safetensors-load-strategy lazy \
 --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":false}' \
 --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[5,6,10,12,15,18,20,24,25,30,36]}' \
 --disable-custom-all-reduce \
 --distributed-executor-backend mp --nnodes 4 --node-rank "$rank" \
 --master-addr "$MASTER_ADDR" --master-port 29557 "${headless[@]}"
echo "launched $NAME rank=$rank host=$host_ip; inspect logs before declaring readiness"
