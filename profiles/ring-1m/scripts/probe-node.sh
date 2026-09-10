#!/usr/bin/env bash
set -euo pipefail
p="$(cd -- "$(dirname -- "$0")/.." && pwd)"
source "${1:?Usage: probe-node.sh /absolute/path/to/node.env}"
: "${RANK:?}" "${MASTER_ADDR:?}" "${MGMT_IFNAME:?}" "${RDMA_HCAS:?}" "${NCCL_DIR:?}"
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -Eq '[0-9]'; then
 echo 'GPU has an existing workload; run this before model serving'; exit 2
fi
docker run --rm --pull never --gpus all --network host --ipc host --memory 6g --memory-swap 6g \
 --ulimit memlock=-1:-1 --cap-add IPC_LOCK --device /dev/infiniband:/dev/infiniband \
 -v "$p/validation:/tests:ro" -v "$NCCL_DIR:/opt/patched-nccl:ro" \
 -e LD_PRELOAD=/opt/patched-nccl/libnccl.so.2 -e VLLM_NCCL_SO_PATH=/opt/patched-nccl/libnccl.so.2 \
 -e NCCL_SKIP_TREE_CONNECT=1 -e NCCL_ALGO=Ring -e NCCL_PROTO=LL,LL128,Simple \
 -e NCCL_NET=IB -e NCCL_NET_PLUGIN=none -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA="$RDMA_HCAS" \
 -e NCCL_IB_ADDR_FAMILY=AF_INET -e NCCL_IB_ROCE_VERSION_NUM=2 \
 -e NCCL_IB_SUBNET_AWARE_ROUTING=1 -e NCCL_IB_SUBNET_PREFIX_LEN=24 \
 -e NCCL_IB_MERGE_NICS=1 -e NCCL_CROSS_NIC=1 -e NCCL_SOCKET_IFNAME="$MGMT_IFNAME" \
 -e GLOO_SOCKET_IFNAME="$MGMT_IFNAME" -e NCCL_CUMEM_ENABLE=0 -e NCCL_NVLS_ENABLE=0 \
 -e NCCL_MIN_NCHANNELS=4 -e NCCL_MAX_NCHANNELS=4 -e NCCL_DEBUG=INFO \
 -e RANK="$RANK" -e WORLD_SIZE=4 -e MASTER_ADDR="$MASTER_ADDR" -e MASTER_PORT=29551 \
 --entrypoint timeout "${IMAGE:-local/dsv41-ring-ssd:fi07-20260911}" 150s python3 -u /tests/ring_vllm_collectives.py
