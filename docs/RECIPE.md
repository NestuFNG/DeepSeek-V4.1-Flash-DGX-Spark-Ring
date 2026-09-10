# Recipe: DeepSeek-V4.1-Flash on four DGX Sparks (vLLM, TP4)

This is the full path from an empty fleet to a serving endpoint. Every file it names is in this repo, and the boot log in the [README](../README.md) records what each step fixed.

## 0. Hardware and layout

**Machines:**
- Four DGX Spark (GB10, 128 GB unified memory each, about 121.7 GiB visible to the OS).
- ConnectX-7 RoCE fabric on 192.168.192.0/24.

**Where the model lives:**
- One node (the head, "Reddie" here) keeps the 510 GB checkpoint on local NVMe at `/var/tmp/models`.
- It exports that directory read-only over NFS. The other three mount it at `/mnt/reddie-models`.
- Nothing is copied to the workers.

| rank | node | IP | weights from |
|---|---|---|---|
| 0 | Reddie (head, API on :8000) | 192.168.192.2 | local NVMe |
| 1 | Spark4 | 192.168.192.4 | NFS |
| 2 | Asusi | 192.168.192.3 | NFS |
| 3 | Bluey | 192.168.192.1 | NFS |

**Why four nodes and a disk patch:**
- The routed experts (296 GB MXFP4) split four ways fit.
- The two Engram n-gram tables (203 GB FP8) do not: stock vLLM puts them in host memory, and on GB10 host memory is the GPU pool.
- With the Engram-on-disk patch, each rank loads 78.79 GiB (measured) and the tables stay on disk. TP2 does not fit either way.

## 1. Download (head node)

`build/dl-ds41.sh` pulls `deepseek-ai/DeepSeek-V4.1-Flash` into `/var/tmp/models/DeepSeek-V4.1-Flash` with `hf download`.

## 2. NFS

- Export `/var/tmp/models` from the head.
- Mount it read-only on each worker: `mount -t nfs -o ro,vers=3 192.168.192.2:/var/tmp/models /mnt/reddie-models`.
- Put the mount in `/etc/fstab`. Two of our workers had it only as a manual mount and lost it after a watchdog reset.

Load times are measured on a cold page cache, because the launcher drops caches before every boot:
- The head reads its 78.79 GiB per rank in about 10 minutes.
- Workers take 10 to 18 minutes over NFS.
- The head waits for the slowest worker before profiling.

## 3. Engine image

Build each image on every node (they are node-local); the chain is incremental.

| image | how | why |
|---|---|---|
| `vllm-dsv41:overlay1` | `build/Dockerfile.overlay` on `vllm/vllm-openai:nightly-8a728663c1c3eeace834a95f5654fa653cc1998c` (the exact merge-base of vLLM branch `dsv41-feat`), plus `_C_stable_libtorch` rebuilt for sm_121a (`build/build_stable_ext.sh`, about 6 minutes on a Spark) | The branch's four kernel changes all live in that one extension. |
| `vllm-dsv41:overlay3` | `build/build_overlay3.sh`: FlashInfer v0.7.0rc1 (`07869c61`) with pinned submodules (cutlass `b46b16d0`, cccl `16bd510c`, spdlog `c3aed4b6`), built with `BUILD_NVEP=0 FLASHINFER_BUILD_NO_PIP=1`; the stale 0.6.18 jit-cache/cubin packages are removed | FlashInfer 0.6.18's SM120 sparse-MLA decode is a fixed table that lacks V4.1's topk of 1152. |
| `vllm-dsv41:overlay4` | `build/build_overlay4.sh`: prebuilds `mxfp8_gemm_cutlass_sm120` with `MAX_JOBS=2` | Its runtime compile (7 CUTLASS files, 22 parallel jobs) exhausted host memory on all four nodes at once in boot 3. |
| `vllm-dsv41:overlay5` | `build/build_overlay5.sh` + `build/prewarm5.py`: rebuilds `sparse_mla_sm120` under the exact runtime environment; `build/verify5.py` checks that both kernels load without compiling | The earlier prewarm was built with debug flags, so its cache key did not match at runtime. |

## 4. Patches (bind-mounted over the image; nothing baked)

Copy these five files to `~/patches/dsv41-boot3/` on every node together with `patch/sm12x-pages/mounts.txt`. The launcher mounts each file over the path the manifest gives.

| file (repo) | mounted over (`vllm/...`) | what it does |
|---|---|---|
| `patch/engram-offset-fix/engram.py` | `models/deepseek_v4_1/common/engram.py` | **Engram on disk**: rows are read on demand from shards 47/48 with `preadv` on a thread pool, then dequantized on the CPU (fp8 x ue8m0 to bf16). This copy includes the rank-offset fix: without it, ranks 1-3 read rank 0's rows ([details](boot3-wedge-and-engram-offset.md)). |
| `patch/weight_utils.py` | `model_executor/model_loader/weight_utils.py` | The weight loader skips the two Engram tables, so the 203 GB is never read at load. |
| `patch/sm12x-pages/attention.py` | `models/deepseek_v4_1/attention.py` | The SWA cache page size and the per-ratio compressed-KV page size come from the backend instead of a literal 32. |
| `patch/sm12x-pages/flashinfer_sparse.py` | `models/deepseek_v4_1/nvidia/flashinfer_sparse.py` | On SM12x, pages hold 64 compressed states: 64 tokens for ratio 1, 128 for ratio 2. There is also a 64-token SWA backend, the only page size FlashInfer's SM120 sparse-MLA kernels are built for. |
| `patch/sm12x-pages/sparse_swa.py` | `v1/attention/backends/mla/sparse_swa.py` | Adds the `get_swa_block_size()` hook (32 by default). |

## 5. Launch

- `launch/dsv41-tp4.sh <rank>` starts one rank.
- `launch/boot_dsv41.sh` (run on any node with SSH to the others) starts ranks 3, 2 and 1, then the head, with the same knobs on all four.
- The serving configuration is `launch/boot8-go.sh`.

Flags that matter, and why:

| flag / env | why |
|---|---|
| `--block-size 128` | Required. vLLM would otherwise pick 64 (the smallest size the patched main backend lists), and the V4 indexer cache only runs 128-token pages on SM12x, so the boot dies at KV init ([details](boot4-block-size.md)). The per-layer 64-token pages from the patches still apply. |
| `--enforce-eager` | The Engram disk lookup is a host-side gather inside the forward pass, so CUDA graphs cannot capture it. |
| `--speculative-config {"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":false}` | DSpark with the checkpoint's own draft layers. Adaptive verification needs CUDA graphs, so the launcher turns it off under eager. |
| `--engram-config '{"cpu_offload": false}'` + `DSV41_ENGRAM_DISK=1` | Engram from disk (32 read threads, chunk 16). |
| `--language-model-only` | Text only; the 32-layer vision encoder is not loaded. |
| `--default-chat-template-kwargs '{"thinking": false}'` | Thinking off by default. A request can still turn it on. |
| `--gpu-memory-utilization 0.80`, `--max-num-seqs 8`, `--max-num-batched-tokens 8192` | Measured to load and profile on 121.7 GiB nodes. |
| `-e MAX_JOBS=2 -e FLASHINFER_NVCC_THREADS=1` | If anything still compiles at runtime, it cannot take the host down. |
| `-e VLLM_USE_FLASHINFER_SAMPLER=0` | Uses the native sampler, so the first request does not JIT-compile one. |
| NCCL: `NCCL_NET=IB`, `NCCL_IB_HCA=rocep1s0f0`, `NCCL_IB_GID_INDEX=3`, RoCE v2, `NCCL_SOCKET_IFNAME=enp1s0f0np0` | Same fabric settings as our other TP4 Spark recipes. |

## 6. Verify

- `tools/postserve.sh <label>` runs, in order:
  - Kai's smoke test.
  - A greedy reference capture, or a comparison against an earlier capture.
  - A garble gate.
  - The fixed-prompt benchmark: `bench/v41bench.py`, prompt set `bench/prompts-v1.json`, C1-C6, plus cold prefill at 2K, 8K, 32K and 64K.
- `bench/v41needle.py` is the long-context needle test.

## 7. Host hardening we recommend

These are system settings, so apply them yourself:
- Enable `dgx-anti-oom` and make its container regex match `vllm_dsv41`.
- Set `vm.min_free_kbytes=1048576` and `vm.watermark_scale_factor=200`, so reclaim runs in the background instead of stalling the node.
- Add the fstab mounts from step 2.
- Raise the head's nfsd threads from 8 to 32.
