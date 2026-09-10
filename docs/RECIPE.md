# Recipe: DeepSeek-V4.1-Flash on four DGX Sparks (vLLM, TP4, DSpark, CUDA graphs)

This is the full path from an empty fleet to the serving endpoint. Every file it names is in this repo. The [README](../README.md) boot log records what each step fixed.

## 0. Hardware and layout

**Machines:**
- Four DGX Spark (GB10, SM 12.1, 128 GB unified memory each, about 121.7 GiB visible to the OS).
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
- With the Engram-on-disk patch, each rank loads 78.79 GiB without DSpark, or 81.36 GiB with the DSpark draft layers (measured). The tables stay on disk.
- TP2 does not fit either way.

## 1. Download (head node)

`build/dl-ds41.sh` pulls `deepseek-ai/DeepSeek-V4.1-Flash` into `/var/tmp/models/DeepSeek-V4.1-Flash`.

## 2. NFS

- Export `/var/tmp/models` from the head.
- Mount it read-only on each worker: `mount -t nfs -o ro,vers=3 192.168.192.2:/var/tmp/models /mnt/reddie-models`.
- Put the mount in `/etc/fstab`. Two of our workers had it only as a manual mount and lost it after a watchdog reset.

Load times (measured, cold page cache, because the launcher drops caches every boot):
- The head reads its weights in about 10 minutes.
- Workers take 10 to 18 minutes over NFS.
- DSpark adds a second pass over all 48 shards for the draft layers.
- The head waits for the slowest worker before profiling.

## 3. Engine image

Build each image on every node; they are node-local.

| image | how | why |
|---|---|---|
| `vllm-dsv41:overlay1` | `build/Dockerfile.overlay` on `vllm/vllm-openai:nightly-8a728663c1c3eeace834a95f5654fa653cc1998c` (the exact merge-base of vLLM branch `dsv41-feat`), plus `_C_stable_libtorch` rebuilt for sm_121a (`build/build_stable_ext.sh`) | The branch's kernel changes all live in that one extension. |
| `vllm-dsv41:overlay3` | `build/build_overlay3.sh`: FlashInfer v0.7.0rc1 (`07869c61`) with pinned submodules; the stale 0.6.18 jit-cache/cubin packages are removed | FlashInfer 0.6.18's SM120 sparse-MLA decode lacks V4.1's topk of 1152. |
| `vllm-dsv41:overlay4` | `build/build_overlay4.sh`: prebuilds `mxfp8_gemm_cutlass_sm120` with `MAX_JOBS=2` | Its runtime compile (7 CUTLASS files, 22 parallel jobs) exhausted host memory on all four nodes at once in boot 3. |
| `vllm-dsv41:overlay5` | `build/build_overlay5.sh` + `build/prewarm5.py`: rebuilds `sparse_mla_sm120` under the exact runtime environment; `build/verify5.py` checks that nothing compiles at runtime | This is the serving image. |

## 4. Patches (bind-mounted over the image; nothing baked)

Copy these seven files and the manifest `patch/mounts.txt` to `~/patches/dsv41-boot3/` on every node. The launcher mounts each file over the site-packages path the manifest gives. `tools/prelaunch-8.sh` does this with md5 checks. The top-level `patch/` files are byte-identical to what the serving boot mounts; `patch/README.md` lists their md5s, and the subfolders hold each fix's diff and test.

| file (repo) | mounted over (`vllm/...`) | what it does |
|---|---|---|
| `patch/engram.py` | `models/deepseek_v4_1/common/engram.py` | **Engram on disk.** Rows are read with `preadv` from shards 47/48 and dequantized on the CPU. Includes the rank-offset fix (without it, ranks 1-3 read rank 0's rows; [details](boot3-wedge-and-engram-offset.md)). Includes one shared read pool, so every row for both Engram layers is in flight at once. Adds `EngramDiskStager`. |
| `patch/model_state.py` | `models/deepseek_v4_1/nvidia/model_state.py` | Stages the Engram rows in `prepare_inputs`, **before the forward**: one GPU hash, one host sync, parallel reads into the persistent `staged_rows` buffer. The forward then has no host round trip, so it can be captured as a CUDA graph. |
| `patch/weight_utils.py` | `model_executor/model_loader/weight_utils.py` | The loader skips the two Engram tables (203 GB never read at load). |
| `patch/attention.py` | `models/deepseek_v4_1/attention.py` | SM12x page sizes: the SWA cache and compressed-KV pages come from the backend. The indexer cache holds 64 states per page (64 tokens at ratio 1, 128 at ratio 2), because DeepGEMM's paged MQA logits only takes 32 or 64 ([details](boot5-indexer-pages.md)). |
| `patch/flashinfer_sparse.py` | `models/deepseek_v4_1/nvidia/flashinfer_sparse.py` | Pages hold 64 compressed states. Adds a 64-token SWA backend, the only page size FlashInfer's SM120 sparse-MLA kernels are built for. |
| `patch/sparse_swa.py` | `v1/attention/backends/mla/sparse_swa.py` | The `get_swa_block_size()` hook. |
| `patch/sparse_attn_indexer.py` | `model_executor/layers/sparse_attn_indexer.py` | SM12x decode top-k uses `top_k_per_row_decode`. `persistent_topk` oversubscribes GB10's 48 SMs on long rows and kills the engine; the generic kernel matches `torch.topk` and is 1.6-3.6x faster on GB10 ([results](../patch/sm12x-indexer-topk/RESULTS.md)). |

## 5. Launch

- `launch/dsv41-tp4.sh <rank>` starts one rank.
- `launch/boot_dsv41.sh` (run on any node with SSH to the others) starts ranks 3, 2 and 1, then the head, with identical knobs.
- `tools/launch.sh <N>` wraps it. It **stops `vllm_dsv41` on every node first, head first**, runs `/root/prelaunch-<N>.sh` if present, then runs `launch/boot<N>-go.sh`. Stopping everything first matters: a new worker that starts while an old head is still listening on the same port joins that head's rendezvous and hangs the new boot.
- The serving configuration is `launch/boot8-go.sh`.

Flags and environment that matter, and why:

| flag / env | why |
|---|---|
| `--compilation-config {"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[5,6,10,12,15,18,20,24,25,30,35,36,40,42,48]}` | CUDA graphs are the throughput fix: eager decode on this model is host-bound (about 200 ms per step, GPUs nearly idle). FULL graphs cover uniform decode batches; mixed batches use breakable PIECEWISE graphs. With DSpark k=5, every decode batch is a multiple of 6 target tokens, or 5 draft tokens, so each has an exact graph and nothing is padded. Padded speculative batches can hang SM120 sparse MLA (FlashInfer #5015). |
| `-e VLLM_USE_BREAKABLE_CUDAGRAPH=1` | Set explicitly on every node, because the eager-break decorator binds when the model imports. |
| `--speculative-config {"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":false}` | DSpark with the checkpoint's own draft layers. Adaptive verification stays off: it forces variable-length decode graphs with padded rows, the #5015 trigger. |
| `--gpu-memory-utilization 0.78` | Leaves host headroom for the graph pool (measured 1.58 GiB) on unified memory. Nodes serve with 11-12 GiB available. |
| `--max-model-len 300000`, `--max-num-seqs 8`, `--max-num-batched-tokens 8192` | 300K context; KV pool 841,005 tokens (2.80x at 300K). |
| `--block-size 128` | Required. vLLM would otherwise pick 64 (the smallest size the patched main backend lists), and the V4 indexer backend refuses it at KV init ([details](boot4-block-size.md)). The per-layer 64-state pages from the patches still apply. |
| `--engram-config '{"cpu_offload": false}'` + `DSV41_ENGRAM_DISK=1` | Engram from disk (32 read threads). |
| `--language-model-only` | Text only; the vision encoder is not loaded. |
| `--default-chat-template-kwargs '{"thinking": false}'` | Thinking off by default. A request can turn it on with `chat_template_kwargs`. |
| `-e MAX_JOBS=2 -e FLASHINFER_NVCC_THREADS=1` | If anything still compiles at runtime, it cannot take the host down. |
| `-e VLLM_USE_FLASHINFER_SAMPLER=0` | Native sampler, so the first request does not JIT-compile one. |
| `-e TILELANG_CACHE_DIR=/cache/tilelang -e TRITON_CACHE_DIR=/cache/triton` | Kernel caches on the host, so they survive container restarts. |
| NCCL: `NCCL_NET=IB`, `NCCL_IB_HCA=rocep1s0f0`, `NCCL_IB_GID_INDEX=3`, RoCE v2, `NCCL_SOCKET_IFNAME=enp1s0f0np0` | Same fabric settings as our other TP4 Spark recipes. |

**1M max context (proof boot, `launch/boot7-go.sh`):**
- Same stack, but eager, `--max-model-len 1048576`, gmu 0.80.
- It served with a 1,078,380-token DSpark KV pool (1.03x at 1M).
- The indexer prefill buffer grows with max context, about 5.2 GiB at 1M, so 1M needs the extra memory from eager mode and gmu 0.80.
- That boot predates the top-k fix, and a 32K request crashed it at decode. The fix is in the patch set above. Its fix is validated on GB10 at row widths up to 300,000; the full 1M-wide case has not been re-run.

## 6. Verify

- `tools/postserve.sh <label>` runs:
  - Kai's smoke test.
  - A greedy reference capture, or a comparison against an earlier capture.
  - A garble gate.
  - The fixed-prompt benchmark: `bench/v41bench.py`, prompt set `bench/prompts-v1.json`, C1-C6.
- `bench/v41needle.py` is the long-context needle test.
- `tools/hangcheck.sh` flags a #5015-style hang: requests running with 0.0 tok/s for 90 s.

## 7. Host hardening we recommend

- **Check the GPU clocks first.** A GB10 can latch below 1 GHz with no visible cause, and only unplugging the adapter for 30-60 s clears it. Two of our four were stuck. Under a 15 s fp16 burn (`tools/recover.sh`), every node should show about 2.2-2.4 GHz, 80 W+ and 75-90 TFLOPS. [details](gpu-clock-latch.md)

These are system settings, so apply them yourself:
- Enable `dgx-anti-oom` and make its container regex match `vllm_dsv41`.
- Set `vm.min_free_kbytes=1048576` and `vm.watermark_scale_factor=200`.
- Add the fstab mounts from step 2.
- Raise the head's nfsd threads from 8 to 32.
