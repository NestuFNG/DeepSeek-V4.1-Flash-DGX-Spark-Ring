# DeepSeek-V4.1-Flash on four NVIDIA DGX Sparks (vLLM, TP4)

**Status (2026-09-10, ~6 AM ET): not serving yet.** Boots 3 and 4 died. Both root causes are identified and reproduced, and boot 5 is staged. Numbers land here as they are measured. Nothing on this page is a projection.

`deepseek-ai/DeepSeek-V4.1-Flash` dropped 2026-09-10 ~2 AM ET.
- 552B-backbone MoE (769B counting the Engram tables), 16B active decode / 8B prefill, 1M context.
- MXFP4 experts, MXFP8 dense, 890 bytes of global KV per token. 510 GB on disk.

It does not fit four GB10s as shipped. The 296 GB of experts split four ways is fine. The problem is the two Engram n-gram tables (203 GB): vLLM host-offloads them into pinned memory, and on a DGX Spark host memory *is* the GPU pool.

This repo is the recipe that makes it fit:

| piece | what |
|---|---|
| `patch/` | **Engram-on-disk**: the per-rank table shard (47 GiB at TP4) stays on NVMe/NFS, and rows are read on demand. `patch/engram-offset-fix/` fixes a rank-offset bug in it (ranks 1-3 read rank 0's rows) and includes a harness that runs the real code against a synthetic 4-rank table. |
| `build/` | How the engine image was made. `vllm-dsv41:overlay5` = vLLM `dsv41-feat` + FlashInfer 0.7.0rc1, with `mxfp8_gemm_cutlass_sm120` and `sparse_mla_sm120` prebuilt under the runtime env, so nothing compiles at boot. |
| `launch/` | `dsv41-tp4.sh <rank>` (per node) and `boot_dsv41.sh` (worker-first fan-out). `bootN-go.sh` are the per-boot knob sets, one change per boot. |
| `bench/` | Fixed prompt set `v1` (8 categories plus a counting ceiling), identical on every boot, run at concurrency C1-C6, plus cold prefill at 2K/8K/32K/64K. |
| `tools/` | Boot poll, post-serve checks (smoke, greedy reference, garble gate, bench), and the block-size reproduction. |
| `docs/` | Post-mortems: boot 3 wedge + Engram offset bug, boot 4 block size. |
| `results/` | Measured logs and bench output, added per boot. |

## Boot log

| boot | change | outcome |
|---|---|---|
| 1 | overlay1, eager, text-only, Engram on disk | KV 1,989,514 tokens. Died in decode warmup: FlashInfer 0.6.18 has no SM120 sparse-MLA decode kernel for `page_block_size=32`. |
| 2 | overlay2 + SWA backend override | Died at KV init: `No common block size for 32`. |
| 3 | overlay3 (FlashInfer 0.7.0rc1) + SM12x page-size patches | Wedged all 4 nodes during profiling. A runtime JIT compile of `mxfp8_gemm_cutlass_sm120` exhausted host memory, and the watchdogs reset the nodes. [docs](docs/boot3-wedge-and-engram-offset.md) |
| 4 | overlay5 (kernels prebuilt) + `MAX_JOBS=2` | No runtime JIT; KV 2,026,695 tokens. Died at KV init: `No common block size for 64`. vLLM picks min(`[128, 64]`) = 64, and the V4 indexer backend takes only 128 on SM12x. [docs](docs/boot4-block-size.md) |
| 5 (staged) | + `--block-size 128` | |
| 6 (staged) | + Engram rank-offset fix | |
| 7 (staged) | + DSpark | |
| 8 (staged) | + vision | |

Fleet: Reddie (head; model on local NVMe, exported over NFS), Asusi, Bluey and Spark4 (workers; read the weights over NFS). ConnectX-7 RoCE fabric 192.168.192.0/24.

Sister repos: [DeepSeek-V4-Flash-Vision-Exp (vLLM, 2x/4x Spark)](https://github.com/tonyd2wild/DeepSeek-v4-Flash-Vision-Exp-DSpark-1M-NVFP4-KV-2x-DGX-Spark) · [DeepSeek-V4-Flash-Vision (SGLang, 2x Spark)](https://github.com/tonyd2wild/DeepSeek-V4-Flash-Vision-SGLang-DGX-Spark)

Credits: the vLLM team for the day-0 `dsv41-feat` branch. The Engram-on-disk idea follows our own PLE-on-disk patch for Qwen3.8-Flash-Next (2026-09-05). Prior art acknowledged at the idea level: vLLM PR #54129 (`VLLM_PLE_MMAP`) and MiaAI-Lab's single-Spark work. No code was copied.
