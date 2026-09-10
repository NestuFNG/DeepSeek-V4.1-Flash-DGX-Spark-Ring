# DeepSeek-V4.1-Flash on four NVIDIA DGX Sparks (vLLM, TP4, DSpark, CUDA graphs)

**Status (2026-09-10): serving.**
- The model dropped at about 2 AM ET, and this stack started serving it at 9:17 AM ET the same day.
- Speed (boot 9, all four GPUs healthy), single stream, temperature 0:
  - count-to-100 **60.8 tok/s**, the bench coding prompt **57.1 tok/s**, count-to-300 **68.2 tok/s**
  - full C1-C6 bench below
- Context: **300K** max context with a **1,032,963-token** KV pool (3.44x at 300K). Tools and vision are on.
- 1M max context was proven on a separate boot, with a 1,078,380-token DSpark KV pool.
- Nothing on this page is a projection.

`deepseek-ai/DeepSeek-V4.1-Flash`:
- 552B-backbone MoE (769B counting the Engram tables), 16B active decode / 8B prefill, 1M context.
- MXFP4 experts, MXFP8 dense, 510 GB on disk.

It does not fit four GB10s as shipped. The 296 GB of experts split four ways is fine. The problem is the two Engram n-gram tables (203 GB FP8): vLLM keeps them in host memory, and on a DGX Spark host memory *is* the GPU pool. This repo keeps them on disk, fixes what that and the GB10's SM 12.1 break in day-0 vLLM, and gets CUDA graphs working around a host-side lookup.

**Full recipe: [docs/RECIPE.md](docs/RECIPE.md)**

## Results

Boot 8, the serving config: 4x DGX Spark TP4, DSpark k=5, FULL_AND_PIECEWISE CUDA graphs, 300K context, gmu 0.78.

About the benchmark:
- Fixed prompt set `bench/prompts-v1.json`, identical on every boot: 8 categories plus a counting ceiling.
- Streaming, temperature 0, thinking off, after a warmup.
- Token counts come from the server's `usage` block, never from stream chunks (DSpark packs several tokens per chunk). TTFT is the first token delta.
- Raw output: `results/boot8/`.

**Throughput by concurrency** (mean of the 8 categories; the counting ceiling is excluded):

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 20.26 | 24.22 | 1.004 |
| C2 | 34.03 | 21.07 | 1.78 |
| C3 | 53.19 | 21.75 | 0.933 |
| C4 | 64.13 | 18.93 | 0.926 |
| C5 | 64.61 | 15.53 | 1.204 |
| C6 | 82.01 | 16.78 | 1.18 |

**Per-stream tok/s by category:**

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 32.92 | 27.44 | 35.27 | 31.0 | 19.86 | 27.03 |
| json | 23.72 | 21.08 | 21.25 | 18.94 | 11.78 | 15.71 |
| narrative | 12.4 | 10.37 | 8.3 | 7.2 | 7.68 | 7.58 |
| prose | 15.23 | 14.08 | 12.05 | 9.92 | 9.23 | 8.6 |
| math | 30.6 | 27.52 | 28.29 | 23.44 | 24.87 | 20.2 |
| reasoning | 22.81 | 16.19 | 17.51 | 14.27 | 15.78 | 17.33 |
| summary | 17.0 | 14.09 | 14.17 | 12.14 | 10.62 | 9.31 |
| format | 39.09 | 37.75 | 37.19 | 34.55 | 24.41 | 28.48 |
| ceiling_count | 40.46 | 41.3 | 44.52 | 33.1 | 27.64 | 34.56 |

**Aggregate at C6, by category:**

| counting | code | tables | math | reasoning | JSON | prose |
|---|---|---|---|---|---|---|
| **190.2** | 138.0 | 125.8 | 107.7 | 87.7 | 69.4 | 44.4 |

All in tok/s.

**TTFT at C1:** 0.44 s (counting) to 1.2 s for short prompts; 2.0 s for the summary prompt, which carries a ~300-token passage.

**What each fix bought (counting prompt, one stream, measured):**

| stack | tok/s |
|---|---|
| eager, no speculation (boot 6) | 5.1 |
| eager + DSpark k=5 (boot 7) | 19.5-22.1 |
| DSpark + CUDA graphs + Engram rows staged before the forward (boot 8) | 41.5 (count-to-100 check; bench ceiling 40.5) |
| **+ GPU clock latch cleared on two nodes (boot 9, also tools + vision + gmu 0.80)** | **60.8** (count-to-100; 68.2 on count-to-300) |

**DSpark acceptance** (vLLM SpecDecoding metrics across the bench): mean acceptance length 3.47 tokens per step.
- It ranges from 1.9 on prose and narrative to 5.6 on counting and tables.
- That spread is why per-stream speed runs from 12 to 40 tok/s by content.

**Memory:**

| | value |
|---|---|
| Weights per rank, with the DSpark draft layers | 81.36 GiB |
| CUDA graphs | 1.58 GiB (captured in 45 s) |
| KV | 3.94 GiB = 841,005 tokens (2.80x at 300K) |
| Host memory available per node while serving | 11-12 GiB |

**1M max context (boot 7):** served at `--max-model-len 1048576` with DSpark: KV 1,078,380 tokens (5.21 GiB, 1.03x at 1M), eager, gmu 0.80.

## What had to be fixed

In boot order. Details in `docs/`.

1. **Engram on disk** (`patch/engram.py`, `patch/weight_utils.py`): the two 101 GB tables stay in the safetensors files, and rows are read on demand. Without this it does not fit.
2. **Runtime JIT wedge** (boot 3). A FlashInfer MXFP8 GEMM compiled at runtime with 22 parallel jobs and exhausted host memory on all four nodes, and the watchdogs reset them. Fix: kernels prebuilt in the image (`build/build_overlay5.sh`) plus `MAX_JOBS=2`. [docs](docs/boot3-wedge-and-engram-offset.md)
3. **Engram rank offset** (found in audit). The disk reader ignored each rank's row offset, so ranks 1-3 read rank 0's rows, silently. [docs](docs/boot3-wedge-and-engram-offset.md)
4. **`No common block size for 64`** (boot 4). vLLM picked the smallest listed block size, and the V4 indexer backend refused it. Fix: `--block-size 128`. [docs](docs/boot4-block-size.md)
5. **DeepGEMM `block_kv == 32 or 64`** (boot 5). The ratio-1 indexer cache had 128 states per block. Fix: SM12x indexer pages of 64 states. [docs](docs/boot5-indexer-pages.md)
6. **Relaunch race.** A new worker joined the still-live old head's rendezvous on the same port. Fix: `tools/launch.sh` stops every node first.
7. **`persistent_topk` on GB10** (boot 7, long context). It oversubscribes the 48 SMs, and its fallback needs 128 KB of shared memory per block (GB10 has 99 KB). Fix: `top_k_per_row_decode`, which is also 1.6-3.6x faster there. [results](patch/sm12x-indexer-topk/RESULTS.md)
8. **Eager mode was the throughput ceiling.** About 200 ms per step, host-bound; the GPUs sat near idle. Fix: the Engram lookup moves out of the forward into `prepare_inputs` (`patch/model_state.py`), with all rows read in parallel. The whole decode step is then captured as a CUDA graph, with exact capture sizes so DSpark batches are never padded (FlashInfer #5015).
9. **GPU clock latch** (2 of 4 Sparks). Reddie and Asusi sat at 630-950 MHz with no visible cause. Every TP step waited for them. Fix: unplug the adapter for 30-60 s; a reboot does not clear it. Result: count 41.5 to 60.8 tok/s, code 32.9 to 57.1. [docs](docs/gpu-clock-latch.md)

## Boot log

| boot | change | outcome |
|---|---|---|
| 1 | overlay1, eager, text-only, Engram on disk | KV 1,989,514. Died in decode warmup: FlashInfer 0.6.18 has no SM120 sparse-MLA decode kernel for `page_block_size=32`. |
| 2 | overlay2 + SWA override | Died at KV init: `No common block size for 32`. |
| 3 | overlay3 (FlashInfer 0.7.0rc1) + SM12x page patches | Wedged all 4 nodes (runtime JIT compile exhausted host memory). |
| 4 | overlay5 (kernels prebuilt) | No runtime JIT; KV 2,026,695. Died at KV init: `No common block size for 64`. |
| 5 | + `--block-size 128`, Engram offset fix, 300K | KV 2,346,690. Died in decode warmup: DeepGEMM `block_kv == 32 or block_kv == 64`. |
| 6 | + indexer pages of 64 states | **Served.** Eager, no speculation: 5.1 tok/s (count). Smoke clean, greedy reference 8/8, garble gate 30/30. |
| 7 | + DSpark k=5 at 1M max context | **Served at 1M.** KV 1,078,380. DSpark eager 19.5-22.1 tok/s. A 32K request then killed it in `persistent_topk` (fix 7). |
| 8 | + CUDA graphs, Engram staged before the forward, top-k fix, gmu 0.78, 300K | Served. Count check 41.5 tok/s, code 32.9 (two GPUs clock-latched, found later). |
| 9 | + tools, vision, gmu 0.80; Reddie and Asusi power-cycled to clear a GPU clock latch | **Serving.** KV 1,032,963 (3.44x at 300K). Count-to-100 60.8 tok/s, code 57.1, count-to-300 68.2. Tool call and image OK. |

## Known limits and next steps

- **Vision and tools.**
  - Both on: up to 4 images per request, with the `deepseek_v41` tool and reasoning parsers.
  - Thinking is off by default; a request can turn it on with `"chat_template_kwargs": {"thinking": true}`.
  - FlashInfer #4973 (vision on SM120) did not reproduce on a single test image; heavier image traffic is untested.
- **Adaptive verification is off.** It pads speculative batches, and padded batches can hang SM120 sparse MLA (FlashInfer #5015, open).
- **Step time.** With all four GPUs healthy, a DSpark step takes about 97 ms on counting (60.8 tok/s at about 5.9 accepted tokens per step), down from about 145 ms. The next levers:
  - the Engram staging, which idles the GPU every step
  - the ~90 all-reduces per step over RoCE
  - the MoE kernel
- **Engram reads scale with tokens per step.** At high concurrency they become the bottleneck. Local copies of each rank's Engram quarter would take NFS out of the path.
- **1M long context after the top-k fix.** The top-k fix is validated on GB10 at row widths up to 300,000. A 1M-token request has not been re-run on the fixed stack.
- **Host hardening** (system settings, not applied here): see [docs/RECIPE.md](docs/RECIPE.md) step 7.

## Repo layout

| path | what |
|---|---|
| `patch/` | The exact files bind-mounted over vLLM (md5s in `patch/README.md`), plus one folder per fix with its diff and test. |
| `build/` | Image chain: overlay1 (branch + sm121 extension), overlay3 (FlashInfer 0.7.0rc1), overlay4/5 (prebuilt kernels). |
| `launch/` | `dsv41-tp4.sh <rank>`, `boot_dsv41.sh` (worker-first fan-out), `bootN-go.sh` per boot (`boot9-go.sh` is the serving config). |
| `tools/` | Launch wrapper, pre-launch patch installer, boot poll, post-serve checks, hang watchdog, bench comparison, reproduction scripts. |
| `bench/` | Fixed prompt set v1, C1-C6 bench, long-context needle test. |
| `docs/` | Recipe and one post-mortem per failure. |
| `results/` | Head logs of every boot, proofs, bench output. |

**Fleet:**
- Reddie is the head: model on local NVMe, exported over NFS.
- Asusi, Bluey and Spark4 are workers.
- ConnectX-7 RoCE fabric, 192.168.192.0/24.

**Sister repos:** [DeepSeek-V4-Flash-Vision-Exp (vLLM, 2x/4x Spark)](https://github.com/tonyd2wild/DeepSeek-v4-Flash-Vision-Exp-DSpark-1M-NVFP4-KV-2x-DGX-Spark) · [DeepSeek-V4-Flash-Vision (SGLang, 2x Spark)](https://github.com/tonyd2wild/DeepSeek-V4-Flash-Vision-SGLang-DGX-Spark)

**Credits:**
- The vLLM team, for the day-0 `dsv41-feat` branch.
- Kai, for the first Engram-on-disk patch and the SM12x page-size patches.
- The Engram-on-disk idea follows our own PLE-on-disk patch for Qwen3.8-Flash-Next.
- Prior art acknowledged at the idea level: vLLM PR #54129 (`VLLM_PLE_MMAP`). No code was copied.
