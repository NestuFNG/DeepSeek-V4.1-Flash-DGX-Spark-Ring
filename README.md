# DeepSeek-V4.1-Flash on four NVIDIA DGX Sparks (vLLM, TP4)

**Status: first boot in progress (2026-09-10, ~3:50 AM ET).** Numbers land here as they are measured. Nothing on this page is a projection.

`deepseek-ai/DeepSeek-V4.1-Flash` dropped 2026-09-10 ~2 AM ET: 552B-backbone MoE (769B counting the Engram tables), 16B active decode / 8B prefill, 1M context, MXFP4 experts, MXFP8 dense, 890 bytes of global KV per token. 510 GB on disk. It does not fit four GB10s as shipped: 296 GB of experts split four ways is fine, but the two Engram n-gram tables (203 GB) are host-offloaded by vLLM into pinned memory, and on a DGX Spark host memory *is* the GPU pool.

This repo is the recipe that makes it fit:

| piece | what |
|---|---|
| `patch/` | **Engram-on-disk**: the per-rank table shard (47 GiB at TP4) stays on NVMe/NFS and rows are read on demand (positional preads on a thread pool, CPU dequant fp8 x ue8m0 -> bf16). 2 files bind-mounted over vLLM. `test_engram_disk.py` is the correctness/speed test (~87K rows/s on Reddie). |
| `build/` | How the engine image was made: vLLM branch `dsv41-feat` is main@`8a728663` + 8 commits touching 4 kernel files, all in `_C_stable_libtorch`. We rebuilt only that extension for sm121 on a Spark (6 minutes) and overlaid the branch's Python tree on the `nightly-8a728663` arm64 image (`Dockerfile.overlay`). |
| `launch/` | `dsv41-tp4.sh <rank>` (per node) and `boot_dsv41.sh` (worker-first fan-out). Same fabric/NCCL env as our GLM TP4 recipe. |
| `results/` | smoke test and measured logs, added per boot. |

Fleet: Reddie (head, model on local NVMe, exports it over NFS), Asusi, Bluey, Spark4 (workers read the weights over NFS). ConnectX-7 RoCE fabric 192.168.192.0/24.

Sister repos: [DeepSeek-V4-Flash-Vision-Exp (vLLM, 2x/4x Spark)](https://github.com/tonyd2wild/DeepSeek-v4-Flash-Vision-Exp-DSpark-1M-NVFP4-KV-2x-DGX-Spark) · [DeepSeek-V4-Flash-Vision (SGLang, 2x Spark)](https://github.com/tonyd2wild/DeepSeek-V4-Flash-Vision-SGLang-DGX-Spark)

Credits: vLLM team for the day-0 `dsv41-feat` branch; the Engram-on-disk idea follows our own PLE-on-disk patch for Qwen3.8-Flash-Next (2026-09-05). Prior art acknowledged at the idea level: vLLM PR #54129 (`VLLM_PLE_MMAP`) and MiaAI-Lab's single-Spark work. No code was copied.
