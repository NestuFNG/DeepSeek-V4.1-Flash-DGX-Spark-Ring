# Attribution and scope

This repository is a fork of [Tech2Wild / Kai's deployment work](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark), not an independent inference engine. Its Git history, original MIT attribution, benchmarks and patches are retained. The original README is [UPSTREAM_README.md](UPSTREAM_README.md).

The new `profiles/ring-1m/` profile was assembled and tested by NestuFNG with Codex assistance on 2026-09-11. It ports SSD Engram and graph prestaging from that upstream onto the pinned official vLLM image, preserves global Engram row ownership, adjusts SM12x sparse attention integration, bounds the indexer workspace by scheduler concurrency, and adds max-thinking validation. Generalized launch scripts have been locally checked; a second clean installation has not been performed.

Modified vLLM Python files retain their Apache-2.0 SPDX and copyright headers and are covered by [Apache-2.0](LICENSES/Apache-2.0.txt). Source hashes before publication-only comment additions are in `profiles/ring-1m/config/runtime-provenance.json`; public file hashes are checked separately. No model arithmetic changed during packaging.

NVIDIA NCCL remains under NVIDIA's license. The small Tree/PAT skip patch is from [FujitsuPolycom/sparkring](https://github.com/FujitsuPolycom/sparkring/blob/b70e127e8bda797e38afd9a1cefe1eb3ca790d2f/spark_transport/nccl/nccl-2.30.7-skip-tree-pat.patch), Apache-2.0. The live prebuilt library's provenance credits Alex Ellis / OpenFaaS for its build recipe. This repository does not distribute that library, load Sparkring's custom transport, or claim authorship of the Ring patch.

DeepSeek supplies the model, tokenizer, official reference code and architecture; vLLM, FlashInfer, PyTorch, NVIDIA CUDA/CUTLASS/CCCL and their contributors supply the runtime and kernels. Downloaded components keep their own licenses. Model weights, containers, credentials and private deployment logs are not distributed here. This is a community deployment profile, with no endorsement by those projects.
