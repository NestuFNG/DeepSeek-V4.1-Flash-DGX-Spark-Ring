# DeepSeek V4.1 Flash on four DGX Sparks — switchless Ring

[中文](profiles/ring-1m/README.zh-CN.md) · [Deploy](profiles/ring-1m/README.md) · [Measured results](profiles/ring-1m/results/README.md) · [Attribution](NOTICE.md)

A community deployment profile for **four 128 GB DGX Sparks in a physical 200G Ring**, serving the full DeepSeek V4.1 Flash checkpoint with vLLM TP4, SSD Engram and DSpark.

This is a fork of [Tech2Wild / Kai's work](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark). Their patches, benchmarks and Git history are retained; [their original README](UPSTREAM_README.md) describes their system. **Our configuration and measurements live under `profiles/ring-1m/`.**

## What this profile contributes

- **1,048,576-token context limit**, six scheduled requests, thinking **ON / max** by default.
- **16 GiB KV per node**, with **4,909,644 logical token slots** reported for the whole TP4 instance. Current cache is `fp8_ds_mla` and FP8 indexer.
- A DeepSeek-specific indexer workspace bound saving a theoretical **4.38 GiB per node**, without changing KV arithmetic or precision.
- Full-file, global-row Engram ownership retained while porting upstream graph prestaging to the pinned official vLLM image.
- Reproducible max-thinking checks for tools, generated image recognition, six independent coding requests and long-context retrieval.
- Pinned runtime sources, patch hashes, checkpoint shard hashes, parameterized configuration and explicit benchmark accounting.

## Current evidence

**Long-input validation is in progress; only completed cases below are claimed.** See [the full table](profiles/ring-1m/results/README.md).

Six mixed coding requests produced **13,551 output tokens** at **75.22 token/s** over the entire draining batch. The measured interval with all six active produced **100.50 token/s**. These include thinking tokens and are not directly comparable to upstream's OFF/temperature-0 benchmarks. We make no throughput-superiority claim.

**Six scheduled requests do not mean six full 1M contexts fit at once.** The logical pool holds about 4.68 × the configured maximum context. Long tests are sequential retrieval checks; output budgets are not measured generated lengths. The current profile keeps FP8 and does not include experimental FP4 kernels.

```mermaid
flowchart LR
  A[Rank 0 / API] --- B[Rank 1]
  B --- C[Rank 2]
  C --- D[Rank 3]
  D --- A
```

Each node keeps weights and Engram on local NVMe. A separate management network handles SSH and rendezvous. See the [deployment guide](profiles/ring-1m/README.md) before choosing interface names or launching containers.

## License

MIT for the original repository and our recipe additions; modified vLLM files keep Apache-2.0. NVIDIA and downloaded dependencies retain their licenses. See [NOTICE.md](NOTICE.md). No weights, container images, credentials or private deployment logs are published. This project is not affiliated with or endorsed by DeepSeek, NVIDIA, vLLM or FlashInfer.
