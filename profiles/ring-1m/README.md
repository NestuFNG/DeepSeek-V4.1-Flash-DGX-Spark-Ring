# Four-Spark Ring / 1M / max-thinking profile

[中文说明](README.zh-CN.md) · [Results](results/README.md) · [Attribution](../../NOTICE.md)

This profile publishes a serving configuration validated on four 128 GB DGX Sparks. It is an experimental deployment recipe, not a general installer. The scripts below parameterize the tested paths; a separate clean-machine installation has not been repeated.

## Validated configuration

| Setting | Value |
|---|---|
| Parallelism | TP4, DP1, one model instance |
| Fabric | Physical A—B—C—D—A Ring, direct 200G links, MTU 9000 |
| Context limit | 1,048,576 input + output tokens |
| Scheduler concurrency | 6 |
| KV allocation | 16 GiB per node; 4,909,644 logical token slots reported |
| KV format | `fp8_ds_mla`: 448 FP8 components + 64 BF16 RoPE components, scales/alignment; FP8 indexer |
| Thinking | ON, effort `max`; temperature 1.0, top_p 0.95 in validation |
| Prefill batch | 4096 tokens |
| Speculation | DSpark k=5, probabilistic draft / block rejection |
| CUDA graphs | FULL_AND_PIECEWISE; captured sizes 5,6,10,12,15,18,20,24,25,30,36 |
| Tools / vision | `deepseek_v41` parser + auto tools; up to 4 images configured |
| Engram | Full original tables on each node's local NVMe; global row offsets preserved |

The pool is shared by the TP4 instance: **do not multiply 4,909,644 by four**. Six simultaneous requests are supported, but six full 1M contexts do not fit resident in this pool. Output reservations and input share the model context limit. A 262,144-token output allowance leaves at most 786,432 input tokens, including templates/tools/images. This release does not claim to have generated 256K output tokens in one request.

## Preparation

1. Install the NVIDIA driver, Docker with NVIDIA Container Toolkit, and ARM64 CUDA 13 development tools if rebuilding NCCL. Verify every physical Ring neighbor and assign each direct logical link its own subnet. Verify MTU 9000 end to end. Keep management connectivity separate. Discover interface names using `ip -br addr` and `rdma link`; logical RDMA device count is not physical port count. This recipe does not reconfigure host networking.
2. Copy `config/node.env.example` to an untracked `node.env` on each host. Set ranks 0/1/2/3 in physical cycle order; set the head's management IP as `MASTER_ADDR`. Point all paths to local storage. The image and NCCL library must be consistent on all nodes.
3. Obtain the full official checkpoint from [ModelScope](https://modelscope.cn/models/deepseek-ai/DeepSeek-V4.1-Flash). `config/modelscope-shards.json` records the per-file source revisions we verified; `config/checkpoint-shards.json` records all 48 expected sizes and SHA256 hashes. Keep the official config, tokenizer, chat template and multimodal assets alongside the shards. No weight download runs automatically and no VPS is needed. Do not rebase or truncate Engram rows: this profile uses full original files and global offsets.
4. Verify each node's local copy once, then run a cheap unchanged-file check on startup:

```bash
source /absolute/path/to/node.env
python3 profiles/ring-1m/scripts/verify-model.py "$MODEL_DIR" \
  --write-stamp "$STATE_DIR/checkpoint-verified.json"
```

This hashes about 510 GB of shards per node; it can take a while. It verifies shard content and metadata presence, not the hash of each metadata file. Do not use NFS for live Engram random reads.

## Build and validate communication

From the repository root on ARM64 Linux, with no model workload competing for memory:

```bash
bash profiles/ring-1m/scripts/build-images.sh
bash profiles/ring-1m/scripts/build-nccl.sh /srv/nccl-patched
```

The image starts from a fixed official vLLM ARM64 digest and FlashInfer sources fixed by commit and archive SHA256. NCCL uses NVIDIA commit `73cf112295c33aee2b895f329f592f2a9b4b0f97` and the credited Sparkring Tree/PAT skip patch. A rebuild can have a different binary hash; correctness must be tested. An already verified compatible library can be used instead. Do not run Spark relay or custom Mesh plugins alongside this profile: the tested runtime selects NCCL's built-in IB transport and Ring.

Run the following on all four hosts at about the same time:

```bash
bash profiles/ring-1m/scripts/probe-node.sh /absolute/path/to/node.env
```

Require all four ranks to finish all-reduce, all-gather, reduce-scatter and broadcast numerical checks. Inspect logs for Ring + IB selection and Tree/PAT skipping. These source/build checks alone cannot prove cable or RoCE correctness.

## Serve

Start workers and head within the distributed initialization timeout, using the same profile on every machine:

```bash
bash profiles/ring-1m/scripts/launch-node.sh /absolute/path/to/node.env
docker logs -f dsv41-ring-1m
```

The launcher refuses an existing container or active GPU workload, checks the verified shard stamp and profile hashes, and requires at least 100 GiB available before starting. It keeps `--restart no`, matching the validated deployment; reboot recovery and service supervision are not configured. It binds port 8041 on the host network. Place it on a trusted network, or add authentication/TLS before exposing it outside that network.

Check `/health` and `/v1/models` on rank 0 and all worker logs before calling the service ready. The configured backend is Chat Completions; universal client/Responses API compatibility is not claimed.

## Run the same validation

```bash
export DSV41_BASE_URL=http://127.0.0.1:8041
export DSV41_RESULTS_DIR=/srv/dsv41-state/validation
python3 profiles/ring-1m/validation/validate_max_1m.py
python3 profiles/ring-1m/validation/validate_long_max.py
# Optional: one C1 and one C6 batch using the upstream coding prompt, still ON/max.
python3 profiles/ring-1m/validation/benchmark_public_prompt_max.py
```

These send real requests: a generated image, an actual synthetic two-tool round trip, six different coding tasks, then approximately 64K/256K/784K/982K input retrieval cases. All use thinking ON/max. Long cases can take tens of minutes. Start with an idle instance; do not mix them with unrelated requests if interpreting metrics. Metrics count thinking tokens as output. The long suite uses unique records, three exact needles and distinct cache salts; it is a retrieval check, not a comprehensive model quality evaluation.

Use `examples/chat-completions.json` as a request body. Preserve returned reasoning and tool call history when continuing agent conversations. The FP4 experiments are excluded from this release: current serving remains FP8 pending upstream official mixed-precision support.

## What changed

The main additional memory change bounds the DeepSeek V4.1 indexer gather workspace from `40 * max_model_len` to `min(40, max_num_seqs) * max_model_len`. At C6/1M that saves a theoretical 4.3828125 GiB per node for this allocation. Both metadata chunking and the operator call the same bound. Non-DeepSeek models retain the old bound. KV arithmetic/precision is unchanged. Run `validation/test_indexer_workspace.py` for real-function allocation/chunk coverage tests.

The SSD implementation keeps official 0909 ownership and row offsets while porting upstream graph prestaging. It requires DP1 and no ubatching. Full runtime copies, the image adaptation patches, source provenance and separate public file hashes are included so incompatibilities fail explicitly instead of silently patching a different vLLM version.
