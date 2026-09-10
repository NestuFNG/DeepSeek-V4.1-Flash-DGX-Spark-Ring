# Ring profile measurements — 2026-09-11

These are NestuFNG's four-Spark Ring measurements. The separate root `results/boot*/` directories are inherited upstream experiments with different prompts, thinking and sampling settings.

**Long-input validation is in progress; only completed cases below are claimed.** Configured context limit: 1,048,576. All results here used thinking ON / effort max, temperature 1, top_p .95. Server usage counts include thinking tokens. No OFF or full-BF16 comparison was run.

## Six independent coding tasks

Batch wall time **180.16 s**, output **13,551 tokens**, whole-batch throughput **75.22 token/s**. Reasoning accounts for **87.5%** of output.

| Request | Output tokens | Thinking tokens | Elapsed s | Client decode token/s |
|---|---:|---:|---:|---:|
| c6_code_0 | 1289 | 967 | 66.81 | 19.39 |
| c6_code_1 | 992 | 724 | 52.43 | 19.60 |
| c6_code_2 | 5054 | 4802 | 180.15 | 28.34 |
| c6_code_3 | 1208 | 992 | 69.97 | 17.72 |
| c6_code_4 | 1082 | 779 | 63.10 | 17.65 |
| c6_code_5 | 3926 | 3587 | 152.00 | 26.14 |

The entire interval between the first and last samples with six active requests was **51.12 s**, with **100.50 token/s** in server counter deltas. About **109.0 s** of the run had only one or two active requests. The batch drains; 75.2 is not a saturated C6 capacity measurement. Both figures are reported to avoid hiding the tail.

Accepted draft tokens plus the bonus token average **3.536 per request step** for this batch. Preemptions: **0**. Samples are approximately one second apart; step estimates are not GPU profiler timings. Full samples and the analysis script are included.

## Long-context retrieval

| Actual input | Actual output | Output allowance | TTFT s | Server prefill token/s | Client decode token/s | Result |
|---|---:|---:|---:|---:|---:|---|
| 65,254 | 204 | 262,144 | 59.17 | 1105.77 | 64.30 | PASS |
| 261,832 | 199 | 262,144 | 248.87 | 1054.47 | 66.59 | PASS |
| 784,046 | 201 | 262,144 | 912.30 | 861.01 | 63.77 | PASS |

Synthetic unique records contain three exact values at approximately 12%, 51% and 89% of the input. All values must match. Each case has a distinct cache salt. The final near-1M input uses a 65,536 output allowance; earlier cases allow 262,144. Actual generated output is much shorter. These are sequential retrieval cases, not six simultaneous 1M contexts or a complete long-context quality suite.

## Tools and images

A real model→tool→model round trip passed: synthetic lookup, multiplication, and a final answer matching 18 units × 375 cents = 6750 cents and the version marker. No external business system was modified. One generated two-shape image was recognized with default ON/max settings. Up to four images are configured, but four-image accuracy was not tested.

## Comparison with upstream boot 10

The upstream [fixed benchmark](../../../bench/v41bench.py) sends the same task to every stream in a category, with distinct short tags. It uses thinking OFF, temperature 0, and 150–256-token output allowances. Its code C6 result is 225.5 token/s and its eight-category C6 mean is 131.86. Our six different tasks, max reasoning, temperature 1, and 262,144-token allowances are not the same workload. The upstream mean draft acceptance length of 3.57 is close to our whole-batch 3.536; that comparison alone does not explain the throughput gap.

The proven explanation is partial tail underutilization. Prompt content, speculative acceptance by category, GPU state, the 1M configuration and actual kernel/communication time remain possible factors requiring matched measurements. We do not claim to outperform upstream in throughput or to have established thinking mode as the sole cause. Source: [upstream snapshot](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/tree/ca662ac35193c69ace9cee37f13a94abf2eff0fc).

## Limits

- 4,909,644 is the engine-reported logical pool for the TP4 instance, not four times that number and not six full resident 1M contexts.
- Current cache is `fp8_ds_mla` plus an FP8 indexer. No FP4 benefit is included.
- Selected package/source/launch hashes agreed on all nodes, but two Docker image IDs were present. Byte-identical images are not asserted.
- The generalized public scripts are syntax/consistency checked; the live tests used the same kernels and serving settings with local paths. A second clean deployment is not yet validated.
- No multi-day soak, reboot recovery, full 256K output generation, or universal agent-client compatibility test is claimed.
