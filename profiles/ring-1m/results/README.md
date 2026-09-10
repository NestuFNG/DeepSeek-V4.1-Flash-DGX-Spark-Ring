# Ring profile measurements — 2026-09-11

These are NestuFNG's four-Spark Ring measurements. The separate root `results/boot*/` directories are inherited upstream experiments with different prompts, thinking and sampling settings.

**All four long-input retrieval cases passed.** Configured context limit: 1,048,576. All results here used thinking ON / effort max, temperature 1, top_p .95. Server usage counts include thinking tokens. No OFF or full-BF16 comparison was run.

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
| 982,126 | 160 | 65,536 | 1214.02 | 810.36 | 60.54 | PASS |

Synthetic unique records contain three exact values at approximately 12%, 51% and 89% of the input. All values must match. Each case has a distinct cache salt. The final near-1M input uses a 65,536 output allowance; earlier cases allow 262,144. Actual generated output is much shorter. These are sequential retrieval cases, not six simultaneous 1M contexts or a complete long-context quality suite.

## Tools and images

A real model→tool→model round trip passed: synthetic lookup, multiplication, and a final answer matching 18 units × 375 cents = 6750 cents and the version marker. No external business system was modified. One generated two-shape image was recognized with default ON/max settings. Up to four images are configured, but four-image accuracy was not tested.

## Comparison with upstream boot 10

The upstream [fixed benchmark](../../../bench/v41bench.py) sends the same task to every stream in a category, with distinct short tags. It uses thinking OFF, temperature 0, and 150–256-token output allowances. Its code C6 result is 225.5 token/s and its eight-category C6 mean is 131.86. Our six different tasks, max reasoning, temperature 1, and 262,144-token allowances are not the same workload. The upstream mean draft acceptance length of 3.57 is close to our whole-batch 3.536; that comparison alone does not explain the throughput gap.

The proven explanation is partial tail underutilization. Prompt content, speculative acceptance by category, GPU state, the 1M configuration and actual kernel/communication time remain possible factors requiring matched measurements. We do not claim to outperform upstream in throughput or to have established thinking mode as the sole cause. Source: [upstream snapshot](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/tree/ca662ac35193c69ace9cee37f13a94abf2eff0fc).

## Same coding prompt, still ON/max

An additional single C1 batch and single C6 batch use the upstream `merge_intervals` prompt. Temperature remains 1.0, top_p .95 and output allowance 262,144. Distinct salts avoid prefix cache reuse. This controls task content, but it still does not reproduce upstream's OFF/temperature-0 measurement, and one batch does not establish a stable performance distribution.

| Concurrency | Batch output token/s | Continuous C6 token/s | Output length range | Accepted + bonus / step | Result |
|---|---:|---:|---:|---:|---|
| 1 | 57.88 | — | 543–543 | 4.371 | PASS |
| 6 | 118.82 | 135.06 | 557–992 | 4.328 | PASS |

## Limits

- 4,909,644 is the engine-reported logical pool for the TP4 instance, not four times that number and not six full resident 1M contexts.
- Current cache is `fp8_ds_mla` plus an FP8 indexer. No FP4 benefit is included.
- Selected package/source/launch hashes agreed on all nodes, but two Docker image IDs were present. Byte-identical images are not asserted.
- The generalized public scripts are syntax/consistency checked; the live tests used the same kernels and serving settings with local paths. A second clean deployment is not yet validated.
- No multi-day soak, reboot recovery, full 256K output generation, or universal agent-client compatibility test is claimed.

## Additional short-concurrency and prefix-cache measurements

The homepage now includes the separate eight-slot candidate measurements, with actual input/output lengths, per-stream decode, batch throughput, strict simultaneous-decode windows and the 33K prefix replay. These do not establish eight resident 500K contexts. See [short C1/C6/C8 raw evidence](short-concurrency-c1-c6-c8.json), [33K prefix replay](prefix-cache-33k.json), and [the homepage tables](../../../README.md). All use ON/max, temperature1/top_p .95; no OFF test was added.

## Eight independent 500K contexts: B12 candidate

The separate B12 long-context run now demonstrates eight simultaneous decoding requests, each with at least 500,000 input tokens. It retains the 1M limit, 16 GiB KV per node, FP8 cache, DSpark K5, prefill chunk 4096 and ON/max settings. B12 includes an experimental CPU prefix-cache boundary correction. The published launcher still uses the B10 six-slot baseline; the B12 launch profile and patch have not been released in this snapshot.

| Workload | Input tokens (sum of requests) | Output tokens | Wall seconds | Whole-batch output token/s | All-eight decode token/s | Acceptance |
|---|---:|---:|---:|---:|---:|---|
| Cold independent retrieval, 8 requests | 4,093,814 | 2,080 | 4290.74 | 0.48 | — | All 8 exact-marker checks passed |
| Hot independent tool agents, 24 requests | 12,298,603 | 4,646 | 61.79 | 75.19 | 122.37 | All 8 three-turn tool workflows passed |
| Hot long code, 8 requests | 4,097,120 | 185,580 | 2011.23 | 92.27 | 100.04 | 6 complete; 2 truncated at the 32,768-token allowance |

All-eight windows total 10.223 seconds / 1,251 tokens for tools and 1233.740 seconds / 123,425 tokens for code. Adjacent samples must contain the same eight client requests, each already generating, and eight running server requests. Decode figures include reasoning tokens. One batch per workload is reported; no stable tuning gain is inferred from these measurements alone.

Cold input took 4286.542 seconds from request arrival until every first token. Dividing total input by that wall time gives 955.039 input token/s, including scheduling and interleaved decode, not a pure GPU prefill measurement. Cold prefill peaked at only two running requests. Seven responses decoded at about 1.28–1.55 token/s while later cold requests were still prefilling. The later hot stages, with 99.926% / 99.945% prefix hits, must not be substituted for this cold-input latency.

The code acceptance check only parses Python, checks five required functions, at least twenty test-method definitions and an exact synthetic project marker. **Generated code and generated tests were not executed.** Six complete cases have 40–49 test-method definitions. Two hit the output allowance; retries at 65,536 tokens are separate and excluded here. This is not an all-eight successful code-generation run. Whole-code timing was reconstructed from the earliest client start through the latest finish; its counter deltas use the first and last monitoring samples rather than exact phase boundaries.

No preemptions were observed. Peak KV usage was 77.67% for hot tools and 79.94% for long code. The minimum available system memory across the monitored B12 run was 2.28 / 4.42 / 5.12 / 6.06 GiB, sorted without node identities. A fresh eight-way long-code run at the larger allowance remains pending.

Evidence: [selected per-request measurements](long-agent-8x500k-b12.json), [identity-free monitoring samples](long-agent-8x500k-b12-samples.json), [local analysis script](analyze-long-agent.py). The public JSON uses a positive field allowlist; private prompts, responses, reasoning, machine logs, addresses and paths are excluded. The analysis script recomputes whole-batch rates and strict eight-way decode windows from the published data.

## Separate retries and shared-parent agents

Both truncated tasks were retried concurrently with a 65,536-token allowance and completed at 22,311 / 24,555 tokens. This separate two-way batch generated 46,866 tokens in 728.196 seconds (64.359 token/s). Both passed the same limited code-structure checks. Both natural output lengths were below the original 32K allowance; stochastic output variation means a causal benefit from increasing the allowance is not established. The original eight-way 6/8 result remains unchanged.

Eight agents then shared one already-cached roughly 512K project, each executing the same three-turn tool workflow. All 24 requests and exact final-answer checks passed: 4,503 output tokens / 53.613 seconds =83.990 token/s whole-batch, with 124.960 token/s during the strict eight-way decode window (8.083 seconds / 1,010 tokens). Prefix hit rate was99.941%, peak KV10.11%, median/max TTFT3.404/11.372seconds, and no preemptions. Unlike the independent-project workload, these agents share a common cached prefix.

The parent correctly aggregated the eight returned summaries using449input and67output tokens in1.805seconds. This is a short aggregation request, not another500Kdecode result. See [separate follow-up measurements](agent8-followup-b12.json). All phases retain ON/max and unchanged B12 serving settings.
