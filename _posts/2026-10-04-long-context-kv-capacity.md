---
title: "When One Request Fills the KV Cache: Diagnosing Long-Context Capacity Limits"
date: 2026-10-04 10:00:00 -0700
categories: [LLM Serving, KV Cache]
tags: [ml-systems, serving, kv-cache, long-context, vllm, sglang, deepseek-v4]
description: The KV cache grows linearly with context length while GPU memory is fixed. How to tell "each request is too big" apart from "too many requests", what operators can change without touching the engine, and five public cases.
---

**Symptom:** request rate is unchanged, yet throughput drops and short requests start missing their latency targets.
**Reality:** a handful of long requests have filled the KV cache. Few requests are running, and the batch is too small to keep decode efficient.

> This note covers **capacity only** — whether the KV cache fits. Three neighbouring problems look similar and are kept separate:
> - **Too many requests:** every request is small, but there are many of them. Also a capacity problem, with a different fix.
> - **Duplicated prefixes:** the same prefix is stored many times. The fix is prefix caching, which cannot help when a single copy already does not fit.
> - **Bandwidth:** long contexts also mean more KV bytes to read per decode step. The two limits often arrive together, but they are diagnosed and fixed differently.
{: .prompt-info }

## 1. Why long context hits a capacity wall

The KV cache trades memory for compute: keys and values of earlier tokens are computed once and read at every
later decode step, instead of being recomputed. The price is memory that grows with every token:

```
KV bytes per token = 2 (K and V) × layers × KV heads × head_dim × bytes per element
KV per request     = KV bytes per token × context length (input + output)
Total KV           = KV per request × concurrent requests
```

| Model (BF16 KV) | Layers | KV heads | head_dim | Per token | 8K request | 128K request |
|---|---|---|---|---|---|---|
| [Llama-3-8B](https://huggingface.co/NousResearch/Meta-Llama-3-8B/blob/main/config.json) | 32 | 8 | 128 | 128 KiB | 1 GiB | 16 GiB |
| [Llama-3-70B](https://huggingface.co/NousResearch/Meta-Llama-3-70B/blob/main/config.json) | 80 | 8 | 128 | 320 KiB | 2.5 GiB | 40 GiB |

Memory is fixed, so concurrency falls as context grows. As an illustration (not a measurement), with about
60 GiB left for the KV cache, Llama-3-8B fits roughly 60 concurrent 8K requests but only 3 at 128K.
[RetroInfer](https://arxiv.org/abs/2505.02922) reports the same order of magnitude: an A100 80GB serving
Llama3-8B supports a maximum batch size of 4 at 128K context.

### Three layers of impact

| | What you see | Is it a problem? |
|---|---|---|
| **① The long request is slow** | Long TTFT (long prefill), somewhat higher TPOT (more KV to read) | No — long requests are expected to be slower |
| **② Others slow down** | Short requests queue longer or are preempted and recomputed; their p99 worsens | Yes |
| **③ Throughput drops** | A few long requests fill the KV cache, fewer requests run, the batch shrinks | Yes |

## 2. Diagnosis

### Step 1 — Compute the expected ceiling before looking at dashboards

vLLM prints the real KV capacity at startup, after weights and activations have been subtracted
(source: [`kv_cache_utils.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_utils.py)):

```
GPU KV cache size: X tokens, Maximum concurrency for N tokens per request: Y x
```

`X` is the total KV capacity in tokens; `Y` is how many requests of length `max_model_len` (`N`) fit at once.

```
expected running ceiling ≈ X ÷ long-tail context length in real traffic (input + output, e.g. p95)
```

- **Use the distribution, not the mean.** If 90% of requests are 2K and 10% are 128K, the mean (~15K) looks
  safe, but a few 128K requests arriving together fill the cache.
- **Count output tokens.** A request keeps growing during decode. For reasoning models with long outputs,
  an input-only estimate is too optimistic.

| What Step 1 shows | What it means | What to check next |
|---|---|---|
| Ceiling is low (single digits) and close to the observed running count | Consistent with long context | Confirm the cache is actually full (Step 2), then check bandwidth (Step 4) |
| Ceiling is high, but few requests run and the cache is full | Running requests are larger than assumed, or `X` is smaller | Compute their actual size: usage × `X` ÷ running. Much larger than assumed → you missed output tokens, or long requests stay longer and dominate the running set (usually still this case). Close to assumed → re-read `X` |
| Ceiling is high, few requests run, cache **not** full | Not a KV problem | Admission caps (`max_num_seqs`, token budget), long prefills taking most of each step's token budget (time, not memory), host overhead |
| Ceiling < 1 | The longest requests cannot fit at all | A deployment issue: startup fails or `max_model_len` rejects them |

### Step 2 — Confirm the KV cache is actually full

| Metric ([vLLM](https://github.com/vllm-project/vllm/blob/main/vllm/v1/metrics/loggers.py)) | Role |
|---|---|
| `vllm:kv_cache_usage_perc` | Primary signal (0–1). Sustained ≥ 0.9 |
| `vllm:num_requests_waiting` or `vllm:num_preemptions` | Confirms it is a bottleneck: at least one is rising, so new requests cannot get blocks |
| `vllm:num_requests_running` | Not a fullness signal — it is what separates "too many" from "too long" once the cache is full |

- **How usage is computed:** `1 − free blocks ÷ total blocks`
  ([`block_pool.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/block_pool.py)). Cached
  prefix blocks that no running request references sit in the free queue as eviction candidates, so usage
  reflects what running requests actually hold.
- **High usage alone is not a bottleneck.** It may just mean the cache is well used. It becomes one when
  requests also start waiting or being preempted.
- **The 90% line:** in the [Red Hat vLLM triage guide](https://developers.redhat.com/articles/2026/03/09/5-steps-triage-vllm-performance),
  a healthy server has zero waiting requests and KV cache usage below 90%.

### Step 3 — Too many, or too long?

Both end with a full cache, and part of the cure is opposite, so it is worth separating them.

```
KV full? (usage ≥ 0.9 sustained, and waiting or preemptions rising)
├─ Yes → is the same prefix stored many times, with a low hit rate?
│    ├─ Yes → duplicated prefixes: fix prefix caching first
│    └─ No  → compare running with the Step 1 ceiling
│         ├─ Many running → too many requests
│         └─ Few running  → check the length distribution
│              ├─ More long requests     → long context (this note)
│              └─ Requests are not long  → KV pool too small (configuration)
└─ No → not a capacity problem: admission caps, bandwidth, host overhead
```

Two signals do most of the work:

1. **Running count at the moment the cache fills** (`vllm:num_requests_running`): many → too many requests; few → too long.
2. **Length distribution and throughput moving together** (`vllm:request_prompt_tokens`, `vllm:request_generation_tokens`):
   if the shift in long requests coincides with the throughput drop, it is context length; if throughput tracks QPS instead, it is request count.

When p99 rises, separate two paths. They apply to any request; to tell interference from expected slowness, look at short requests.

| Requests are... | Evidence |
|---|---|
| Blocked from entering | Cache full + `vllm:request_queue_time_seconds` rises |
| Preempted after entering | Cache full + `vllm:num_preemptions` rising + longer `request_prefill_time_seconds` / `request_decode_time_seconds` |

`request_queue_time` only measures the first wait, from first queued to first scheduled; time spent re-queued
after a preemption lands in the prefill or decode interval instead
([`stats.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/metrics/stats.py)).

These histograms are labeled only by model and engine, not by request length, so isolating short requests needs
per-request data (client-side results or request logs).

### Step 4 — Check whether bandwidth has also hit its ceiling

Long contexts raise the bytes read per decode step. If `DRAMA` ([DCGM](https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/feature-overview.html) field 1005, DRAM active) is high and TPOT misses its target, bandwidth
is saturated as well, and freeing capacity alone will not restore throughput. RetroInfer shows how close the
two limits sit: on the same A100 at 128K, the cache fits 4 requests, but scaling the batch beyond 3 gives
only marginal throughput gains because memory bandwidth saturates.

### Step 5 — Sweep context length (before launch, and after each change)

Fix concurrency and sweep context length (e.g. 8K / 32K / 64K / 128K), recording throughput, TTFT, TPOT,
preemptions and running count. It is the same idea as sweeping concurrency to find a saturation point, on a different axis.

1. **Causality:** if the problem appears with length, it is long context; if only with concurrency, it is request count.
2. **The real limit:** the length where preemptions start rising and TPOT degrades. On capacity alone it should
   sit close to Step 1, since `X` is already net of weights. It can land earlier when outputs are long (Step 1 counted
   only input) or bandwidth saturates first (Step 4); block granularity costs little, since
   [PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html) wastes under 4%.
3. **Use it:** to set `max_model_len`, the boundary for a separate long-request pool, and a realistic SLO for long requests.

### Step 6 — Validate by ablation

Change one thing (for example FP8 KV), re-run the sweep and check that the running ceiling and preemption
onset move in proportion to the change in KV bytes per token.

## 3. Workloads most exposed

| Trait | Why | Example                                              |
|---|---|------------------------------------------------------|
| Long documents / codebases | The input itself is long | Contract review, a whole repository as context       |
| Reasoning models | Long outputs; KV keeps growing during decode | Long chains of thought                               |
| Long-running agents | The trajectory grows with every step | Many tool calls, each result appended to the context |
| Very long user histories | The history itself is long | Generative recommendations|

**Less exposed:** short contexts (a few thousand tokens), and models whose per-token KV is already small
(MLA, many linear-attention layers).

## 4. What helps, and what does not

### A. Ways to shrink or move the KV cache

| Approach | Method | Who changes | Saves capacity | Saves bandwidth | Cost |
|---|---|---|---|---|---|
| Management  | [PagedAttention](https://arxiv.org/abs/2309.06180) | Engine | Removes waste, not KV itself: 60–80% waste in earlier systems → under 4% ([vLLM blog](https://blog.vllm.ai/2023/06/20/vllm.html)) | — | Kernels must handle non-contiguous blocks |
| Compression | [MQA](https://arxiv.org/abs/1911.02150) / [GQA](https://arxiv.org/abs/2305.13245): query heads share KV heads | Model | ✅ | ✅ | Decided at training time |
| Compression | [MLA](https://arxiv.org/abs/2405.04434): K and V stored as a low-dimensional latent | Model | ✅ | ✅ | Decided at training time |
| Compression | Along the sequence: several tokens' KV merged into one entry ([DeepSeek-V4](https://arxiv.org/abs/2606.19348)) | Model | ✅ | ✅ | Decided at training time |
| Compression | [Gist tokens](https://arxiv.org/abs/2304.08467): a long prompt compressed into a few learned tokens | Input | ✅ | ✅ | A trained compressor; information loss |
| Quantization | FP8 / [NVFP4](https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/) / [2-bit (KIVI)](https://arxiv.org/abs/2402.02750) | Engine | ✅ | ✅ | Accuracy loss; kernel support varies |
| Eviction | Keep attention sinks + a recent window ([StreamingLLM](https://arxiv.org/abs/2309.17453)), or heavy hitters ([H2O](https://arxiv.org/abs/2306.14048), [SnapKV](https://arxiv.org/abs/2404.14469)) | Engine | ✅ | ✅ | Evicted tokens never come back |
| Offload | Move KV to CPU memory or SSD, bring it back when needed (e.g. [LMCache](https://github.com/LMCache/LMCache)) | Engine | ✅ on GPU | ❌ adds PCIe traffic | Latency for capacity |

Two approaches that **do not** solve this problem:

- **Sparse attention** reads only the selected KV entries each step, but the unselected ones must still be
  stored. It saves bandwidth, not capacity — see the SparseServe case in Section 5.
- **Prefix reuse** ([prefix caching](https://docs.vllm.ai/en/latest/design/prefix_caching/), [RadixAttention](https://arxiv.org/abs/2312.07104))
  helps only when content repeats across requests. When one copy already does not fit, there is nothing to share.

### B. Without modifying the engine

**0. Turn on FP8 KV.** Half the bytes per token of BF16, so both capacity and bytes read are halved.
vLLM `--kv-cache-dtype fp8` ([`fp8` = `fp8_e4m3`](https://github.com/vllm-project/vllm/blob/main/vllm/config/cache.py));
SGLang `--kv-cache-dtype fp8_e4m3`.

**1. Give the KV cache more memory.** vLLM `--gpu-memory-utilization` (default 0.92);
SGLang `--mem-fraction-static` (the fraction for weights plus the KV pool). Too high risks OOM.

**2. Set the maximum length to real need.** `--max-model-len` rejects requests above the limit so they cannot
collapse concurrency. It does **not** shrink each request's KV — blocks are allocated on demand — it only
keeps the longest ones out. Send those to a separate deployment.

**3. Choose a model with small per-token KV.** GQA, MLA or hybrid attention at comparable quality. Only
possible at model-selection time.

**4. Shorten the input.** Retrieve only relevant passages, summarize old turns, compact agent history
periodically. This changes the application, not the server.

**5. Add a CPU / SSD tier.** Preempted or idle KV is moved off the GPU and brought back instead of
recomputed. ⚠️ It does not make a single running request fit: attention still needs that request's KV on the
GPU. What it saves is recomputation and prefix reuse. Letting running requests keep KV in CPU memory
requires sparse attention as well (the SparseServe approach).

**6. More tensor parallelism, or a larger GPU.** TP splits KV heads across GPUs, so each holds less. It stops
helping once TP exceeds the number of KV heads: vLLM computes `max(1, kv_heads // tp)` and replicates heads
beyond that ([`model.py`](https://github.com/vllm-project/vllm/blob/main/vllm/config/model.py)). Llama-3-70B has
8 KV heads, so TP=16 does not reduce per-GPU KV further than TP=8. The costs are communication and money.

### C. High exposure ≠ should change: three checks

| Check | Question | If it fails |
|---|---|---|
| ① Is a single copy really too large? | Is the same prefix stored many times? | Fix prefix reuse first — lossless and cheap. Come back if the cache is still full |
| ② Is the accuracy cost acceptable? | Quantization and input compression are all lossy | Evaluate offline on your own task |
| ③ Is there transfer bandwidth? | Offload depends on PCIe / network bandwidth | Offload only cold data, or quantize instead |

## 5. Public cases, by the decision they inform

Each case answers one question you will face in Sections 2 and 4. Numbers are as reported in the linked sources.

### "Is my ceiling estimate realistic, and will freeing capacity be enough?" — RetroInfer ([arXiv 2505.02922](https://arxiv.org/abs/2505.02922), §2)

- **Setting:** A100 80GB, Llama3-8B (the 1048K-context variant used throughout the paper's analysis), 128K context.
- **Observation:** maximum batch size 4 — beyond that, out of memory; and beyond 3, throughput gains become marginal because memory bandwidth saturates.
- **Use it when:** checking Step 1 and Step 4. The measured ceiling (4) is in line with the per-token arithmetic (Section 1's
  illustration gives 3), but bandwidth saturates one request earlier — so after freeing capacity, check `DRAMA` before expecting throughput to follow.

### "Will a model with smaller KV solve it?" — DeepSeek-V4 on vLLM ([blog](https://vllm.ai/blog/2026-04-24-deepseek-v4), [report](https://arxiv.org/abs/2606.19348)) and Together AI ([blog](https://www.together.ai/blog/serving-deepseek-v4-why-million-token-context-is-an-inference-systems-problem))

- **The model's part:** KV is compressed along the sequence. *c4a* merges 8 tokens into one entry with stride 4
  (~1/4); *c128a* merges 128 tokens with stride 128 (~1/128); a 128-token sliding window keeps local information.
  At 1M context with BF16 KV, vLLM estimates **9.62 GiB per sequence, about 8.7× smaller than the 83.9 GiB** of a
  61-layer DeepSeek-V3.2-style stack. The report estimates V4-Pro at 10% and V4-Flash at 7% of V3.2's KV cache at
  1M tokens, crediting the hybrid attention together with storage-precision optimizations — so it is not directly
  comparable to vLLM's BF16-only number.
- **The engine's part (vLLM):** V4 keeps several kinds of cache (compressed KV, sliding-window KV, indexer KV and
  compressor state) with different page sizes, and separate pools would fragment. vLLM fixes every compressed layer's logical block at 256 native token positions
  and fits the five-way cache stack into three page sizes, each backed by one shared pool.
- **The engine's part (Together AI):** their initial V4 path stored the full sliding-window state, about 3.8 KB
  per token against 3.4 KB on their V3 path. Keeping only the sliding-window states most likely to be reused raised
  total KV capacity on one NVIDIA HGX B200 node from roughly **1.2M to 3.7M tokens**, with the model unchanged.
- **Use it when:** choosing a model (Section 4B, item 3). A smaller KV on paper is potential; how much you get
  depends on how the engine stores, recomputes and evicts the different cache types. Judge it by the KV capacity
  your engine version actually reports (`X` in Step 1), not by the paper's ratio.

### "How much does quantizing the KV buy, and what must I evaluate?" — NVFP4 KV cache ([NVIDIA blog](https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/))

- **Mechanism:** KV stored in 4-bit, dequantized to FP8 before attention; new K and V are quantized on append.
- **Result:** up to 50% less memory than FP8 KV, effectively doubling the context budget; under 1% accuracy
  loss on LiveCodeBench, MMLU-PRO, MBPP and RULER 64K; up to 3× better TTFT and 20% higher cache-hit rate
  (Qwen3-Coder-480B-A35B), because the same memory holds more reusable KV.
- **Use it when:** FP8 KV is already on (Section 4B, item 0) and the cache is still full, on NVIDIA Blackwell GPUs —
  the format the blog describes targets Blackwell. The evaluation should include a long-context benchmark, as
  NVIDIA's did (RULER 64K, on Qwen3-480B-A35B against BF16 and FP8), not only short tasks.

### "Will sparse attention free capacity?" — SparseServe ([arXiv 2509.24626](https://arxiv.org/abs/2509.24626))

- **Problem:** dynamic sparse attention reads only selected KV blocks each step, so the bottleneck shifts
  from HBM bandwidth to HBM capacity — unselected KV must still stay in HBM, limiting batch size.
- **Fix:** an HBM–DRAM hierarchy. Moving small KV blocks with `cudaMemcpy` reaches under 4 GB/s on an A100 40GB
  (PCIe Gen4 D2H peak: 32 GB/s), so a fused GPU-direct loading kernel (FlashH2D) brings it above 20 GB/s.
- **Result:** built on vLLM; up to **9.26× lower mean TTFT** and **3.14× higher token throughput** than vanilla vLLM.
- **Use it when:** sparse attention is proposed as the fix for a full cache. On its own it saves bandwidth, not
  capacity; it only helps capacity when paired with moving unselected KV off the GPU, and then the transfer path
  becomes the problem to solve.

## 6. The same problem outside LLM serving

Cost that grows with the input against a fixed memory budget is an old problem. PagedAttention itself is
inspired by virtual memory and paging in operating systems ([Kwon et al., 2023](https://arxiv.org/abs/2309.06180)).

### Recommender systems (RecSys 2026)

Three industry papers from RecSys 2026 run into the same pattern: something grows without bound, and memory
or I/O does not. The LLM-side analogue in each heading is my mapping, not the authors'.

#### Compress the input: Token Factory, Google ([arXiv 2606.19635](https://arxiv.org/abs/2606.19635))

- **What grows:** the prompt of a large recommendation model (PLUM, built on Gemini). Textualized as Semantic
  IDs plus dense features, each watch-history item costs 12 tokens in the ranking baseline (8 for the Semantic ID,
  1 for the channel, 3 for dense features), so 200 items need a 1,536-token prompt, and older items are truncated.
- **How it fits:** small learned networks ("token makers") map all features of one item into a single soft token,
  so 200 items fit in a 480-token prompt. Longer histories can be compressed further, e.g. attention pooling over
  every *K* items into one token. Query-level soft tokens are shared across candidates and can be prefix-cached.
- **Result:** ranking AUC matches the baseline after about 1.5M steps, and training is about 200% faster with
  the prompt at ~30% of its original length. In retrieval (768 → 256 tokens), offline Recall@10 +2.0% and online
  unique impressions +16.8%.
- **LLM analogue:** gist tokens, and sequence-direction compression like DeepSeek-V4, which also merges every
  *K* positions into one entry.

#### Quantize what you store: Dual-purpose Semantic IDs, YouTube / Google DeepMind ([arXiv 2607.24865](https://arxiv.org/abs/2607.24865))

- **What grows:** dense content embeddings attached to every item in a user's history. At 200 items × 256
  dimensions that is 51,200 floats (200 KB in FP32) per training example, logged, stored and joined across
  billions of examples.
- **How it fits:** each embedding is quantized into a *K*-token Semantic ID (*K* × log₂*V* bits instead of
  *d* × 32), typically 50–100× smaller. The same IDs act as a learned identity and are decoded back into an
  approximate embedding inside the model graph, so dense vectors no longer have to be logged or joined.
- **Result:** ingesting raw 64-dimensional embeddings cut training throughput by 28.2% (16.80 → 12.07 steps/s);
  decoding Semantic IDs recovered it to 15.41 steps/s, and a larger codebook raised Hit Rate@100 above the
  raw-embedding arm (0.2870 vs 0.2844). Deployed in YouTube ranking and retrieval, with online satisfied engagement
  +0.06% to +0.09% sitewide.
- **LLM analogue:** KV quantization — store compact codes and reconstruct on demand.

#### Evict under a fixed budget: MPZCH, Meta ([arXiv 2602.17050](https://arxiv.org/abs/2602.17050))

- **What grows:** the ID space of embedding tables, which can reach tens of billions of rows. Tables cannot grow
  indefinitely, so IDs are hashed into a fixed-size table, and colliding IDs share a row.
- **How it fits:** linear probing plus an identities tensor recording which ID owns each slot. Eviction is lazy:
  only when a new ID needs a slot, it takes the first expired slot in its probe range (TTL policy), or the
  oldest one (LRU). The evicted row's embedding *and* optimizer state are reset, so the new ID does not inherit a
  stale representation.
- **Result:** with 150M IDs, a 200M-row table (1.33×) and probe depth 256 reach zero collisions, where the
  baseline hash still collides on 29.6%. A 4-billion-row user table for a model serving ~3 billion monthly users
  improved Normalized Entropy on 14 of 17 tasks. For item tables (24 h TTL for posts, 72 h for owners), new videos
  got 0.83% more impressions.
- **LLM analogue:** KV eviction under a fixed budget, and PagedAttention's block table — an indirection that
  decides who owns which slot.

## References

**Papers**
- Kwon et al., [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) (SOSP 2023)
- Chen et al., [RetroInfer: A Vector Storage Engine for Scalable Long-Context LLM Inference](https://arxiv.org/abs/2505.02922) (2025)
- Zhou et al., [SparseServe: Unlocking Parallelism for Dynamic Sparse Attention in Long-Context LLM Serving](https://arxiv.org/abs/2509.24626) (2025)
- DeepSeek-AI, [DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence](https://arxiv.org/abs/2606.19348) (2026)
- DeepSeek-AI, [DeepSeek-V2](https://arxiv.org/abs/2405.04434) (2024) — MLA
- Shazeer, [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) (2019) — MQA
- Ainslie et al., [GQA](https://arxiv.org/abs/2305.13245) (2023)
- Mu et al., [Learning to Compress Prompts with Gist Tokens](https://arxiv.org/abs/2304.08467) (2023)
- Liu et al., [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750) (2024)
- Xiao et al., [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453) (2023)
- Zhang et al., [H2O: Heavy-Hitter Oracle for Efficient Generative Inference of LLMs](https://arxiv.org/abs/2306.14048) (2023)
- Li et al., [SnapKV: LLM Knows What You are Looking for Before Generation](https://arxiv.org/abs/2404.14469) (2024)
- Zheng et al., [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104) (2023)

**Engineering blogs and guides**
- NVIDIA, [DCGM Feature Overview — profiling metrics](https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/feature-overview.html)
- vLLM, [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html) (2023)
- vLLM, [DeepSeek V4 in vLLM: Efficient Long-context Attention](https://vllm.ai/blog/2026-04-24-deepseek-v4) (2026)
- Together AI, [Serving DeepSeek-V4: why million-token context is an inference systems problem](https://www.together.ai/blog/serving-deepseek-v4-why-million-token-context-is-an-inference-systems-problem) (2026)
- Alvarez, Chen and Mao (NVIDIA), [Optimizing Inference for Long Context and Large Batch Sizes with NVFP4 KV Cache](https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/) (2025)
- Whyte-Gray, Bathusha, Goin and Kamra (Red Hat), [5 steps to triage vLLM performance](https://developers.redhat.com/articles/2026/03/09/5-steps-triage-vllm-performance) (2026)

**Source code and issues**
- vLLM: [`metrics/loggers.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/metrics/loggers.py), [`metrics/stats.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/metrics/stats.py), [`core/block_pool.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/block_pool.py), [`core/kv_cache_utils.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_utils.py), [`config/cache.py`](https://github.com/vllm-project/vllm/blob/main/vllm/config/cache.py), [`config/model.py`](https://github.com/vllm-project/vllm/blob/main/vllm/config/model.py)
- [LMCache](https://github.com/LMCache/LMCache)

**RecSys 2026**
- Chen et al., [Token Factory: Efficiently Integrating Diverse Signals into Large Recommendation Models](https://arxiv.org/abs/2606.19635) (RecSys 2026, [DOI](https://doi.org/10.1145/3773078.3831865))
- Li et al., [Tokens are All You Need: Dual-purpose Semantic IDs for Achieving LLM-Level I/O Efficiency in Recommendation Systems](https://arxiv.org/abs/2607.24865) (RecSys 2026, [DOI](https://doi.org/10.1145/3773078.3831900))
- Zhao et al., [Multi-Probe Zero Collision Hash (MPZCH): Mitigating Embedding Collisions and Enhancing Model Freshness in Large-Scale Recommenders](https://arxiv.org/abs/2602.17050) (RecSys 2026, [DOI](https://doi.org/10.1145/3773078.3831866))

Numbers in Sections 5 and 6 are quoted from the linked sources; arithmetic marked illustrative is mine, not a measurement.
