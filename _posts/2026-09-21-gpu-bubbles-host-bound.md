---
title: "GPU Bubbles Without Slow Kernels: Diagnosing Host-Bound LLM Serving"
date: 2026-09-21 10:00:00 -0700
categories: [LLM Serving, Host Overhead]
tags: [ml-systems, serving, profiling, cuda-graph, nsys, dcgm, sglang, vllm]
pin: true
description: The GPU is idle, yet compute, bandwidth, interconnect and capacity are all unsaturated. A six-step diagnosis, the workloads most exposed to it, what engines and operators can each do, and four cases from sglang-omni.
---

**Symptom:** doubling concurrency barely moves throughput, while latency nearly doubles.
**Reality:** the kernels are not slow. The gaps between them are — the GPU is waiting on the host.

> "Flat throughput, rising latency" is the shared symptom of almost every serving bottleneck
> (KV capacity, HBM bandwidth and interconnect all look the same), so it does not discriminate.
> The evidence that does: **no device resource is saturated, and the GPU is still idle.**
{: .prompt-info }

## 1. Profiling: a six-step diagnosis

**Measure near the saturation point.** At low load an idle GPU is expected — it simply has not been
given enough work, and utilization numbers taken there mean nothing. Sweep concurrency first and
find the point where throughput stops increasing.

### Step 1 — Confirm the GPU is idle, then rule out all four device resources

- Low **GRACT** ([DCGM](https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/feature-overview.html) field 1001, graphics engine active) is the positive evidence that the GPU is idle.
- Then rule out each resource class:

  | Resource | Signal |
  |---|---|
  | Compute | `TENSO` (1004, tensor pipe active) |
  | Memory bandwidth | `DRAMA` (1005, DRAM active) |
  | Interconnect | NVLink TX/RX (1011, 1012) |
  | Capacity | ⚠️ **Not visible in DCGM.** The KV cache pool is preallocated at startup, so memory usage is flat. Use the engine's own [metrics](https://docs.vllm.ai/en/latest/design/metrics/): `vllm:kv_cache_usage_perc` and `vllm:num_preemptions_total` |

- All four low, with low GRACT → the bottleneck is **off-device**. Step 3 decides whether that
  means the host cannot keep up, or there is simply no load reaching the engine.

### Step 2 — Quantify the bubbles on an Nsight Systems timeline

Take cumulative kernel time on the GPU row divided by wall time, and cross-check it against the DCGM
utilization. Two unrelated instruments agreeing on the order of magnitude is what makes the bubble real.

### Step 3 — Align the GPU row with the CUDA API row directly above it

A bubble only says the GPU is idle. What the host is doing during that interval says why.
There are four possible answers:

```
CUDA API  |  cudaLaunchKernel × 200 ... |cudaStreamSync————|  ...
          └──────────── host activity in this interval ─────────┘
                              ↕  aligned on the time axis
GPU       |██|██|██|██|██|██| |                          | |██|
          └──── kernels ────┘   └──────── bubble ─────────┘
```

| Above the bubble | Diagnosis |
|---|---|
| (1) A long run of `cudaLaunchKernel` | **Launch-bound**: the host issues kernels more slowly than the GPU executes them |
| (2) A long `cudaStreamSynchronize` or D2H copy | **Synchronization point**: the host is blocked waiting for a result |
| (3) An empty CUDA API row, but a busy Python thread | **Host overhead**: scheduling, sampling, detokenization |
| (4) Nothing, and the host is idle too | **Insufficient load**: fix the benchmark, not the engine |

### Step 4 — Attribute to code

- **Launch-bound** → which region is kernel-dense, and can it be captured in a CUDA Graph?
- **Synchronization point** → which line triggers it? Use the PyTorch profiler to map ATen ops back to Python
  source lines. This is where it's indispensable.
  - vLLM: enable the torch profiler as described in the
    [profiling docs](https://docs.vllm.ai/en/latest/contributing/profiling/), then call
    `/start_profile` and `/stop_profile`
  - SGLang: set `SGLANG_TORCH_PROFILER_DIR`, then call `/start_profile` and `/stop_profile`
- **Host overhead** → which Python code path dominates?
- **Insufficient load** → skip this step.

### Step 5 — Fix synchronization before launch overhead

A single synchronization point inside a region prevents that region from being captured. When both are
present, this is an ordering, not a choice.

### Step 6 — Validate by ablation

Change one thing, re-measure, and check whether the bubble is gone. **If it is not, the attribution
was wrong — go back to Step 3.**

## 2. Workloads most exposed to GPU bubbles

Each of these either shortens the GPU side of a step or adds work on the host side:

- **Small models (≲8B):** fewer weights make each GPU step short, while the host's per-step work
  (scheduling, sampling, launching kernels) stays the same. The ratio flips and the host becomes the bottleneck.
- **Short outputs (tens to ~100 tokens):** a request is only a few decode steps, each with little GPU work,
  so per-step host cost dominates. A larger batch amortizes it — as long as the host's per-step work does
  not grow with the batch. MOSS-TTS in Section 4 (89 tokens per request) is the counterexample: its
  per-step sampling ran per request, so doubling concurrency added only 3.83% throughput.
- **Multi-stage pipelines** (TTS: text encoder → AR → vocoder; omni models): stages are orchestrated by
  the host, and every stage boundary is a synchronization point by construction.
- **Decoding logic that needs a host-side decision every step:** when the next step depends on a value the
  GPU just produced and the logic consuming it runs on the host, that value is copied back every step —
  one synchronization per step, and often a shape that changes from step to step, which blocks CUDA Graph
  capture.
  - *Example — speculative decoding:* each step verifies k draft tokens, and the number accepted differs
    per request. Acceptance, KV cache rollback and sequence bookkeeping have traditionally run on the host,
    so the step waits on a D2H copy and the next batch has a data-dependent shape. Engines work around it by
    padding every request to k + 1 slots and masking rejected tokens (static shapes, so graphs still apply,
    at the cost of computing tokens that get thrown away), and by moving acceptance onto the GPU.
- **Low concurrency / batch size 1:** this is often a property of the deployment rather than a bug — the fix is more concurrency, not new code.
- **High tensor-parallel degree:** TP divides each GPU's compute by N but not the host work — kernel
  launches per step stay the same and NCCL calls are added. All ranks synchronize every step, so a hiccup
  on one host thread stalls all N GPUs.
- **High QPS with very short outputs:** per-request host cost dominates — see below.

### Short outputs vs. high QPS with very short outputs

The two look alike but have different bottlenecks. Host cost per request splits into two terms:

```
host cost per request = C_request + C_step × steps / batch
  C_request : HTTP parsing, tokenization, admission and retirement — once per request, not shared
  C_step    : scheduling, input preparation, kernel launches — once per step, shared by the batch
```

| | Short outputs | High QPS + very short outputs (1–5 tokens) |
|---|---|---|
| Dominant term | `C_step`: a few steps, each light on the GPU | `C_request`: requests arrive faster than the host can process them |
| Does a larger batch help? | Yes, if per-step host work is fixed rather than per request — short requests finish together with no long tail, which is ideal for batching | Barely — batching amortizes only per-step cost, and there are hardly any steps |
| What helps | Larger batches, CUDA Graphs | Take request handling off the engine loop (process isolation), then parallelize it (more API server processes, or more replicas) |
| Typical workloads | Short generations, TTS utterances | Classification, moderation, reranking |

## 3. What engines already do, and what is left to the operator

### A. Engine-side optimizations

| Technique | Mechanism | vLLM | SGLang |
|---|---|---|---|
| CUDA Graphs | Hundreds of `cudaLaunchKernel` calls replaced by one `cudaGraphLaunch` | On by default; piecewise and full modes. [Qwen2.5-0.5B](https://www.linkedin.com/pulse/efficiently-serving-llms-part-4-how-cuda-graphs-make-vllm-thomas-4ofuc): +13% throughput, −17.3% ITL (combined with async scheduling) | On by default for decode, captured per batch size |
| Overlap scheduling | The host prepares batch N+1 while the GPU runs batch N | [Multi-step scheduling](https://blog.vllm.ai/2024/09/05/perf-update.html) +28% (2024), later replaced by async scheduling | [Zero-overhead batch scheduler](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/) 1.1× (2024), "most significant for small models and large TP" |
| Asynchronous output processing | Detokenization and response assembly overlap with the next forward pass | [v0.6.0](https://blog.vllm.ai/2024/09/05/perf-update.html): −8.7% TPOT | Covered by process isolation (next row) |
| Process isolation | Tokenization, detokenization and HTTP moved out of the engine loop | [V1 `EngineCore`](https://vllm.ai/blog/2025-01-27-v1-alpha-release) 1.7× (2025; motivation: an 8B decode step down to ~5 ms) | Built in: tokenizer, scheduler and [detokenizer](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/detokenizer_manager.py) run as separate processes |
| Reduced Python overhead | Object pooling | [v0.6.0](https://blog.vllm.ai/2024/09/05/perf-update.html): +24% | — |
| Input preparation on the GPU | Triton kernels for input preparation, targeting zero CPU–GPU synchronization | [Model Runner V2](https://vllm.ai/blog/2026-03-24-mrv2) (2026): Qwen3-0.6B on GB200 +56% | — |


### B. Without modifying the engine

**0. Confirm the relevant features are actually on.** Built into the engine does not mean enabled in
your deployment. Overlap scheduling is on by default in both engines, but a flag can switch it off
(vLLM `--no-async-scheduling`, SGLang `--disable-overlap-schedule`), and an engine may turn it off for
configurations it does not support. Check the effective configuration at startup.

**1. Verify that CUDA Graphs cover your traffic.** This is the most common silent failure. Graphs are
captured per batch-size bucket, and a batch outside the captured sizes falls back to eager entirely.

**2. Increase batch size.** The most direct way to lengthen the GPU side of each step: vLLM
`--max-num-seqs` / `--max-num-batched-tokens`, SGLang `--max-running-requests`. Keep the CUDA Graph
capture range in step with it (item 1). Speculative decoding also raises tokens per step (`T = B × (k+1)`)
without adding sequences.

**3. Upgrade the engine.** Each generation above removes a different source of bubbles; running a few
versions behind means missing entire layers of optimization.

**4. Prefer more replicas over higher TP.** TP divides GPU work by N but not host work — the number of
kernel launches is unchanged, and NCCL calls are added. This requires the model to fit on fewer GPUs,
at the cost of higher single-request latency and N copies of the weights.

**5. Cut per-step and per-request host work.** Avoid streaming where it is not needed; use structured
output sparingly. ⚠️ Both engines already run detokenization and HTTP outside the engine loop (vLLM V1's
`EngineCore`, SGLang's separate tokenizer and detokenizer processes), so the gains here are much smaller than
they used to be.

## 4. Cases from sglang-omni

[sglang-omni](https://github.com/sgl-project/sglang-omni) serves multi-stage TTS and omni models, and it
matches nearly every high-risk trait in Section 2: small models, short outputs, multiple stages and
per-step host decisions. The cases below are public issues and PRs from the repository: first an
investigation that walks through all six steps, then four PRs, each an example of one Step 3 outcome.
Each case links to its source, where you can follow how the cause was found, how it was fixed and what
the fix measured.

### A full pass: MOSS-TTS Delay ([#1232](https://github.com/sgl-project/sglang-omni/issues/1232))

| Step | Observation                                                                                                                                                                                                                                                                                                                      |
|---|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Saturation | c16 → c32: throughput +3.83%, mean latency +92.24%, output length constant at 89 tokens                                                                                                                                                                                                                                          |
| Step 1 | DCGM at c16: `SMACT 34.26%`, `SMOCC 4.72%`, `DRAMA 21.70%`, `TENSO 2.04%`; 291.5 W of 700 W, no clock throttling → no resource saturated.  GRACT was not collected at the time; `SMACT` was used instead, and it is a different quantity (averaged over SMs). |
| Step 2 | Per decode cycle (nsys): graphed backbone 12.7%, non-graph GPU activity 21.5%, **no GPU activity 65.9%**. Total GPU activity of 34.2% (**c32 trace**) **agrees to the order of magnitude** with `SMACT`'s 34.26% (**c16 counters**) — different operating points, different definitions, mechanically unrelated instruments. Order-of-magnitude agreement is the useful signal here, not the decimals (see the note in Step 1)                                                                                                                                            |
| Step 3 | Per decode step: **~2,954 `cudaLaunchKernel` calls against a single `cudaGraphLaunch`**; synchronization APIs account for only 2.47% of idle time → launch-bound, plus host-side tensor materialization                                                                                                                          |
| Step 4 | The backbone is captured, but the per-step sampling and feedback chain runs eagerly outside the graph, with `any()`, `.item()` and `nonzero()` on the hot path                                                                                                                                                                   |
| Step 5 | Remove the data-dependent synchronizations first (small in time, but they block capture), then capture the sampling chain per batch bucket and sampling signature                                                                                                                                                                |

### (1) Launch-bound: the Qwen3-TTS code predictor ([#1134](https://github.com/sgl-project/sglang-omni/pull/1134))

- **Symptom:** 67.7% of the serving loop spent in `_collect_codes`; throughput *decreased* with
  concurrency (3.4 QPS at c8 → 2.8 at c32).
- **Root cause:** the talker backbone was graphed, but after each semantic token the code predictor ran
  `num_code_groups - 1` serial iterations of lm_head, sampling, embedding and predictor layers in eager
  mode — thousands of Python-level kernel launches per step.
- **Fix:** capture the whole predictor chain as one CUDA Graph per (batch bucket, sampling signature),
  with `top_k` quantized to a fixed ladder so requests share graphs.
- **Result:** on one H100, **~1.9×** at c8 (graph-on/off arm means: 4.56 vs 2.39 req/s),
  **~3×** at c32 (5.94 vs 2.00 req/s, **a single A/B pair**); 256 SeedTTS EN samples per leg.
- **Lesson:** Section 3, item 1. A captured backbone does not mean a captured step.

### (2) Synchronization point: three D2H copies per step in Higgs TTS ([#564](https://github.com/sgl-project/sglang-omni/issues/564) → [#572](https://github.com/sgl-project/sglang-omni/pull/572))

- **Hypothesis (#564):** three `.cpu()` calls per decode step — three `cudaStreamSynchronize` calls —
  estimated at 5–15% of end-to-end latency.
- **Fix:** coalesce them into a single D2H copy, with byte-identical output.
- **Step 6 ablation:** two matched A/Bs on H200 at c1 were within noise (all gains below 1%).
  Per-call timing showed why:

  ```
  decode step ~4.88 ms: GPU forward ~3.72 ms, host serialization gap ~1.10 ms

  call 1  _cg_was_done               3713 µs  ← waiting for this step's forward pass
  call 2  _cg_codes_BN                 31 µs
  call 3  _cg_active_generation_done   16 µs
  ```

  Only the first call actually blocks, and what it waits for is compute that has to happen anyway.
  The two removed calls were worth 45 µs — **a 0.9% ceiling**.
- **Lesson:** the cost of synchronization depends on what it waits for, not how many times it happens.

### (3) Host overhead: reference-audio encoding on the CPU in MOSS-TTS Delay ([#1222](https://github.com/sgl-project/sglang-omni/pull/1222))

- **Symptom:** under concurrency, the preprocessing stage encoded reference audio on the CPU and could
  not feed the AR stage fast enough.
- **Fix:** decouple the codec from the processor and run the preprocessing codec on the GPU.
- **Result:** A800 c16 **2.9958 → 4.4395 QPS** (+48.19%), three runs per group with 1088 requests
  per run and preprocessing concurrency 8. GPU memory increased by approximately 3.1 GiB.
- **Lesson:** Section 2, multi-stage pipelines. CPU preprocessing can leave the next GPU stage waiting.

### (4) Insufficient load: the default admission cap in Higgs TTS ([#756](https://github.com/sgl-project/sglang-omni/pull/756))

- **Symptom:** in the [independent H100 cross-check](https://github.com/sgl-project/sglang-omni/pull/756#issuecomment-4759834453),
  `16/16` at c16 → c32 gives 15.23 → 15.28 req/s (**+0.33%**), while mean RTF rises
  0.2489 → 0.5167 (about 2.08×) — the same plateau symptom that can suggest host overhead.
- **Root cause:** the default `max_running_requests / cuda_graph_max_bs` was `16/16`. The client sent 32
  concurrent requests, but the engine admitted only 16 and the rest queued. The GPU was not starved by
  the host; the work never reached it.
- **Fix:** raise the default to `64/64`, and size model-side buffers from `max_running_requests`
  (they were sized from a stale constant, which crashed capture at 128).
- **Result:** lifting the cap to **`32/32` alone** gives **+38.3% at c32 on H200**
  (15.377 → 21.261 req/s, RTF −28.8%, no WER regression observed in the checked runs).
  `64/64` became the default because it extends the plateau to higher concurrency —
  **26.33 vs 20.44 req/s at c64** for `64/64` vs `32/32` in the independent H100 cross-check.
  It did not win at every point: at c32 on H100, `64/64` gave 20.64 vs 21.30 req/s for `32/32`.
- **Lesson:** before suspecting the host, check that the engine's `#running-req` actually reaches the
  intended concurrency.

## Takeaway

> Low GPU utilization is not a slow GPU — align the bubble with the host timeline to see what it is waiting for.
{: .prompt-tip }

## References

**Engines and tooling**
- NVIDIA, [DCGM Feature Overview — profiling metrics](https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/feature-overview.html)
- vLLM, [Metrics](https://docs.vllm.ai/en/latest/design/metrics/), [Engine Arguments](https://docs.vllm.ai/en/stable/configuration/engine_args/) and [Profiling](https://docs.vllm.ai/en/latest/contributing/profiling/)
- SGLang, [Production Metrics](https://docs.sglang.io/docs/references/production_metrics)
- vLLM, [vLLM v0.6.0: 2.7x Throughput Improvement and 5x Latency Reduction](https://blog.vllm.ai/2024/09/05/perf-update.html) (2024)
- vLLM, [vLLM V1: A Major Upgrade to vLLM's Core Architecture](https://vllm.ai/blog/2025-01-27-v1-alpha-release) (2025)
- vLLM, [Model Runner V2](https://vllm.ai/blog/2026-03-24-mrv2) (2026)
- LMSYS, [SGLang v0.4: Zero-Overhead Batch Scheduler](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/) (2024)
- Elizabeth Thomas, [Efficiently Serving LLMs (Part 4): How CUDA Graphs make vLLM think faster](https://www.linkedin.com/pulse/efficiently-serving-llms-part-4-how-cuda-graphs-make-vllm-thomas-4ofuc) (2025)
- Modal, [Host overhead is killing your inference efficiency](https://modal.com/blog/host-overhead-inference-efficiency)

**sglang-omni cases**
- [#1232](https://github.com/sgl-project/sglang-omni/issues/1232) Profile GPU-side bottlenecks in MOSS-TTS-v1.5 Delay
- [#756](https://github.com/sgl-project/sglang-omni/pull/756) Raise Higgs TTS AR server default to 64
- [#1134](https://github.com/sgl-project/sglang-omni/pull/1134) CUDA-graph the Qwen3-TTS code-predictor chain
- [#564](https://github.com/sgl-project/sglang-omni/issues/564) / [#572](https://github.com/sgl-project/sglang-omni/pull/572) Batch the per-step D2H syncs in Higgs TTS
- [#1222](https://github.com/sgl-project/sglang-omni/pull/1222) Run MOSS-TTS Delay reference encoding on GPU

Numbers in Section 4 are quoted from these PRs and issues; figures in Section 3 are as reported by the linked sources.
