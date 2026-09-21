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

- All four low, with low GRACT → the bottleneck is off-device: **host-bound**.

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
  source lines. This is the only branch that needs it.
- **Host overhead** → which Python code path dominates?
- **Insufficient load** → skip this step.

### Step 5 — Fix synchronization before launch overhead

A single synchronization point inside a region prevents that region from being captured. When both are
present, this is an ordering, not a choice.

### Step 6 — Validate by ablation

Change one thing, re-measure, and check whether the bubble is gone. **If it is not, the attribution
was wrong — go back to Step 3.**

## 2. Workloads most exposed to GPU bubbles

- **Small models (≲8B)**
- **Short outputs** (tens to ~100 tokens): per-request fixed cost is barely amortized
- **Multi-stage pipelines** (TTS: text encoder → AR → vocoder; omni models): stages are orchestrated
  by the host, and stage boundaries introduce synchronization points by construction
- **Decoding logic that needs a host-side decision every step**
- **Low concurrency / batch size 1**
- **High tensor-parallel degree**
- **High QPS with very short outputs**

The last one deserves a closer look. Per-request host cost (HTTP parsing, tokenization, scheduling
admission and retirement) is the same for every request. What changes is how many tokens it is
amortized over:

```
1000 output tokens → cost spread over 1000 tokens → negligible per token
   3 output tokens → cost spread over    3 tokens → 333× higher per token
```

The absolute cost is unchanged; the denominator collapsed.

## 3. What engines already do, and what is left to the operator

### A. Engine-side optimizations

| Year | Technique | Mechanism | Reported gain |
|---|---|---|---|
| 2023 | CUDA Graphs | Hundreds of `cudaLaunchKernel` calls replaced by one `cudaGraphLaunch` | [llama.cpp](https://developer.nvidia.com/blog/optimizing-llama-cpp-ai-inference-with-cuda-graphs/) (batch 1) 1.2×; [Qwen2.5-0.5B on vLLM](https://www.linkedin.com/pulse/efficiently-serving-llms-part-4-how-cuda-graphs-make-vllm-thomas-4ofuc) +13% throughput, −17.3% ITL (combined with async scheduling) |
| 2024 | Overlap scheduling | The host prepares batch N+1 while the GPU runs batch N | [SGLang zero-overhead scheduler](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/) 1.1× ("most significant for small models and large TP"); [vLLM multi-step scheduling](https://blog.vllm.ai/2024/09/05/perf-update.html) +28% |
| 2024 | Asynchronous output processing | Detokenization and response assembly overlap with the next forward pass | [vLLM v0.6.0](https://blog.vllm.ai/2024/09/05/perf-update.html) −8.7% TPOT |
| 2024 | Reduced Python overhead | Object pooling | [vLLM v0.6.0](https://blog.vllm.ai/2024/09/05/perf-update.html) +24% |
| 2025 | Process isolation | Tokenization, detokenization and HTTP moved out of the engine loop | [vLLM V1 `EngineCore`](https://vllm.ai/blog/2025-01-27-v1-alpha-release) 1.7× (motivation: an 8B decode step down to ~5 ms) |
| 2026 | Input preparation on the GPU | Triton kernels for input preparation, targeting zero CPU–GPU synchronization | [vLLM Model Runner V2](https://vllm.ai/blog/2026-03-24-mrv2): Qwen3-0.6B on GB200 +56% |

### B. Without modifying the engine

**0. Confirm the overlap features are actually enabled.** Built into the engine does not mean enabled
in your deployment. vLLM enables async scheduling by default from v0.14 (`--async-scheduling` on older
versions; see [engine arguments](https://docs.vllm.ai/en/stable/configuration/engine_args/)); SGLang's overlap scheduler is on by default — make sure `--disable-overlap-schedule` is not set.

**1. Verify that CUDA Graphs cover your traffic.** This is the most common silent failure. Graphs are
captured per batch-size bucket, and a batch outside the captured sizes falls back to eager entirely.

- Does the real batch-size distribution fall inside the captured sizes? (SGLang `--cuda-graph-max-bs`;
  vLLM `--cudagraph-capture-sizes` and the startup log)
- Is it running full or piecewise graphs? Full graphs require attention-backend support and fall back
  silently when it is missing.
- vLLM `--enforce-eager` disables CUDA Graphs completely.

**2. Increase batch size.** The most direct way to lengthen the GPU side of each step. Speculative
decoding also raises tokens per step (`T = B × (k+1)`) without adding sequences.

**3. Upgrade the engine.** Each generation above removes a different source of bubbles; running a few
versions behind means missing entire layers of optimization.

**4. Prefer more replicas over higher TP.** TP divides GPU work by N but not host work — the number of
kernel launches is unchanged, and NCCL calls are added. This requires the model to fit on fewer GPUs,
at the cost of higher single-request latency and N copies of the weights.

**5. Cut per-step and per-request host work.** Avoid streaming where it is not needed; use structured
output sparingly. ⚠️ Since vLLM V1's process isolation, the gains here are much smaller.

## 4. Cases from sglang-omni

[sglang-omni](https://github.com/sgl-project/sglang-omni) serves multi-stage TTS and omni models, and it
matches nearly every high-risk trait in Section 2: small models, short outputs, multiple stages and
per-step host decisions. Below is one full pass through the six steps, followed by one case for each
Step 3 outcome.

### A full pass: MOSS-TTS Delay ([#1232](https://github.com/sgl-project/sglang-omni/issues/1232), my investigation)

| Step | Observation |
|---|---|
| Saturation | c16 → c32: throughput +3.83%, mean latency +92.24%, output length constant at 89 tokens |
| Step 1 | DCGM at c16: `SMACT 34.26%`, `SMOCC 4.72%`, `DRAMA 21.70%`, `TENSO 2.04%`; 291.5 W of 700 W, no clock throttling → no resource saturated |
| Step 2 | Per decode cycle (nsys): graphed backbone 12.7%, non-graph GPU activity 21.5%, **no GPU activity 65.9%**. Total GPU activity of 34.2% matches DCGM's 34.26% |
| Step 3 | Per decode step: **~2,954 `cudaLaunchKernel` calls against a single `cudaGraphLaunch`**; synchronization APIs account for only 2.47% of idle time → launch-bound, plus host-side tensor materialization |
| Step 4 | The backbone is captured, but the per-step sampling and feedback chain runs eagerly outside the graph, with `any()`, `.item()` and `nonzero()` on the hot path |
| Step 5 | Remove the data-dependent synchronizations first (small in time, but they block capture), then capture the sampling chain per batch bucket and sampling signature |

A by-product: traces collected through `/start_profile` contained **zero CPU operator events**. Kineto's
CPU callbacks are thread-local, and the profiler was started on a different thread from the one running
the model. Fixed in [#1304](https://github.com/sgl-project/sglang-omni/pull/1304) (ATen events: 0 → ~494k).
**Validate the instrument before trusting Step 3.**

### (1) Launch-bound: the Qwen3-TTS code predictor ([#1134](https://github.com/sgl-project/sglang-omni/pull/1134))

- **Symptom:** 67.7% of the serving loop spent in `_collect_codes`; throughput *decreased* with
  concurrency (3.4 QPS at c8 → 2.8 at c32).
- **Root cause:** the talker backbone was graphed, but after each semantic token the code predictor ran
  `num_code_groups - 1` serial iterations of lm_head, sampling, embedding and predictor layers in eager
  mode — thousands of Python-level kernel launches per step.
- **Fix:** capture the whole predictor chain as one CUDA Graph per (batch bucket, sampling signature),
  with `top_k` quantized to a fixed ladder so requests share graphs.
- **Result:** **~1.9×** at c8, **~3×** at c32.
- **Lesson:** Section 3, item 1. A captured backbone does not mean a captured step.

### (2) Synchronization point: three D2H copies per step in Higgs TTS ([#564](https://github.com/sgl-project/sglang-omni/issues/564) → [#572](https://github.com/sgl-project/sglang-omni/pull/572))

- **Hypothesis (#564):** three `.cpu()` calls per decode step — three `cudaStreamSynchronize` calls —
  estimated at 5–15% of end-to-end latency.
- **Fix:** coalesce them into a single D2H copy, with byte-identical output.
- **Step 6 ablation:** the A/B result was within noise. Per-call timing showed why:

  ```
  decode step 4.88 ms = GPU forward 3.72 ms (76%) + host serialization gap 1.10 ms (24%)

  call 1  _cg_was_done               3713 µs  ← waiting for this step's forward pass
  call 2  _cg_codes_BN                 31 µs
  call 3  _cg_active_generation_done   16 µs
  ```

  Only the first call actually blocks, and what it waits for is compute that has to happen anyway.
  The two removed calls were worth 45 µs — **a 0.9% ceiling**.
- **Lesson:** the cost of synchronization depends on what it waits for, not how many times it happens.
  The recoverable time is the 1.1 ms host gap, and the fix for that is overlap scheduling (Higgs had
  `disable_overlap_schedule=True`). Coalescing is a prerequisite for enabling overlap, not a gain on its own.

### (3) Host overhead: reference-audio encoding on the CPU in MOSS-TTS Delay ([#1222](https://github.com/sgl-project/sglang-omni/pull/1222))

- **Symptom:** under concurrency, the preprocessing stage encoded reference audio on the CPU and could
  not feed the AR stage fast enough.
- **Fix:** decouple the codec from the processor and run the preprocessing codec on the GPU.
- **Result:** A800 c16 **2.996 → 4.440 QPS** (+48%). My independent H100 measurement (#1232):
  non-AR time per request **2.33 s → 0.41 s (−82.6%)**.
- **Lesson:** Section 2, multi-stage pipelines. Host work at a stage boundary issues no CUDA calls, so on
  the timeline it appears as a bubble with an empty API row.

### (4) Insufficient load: the default admission cap in Higgs TTS ([#756](https://github.com/sgl-project/sglang-omni/pull/756), my PR)

- **Symptom:** c16 → c32 throughput +1.8%, latency doubled — indistinguishable from a host-bound symptom.
- **Root cause:** the default `max_running_requests / cuda_graph_max_bs` was `16/16`. The client sent 32
  concurrent requests, but the engine admitted only 16 and the rest queued. The GPU was not starved by
  the host; the work never reached it.
- **Fix:** raise the default to `64/64`, and size model-side buffers from `max_running_requests`
  (they were sized from a stale constant, which crashed capture at 128).
- **Result:** c32 throughput 15.377 → 21.261 req/s (**+38.3%**), RTF −28.8%, no WER regression.
- **Lesson:** before suspecting the host, check that the engine's `#running-req` actually reaches the
  intended concurrency. At c128 it peaked at 78.

## Takeaway

> Low GPU utilization is not a slow GPU — align the bubble with the host timeline to see what it is waiting for.
{: .prompt-tip }

## References

**Engines and tooling**
- NVIDIA, [DCGM Feature Overview — profiling metrics](https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/feature-overview.html)
- vLLM, [Metrics](https://docs.vllm.ai/en/latest/design/metrics/) and [Engine Arguments](https://docs.vllm.ai/en/stable/configuration/engine_args/)
- vLLM, [vLLM v0.6.0: 2.7x Throughput Improvement and 5x Latency Reduction](https://blog.vllm.ai/2024/09/05/perf-update.html) (2024)
- vLLM, [vLLM V1: A Major Upgrade to vLLM's Core Architecture](https://vllm.ai/blog/2025-01-27-v1-alpha-release) (2025)
- vLLM, [Model Runner V2](https://vllm.ai/blog/2026-03-24-mrv2) (2026)
- LMSYS, [SGLang v0.4: Zero-Overhead Batch Scheduler](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/) (2024)
- NVIDIA, [Optimizing llama.cpp AI Inference with CUDA Graphs](https://developer.nvidia.com/blog/optimizing-llama-cpp-ai-inference-with-cuda-graphs/) (2024)
- Elizabeth Thomas, [Efficiently Serving LLMs (Part 4): How CUDA Graphs make vLLM think faster](https://www.linkedin.com/pulse/efficiently-serving-llms-part-4-how-cuda-graphs-make-vllm-thomas-4ofuc) (2025)
- Modal, [Host overhead is killing your inference efficiency](https://modal.com/blog/host-overhead-inference-efficiency)

**sglang-omni cases**
- [#1232](https://github.com/sgl-project/sglang-omni/issues/1232) Profile GPU-side bottlenecks in MOSS-TTS-v1.5 Delay (mine)
- [#1304](https://github.com/sgl-project/sglang-omni/pull/1304) Fix missing CPU operator events in scheduler-thread profiling (mine)
- [#756](https://github.com/sgl-project/sglang-omni/pull/756) Raise Higgs TTS AR server default to 64 (mine)
- [#1134](https://github.com/sgl-project/sglang-omni/pull/1134) CUDA-graph the Qwen3-TTS code-predictor chain
- [#564](https://github.com/sgl-project/sglang-omni/issues/564) / [#572](https://github.com/sgl-project/sglang-omni/pull/572) Batch the per-step D2H syncs in Higgs TTS
- [#1222](https://github.com/sgl-project/sglang-omni/pull/1222) Run MOSS-TTS Delay reference encoding on GPU

Numbers in Section 4 are quoted from these PRs and issues; figures in Section 3 are as reported by the linked sources.
