---
title: "Assembling a GEAK-Style Optimization Loop on NVIDIA"
date: 2026-11-01 10:00:00 -0700
categories: [LLM Serving, Agents]
tags: [ml-systems, serving, agents, optimization, geak, vllm, sglang, tensorrt-llm]
description: A design for assembling a GEAK-style optimization loop for vLLM and SGLang on NVIDIA from existing tools.
---

## 3. How I'd assemble a GEAK-like loop for vLLM/SGLang on NVIDIA

This is a design, not something I've run.

### 3.1 Principles
- **Deterministic orchestration.** A plain Python script owns stage order, budgets, and gates, and writes all state to files. The LLM is called only at decision points. (GEAK's choice; the opposite of TRT-LLM's LLM coordinator.)
- **Humans set the objective and constraints.** Stage 1 fixes the metric (for example goodput) and treats the SLA and quality bar as hard constraints. The agent can't swap objectives.
- **One metric per track.** The config track optimizes goodput and the kernel track optimizes wall time. Single-metric search is where agents are strongest.
- **Trade-offs go through a written brief.** Latency vs. throughput, whether to quantize, whether to change the model: newer models reason about these reasonably well, but the context they need (traffic distribution, which SLA matters most, quality budget, GPU cost, whether the model can be retrained) mostly isn't in the code or the traces. A human writes a `tradeoff_brief.md`; the agent proposes options with data; the human decides.
- **Writer and reviewer are different agents.** Each round's diff is reviewed by a second agent, following LMSYS's Claude Code + Codex Review loop.
- **Reuse before building.** Most stages already have a tool.

### 3.2 Stage-by-stage mapping

| # | Stage | What it does | Existing tool I'd reuse | What's missing |
|---|---|---|---|---|
| 1 | Setup | Pin model, workload, SLA; launch; measure the baseline **N times to get a noise band** | `vllm bench serve` / SGLang `bench_serving`; BBuf `llm-serving-auto-benchmark` config format | A standard noise-band method |
| 2 | Profile | Steady-state trace, prefill and decode separated, per-layer breakdown | BBuf `llm-torch-profiler-analysis` + `llm-pipeline-analysis`; SGLang `generate-profile` | — |
| 3 | Strategize | Amdahl table; route to **config / kernel / host overhead** | BBuf `model-compute-simulation` (FLOPs/MFU → distance to roofline); PR-history query ("has someone already fixed this?") | Routing rules written as code, not left to the LLM |
| 4 | Config track | One flag at a time, keep or revert by goodput | vllm-autotune loop rules; AIConfigurator/AISimulate to prune offline | SGLang support for the autotune loop |
| 5 | Kernel track | Extract hot kernels as frozen-shape tasks; optimize | KDA-Pilot (SGLang); KernelAgent; vLLM `kernel-microbenchmark` | — |
| 6 | **Integrate + end-to-end gate** | Swap the kernel into the server, **re-measure end to end**, apply the three gates | — | ⭐ **I didn't find a standalone, framework-agnostic tool for this** |
| 7 | Finalize | Patch, launch command, report; the same script re-measures | — | A report template |

### 3.3 The gap: the end-to-end gate

The clearest gap I found is stage 6. LMSYS reports that KDA-Pilot kernels ran **1.13–2.75× faster on B200**, yet the full model gained only **1.02–1.13×**. Amdahl's law explains it: the kernel is a small share of total time, while host overhead, communication, and scheduling are untouched. They also list ways a gain can be fake: shapes edited after seeing results, fast-math settings that differ between baseline and candidate, only one side using a lighter wrapper, a silent fallback, or CUDA graphs turned off.

So the first thing I'd build isn't another kernel generator. It's a small, framework-agnostic **`e2e-regression-gate`** skill: given a baseline and a candidate launch command plus a frozen workload, it checks engagement from the startup logs, runs interleaved A/B rounds to get a noise band, compares temperature-0 outputs token by token, and returns PASS only if all three hold. It's useful whether the change came from an agent or a human, which matters on NVIDIA, where most optimization is still done by hand.

**Hardware:** an A100 80GB with Qwen3-8B should be enough to exercise this end to end; as designed, none of it depends on Hopper or Blackwell features.

---

<!-- Full GEAK walkthrough moved from part 1. Reuse or cut. -->

## 2. What AMD built: GEAK's end-to-end workflow

### 2.1 Architecture

GEAK runs inside Claude Code, but **Claude doesn't drive it. A deterministic JavaScript workflow does.** Stage order, budgets, stop conditions, and gates are all code. LLM roles are called only when a judgment is needed: which kernel to attack, how to rewrite it, whether a route is worth it.

```
User task: "raise Qwen3 SGLang throughput on MI300X"
                       │
             Claude Code (Workflow tool)
                       │
   Deterministic JS orchestration: stage order, budgets, stop rules, gates
     │          │           │              │              │
  Director   Profiler   Architect     Config Tuner   Kernel Extractor / Integrator
                                                          │
                                       recursive kernel_workflow
                                 (Director → TechLead → specialists:
                                  algorithm / memory / compute / host-runtime)
     perf_knowledge/  (AMD operator and backend notes, read-only)
     adapters/        (sglang, vllm, atom)
```

### 2.2 The seven stages

| # | Stage | What happens | Why it's designed this way |
|---|---|---|---|
| 1 | **Setup** | Isolated workspace, warm server, measure the real baseline **and its noise band**. Weights and installed packages are read-only. | Without knowing the noise, you can't tell a real gain from luck. |
| 2 | **Profile** | Capture traces under the real workload; build a hotspot table ranked by Amdahl, **separately for prefill and decode**. | Prefill tends to be compute-bound and decode memory-bound, so their bottlenecks differ. |
| 3 | **Strategize** | Route each hotspot by `GPU-time share × achievable speedup` to one of four tracks: **config, head kernel, editable kernel, host overhead**. | A kernel with a small time share isn't worth it however fast it gets; one already near the roofline isn't either. |
| 4 | **ConfigSweep** | Change one server flag at a time (attention backend, graph mode, scheduler/memory knobs, optional low precision); keep only what helps; re-profile. | Cheapest wins first, and one variable at a time keeps every gain attributable. |
| 5 | **HeadKernel** | For the heaviest GEMM/attention: extract an **immutable unit test** from the real workload → race existing backends → recursively optimize → integrate → end-to-end gate. | An immutable test stops the agent from "changing the question". |
| 6 | **Milestone** | Loop over the remaining editable kernels above threshold: extract → optimize → integrate → re-profile, until the budget or the Amdahl stop rule is hit. | Every change shifts the hotspot distribution, so re-profile each time. |
| 7 | **Finalize** | Emit an overlay, `final_patch.diff`, `final_launch.sh`, and `final_report.md`. The Director **re-measures against the real baseline** to set the official number. | One role, one method, for the final number. |

### 2.3 Three gates every change must pass
1. **Engagement proof**: show the new kernel actually ran. This catches silent fallbacks, such as a backend quietly reverting or CUDA graphs being disabled.
2. **Clears the noise band**: in back-to-back A/B runs, the gain must exceed the measured noise.
3. **Quality unchanged**: greedy outputs match; low-precision kernels must pass a task-accuracy gate.

Changes land as a **rollback-able overlay**, never as edits to installed packages.

### 2.4 Why did AMD build this first?

Two explanations come up. Some people see it as riding the agent wave. I find a second explanation more convincing: **the ecosystem gap.** CUDA already has many hand-tuned kernels and mature profiling tools, so there's less low-hanging fruit for an agent to pick. ROCm has more unoptimized paths and fewer experts to hand-tune them, so automation pays off more. Either way, GEAK's design principles transfer to NVIDIA even if the tool itself doesn't.
