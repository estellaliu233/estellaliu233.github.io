---
title: "Inference Optimization Agents: A Field Guide, What AMD Built, and How I'd Assemble One for vLLM/SGLang on NVIDIA"
date: 2026-10-31 10:00:00 -0700
categories: [LLM Serving, Agents]
tags: [ml-systems, serving, agents, optimization, geak, vllm, sglang, tensorrt-llm]
description: Four layers of inference optimization tooling, from framework skills to complete workflows, a walkthrough of AMD's GEAK, and a design for assembling something similar on NVIDIA.
---

> This is a reading-based survey. I haven't run these tools yet: the descriptions come from their docs, READMEs, and skill files, and section 3 is an untested design. Items marked **[abstract only]** are ones I've only read the abstract or summary of.
{: .prompt-info }

## Why I wrote this

AMD recently shipped **GEAK**, an end-to-end inference optimization agent. You point it at a running vLLM or SGLang server, and it profiles the workload, finds the bottlenecks, tunes configs or rewrites kernels, and verifies that each change is a real end-to-end win. It only runs on AMD GPUs.

That made me ask: what's the NVIDIA equivalent for vLLM and SGLang? The short answer is that **there isn't a single product, but most of the parts already exist.** These notes map the landscape, walk through what GEAK does, and sketch how I'd assemble something similar from existing tools.

---

## 1. The landscape: four layers

"Optimization agent" covers very different things. Some help you write one kernel correctly; others run a whole optimization loop. I group the serving-focused tools into four layers, from the narrowest scope to the widest:

1. **Skills inside a framework:** vLLM and SGLang. They help developers do things correctly.
2. **Collaboration process:** TensorRT-LLM. It organizes skills and agents into a process.
3. **Cross-framework tools:** BBuf. They compare frameworks and explain the gaps.
4. **Complete workflow:** GEAK. Code controls the flow, and the LLM steps in only when a judgment is needed.

Everything that doesn't fit these layers is listed in section 1.5.

---

### 1.1 Skills inside a framework: vLLM and SGLang

**What it does.** These skills ship with the framework's own repo and encode how that project's developers work: how to write a kernel, debug a crash, benchmark a change, or prepare a PR. They help a developer do one task correctly. They don't run an optimization loop.

| Framework | Skills that matter for performance work | Others | Links |
|---|---|---|---|
| **vLLM** (9 skills) | `kernel-microbenchmark`, `model-microbenchmark`, `triton-kernel-writing`, `debug-ima` | `check-api-compat`, `ci-fails-buildkite`, `model-recipes`, `pr-checklist`, `transformers-upgrade` | [skills](https://github.com/vllm-project/vllm/tree/main/.agents/skills) |
| **SGLang** (28 skills) | `generate-profile`, `llm-torch-profiler-analysis`, `add-jit-kernel`, `add-sgl-kernel`, `kl-consistency-test`, `debug-cuda-crash`, `debug-distributed-hang`, `sglang-prod-incident-triage` | CI, cookbook, code style, release, and other project workflows | [skills](https://github.com/sgl-project/sglang/tree/main/.agents/skills) |

⭐ The best practitioner write-up I found is [LMSYS: Agent-Assisted SGLang Development](https://www.lmsys.org/blog/2026-07-02-agent-assisted-sglang-development/) ([Chinese](https://zhuanlan.zhihu.com/p/2056522862169678247)).

---

### 1.2 Collaboration process: TensorRT-LLM

**What it does.** TRT-LLM ships about 30 skills and 11 sub-agents in its [`.claude/`](https://github.com/NVIDIA/TensorRT-LLM/tree/main/.claude) directory and organizes them into a process. A [`perf-optimization` coordinator](https://github.com/NVIDIA/TensorRT-LLM/blob/main/.claude/skills/perf-optimization/SKILL.md) plans and delegates but is told *not to write code*. Specialists (profiling, CUDA graph, Triton, CuTe, CUDA, TileIR) do the work. Before trying anything, it checks a [**casebook**](https://github.com/NVIDIA/TensorRT-LLM/blob/main/.claude/skills/perf-optimization-casebook/SKILL.md) of past wins, each written as a precedent: when it applies, the mechanism, the expected gain, the accuracy risk, how to verify it, and how to roll it back. There is a user-specified mode and an **autopilot** mode. It runs on NVIDIA and serves **TensorRT-LLM only**.

**How to try it:** clone the repo and start Claude Code inside it. Skills and agents are discovered automatically; type `/skills` and `/agents` to check what loaded. Then either ask in plain language ("optimize this workload to TPOT < X ms") or invoke a skill directly, for example `/perf-analysis`.

---

### 1.3 Cross-framework tools: BBuf

**What it does.** [BBuf AI-Infra-Auto-Driven-SKILLS](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS) isn't tied to one framework. It runs the same model and workload across frameworks, compares them, and helps explain why one is behind. It covers SGLang, vLLM, TensorRT-LLM, and TokenSpeed.

| Skill or resource | What it does |
|---|---|
| `llm-serving-auto-benchmark` | Finds the best launch command for the same model, workload, GPU budget, and SLA **across frameworks**. You give it base flags plus a search space; it generates a bounded candidate list (baseline first), validates flag names before launching, records failed candidates, and picks the best one that meets the SLA. |
| `llm-torch-profiler-analysis` | Produces a kernel table and overlap and fusion opportunities, cross-checked against known optimizations in SGLang, vLLM, TRT-LLM, and FlashInfer. |
| `llm-pipeline-analysis` | Inspects torch profiler traces at the forward-pass, layer, and kernel level. |
| `llm-serving-capacity-planner` | Explains startup memory, KV budget, concurrency, and OOM causes from logs. |
| `model-compute-simulation` | Estimates operator shapes, FLOPs, and MFU, and maps kernels back to ops. |
| PR history | A searchable history of optimization PRs for 118 models (44 in vLLM, 45 in SGLang, ...). |

Write-ups: [intro (Zhihu)](https://zhuanlan.zhihu.com/p/2025216177463370060), [v0.1.0 (Zhihu)](https://zhuanlan.zhihu.com/p/2042740770457772060).

**How to install:**
```text
/plugin marketplace add BBuf/AI-Infra-Auto-Driven-SKILLS
/plugin install ai-infra-auto-driven-skills@ai-infra-auto-driven-skills
```
To search the PR history: `python3 scripts/query.py --framework vllm "qwen3 fused qk norm"`.

---

### 1.4 Complete workflow: GEAK

**What it does.** You give it a running server, a fixed workload, and a target metric. It runs the whole loop: measure the baseline → profile → decide what to try (configs first, then kernels) → apply one change → re-measure end to end → keep or revert. Code controls the flow, and LLM roles are called only at decision points. This layer answers *"did the service actually get faster?"* rather than *"did a kernel get faster?"*

[AMD GEAK](https://github.com/AMD-AGI/GEAK) (`e2e_workflow`) runs on AMD Instinct/Radeon + ROCm 6+ and supports vLLM, SGLang, and ATOM. Section 2 walks through it. Further reading: [pipeline docs](https://rocm.docs.amd.com/projects/geak/en/latest/conceptual/geak-pipeline.html), [original GEAK paper (2025, Triton kernel agent)](https://arxiv.org/abs/2507.23194), and its companion [Magpie skill](https://github.com/AMD-AGI/Magpie/blob/main/skills/magpie/SKILL.md) (benchmarking and analysis on AMD, with a 9-step loop and explicit "don't claim success unless..." rules).

[LLM4LLM](https://arxiv.org/abs/2608.21836) **[abstract only]**, no code found, is a similar idea on NVIDIA: it splits an inference script into prefill/decode optimization tasks and accepts a patch only after verifying it inside the model.

---

### 1.5 Outside the four layers

These tools are useful, but each covers one piece of the work:

1. **Config tuning:** vllm-autotune, AIConfigurator/AISimulate. They change launch flags only, never code.
2. **Kernel agents:** KDA-Pilot, KernelAgent, Kernel Forge, Triton8. They optimize one hot kernel inside a harness.
3. **RL-trained kernel models:** CUDA Agent, Kevin. They train the model itself to write kernels.
4. **Skill-building guides:** Hugging Face. They show how to distill experience into a skill.
5. **Benchmarks for agents:** KernelBench, InferenceBench, SWE-Serve. They grade whether a speedup is real.

**Config tuning**
- [vllm-autotune](https://github.com/SummerTea/vllm-autotune): the agent may edit only `config.yaml`, and goodput under the SLO decides keep or revert. It's a new project (created in October 2026) with no visible adoption yet, and I've only read its design.
- [AIConfigurator](https://github.com/ai-dynamo/aiconfigurator) → [AISimulate](https://github.com/ai-dynamo/aisimulate) (NVIDIA Dynamo): not an LLM agent. A performance model searches disaggregated-serving configs offline, with no GPU needed. See the [paper](https://arxiv.org/abs/2601.06288) and the [Dynamo 1.0 blog](https://developer.nvidia.com/blog/nvidia-dynamo-1-production-ready/).

**Kernel agents**
- [KDA-Pilot](https://github.com/BBuf/KDA-Pilot) (NVIDIA B200; SGLang): turns real SGLang kernels into reproducible tasks with frozen production shapes, a shared ABI, wall-time measurement, and correctness gates. Examples: `rotary_embedding` 1.49×, `norm_infer` 1.35×.
- [KernelAgent](https://github.com/meta-pytorch/KernelAgent) (Meta PyTorch; NVIDIA, Intel XPU): converts a PyTorch program into verified Triton kernels. See the [PyTorch blog](https://pytorch.org/blog/kernelagent-hardware-guided-gpu-kernel-optimization-via-multi-agent-orchestration/).
- [Kernel Forge](https://github.com/TheJoshBrod/KernelForge) (NVIDIA, AMD): profiles a whole PyTorch model per operator and swaps in a generated kernel only if it measures faster. See the [paper](https://arxiv.org/abs/2607.24762).
- [Triton8](https://github.com/IntelLabs/Triton8) (Intel GPU): Claude Code drives a Triton optimization loop with a curated knowledge base.
- Paper-only **[abstract only]**: [Astra](https://arxiv.org/abs/2509.07506), [KernelOPT](https://arxiv.org/abs/2609.30059), [FACT](https://arxiv.org/abs/2604.26666), [KernelPro](https://arxiv.org/abs/2606.26453).

What practitioners tell me matches what I found online: **kernel-level agents get the most real use** (KDA-Pilot in particular). Kernel tasks have clear boundaries: you can check correctness against a reference and measure the speedup directly. End-to-end systems are rarer, especially on NVIDIA, where hand-written skills and profiling tools are already good enough that experienced engineers wire them together themselves.

Write-ups on the kernel approach:
- [Harness Engineering for LLM-Driven GPU Kernel Generation](https://arxiv.org/abs/2607.17979) (MLSys 2026 FlashInfer contest, B200): the agent proposes patches and reads profiles, while **humans own the evaluation harness and acceptance rules**.
- [Cursor × NVIDIA: multi-agent kernel optimization](https://cursor.com/blog/multi-agent-kernels): 235 Blackwell kernels optimized in 3 weeks, 38% geomean speedup **[abstract only]**.
- [blopig: Writing Triton kernels with Claude](https://www.blopig.com/blog/2026/01/democratising-the-dark-arts-writing-cuda-kernels-with-claude/): the first draft was about 95% right; edge cases and masking still needed a human.
- [NVIDIA: kernel generation with DeepSeek-R1 + inference-time scaling](https://developer.nvidia.com/blog/automating-gpu-kernel-generation-with-deepseek-r1-and-inference-time-scaling/): a generate → verify → refine loop.

**RL-trained kernel models** (not something you plug into a serving stack today)
- [CUDA Agent](https://arxiv.org/abs/2602.24286): large-scale agentic RL for CUDA kernel writing.
- [Kevin](https://arxiv.org/abs/2507.11948): multi-turn RL for kernel generation.

**Skill-building guides**
- [HF: Custom Kernels from Codex and Claude](https://huggingface.co/blog/custom-cuda-kernels-agent-skills) and [tensormux/kernel-skills](https://github.com/tensormux/kernel-skills): a skill of about 550 tokens that teaches agents to write production CUDA kernels tuned for H100/A100/T4.
- [HF: upskill](https://huggingface.co/blog/upskill): build with a strong model interactively, distill the trace into a skill, and transfer it to smaller models.

**Benchmarks for agents**
- [KernelBench](https://github.com/ScalingIntelligence/KernelBench): kernel generation from PyTorch reference programs.
- [InferenceBench](https://arxiv.org/abs/2607.20468) **[abstract only]**: agents vs. classic search (SMAC/TPE) on inference tuning.
- [SWE-Serve](https://arxiv.org/abs/2609.26777) **[abstract only]**: SWE-style tasks on serving systems.
- [Production-kernel benchmark](https://arxiv.org/abs/2607.14541) **[abstract only]**: whether LLM-generated kernels hold up under production traces.

Survey and lists: [survey](https://arxiv.org/abs/2601.15727), [awesome-LLM-driven-kernel-generation](https://github.com/flagos-ai/awesome-LLM-driven-kernel-generation), [Awesome-GPU-Kernel-Optimization](https://github.com/luckily268/Awesome-GPU-Kernel-Optimization).

---

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
| 6 | **Integrate + end-to-end gate** | Swap the kernel into the server, **re-measure end to end**, apply the three gates | — | ⭐ **I didn't find a tool for this** |
| 7 | Finalize | Patch, launch command, report; the same script re-measures | — | A report template |

### 3.3 The gap: the end-to-end gate

The clearest gap I found is stage 6. LMSYS reports that KDA-Pilot kernels ran **1.13–2.75× faster on B200**, yet the full model gained only **1.02–1.03×**. Amdahl's law explains it: the kernel is a small share of total time, while host overhead, communication, and scheduling are untouched. They also list ways a gain can be fake: shapes edited after seeing results, fast-math settings that differ between baseline and candidate, only one side using a lighter wrapper, a silent fallback, or CUDA graphs turned off.

So the first thing I'd build isn't another kernel generator. It's a small, framework-agnostic **`e2e-regression-gate`** skill: given a baseline and a candidate launch command plus a frozen workload, it checks engagement from the startup logs, runs interleaved A/B rounds to get a noise band, compares temperature-0 outputs token by token, and returns PASS only if all three hold. It's useful whether the change came from an agent or a human, which matters on NVIDIA, where most optimization is still done by hand.

**Hardware:** an A100 80GB with Qwen3-8B should be enough to exercise this end to end; as designed, none of it depends on Hopper or Blackwell features.

---

*Next: a hands-on run with Qwen3-8B, to test this design and see which of these gaps hold up.*

Numbers in this note are quoted from the linked sources; I haven't reproduced them.
