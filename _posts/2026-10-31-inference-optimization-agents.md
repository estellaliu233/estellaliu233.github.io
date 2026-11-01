---
title: "From Skills to Workflows: How Inference Optimization Agents Are Built Today"
date: 2026-10-31 10:00:00 -0700
categories: [LLM Serving, Agents]
tags: [ml-systems, serving, agents, optimization, geak, vllm, sglang, tensorrt-llm]
description: A beginner's map of inference optimization tooling, from framework skills to complete workflows like AMD's GEAK, with where to start in each layer.
---

## Why I wrote this

When I started reading about inference optimization agents, I couldn't tell these tools apart. "Agent" could mean a markdown file that teaches Claude to write a kernel, or a program that runs a whole optimization loop. There are also a lot of skills and agents out there, and I didn't know where to start. So I wrote this short summary, which introduces some of the best-known tools in four layers.

---

## 1. Four layers, and where to start

I group the serving-focused tools into four layers, from the narrowest scope to the widest:

| Layer | Tools | Use it when | Start with |
|---|---|---|---|
| **Skills inside a framework** | vLLM, SGLang | You work on that framework: profiling a run, writing a kernel, debugging a crash | `generate-profile` + `llm-torch-profiler-analysis` (SGLang), `kernel-microbenchmark` (vLLM) |
| **Collaboration process** | TensorRT-LLM | You serve on TensorRT-LLM and want an agent to plan and delegate the work | `/perf-analysis` |
| **Cross-framework tools** | BBuf | You want to compare frameworks or explain why one is behind | `llm-serving-auto-benchmark` |
| **Complete workflow** | GEAK | You're on AMD GPUs and want the whole optimization loop run for you | `e2e_workflow` |

---

### 1.1 Skills inside a framework: vLLM and SGLang

**What it does.** These skills ship with the framework's own repo ([vLLM skills](https://github.com/vllm-project/vllm/tree/main/.agents/skills), [SGLang skills](https://github.com/sgl-project/sglang/tree/main/.agents/skills)) and encode how that project's developers work: how to write a kernel, debug a crash, benchmark a change, or prepare a PR. They help a developer do one task correctly. They don't run an optimization loop.

| Framework | Skills that matter for performance work | Others | Links |
|---|---|---|---|
| **vLLM** | `kernel-microbenchmark`, `model-microbenchmark`, `triton-kernel-writing`, `debug-ima` | `check-api-compat`, `ci-fails-buildkite`, `model-recipes`, `pr-checklist`, `transformers-upgrade` | [skills](https://github.com/vllm-project/vllm/tree/main/.agents/skills) |
| **SGLang** | `generate-profile`, `llm-torch-profiler-analysis`, `add-jit-kernel`, `add-sgl-kernel`, `kl-consistency-test`, `debug-cuda-crash`, `debug-distributed-hang`, `sglang-prod-incident-triage` | CI, cookbook, code style, release, and other project workflows | [skills](https://github.com/sgl-project/sglang/tree/main/.agents/skills) |

**How they differ.**

| | vLLM | SGLang |
|---|---|---|
| **Performance focus** | Microbenchmarks: a single kernel or a model's forward pass | Traces: capture a whole server run, then analyze the hotspots |
| **Unique directions** | API compatibility checks, deployment recipes, Transformers upgrades | Diffusion, production incident triage, distributed hangs, code conventions |

**Where to start.** Read these three SGLang skills:

1. [`generate-profile`](https://github.com/sgl-project/sglang/blob/main/.agents/skills/generate-profile/SKILL.md): launches a server, validates accuracy, and captures a trace.
2. [`llm-torch-profiler-analysis`](https://github.com/sgl-project/sglang/blob/main/.agents/skills/llm-torch-profiler-analysis/SKILL.md): turns a trace into kernel, overlap-opportunity, and fuse-pattern tables.
3. [`debug-cuda-crash`](https://github.com/sgl-project/sglang/blob/main/.agents/skills/debug-cuda-crash/SKILL.md): shows what a detailed skill looks like.

**What a useful skill should answer.** This checklist comes from [LMSYS: Agent-Assisted SGLang Development](https://www.lmsys.org/blog/2026-07-02-agent-assisted-sglang-development/):

| Question | What the skill should capture |
|---|---|
| When to use it | Trigger scenarios, supported models and hardware, when it must stop |
| How to start | Preflight checks, environment variables, repo state, dependency checks, model config |
| How to verify | Benchmark and profile commands, test entry points, accuracy gates |
| How to judge | Output tables, failure modes, priorities, risk classification, fallback conditions |
| How to deliver | Artifact directory, result schema, PR description, reproduction commands, review requirements |

Use these examples and this checklist when you use or modify a skill. The LMSYS post covers three things: SGLang's in-repo skills, BBuf's cross-framework skills, and KDA-Pilot's kernel workflow. It gave me a lot of insight when I was just getting started.

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

The best practitioner write-up I found is by BBuf: [LMSYS: Agent-Assisted SGLang Development](https://www.lmsys.org/blog/2026-07-02-agent-assisted-sglang-development/) ([Chinese](https://zhuanlan.zhihu.com/p/2056522862169678247)). It also covers SGLang's in-repo skills and KDA-Pilot.

Other write-ups: [intro (Zhihu)](https://zhuanlan.zhihu.com/p/2025216177463370060), [v0.1.0 (Zhihu)](https://zhuanlan.zhihu.com/p/2042740770457772060).

**How to install:**
```text
/plugin marketplace add BBuf/AI-Infra-Auto-Driven-SKILLS
/plugin install ai-infra-auto-driven-skills@ai-infra-auto-driven-skills
```
To search the PR history: `python3 scripts/query.py --framework vllm "qwen3 fused qk norm"`.

**What happened to the SOTA loop.** The LMSYS post describes an SGLang SOTA Performance Loop built on Humanize/RLCR: fair benchmark, gap decision, profile, patch, revalidate. It existed as the `sglang-sota-humanize-loop` skill, but [PR #98](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/pull/98) removed it and its vLLM counterpart in October 2026, saying "the benchmark, profiler, kernel, and general agent workflows cover this work." What remains are Codex Goal prompts for [B200](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/blob/main/prompts/sglang-sota-b200-codex-goal-prompts.md) and [H200](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/blob/main/prompts/sglang-sota-h200-codex-goal-prompts.md), which write the process directly into a persistent goal. So the loop is now a pattern you assemble rather than a tool you install. I read this as the opposite of GEAK's direction: BBuf moved the process from a skill into a prompt, while GEAK moved it into code.

---

### 1.4 Complete workflow: GEAK

**What it does.** You give it a running server, a fixed workload, and a target metric. It runs the whole loop: measure the baseline → profile → decide what to try (configs first, then kernels) → apply one change → re-measure end to end → keep or revert. Code controls the flow, and LLM roles are called only at decision points. This layer answers *"did the service actually get faster?"* rather than *"did a kernel get faster?"*

[AMD GEAK](https://github.com/AMD-AGI/GEAK) (`e2e_workflow`) runs on AMD Instinct/Radeon + ROCm 6+ and supports vLLM, SGLang, and ATOM. Further reading: [pipeline docs](https://rocm.docs.amd.com/projects/geak/en/latest/conceptual/geak-pipeline.html), [original GEAK paper (2025, Triton kernel agent)](https://arxiv.org/abs/2507.23194), and its companion [Magpie skill](https://github.com/AMD-AGI/Magpie/blob/main/skills/magpie/SKILL.md) (benchmarking and analysis on AMD, with a 9-step loop and explicit "don't claim success unless..." rules).

GEAK runs inside Claude Code, but **Claude doesn't drive it. A deterministic JavaScript workflow does.** Stage order, budgets, stop conditions, and gates are all code. LLM roles step in for judgment calls: which kernel to attack, how to rewrite it, whether a route is worth it.

Every change must pass three gates, which are worth copying even if you never use GEAK:

1. **Engagement proof:** the new kernel actually ran. This catches silent fallbacks.
2. **Clears the noise band:** in back-to-back A/B runs, the gain exceeds the measured noise.
3. **Quality unchanged:** greedy outputs match, and low-precision kernels pass a task-accuracy gate.

**How to try it:** you need AMD GPUs. The workflow spawns many sub-agents and runs commands on the box, and its README asks you to auto-approve Claude Code permissions, so use a dedicated machine. Then ask for a run in plain language:
```text
use path_to_GEAK/e2e_workflow to optimize inference for /models/Qwen3.5-27B-FP8, sglang, ISL/OSL=1024, conc=64, gpus 0,1,2,3
```

[LLM4LLM](https://arxiv.org/abs/2608.21836) **[abstract only]**, no code found, is a similar idea on NVIDIA: it splits an inference script into prefill/decode optimization tasks and accepts a patch only after verifying it inside the model.
---

## 2. Letting an agent run for longer

Once a single task works, you can let the agent keep going. Both [Codex](https://github.com/openai/openai-cookbook/blob/main/examples/codex/using_goals_in_codex.ipynb) and [Claude Code](https://code.claude.com/docs/en/goal) have a `/goal` command: the agent keeps working until a completion condition is met or it can't continue. It suits work where the end state is checkable but the next step depends on the last result, such as benchmark-driven tuning.

OpenAI's guide says a strong goal defines six things:

| Part | Question | Example |
|---|---|---|
| **Outcome** | What must be true when it's done? | A baseline, noise band, and hotspot table exist |
| **Evidence** | What proves it? | Benchmark files, the trace, the final report |
| **Constraints** | What must not regress or change? | The workload stays fixed; no source edits |
| **Boundaries** | What may it use? | One GPU, one artifact directory |
| **Iteration** | How does it pick the next step? | Change one variable, keep it only if the gain beats the noise |
| **Stop conditions** | When should it stop and report? | The GPU is busy, the trace is empty, the budget is gone |

I'd start read-only: baseline, profile, and report, with no code changes. For Claude Code, note that the evaluator judges only what appears in the conversation and doesn't run commands itself, so ask the agent to print the evidence:

```text
/goal Follow ./plan.md for a read-only analysis. Done when the artifact directory
has a baseline, noise band, prefill/decode hotspot table, and final_report.md, and
the output of `python gate_readonly.py` shows PASS in this conversation. Do not edit
source code. If the GPU is unavailable or the evidence is untrustworthy, stop and
report why, or stop after 20 turns.
```

`plan.md` and `gate_readonly.py` are files you'd write yourself. BBuf's [H200 prompts](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/blob/main/prompts/sglang-sota-h200-codex-goal-prompts.md) are a full example, but they carry 20-plus rules tuned to the author's machines. Don't try to write that many up front: start with a handful, and add one each time a run goes wrong.

---

*Next: a hands-on run with Qwen3-8B, and how I'd assemble a GEAK-style loop on NVIDIA.*

## References

**Framework skills and collaboration processes**
- vLLM, [`.agents/skills`](https://github.com/vllm-project/vllm/tree/main/.agents/skills)
- SGLang, [`.agents/skills`](https://github.com/sgl-project/sglang/tree/main/.agents/skills)
- LMSYS, [Agent-Assisted SGLang Development](https://www.lmsys.org/blog/2026-07-02-agent-assisted-sglang-development/) (2026); [Chinese version](https://zhuanlan.zhihu.com/p/2056522862169678247)
- NVIDIA, [TensorRT-LLM `.claude/`](https://github.com/NVIDIA/TensorRT-LLM/tree/main/.claude): [`perf-optimization` coordinator](https://github.com/NVIDIA/TensorRT-LLM/blob/main/.claude/skills/perf-optimization/SKILL.md), [performance casebook](https://github.com/NVIDIA/TensorRT-LLM/blob/main/.claude/skills/perf-optimization-casebook/SKILL.md)

**Cross-framework tools**
- BBuf, [AI-Infra-Auto-Driven-SKILLS](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS); write-ups: [intro (Zhihu)](https://zhuanlan.zhihu.com/p/2025216177463370060), [v0.1.0 (Zhihu)](https://zhuanlan.zhihu.com/p/2042740770457772060)
- BBuf, [PR #98: removal of the SOTA loop skills](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/pull/98) (2026); Codex Goal prompts for [B200](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/blob/main/prompts/sglang-sota-b200-codex-goal-prompts.md) and [H200](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/blob/main/prompts/sglang-sota-h200-codex-goal-prompts.md)

**End-to-end workflows**
- AMD, [GEAK](https://github.com/AMD-AGI/GEAK): [pipeline docs](https://rocm.docs.amd.com/projects/geak/en/latest/conceptual/geak-pipeline.html), [original paper](https://arxiv.org/abs/2507.23194) (2025)
- AMD, [Magpie skill](https://github.com/AMD-AGI/Magpie/blob/main/skills/magpie/SKILL.md)
- [LLM4LLM](https://arxiv.org/abs/2608.21836) **[abstract only]**

**Config tuning**
- [vllm-autotune](https://github.com/SummerTea/vllm-autotune)
- NVIDIA Dynamo, [AIConfigurator](https://github.com/ai-dynamo/aiconfigurator), [AISimulate](https://github.com/ai-dynamo/aisimulate), [paper](https://arxiv.org/abs/2601.06288), [Dynamo 1.0 blog](https://developer.nvidia.com/blog/nvidia-dynamo-1-production-ready/)

**Kernel agents**
- BBuf, [KDA-Pilot](https://github.com/BBuf/KDA-Pilot)
- Meta PyTorch, [KernelAgent](https://github.com/meta-pytorch/KernelAgent), [blog](https://pytorch.org/blog/kernelagent-hardware-guided-gpu-kernel-optimization-via-multi-agent-orchestration/)
- [Kernel Forge](https://github.com/TheJoshBrod/KernelForge), [paper](https://arxiv.org/abs/2607.24762)
- Intel Labs, [Triton8](https://github.com/IntelLabs/Triton8)
- **[abstract only]** [Astra](https://arxiv.org/abs/2509.07506), [KernelOPT](https://arxiv.org/abs/2609.30059), [FACT](https://arxiv.org/abs/2604.26666), [KernelPro](https://arxiv.org/abs/2606.26453)

**Kernel write-ups**
- [Harness Engineering for LLM-Driven GPU Kernel Generation](https://arxiv.org/abs/2607.17979) (2026)
- Cursor, [Multi-agent kernel optimization with NVIDIA](https://cursor.com/blog/multi-agent-kernels) **[abstract only]**
- blopig, [Writing Triton kernels with Claude](https://www.blopig.com/blog/2026/01/democratising-the-dark-arts-writing-cuda-kernels-with-claude/) (2026)
- NVIDIA, [Kernel generation with DeepSeek-R1 and inference-time scaling](https://developer.nvidia.com/blog/automating-gpu-kernel-generation-with-deepseek-r1-and-inference-time-scaling/)

**RL-trained kernel models**
- [CUDA Agent](https://arxiv.org/abs/2602.24286)
- [Kevin](https://arxiv.org/abs/2507.11948)

**Skill-building guides**
- Hugging Face, [Custom Kernels from Codex and Claude](https://huggingface.co/blog/custom-cuda-kernels-agent-skills); [tensormux/kernel-skills](https://github.com/tensormux/kernel-skills)
- Hugging Face, [upskill](https://huggingface.co/blog/upskill)

**Benchmarks for agents**
- [KernelBench](https://github.com/ScalingIntelligence/KernelBench)
- [InferenceBench](https://arxiv.org/abs/2607.20468) **[abstract only]**
- [SWE-Serve](https://arxiv.org/abs/2609.26777) **[abstract only]**
- [Production-kernel benchmark](https://arxiv.org/abs/2607.14541) **[abstract only]**

**Long-running goals**
- OpenAI Cookbook, [Using Goals in Codex](https://github.com/openai/openai-cookbook/blob/main/examples/codex/using_goals_in_codex.ipynb)
- Anthropic, [Keep Claude working toward a goal](https://code.claude.com/docs/en/goal)

**Survey and lists**
- [Survey](https://arxiv.org/abs/2601.15727)
- [awesome-LLM-driven-kernel-generation](https://github.com/flagos-ai/awesome-LLM-driven-kernel-generation)
- [Awesome-GPU-Kernel-Optimization](https://github.com/luckily268/Awesome-GPU-Kernel-Optimization)

Numbers in this note are quoted from the linked sources; I haven't reproduced them.
