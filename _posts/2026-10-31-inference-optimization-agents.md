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

These skills ship with the framework's own repo ([vLLM skills](https://github.com/vllm-project/vllm/tree/main/.agents/skills), [SGLang skills](https://github.com/sgl-project/sglang/tree/main/.agents/skills)) and encode how that project's developers work: how to write a kernel, debug a crash, benchmark a change, or prepare a PR. They help a developer do one task correctly. They don't run an optimization loop.

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

I'm not a heavy TensorRT-LLM user, but its differences from vLLM and SGLang stand out: TensorRT-LLM is more automated. **vLLM and SGLang provide tools. TensorRT-LLM also provides a process for organizing agents around those tools.** Its [`.claude/`](https://github.com/NVIDIA/TensorRT-LLM/tree/main/.claude) directory ships 32 skills and 11 sub-agents, and four parts make up the process:

| Part | Role |
|---|---|
| **[Coordinator](https://github.com/NVIDIA/TensorRT-LLM/blob/main/.claude/skills/perf-optimization/SKILL.md)** | Sets the direction, assigns work, reads the results, and decides the next round, but doesn't write code |
| **Specialists** | Handle specific work such as profiling, CUDA graphs, and Triton |
| **[Casebook](https://github.com/NVIDIA/TensorRT-LLM/blob/main/.claude/skills/perf-optimization-casebook/SKILL.md)** | Records how similar problems were solved, and gets checked before any change |
| **Loop** | Change → profile → analyze → change again |

The coordinator covers two modes:

| | User-specified mode | Autopilot mode |
|---|---|---|
| **What you say** | "Apply CUDA Graph to my model" | A target metric and value, such as a TPOT goal |
| **Scope of changes** | One optimization | Several optimizations, applied one after another |

**Compared with vLLM and SGLang,** this is how I read the trade-offs.

| | Details |
|---|---|
| **Advantages** | You make fewer step-by-step decisions. Roles are clear, so the agent that sets the direction isn't also writing the code. Past wins are collected in one place and checked first. |
| **Disadvantages** | The report names the specialist used, what changed, and the before-and-after metrics, and you can test one specialist on its own, which may be easier than working through a skill hundreds of lines long. The hard part is the handoffs: changing one step can affect every task after it. So debugging and improving the whole process on your own can be harder than with vLLM or SGLang skills. |

---

### 1.3 Cross-framework tools: BBuf

[BBuf AI-Infra-Auto-Driven-SKILLS](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS) isn't tied to one framework. It runs the same model and workload across frameworks, compares them, and helps explain why one is behind. It covers SGLang, vLLM, TensorRT-LLM, and TokenSpeed.

| Skill or resource | What it does |
|---|---|
| `llm-serving-auto-benchmark` | Finds the best launch command for the same model, workload, GPU budget, and SLA **across frameworks**. You give it base flags plus a search space; it generates a bounded candidate list (baseline first), validates flag names before launching, records failed candidates, and picks the best one that meets the SLA. |
| `llm-torch-profiler-analysis` | Produces a kernel table and overlap and fusion opportunities, cross-checked against known optimizations in SGLang, vLLM, TRT-LLM, and FlashInfer. |
| `llm-pipeline-analysis` | Inspects torch profiler traces at the forward-pass, layer, and kernel level. |
| `llm-serving-capacity-planner` | Explains startup memory, KV budget, concurrency, and OOM causes from logs. |
| `model-compute-simulation` | Estimates operator shapes, FLOPs, and MFU, and maps kernels back to ops. |
| PR history | A searchable knowledge base of 118 model-family dossiers built from optimization PRs (45 for SGLang, 44 for vLLM, ...). |

BBuf also wrote an excellent practitioner's guide: [LMSYS: Agent-Assisted SGLang Development](https://www.lmsys.org/blog/2026-07-02-agent-assisted-sglang-development/) ([Chinese](https://zhuanlan.zhihu.com/p/2056522862169678247)). It also covers SGLang's in-repo skills and KDA-Pilot.

Other write-ups: [intro (Zhihu)](https://zhuanlan.zhihu.com/p/2025216177463370060), [v0.1.0 (Zhihu)](https://zhuanlan.zhihu.com/p/2042740770457772060).

The LMSYS post also describes the SOTA loop, where SOTA means the best reproducible result under fixed conditions. It compares frameworks, explains the gap, patches SGLang, and re-tests. BBuf used to ship it as the `sglang-sota-humanize-loop` skill, but [PR #98](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/pull/98) removed it in October 2026. I read this as the opposite of AMD's direction: BBuf moved the process from a skill into a prompt, while GEAK moved it into code.

**How to run it end to end.** Write the whole process into one persistent `/goal`. Here is a shortened version of the [Nemotron-3-Nano prompt](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/blob/main/prompts/sglang-sota-h200-codex-goal-prompts.md#L42-L71), in my own words. The original has more than 20 rules:

```text
/goal Keep optimizing SGLang serving for Nemotron-3-Nano-30B-A3B-FP8 on one H200
until it matches or beats the best reproducible vLLM and TensorRT-LLM results under
the same GPU budget, precision, workload, and SLA. Prove it with benchmarks, profiles,
a final report, and GSM8K/MMLU accuracy tables.

Rules:
- Confirm the repos are clean and check open PRs before setting the baseline.
- Before each measurement, confirm no one else is using the GPU.
- Use one GPU only, and don't choose a patch until the fair benchmark is done.
- If SGLang is more than 1% behind, profile first, using
  llm-torch-profiler-analysis and llm-pipeline-analysis.
- Patch only paths the evidence supports, then re-run the same workload.
- Update the manifest every round: evidence, failed attempts, next step.
- Clean up only what this task started or downloaded.
- Stop and report if GPUs stay unavailable, the evidence can't be trusted, the
  budget runs out, or no defensible next patch exists.
```

**How a run might use these tools.** This is my own illustration, not a recorded run:

1. `llm-serving-auto-benchmark` finds the best command for each framework and shows SGLang is behind.
2. `llm-torch-profiler-analysis` and `llm-pipeline-analysis` trace the gap to the stage, layer, and kernel responsible.
3. If the cause is a config or routing problem, the agent fixes it in SGLang, using in-repo skills such as `write-sglang-test` to add coverage.
4. If the cause is one hot kernel, it can become a KDA-Pilot task. The agent optimizes the kernel in isolation, then the change is integrated back into SGLang.
5. The agent re-runs the same benchmark and accuracy checks, then decides what to try next.

[Kernel Design Agents](https://github.com/NVlabs/kda) (KDA) is an agent workflow for writing high-performance CUDA kernels, and it won the MLSys 2026 FlashInfer Kernel Contest. [KDA-Pilot](https://github.com/BBuf/KDA-Pilot) applies it to real SGLang kernels, with frozen shapes, matching interfaces for the old and new versions, and correctness gates. Steps 3 and 4 are my suggestion: if you want the agent to use them, name them in your goal.

You set the goal and review the result. Inside the goal, the agent decides each next step.

---

### 1.4 Complete workflow: GEAK

You give it a serving backend, model weights, and a target workload, and it works to raise serving throughput. It runs the whole loop: measure the baseline → profile → rank bottlenecks by `GPU-time share × achievable speedup` → try the cheapest options first (configs, then kernels) → apply one change → re-measure end to end → keep or revert. An isolated kernel speedup is only a hypothesis, and the end-to-end A/B result is the verdict. This layer answers *"did the service actually get faster?"* rather than *"did a kernel get faster?"*

[AMD GEAK v4](https://www.amd.com/en/developer/resources/technical-articles/2026/geak-v4.html) ([Chinese](https://zhuanlan.zhihu.com/p/2076730119767450356), [repo](https://github.com/AMD-AGI/GEAK)) runs on AMD Instinct/Radeon + ROCm 6+ and supports vLLM, SGLang, and ATOM. Further reading: [pipeline docs](https://rocm.docs.amd.com/projects/geak/en/latest/conceptual/geak-pipeline.html) and its companion [Magpie skill](https://github.com/AMD-AGI/Magpie/blob/main/skills/magpie/SKILL.md) (benchmarking and analysis on AMD, with a 9-step loop and explicit "don't claim success unless..." rules).

GEAK runs inside Claude Code, but Claude doesn't drive it. **A deterministic JavaScript workflow does.** Stage order, budgets, stop conditions, and gates are all code. LLM roles step in for judgment calls: which kernel to attack, how to rewrite it, whether a route is worth it. GEAK has specialist roles too, so the main difference from TensorRT-LLM is enforcement: TensorRT-LLM's process still relies on an LLM following written instructions, while GEAK's is enforced by code.

It works at two levels. `e2e_workflow` handles the server, profiling, and end-to-end checks, and it calls `kernel_workflow` when a kernel is worth rewriting. A separate Verify Engineer then re-measures each claimed speedup in a clean workspace, so no agent's self-reported result counts. Skills and past lessons can suggest candidates, but they can't replace measurement.

Every change must pass three gates, which are worth copying even if you never use GEAK:

1. **Engagement proof:** the new kernel actually ran. This catches silent fallbacks.
2. **Clears the noise band:** in back-to-back A/B runs, the gain exceeds the measured noise.
3. **Quality unchanged:** greedy outputs match, and low-precision kernels pass a task-accuracy gate.

**How to try it:** you need AMD GPUs. Ask for a run in plain language:
```text
use path_to_GEAK/e2e_workflow to optimize inference for /models/Qwen3.5-27B-FP8, sglang, ISL/OSL=1024, conc=64, gpus 0,1,2,3
```

AMD reports throughput gains of +60% on Qwen3.5-27B-FP8, +96% on Qwen3-14B-FP8, and +42.2% on MiniMax-M3-MXFP8, on MI300X-class systems. I read GEAK v4 as more enterprise-oriented than the other layers: AMD pitches it as a way for customers to get faster, cheaper inference on AMD GPUs without building a dedicated kernel engineering team.

I didn't find independent hands-on reviews of GEAK v4. Its GitHub issues show that a run can be long and costly ([one reported run](https://github.com/AMD-AGI/GEAK/issues/402) took 11 hours and cost $237 with no usable result), and that [a run that stops early can look like one that found no gain](https://github.com/AMD-AGI/GEAK/issues/492). So I'd watch cost and check why a run ended, and you may also need to decide which skills to enable.

---

## 2. Letting an agent run for longer

Once a single task works, you can let the agent keep going. BBuf's post mentions [Codex Goal](https://github.com/openai/openai-cookbook/blob/main/examples/codex/using_goals_in_codex.ipynb), and [Claude Code](https://code.claude.com/docs/en/goal) has a similar `/goal` command: the agent keeps working until a completion condition is met or it can't continue. It suits work where the end state is checkable but the next step depends on the last result, such as benchmark-driven tuning.

OpenAI's guide says a strong goal defines six things:

| Part | Question | Example |
|---|---|---|
| **Outcome** | What must be true when it's done? | A baseline, noise band, and hotspot table exist |
| **Evidence** | What proves it? | Benchmark files, the trace, the final report |
| **Constraints** | What must not regress or change? | The workload stays fixed; no source edits |
| **Boundaries** | What may it use? | One GPU, one artifact directory |
| **Iteration** | How does it pick the next step? | Change one variable, keep it only if the gain beats the noise |
| **Stop conditions** | When should it stop and report? | The GPU is busy, the trace is empty, the budget is gone |

Iteration is the part that's easiest to leave vague, so here is an example for improving throughput:

| Result of this round | Next step |
|---|---|
| The gain beats the noise, and all constraints hold | Keep the change and re-profile, because the hotspot may have moved |
| The gain doesn't beat the noise | Roll back and try the next-ranked option |
| Throughput improves, but a constraint is violated | Roll back and record why it failed |
| The new path didn't actually run | Check for a fallback, and don't count this result as valid data |
| Three rounds in a row bring no real gain | Switch routes. For example, if the hot attention kernel got faster but overall throughput didn't improve, it may not be the real bottleneck, so check other bottlenecks |
| Every route is exhausted, with no new evidence | Trigger the stop condition and report |

The wording matters too:

- **Too vague:** Keep optimizing until performance improves.
- **Better:** Change one variable per round, and keep a change only if the re-test gain beats the noise. Try the path with the highest expected gain from the profile first. After three failed rounds on one route, switch routes. Write each round's results to the manifest.

I'd start read-only: baseline, profile, and report, with no code changes. For Claude Code, note that the evaluator judges only what appears in the conversation and doesn't run commands itself, so ask the agent to print the evidence.

BBuf's [H200 prompts](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/blob/main/prompts/sglang-sota-h200-codex-goal-prompts.md) are a full example, but they carry 20-plus rules. Don't try to write that many up front. Start with a handful. When a run goes wrong, prefer a script or test that catches the mistake, and add a rule only if the mistake keeps coming back.

## References

**Framework skills and collaboration processes**
- vLLM, [`.agents/skills`](https://github.com/vllm-project/vllm/tree/main/.agents/skills)
- SGLang, [`.agents/skills`](https://github.com/sgl-project/sglang/tree/main/.agents/skills): [`generate-profile`](https://github.com/sgl-project/sglang/blob/main/.agents/skills/generate-profile/SKILL.md), [`llm-torch-profiler-analysis`](https://github.com/sgl-project/sglang/blob/main/.agents/skills/llm-torch-profiler-analysis/SKILL.md), [`debug-cuda-crash`](https://github.com/sgl-project/sglang/blob/main/.agents/skills/debug-cuda-crash/SKILL.md)
- LMSYS, [Agent-Assisted SGLang Development](https://www.lmsys.org/blog/2026-07-02-agent-assisted-sglang-development/) (2026); [Chinese version](https://zhuanlan.zhihu.com/p/2056522862169678247)
- NVIDIA, [TensorRT-LLM `.claude/`](https://github.com/NVIDIA/TensorRT-LLM/tree/main/.claude): [`perf-optimization` coordinator](https://github.com/NVIDIA/TensorRT-LLM/blob/main/.claude/skills/perf-optimization/SKILL.md), [performance casebook](https://github.com/NVIDIA/TensorRT-LLM/blob/main/.claude/skills/perf-optimization-casebook/SKILL.md)

**Cross-framework tools**
- BBuf, [AI-Infra-Auto-Driven-SKILLS](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS); write-ups: [intro (Zhihu)](https://zhuanlan.zhihu.com/p/2025216177463370060), [v0.1.0 (Zhihu)](https://zhuanlan.zhihu.com/p/2042740770457772060)
- BBuf, [PR #98: removal of the SOTA loop skills](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/pull/98) (2026); [H200 Codex Goal prompts](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/blob/main/prompts/sglang-sota-h200-codex-goal-prompts.md)

**End-to-end workflows**
- AMD, [GEAK v4: From Faster Kernels to Faster Workloads](https://www.amd.com/en/developer/resources/technical-articles/2026/geak-v4.html) (2026); [Chinese version](https://zhuanlan.zhihu.com/p/2076730119767450356)
- AMD, [GEAK](https://github.com/AMD-AGI/GEAK): [pipeline docs](https://rocm.docs.amd.com/projects/geak/en/latest/conceptual/geak-pipeline.html), issues [#402](https://github.com/AMD-AGI/GEAK/issues/402) and [#492](https://github.com/AMD-AGI/GEAK/issues/492)
- AMD, [Magpie skill](https://github.com/AMD-AGI/Magpie/blob/main/skills/magpie/SKILL.md)

**Kernel agents**
- [Kernel Design Agents](https://github.com/NVlabs/kda)
- BBuf, [KDA-Pilot](https://github.com/BBuf/KDA-Pilot)

**Long-running goals**
- OpenAI Cookbook, [Using Goals in Codex](https://github.com/openai/openai-cookbook/blob/main/examples/codex/using_goals_in_codex.ipynb)
- Anthropic, [Keep Claude working toward a goal](https://code.claude.com/docs/en/goal)

Numbers in this note are quoted from the linked sources; I haven't reproduced them.
