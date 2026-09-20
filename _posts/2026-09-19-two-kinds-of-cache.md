---
title: "Two Kinds of Cache: KV Cache in LLMs vs Embedding Cache in Recommenders"
date: 2026-09-19 10:00:00 -0700
categories: [LLM Inference, KV Cache]
tags: [ml-systems, caching, kv-cache, recsys, serving]
pin: true
math: true
mermaid: true
description: Both systems hit the same wall — the data you need does not fit where you need it. They solve it differently, and the contrast explains a lot about each.
---

> This post is a **template**, not a finished piece. The structure is what matters: a concrete
> hook, a mechanism, numbers, and a takeaway. Replace the content with your own and delete this
> note.
{: .prompt-warning }

## The wall both systems hit

An LLM serving 100 concurrent requests and a recommender ranking 10,000 candidates have almost
nothing in common at the model level. At the memory level they have the same problem: **the bytes
you need on the next microsecond are not in the memory you are reading from.**

For an LLM it is the KV cache. For a recommender it is the embedding table.

| | KV cache | Embedding table |
|---|---|---|
| What it holds | keys and values for every token generated so far | one row per user / item / feature value |
| Size driver | batch size × sequence length × layers | catalog size × dimension |
| Lifetime | one request | months, updated continuously |
| Sharing | private to a request | shared across all requests |
| Access pattern | sequential, predictable | random, heavily skewed |

## How LLM serving handles it

TODO: PagedAttention — fixed-size blocks, an OS-style page table, near-zero fragmentation.
Include the fragmentation numbers from the vLLM paper, and what you measured yourself.

## How recommender serving handles it

TODO: the memory hierarchy — hot rows on GPU, warm in host memory, the long tail on SSD.
Where HugeCTR / Persia / Monolith draw the line, and why the skew in item popularity makes it work.

## What the contrast teaches

TODO: one paragraph. Something like: a private, short-lived cache is an *allocation* problem, so
the win comes from eliminating fragmentation; a shared, long-lived cache is a *hit rate* problem,
so the win comes from the eviction policy. Both are memory problems dressed up as model problems.

## Takeaway

> TODO: one sentence a reader could repeat to someone else tomorrow.
{: .prompt-tip }
