---
# the default layout is 'page'
icon: fas fa-info-circle
order: 5
---

I work on LLM systems: how models are trained and served, and where the time and the
memory actually go. Most of what I write here is hands-on — reading a scheduler, running a benchmark, taking apart a
KV cache implementation — and much of it started as something I got wrong the first time.

I've also worked on search, recommendation and advertising: embedding, retrieval and
ranking models serving traffic under tight latency budgets. Those notes live in the
[Ads & Reco](/adsreco/) tab. The two fields look unrelated, but the hard parts rhyme — at serving time, a 100 GB
embedding table and a 100 B parameter model hit the same wall, and it is a memory wall, not a
compute one.

## What you'll find here

- **[LLM Systems](/)** — inference, KV cache, batching and scheduling, quantization, serving.
- **[Ads & Reco](/adsreco/)** — mind maps and paper notes from a systematic pass through the
  search / recommendation / advertising literature.

## Elsewhere

- LinkedIn: [estellaxinyuan](https://www.linkedin.com/in/estellaxinyuan/)
- Google Scholar: [publications](https://scholar.google.com/citations?user=mYGmeOgAAAAJ)
