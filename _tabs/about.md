---
# the default layout is 'page'
icon: fas fa-info-circle
order: 5
---

<!-- Draft — rewrite in your own words whenever you like. -->

I work on **LLM systems**: how models are served, where the time and the memory actually go, and
what it takes to make inference fast enough to be worth running.

Most of what I write here is hands-on — reading a scheduler, running a benchmark, taking apart a
KV cache implementation — and most of it started as something I got wrong the first time.

Before this I spent my time in search, recommendation and advertising: embedding, retrieval and
ranking models serving traffic under tight latency budgets. Those notes live in the
[Ads & Reco](/adsreco/) tab. The two fields look unrelated, but the hard parts rhyme — a 100 GB
embedding table and a 100 B parameter model hit the same wall, and it is a memory wall, not a
compute one.

## What you'll find here

- **[LLM Systems](/)** — inference, KV cache, batching and scheduling, quantization, serving.
- **[Ads & Reco](/adsreco/)** — mind maps and paper notes from a systematic pass through the
  search / recommendation / advertising literature.

## Elsewhere

- GitHub: [@estellaliu233](https://github.com/estellaliu233)
- LinkedIn: [estellaxinyuan](https://www.linkedin.com/in/estellaxinyuan/)
- Google Scholar: [publications](https://scholar.google.com/citations?user=mYGmeOgAAAAJ)
