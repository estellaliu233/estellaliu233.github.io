---
title: "Embedding: A Map of the Literature"
date: 2026-09-19 09:00:00 -0700
categories: [Ads & Reco, Embedding]
tags: [recsys, embedding, graph-embedding, retrieval]
math: true
mermaid: true
description: How items, users and queries become vectors — the families of ideas and what each one actually solves.
---

A map of the embedding literature in search, recommendation and advertising, organized the way I
learned it. Each branch is a family of ideas; the notes behind them go paper by paper.

```mermaid
mindmap
  root((Embedding))
    Sequence
      Word2vec
        Skip-gram / CBOW
        Negative sampling
        Subsampling
      Item2vec
    Graph — shallow
      DeepWalk
      LINE
      node2vec
      NetMF — matrix factorization view
    Graph — GNN
      GCN
      GraphSAGE — inductive
      GAT
    Industrial
      EGES — side info, cold start
      PinSage — web scale
      GATNE — multiplex
    Compression
      Hashing trick
      QR embedding
      DHE
      Mixed dimension
    Retrieval
      DSSM
      YouTube DNN
      Two-tower + logQ
      Mixed negative sampling
      Facebook EBR
```

### The one thing to remember

| Family | Core trick | Solves |
|---|---|---|
| Word2vec | negative sampling + subsampling | full softmax over a huge vocabulary is too slow |
| Shallow graph | random walks as sentences | no explicit objective for graph structure |
| GNN | learn an aggregation function | new nodes have no vector (cold start) |
| Compression | share rows / drop the table | embedding tables do not fit in memory |
| Two-tower | in-batch negatives + logQ correction | candidate set is the whole catalog |
