---
title: "Embedding: A Map of the Literature"
date: 2026-09-19 09:00:00 -0700
categories: [Ads & Reco, Embedding]
tags: [recsys, embedding, graph-embedding, gnn, retrieval, two-tower]
description: How items, users and queries become vectors — a knowledge graph of the embedding models behind search, recommendation and advertising, paper by paper.
---

A knowledge graph of the embedding models I studied for search, recommendation and advertising. Each
branch is a family of ideas; under every paper are the same four questions — what problem it solved, how
it works, where it falls short (and what came next), and how it shows up in an ML system design.

Click a node to expand or collapse it; scroll to zoom, drag to pan.

<style>
  .markmap-svg { display: block; width: 100%; height: 80vh; min-height: 520px; }
  .markmap-src { display: none !important; }
  html[data-mode="dark"] .markmap-svg { --markmap-text-color: #d4d4d4; --markmap-code-bg: #2b2b2b; }
  @media (prefers-color-scheme: dark) {
    html:not([data-mode="light"]) .markmap-svg { --markmap-text-color: #d4d4d4; --markmap-code-bg: #2b2b2b; }
  }
</style>

<svg id="embedding-map" class="markmap-svg" aria-label="Embedding models knowledge graph"></svg>

<pre id="embedding-map-src" class="markmap-src" hidden>
# Embedding models
## A. Sequence embeddings
### Word2vec (2013 · Google) — skip-gram + negative sampling + subsampling
#### Basics
- Skip-gram predicts context words from the center word; CBOW predicts the center word from its context
- Trains two tables, W_in and W_out; W_in is usually kept as the word vectors
- No hidden layer and no dense matrix multiply → over 100B words per day on a single machine
- Two ways to approximate the softmax
  - Hierarchical softmax: a Huffman tree, ~log₂W nodes per prediction
  - Negative sampling: a simplified NCE; k negatives drawn from unigram^0.75
#### Problem it solved
- Full softmax costs O(W) per gradient, with W = 10⁵–10⁷
- Frequent, uninformative words ("the", "a") dominate compute while rare words learn poorly
- Single-word vectors cannot express idiomatic phrases ("Air Canada" ≠ "Air" + "Canada")
#### How it works
- Negative-sampling loss: one positive and k negative binary cross-entropy terms per training pair
- Subsampling of frequent words
- Phrase detection: merge high-association bigrams into single tokens, repeat for longer phrases
#### Limitations → what came next
- One vector per word, independent of context → BERT in NLP; DIN target attention and MIND multi-interest retrieval in recommendation
- No vector for unseen words or new items → subword tokenization (WordPiece, BPE); for items: EGES side information, GraphSAGE / PinSage, multimodal encoders, Semantic IDs (TIGER)
- Learns co-occurrence, not clicks or conversions → end-to-end ID embeddings (YouTube DNN, two-tower models)
#### In ML system design
- Baseline: a new retrieval or embedding method is expected to beat item2vec first
- Features and initialization: candidate-to-history similarity as a ranking feature; pretrained vectors to initialize ID embeddings
- Still a production retrieval channel: item2vec i2i retrieval is cheap to train, quick to refresh, and complements two-tower and graph retrieval
## B. Shallow graph embeddings (random walks / edge sampling)
### DeepWalk (2014 · Stony Brook) — truncated random walks as sentences + skip-gram
#### Problem it solved
- Graphs are sparse and discrete; an adjacency matrix cannot be fed to LR or a DNN directly
- Spectral methods need a global eigendecomposition, O(|V|³), and do not scale to million-node graphs
- Collective classification is tied to one label space; a new task means starting over
#### How it works
- Input: an unweighted graph; output: a |V| × d embedding matrix Φ (d = 128)
- Random walks → node sequences → skip-gram; nodes with similar neighborhoods co-occur in walks (the distributional hypothesis)
- Hyperparameters: γ = 80 walks per node, walk length t = 40, window w = 10, dimension d = 128
- Evaluation: BlogCatalog (10,312 nodes, 333,983 edges, 39 labels), Flickr, YouTube (1.1M nodes); one-vs-rest classifiers on Φ, scored by Micro-F1 and Macro-F1
#### Limitations → what came next
- Uniform walks give no control over local vs. distant exploration → node2vec (p, q)
- No node features and no parameter sharing; parameters grow as O(|V|·d) → EGES (side information), GraphSAGE (a shared aggregator)
- Transductive: a node added after training has no row, so the model must be retrained → GraphSAGE / PinSage
#### In ML system design
- i2i retrieval: co-click / co-purchase item graph → walks → skip-gram → ANN; walks stitch behavior across users and sessions, connecting two-hop items and easing co-occurrence sparsity
- Topology-only graphs: friend recommendation, fraud-ring detection, community detection
- Prefer a GNN when node features are rich, new items need embeddings immediately, or the objective must align end to end
### LINE (2015 · MSRA) — first- and second-order proximity + alias edge sampling
#### Problem it solved
- DeepWalk handles only undirected, unweighted graphs and defines neighborhoods implicitly → explicit first- and second-order proximity on directed, weighted graphs
#### How it works
- Input: an edge list (src, dst, weight); undirected edges are written in both directions; the |V| × |V| matrix is never materialized
- First order: directly connected nodes should be similar (symmetric score u_A · u_B)
- Second order: nodes that share neighbors should be similar; each node has a vertex vector (kept) and a context vector (discarded)
- Edge sampling: edge weights span orders of magnitude, so multiplying them into the gradient explodes its variance; instead sample edges in proportion to weight and train them as unweighted
- Alias method: O(1) discrete sampling after O(n) setup — an edge table (|E| buckets) and a node table for negatives (degree^0.75)
- Output: first- and second-order models are trained separately and concatenated (e.g. 128 + 128 = 256 dims)
#### Limitations
- Same as DeepWalk: no node features, no parameter sharing, transductive; industry moved on to GraphSAGE / PinSage
### node2vec (2016 · Stanford) — second-order biased walks between BFS and DFS
#### Problem it solved
- Two kinds of similarity need different exploration
  - Homophily (same community, frequent interaction) → DFS-like walks that move outward
  - Structural equivalence (same role, e.g. hubs or bridges, even far apart) → BFS-like walks that stay local
#### How it works
- After stepping t → v, the next node x is weighted by α_pq(t, x) · w_vx
  - Distance 0 (back to t): 1/p — a large p discourages backtracking
  - Distance 1 (x is also a neighbor of t): 1
  - Distance 2 (moving away from t): 1/q — q < 1 is DFS-like, q > 1 is BFS-like
- p = q = 1 recovers DeepWalk; the model and loss are unchanged, only the walk is
- Edge features from binary operators on node vectors: average, Hadamard, weighted-L1, weighted-L2
#### Limitations
- Second-order alias tables are built per directed edge: O(|E| × average degree) memory, which can exceed the graph itself
- Same as DeepWalk: no node features, no parameter sharing, transductive
#### In ML system design
- Friend / follow recommendation and fraud rings, where node features are weak
- Cold-start or low-data settings without enough behavior to train a two-tower model
### struc2vec (2017 · UFRJ) — similarity by structural role, not proximity (DTW over degree sequences + multilayer walks)
### NetMF (2018 · Tsinghua / MSR) — random walks + SGNS as implicit matrix factorization
- Shows that DeepWalk, LINE, PTE and node2vec all implicitly factorize a closed-form matrix
- Part of their gap comes from sampling noise in the walks; factorizing the matrix explicitly with SVD improves results by up to 50%
- It materializes a dense |V| × |V| matrix, so it is a theoretical contribution rather than a production method (NetSMF sparsifies it)
- Background: any matrix factors as M = UΣVᵀ, with U and V orthogonal and Σ diagonal, holding the singular values in descending order
### EGES (2018 · Alibaba) — behavior graph + weighted DeepWalk + side information
#### Problem it solved
- Taobao's i2i retrieval, which drives 40% of home-page recommendation traffic
  - Scalability (billions of items and users) → subgraph partitioning, distributed walks, 100 GPUs
  - Sparsity → walks generate more sequences; side information adds signal
  - Cold start (millions of new items per hour) → side-information fusion
- Why a graph rather than CF: CF sees only direct co-occurrence; walks capture higher-order similarity
#### How it works
- Weighted, directed DeepWalk with side-information fusion at the input: item embedding = learned weighted sum of ID, category, brand and shop embeddings
- A new item has no ID embedding, but its category, brand and shop were seen in training
- Chicken-and-egg: a new item's fusion weights are untrained too, so they are initialized from category means or defaults
#### Blending ID and content embeddings
- ① Learned weighted fusion (EGES): automatic, but a new item's weights are untrained
- ② Residual with a zero-initialized ID embedding: content + ID, with ID starting at 0 — a smooth transition with no switching logic
- ③ Gating by interaction count: w = n / (n + k) — interpretable, like Bayesian CTR smoothing; k is a prior to tune
- ④ ID dropout during training: forces the content path to learn; combines with ①–③
#### In ML system design — the i2i channel template
- Graph: split sessions (30 min–1 h), connect consecutive actions with directed edges weighted by transition counts, merge into one global graph
- Cleaning: drop accidental clicks (dwell-time threshold), spam and crawler users, re-listed item IDs; prune low-frequency edges and very high-degree nodes (phone cases co-occur with everything)
- Training: weighted walks + skip-gram with negative sampling + side information
- Serving: offline embeddings → ANN top-K per item → key-value store (trigger item → similar items) → fetch by the user's recent triggers
- Evaluation: offline gains do not always carry over online; always A/B test, and prefer hit rate / Recall@K over link-prediction AUC
## C. Graph neural networks
### The paradigm shift
- Shallow methods memorize one answer per node; GNNs learn how to compute the answer — so a new node gets an embedding immediately
### Scalability map
#### Four scale problems
- A. Training throughput: hundreds of billions of samples
- B. Neighborhood explosion: K-hop neighborhoods grow exponentially
- C. Parameter storage: |V| × d tables (200M items × 128 dims ≈ 100 GB in fp32)
- D. Inference and retrieval: embedding the whole corpus and searching 100M+ candidates online
#### Which method addresses which
- DeepWalk: A, partly — local walks, lock-free asynchronous SGD
- LINE: A — edge sampling + alias tables, O(|E| + |V|) memory
- node2vec: makes memory worse — second-order alias tables
- NetMF: the counterexample — a dense |V| × |V| matrix
- GCN: creates B — full-batch training, no minibatches
- GAT: worse still — all neighbors, times the number of heads
- GraphSAGE: B — fixed-size neighbor sampling bounds the cost per batch
- PinSage: B + D — random-walk sampling, MapReduce inference, a producer–consumer pipeline; 3B nodes in production
- EGES: A — subgraph partitioning, distributed walks, 100 GPUs
- QR / mixed dimension / DHE: C
- ANN (Faiss, HNSW): D
### GCN (2017 · Amsterdam) — first-order spectral approximation + renormalization trick
#### Problem it solved
- Skip-gram methods are multi-stage pipelines (walk → embed → classify) that cannot be optimized end to end and ignore node features; GCN unifies structure and features in one model and establishes message passing
- Not solved: cold start — vanilla GCN is transductive, which is what GraphSAGE addresses
#### How it works — a prohibited-item detection example
- Setup: 4 items in a chain 0–1–2–3; an edge means co-browsed by the same high-risk account, or shared shipping or payment details; 3 features (impressions, discount, banned-word score); very few labels, so training is semi-supervised
- Add self-loops, Ã = A + I, and normalize symmetrically, Â = D̃^(-1/2) Ã D̃^(-1/2)
- High-degree nodes get smaller self-weights (0.333 vs. 0.5) — normalization damps popular items
- One layer: H^(l+1) = σ(Â H^(l) W^(l))
- Two layers: Z = softmax(Â · ReLU(Â X W⁽⁰⁾) · W⁽¹⁾); shapes 4×3 → 4×16 → 4×2
- Every layer mixes neighbors again; without it, extra layers are just a deeper MLP with no new graph information
- Loss: cross-entropy on labeled nodes only (masked) — structure is used everywhere, labels supervise a few nodes
#### In ML system design
- Not used for large-scale retrieval: full-graph, transductive, no minibatches ("how does a 100M-node adjacency matrix fit in GPU memory?")
- New items must be retrievable within a day → GCN is enough; within minutes → an inductive model (GraphSAGE / PinSage)
- The graph fits in GPU memory (10K–100K nodes) → GCN / GAT; tens of millions of nodes or more → sampling
- Good fits: fraud-subgraph analysis, content moderation, community detection, knowledge-graph completion, offline feature generation
### GraphSAGE (2017 · Stanford) — sample and aggregate; learn a function that generates embeddings
#### Problem it solved
- Production graphs keep changing (new posts, users and videos every day); DeepWalk, node2vec, LINE and matrix factorization are transductive and need extra SGD for every new node
#### How it works
- Learn a function: sample a fixed number of neighbors, aggregate them, concatenate with the node's own representation, apply a shared W, then L2-normalize
- Parameters do not depend on |V|; any new node with features gets an embedding in one forward pass
- Input: an edge list (undirected, unweighted, homogeneous) + an N × C node-feature matrix (category, brand, text and image vectors, price, age, CTR) + K = 2, S₁ = 25, S₂ = 10
- A minibatch is a sampled tree, not the graph: 2 targets × 3 neighbors × 2 neighbors → 20 nodes' features
- Concatenating instead of mixing doubles the input width of W and keeps the node's own signal from being diluted, much like a skip connection
- Loss: unsupervised, word2vec-style negative sampling over random-walk co-occurrence, with embeddings computed by aggregators instead of looked up; or supervised cross-entropy
- Output: L2-normalized vectors, so the inner product equals cosine similarity and they go straight into an ANN index
#### GCN vs. GraphSAGE vs. GAT
- GCN: all neighbors; fixed degree-normalized weights; sum then W; transductive
- GraphSAGE: sampled neighbors; the aggregator decides; concatenate self and neighbors, then W; inductive
- GAT: all neighbors; learned attention weights; weighted sum then W; inductive
#### Limitations — stated precisely
- It is inductive only when nodes have features; without features it cannot help
- K = 2 is the sweet spot; deeper models explode the neighborhood and over-smooth
- The paradigm lives on (PinSage, Uber Eats); the original model is not deployed directly
#### In ML system design
- Use it for i2i / u2i retrieval (cross-user, higher-order signals that complement a two-tower model), new-item cold start, relationship-heavy domains, and scoring new accounts for risk
- Avoid it when there are no node features, the graph fits in memory, the graph is static, or engineering capacity is limited (sampling, a graph store and full-corpus inference are heavier than a two-tower model)
- Graph models bring cross-user structure and transfer under sparse behavior; two-tower models bring personalization, freshness and end-to-end alignment
### GAT (2018 · Cambridge / Mila) — masked self-attention over neighbors, multi-head
#### Problem it solved
- Spectral filters depend on the Laplacian eigenbasis and do not transfer across graphs; GCN's neighbor weights are fixed by degree
- GraphSAGE samples, so it never sees all neighbors, and its LSTM aggregator assumes an ordering
#### How it works
- Same skeleton as GCN; only the coefficient changes: 1/n for GraphSAGE-mean, 1/√(d_i·d_j) for GCN, a learned α_ij for GAT
- Score e_ij = LeakyReLU(aᵀ[W h_i ‖ W h_j]) with a a vector and negative slope 0.2; softmax over the neighborhood; h'_i = σ(Σ_j α_ij W h_j)
- Additive rather than dot-product attention; the key and the value are the same vector, W h_j
- Dot-product attention scores qᵀk/√d in one matrix multiply and parallelizes well; additive attention uses vᵀtanh(W₁q + W₂k), which is more flexible but slower
#### Limitations
- The full architecture does not suit large graphs (all neighbors × heads); it works for small graphs or when interpretability matters
#### In ML system design
- When neighbor importance varies (e.g. noisy co-occurrences), add attention to the aggregation layer without adopting the full GAT architecture
- Analogy to DIN: DIN attends over user history with the target item; GAT attends over neighbors with the center node
- Interpretability: α_ij explains "because your friend X bought it", or which relationship drove a risk score
### PinSage (2018 · Pinterest / Stanford) — a 3B-node GCN: random-walk sampling, curriculum hard negatives, MapReduce inference
#### Problem it solved
- GCNs operate on the full graph Laplacian and GraphSAGE still held the whole graph in GPU memory; DeepWalk and node2vec are unsupervised, featureless, and grow with the graph; pure content embeddings ignore the graph
#### Input and training data
- A pin–board bipartite graph: 2B pins, 1B boards, 18B edges
- Pin features, concatenated: visual 4096-d (VGG-16 fc6), text 256-d (word2vec on annotations), log degree
- Supervised pairs: a user engaged with pin q and then immediately with pin i — 1.2B positive pairs, 7.5B training examples in total
- 500 random negatives shared per minibatch + 6 hard negatives per pin
- Training on a 300M-item subgraph was already as good as the full graph, in one-sixth of the time
#### Six technical details
- ① Importance-based neighborhoods: short random walks from u, keep the top-T nodes by visit count (approximates Personalized PageRank); fixed size, plus an importance score per neighbor
- ② Importance pooling: a weighted mean using L1-normalized visit counts
- ③ Convolve: each neighbor through a dense layer + ReLU (1024 → 2048), weighted pooling, concatenate with self, dense + ReLU, L2-normalize; K = 2, hidden 2048, output 1024
- ④ Hard negatives with a curriculum: 500 random negatives resolve 1 in 500, but choosing the top 1,000 of 2B items needs about 1 in 2M; hard negatives come from PPR ranks 2,000–5,000, and epoch n adds n − 1 of them
- ⑤ Max-margin ranking loss, max(0, z_q·z_n − z_q·z_i + Δ) — retrieval is a ranking problem, not a classification problem
- ⑥ Engineering: producer–consumer minibatches (the CPU samples, re-indexes and gathers features while the GPU computes; training time nearly halved) and MapReduce inference that computes each node once
#### Output
- A 1024-d, L2-normalized embedding per pin (~4 TB in total), served through LSH-based nearest-neighbor search
#### In ML system design
- The paradigm is still standard: content features as node inputs, walk-based importance sampling, supervised objectives from real behavior, curriculum hard negatives, offline full inference + ANN
- A modern version: a user–item bipartite graph; CLIP / BERT / multimodal encoders; PPR sampling with time decay; importance pooling or attention; max-margin or sampled softmax with logQ correction; in-batch plus hard negatives; Spark / Ray or GPU batch inference; Faiss / ScaNN / HNSW
- Worth raising when asked whether graph methods reach production, how to choose neighbors or negatives, or how to engineer cold start; overkill for small graphs without content features
### GATNE (2019 · Tsinghua / Alibaba) — multiplex heterogeneous networks with attributes: base + edge-type attention + attribute embeddings
## D. ID embedding compression (model side)
### Overview
- Changes how embeddings are parameterized so the table itself shrinks — complementary to system-side sharding, caching and eviction
- Three directions: compose small tables (QR) → replace the table with a network (DHE) → allocate dimensions by frequency (mixed dimension)
### QR / compositional embeddings (2020 · Facebook) — two small tables compose a unique vector per ID
#### Problem it solved
- DLRM-style models have features with |S| ≈ 10⁷ categories and D ≈ 100; embedding tables are the main memory bottleneck
- The hashing trick (i mod m) maps very different categories to the same vector
- Codebook methods still store O(|S|·D) codes and can only shrink D
#### How it works
- Quotient–remainder trick: W₁ (m × D) indexed by i mod m and W₂ (|S|/m × D) indexed by i div m, combined element-wise; memory drops from O(|S|·D) to O(|S|/m·D + m·D), at best O(√|S|·D)
- Complementary partitions generalize it (multi-level QR, Chinese-remainder partitions, business attributes), down to O(k·|S|^(1/k)·D)
- Combining operators: concatenation (provably unique), addition, element-wise product (best overall); a path-based variant with small MLPs did not do better
- In practice, compress only tables above a row-count threshold
#### Results (Criteo Kaggle; DCN and DLRM)
- At 4 hash collisions, validation loss falls between the hashing trick and the full table
- With up to 60 collisions (~15× smaller), it matches or beats the hashing trick at 4 collisions; with at most 4 collisions it stays within 0.3% (DCN) and 0.7% (DLRM) of the full table
#### Limitations
- Ignores category frequency; imposed partitions can cost accuracy; IDs that share a remainder share parameters → DHE and mixed dimension push further
#### In ML system design
- The cheapest drop-in replacement for the hashing trick on huge ID features (user, device, feature crosses); Monolith uses QR as its online baseline
- Start with the largest tables, keep small tables full, and combine with FP16 and row-wise optimizers
### DHE (2021 · Google) — a dense hash encoding + deep MLP computes embeddings on the fly, with no table
#### Problem it solved
- Huge vocabularies (video IDs have no subword trick), IDs that appear daily, and heavily skewed distributions
- A table for 1B IDs × 100 dims is ~400 GB and cannot handle unseen values; hashing collides, and adding hash functions stops helping
#### How it works
- A good encoding is unique, treats all values as equally similar, is high-dimensional and has high entropy
- k = 1024 universal hash functions map each value into [1, 10⁶], normalized to [−1, 1] (uniform) or made Gaussian with Box–Muller; deterministic, with nothing to store
- A deep embedding network turns the encoding into a d-dim vector; parameters are independent of vocabulary size, and about 5 equal-width hidden layers work best
- The network underfits rather than overfits: Mish activations and BatchNorm help, dropout does not
- Appending generalizable side features (e.g. genre) to the encoding lets new and similar IDs share signal
#### Results
- Matches full-table AUC at 1/4 the model size in most settings and beats hashing baselines (QR is slightly better on the very sparse Amazon data)
- Only DHE keeps improving as k grows
- About 9× slower than a table lookup even on GPU
#### In ML system design
- Fits huge or fast-changing ID spaces and memory-constrained (on-device) settings; a poor fit for latency-critical, large-batch online ranking
- Hybrid: a table for head IDs, DHE for tail and new IDs
## E. Two-tower models and embedding-based retrieval
### Lineage
- DSSM → YouTube DNN → sampling-bias-corrected two-tower → mixed negative sampling → Facebook EBR
### DSSM (2013 · Microsoft) — the first two-tower model: letter-trigram word hashing + DNN + softmax on clicks
#### Problem it solved
- Keyword matching misses vocabulary mismatch; LSA, PLSA and LDA are unsupervised and loosely tied to ranking quality; earlier click-trained models had to prune the vocabulary heavily
#### How it works
- Word hashing: good → #good# → letter trigrams #go, goo, ood, od#; a 500K-word vocabulary becomes 30,621 dims with 22 collisions, and inflections and unseen words are handled naturally
- Bag of words → ~30K → 300 → 300 → 128 with tanh; query and document are encoded separately
- Softmax over cosine similarity, with 4 random unclicked documents per click
- Data: ~100M (query, clicked title) pairs; evaluated on 16,510 queries
#### Results
- NDCG@1 of 0.362, vs. 0.319 for TF-IDF and 0.308 for BM25
#### Limitations → what came next
- Bag of words loses word order (→ CLSM, LSTM-DSSM, BERT two-tower models); titles only; random negatives are easy; letter trigrams do not carry over directly to Chinese
#### In ML system design
- All three two-tower ingredients appear here: separate encoders, cosine / inner-product scoring, softmax with sampled negatives
- n-gram hashing for unseen words lives on in Facebook EBR
### YouTube DNN (2016 · Google) — retrieval as extreme multiclass classification + sampled softmax + ANN; ranking with weighted LR
#### Problem it solved
- Scale, freshness and noisy implicit feedback; the previous retrieval stage was matrix factorization trained with a rank loss
#### Retrieval
- Averaged watch and search embeddings + geography, device and demographics + example age → a ReLU tower (up to 2048 → 1024 → 512 → 256)
- Sampled softmax with importance weighting, more than 100× faster than a full softmax
- Serving: a user vector + ANN over video vectors (the softmax output weights)
- Example age: the sample's age during training, set to zero at serving time, removes the bias toward the training window's average popularity
- Training data: all watches (including off-site), a fixed number of samples per user, unordered search tokens, and predicting the next watch rather than a randomly held-out one
#### Ranking
- Weighted logistic regression with positives weighted by watch time, so the learned odds approximate expected watch time
- 1024 → 512 → 256 was best within the serving budget (weighted per-user loss 34.6%)
#### Limitations → what came next
- Videos are output-layer embeddings with a fixed vocabulary and no content features → the sampling-bias-corrected two-tower model
- The user tower is plain average pooling → sequence and multi-interest models
#### In ML system design
- The reference case for retrieval: sample construction, example age as a debiasing feature, sampled softmax, user vector + ANN
- Turning a continuous target into sample weights is a reusable trick
### Sampling-bias-corrected two-tower (2019 · Google) — in-batch softmax + logQ correction + streaming frequency estimation
#### Problem it solved
- Item towers with content features cannot afford thousands of sampled negatives
- In-batch negatives follow a power law and over-penalize popular items
- In streaming training, the vocabulary and distribution keep shifting, so there is no fixed frequency table
#### How it works
- Score s(x, y) = ⟨u, v⟩ / τ with L2-normalized towers, corrected to s − log p_j, where p_j is the probability that item j appears in a batch
- Streaming estimate: array A holds the step an item was last seen and array B the estimated gap, B ← (1 − α)·B + α·(t − A), p̂ = 1/B; multiple hash arrays, taking the max of B
- ID embeddings shared across the seed, candidate and history; unseen IDs go to hash buckets; data trained day by day in order; an index of ~10M videos rebuilt every few hours
- Reward-weighted batch softmax loss
#### Results
- YouTube offline Recall@50 from 0.4586 to 0.5322 with the correction
- Online engagement +0.20% without the correction, +0.37% with it
#### Limitations → what came next
- In-batch negatives only contain items that appear in the training data (selection bias) → mixed negative sampling
- logQ corrects the sampling probability, not exposure bias; the temperature needs careful tuning
#### In ML system design
- The standard two-tower recipe: L2 normalization + temperature + in-batch softmax + logQ + shared ID embeddings + hash buckets for unseen IDs + daily training + periodic index rebuilds
### Mixed negative sampling (2020 · Google) — in-batch negatives + uniformly sampled corpus negatives (Google Play)
#### Problem it solved
- Training logs from the old system concentrate on popular apps; apps that were never engaged are never negatives, so low-quality long-tail apps cannot be pushed down
#### How it works
- Add B′ items sampled uniformly from the full index; logits become B × (B + B′)
- The effective sampling distribution mixes the batch (unigram) and uniform distributions in proportion B : B′, with logQ correction
#### Results
- Recall@10: 0.4283 for the MLP baseline, 0.3987 for a two-tower model with batch negatives, 0.4473 with mixed negative sampling; B′ = 8192 was best at 0.4780
- Online: −1.46% with batch negatives only, +1.54% with mixed negative sampling
#### Limitations → what came next
- B′ is tuned empirically; uniform negatives are mostly easy → hard-negative mining in Facebook EBR; a full item-feature stream must be maintained
#### In ML system design
- In-batch negatives + logQ correction + uniform corpus negatives, with B′ tuned on offline recall
- "Why is the new two-tower model worse than the old one?" → most likely selection bias
### Facebook EBR (2020 · Facebook) — unified embeddings (text + social + location) + hard mining + ANN inside the inverted index
#### Problem it solved
- Social-search intent depends on the searcher and their context, not just the query text
- Embedding retrieval has to merge with Boolean term matching rather than run as a second index
#### How it works
- Two towers over text (character trigrams + word n-grams), searcher features and context; cosine similarity; triplet loss with a carefully tuned margin
- Negatives: random documents work far better than impressed-but-unclicked ones (−55% absolute recall on people search); clicks and impressions work equally well as positives
- Unified embeddings vs. text only: +18% recall on event search, +16% on group search
- Serving: Faiss coarse quantization (IVF / IMI) + PQ / OPQ inside the Unicorn retrieval engine, supporting hybrid Boolean + embedding queries
- ANN tuning: compare recall by documents scanned, retune after model changes, always try OPQ, and set PQ bytes to d/4
#### Results
- Online hard-negative mining: recall +8.38% on people, +7% on groups, +5.33% on events, with at most 2 hard negatives per positive
- Offline hard mining: the hardest negatives alone do worse than random; sampling from ANN ranks 101–500 works best; easy and hard negatives mixed at about 100:1
- Hard positives mined from failed sessions match click-based recall with 4% of the data
- Embedding ensembles, both weighted concatenation and cascades, add further recall
#### Limitations
- Triplet loss compares one negative at a time; online gains are not reported numerically; quantization error interacts with model difficulty
#### In ML system design
- A complete search EBR checklist: samples, features, loss, KNN and ANN recall, serving, cosine similarity as a ranking feature, relevance filtering
- Never train on hard negatives alone
</pre>

<script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/markmap-lib@0.18.12/dist/browser/index.iife.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/markmap-view@0.18.12/dist/browser/index.min.js"></script>
<script>
  (function () {
    var src = document.getElementById("embedding-map-src");
    var svg = document.getElementById("embedding-map");
    if (!src || !svg || !window.markmap || !window.markmap.Transformer) return;
    var mm = window.markmap;
    var result = new mm.Transformer().transform(src.textContent);
    var options = mm.deriveOptions({ initialExpandLevel: 3, maxWidth: 380, colorFreezeLevel: 2 });
    mm.Markmap.create(svg, options, result.root);
  })();
</script>

## References

**Sequence and shallow graph embeddings**
- Mikolov et al., [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546) (NeurIPS 2013)
- Perozzi et al., [DeepWalk: Online Learning of Social Representations](https://arxiv.org/abs/1403.6652) (KDD 2014)
- Tang et al., [LINE: Large-scale Information Network Embedding](https://arxiv.org/abs/1503.03578) (WWW 2015)
- Grover & Leskovec, [node2vec: Scalable Feature Learning for Networks](https://arxiv.org/abs/1607.00653) (KDD 2016)
- Ribeiro et al., [struc2vec: Learning Node Representations from Structural Identity](https://arxiv.org/abs/1704.03165) (KDD 2017)
- Qiu et al., [Network Embedding as Matrix Factorization: Unifying DeepWalk, LINE, PTE, and node2vec](https://arxiv.org/abs/1710.02971) (WSDM 2018)
- Wang et al., [Billion-scale Commodity Embedding for E-commerce Recommendation in Alibaba](https://arxiv.org/abs/1803.02349) (KDD 2018)

**Graph neural networks**
- Kipf & Welling, [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) (ICLR 2017)
- Hamilton et al., [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216) (NeurIPS 2017)
- Veličković et al., [Graph Attention Networks](https://arxiv.org/abs/1710.10903) (ICLR 2018)
- Ying et al., [Graph Convolutional Neural Networks for Web-Scale Recommender Systems](https://arxiv.org/abs/1806.01973) (KDD 2018)
- Cen et al., [Representation Learning for Attributed Multiplex Heterogeneous Network](https://arxiv.org/abs/1905.01669) (KDD 2019)

**Embedding compression**
- Shi et al., [Compositional Embeddings Using Complementary Partitions for Memory-Efficient Recommendation Systems](https://arxiv.org/abs/1909.02107) (KDD 2020)
- Kang et al., [Learning to Embed Categorical Features without Embedding Tables for Recommendation](https://arxiv.org/abs/2010.10784) (KDD 2021)

**Two-tower models and embedding-based retrieval**
- Huang et al., [Learning Deep Structured Semantic Models for Web Search using Clickthrough Data](https://doi.org/10.1145/2505515.2505665) (CIKM 2013)
- Covington et al., [Deep Neural Networks for YouTube Recommendations](https://doi.org/10.1145/2959100.2959190) (RecSys 2016)
- Yi et al., [Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations](https://doi.org/10.1145/3298689.3346996) (RecSys 2019)
- Yang et al., [Mixed Negative Sampling for Learning Two-tower Neural Networks in Recommendations](https://doi.org/10.1145/3366424.3386195) (WWW Companion 2020)
- Huang et al., [Embedding-based Retrieval in Facebook Search](https://arxiv.org/abs/2006.11632) (KDD 2020)

Numbers in the map are as reported in each paper. The mind map is rendered with [markmap](https://markmap.js.org/).
