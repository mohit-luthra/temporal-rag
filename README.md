# Temporal RAG

**Exponential temporal decay weighting for time-aware document retrieval.**

Standard RAG scores documents by semantic similarity alone. A 2019 strategy document and a 2024 strategy document on the same topic receive identical retrieval scores for the same query. This repository introduces a single modification to the scoring function that fixes this — no retraining, no architectural changes, no additional models.

---

## The Problem

In any domain where knowledge evolves over time — marketing strategy, financial analysis, competitive intelligence, policy — a knowledge base accumulates documents from multiple periods on the same topic. Standard retrieval has no mechanism to prefer the current one.

```
Normal RAG:   score = cosine_similarity(query, document)

              A 2019 document and a 2024 document on the same topic
              receive the same score for the same query.
              The LLM gets outdated evidence.
```

## The Fix

```
Temporal RAG: score = cosine_similarity(query, document)
                      x exp( -log(2) / h * max(0, days_since_document) )

              h = half-life in days (domain-specific, set by practitioner)
              Recent documents score higher. Old documents decay smoothly.
              No training. No new models. One extra line of math.
```

The `max(0, ...)` operator ensures future-dated documents receive a neutral weight of 1.0 rather than an inflated boost. The decay function is identical to the adstock carry-over formula used in marketing mix modelling, where `h` is the half-life: the age at which a document retains exactly 50% of its original temporal weight.

---

## Results

Evaluated on a 99-query benchmark over a 47-document marketing analytics corpus spanning 2017 to 2024. Seven topic areas with 5 to 8 documents per topic. Both systems use `all-MiniLM-L6-v2` dense embeddings.

### Main results (h = 1825 days)

| Metric | Normal RAG | 95% CI | Temporal RAG | 95% CI | Delta |
|---|---|---|---|---|---|
| Precision@1 | 0.424 | [0.323-0.525] | 0.566 | [0.465-0.667] | +14.2 pp |
| MRR | 0.515 | [0.430-0.601] | 0.685 | [0.607-0.759] | +17.0 pp |
| NDCG@3 | 0.549 | [0.465-0.633] | 0.725 | [0.651-0.795] | +17.6 pp |
| Failures (of 99) | 57 | | 43 | | -14 |

Bootstrap 95% CI over 5,000 resamples.

### Half-life sensitivity

| h (days) | P@1 | MRR | NDCG@3 |
|---|---|---|---|
| 365 (aggressive) | 0.394 | 0.520 | 0.563 |
| 730 | 0.495 | 0.621 | 0.664 |
| 1095 | 0.505 | 0.648 | 0.697 |
| **1825 (best)** | **0.566** | **0.685** | **0.725** |
| no decay (baseline) | 0.424 | 0.515 | 0.549 |

The crossover from harmful to helpful occurs near h = 700 days, consistent with the annual planning cycle of the domain.

### Per-domain results (h = 730 days)

| Domain | Normal RAG | Temporal RAG | Delta |
|---|---|---|---|
| Digital mix | 0.286 | 0.571 | +0.286 |
| TV media | 0.357 | 0.571 | +0.214 |
| Influencer | 0.500 | 0.714 | +0.214 |
| Competitive | 0.286 | 0.357 | +0.071 |
| Attribution | 0.533 | 0.533 | +0.000 |
| MMM methodology | 0.643 | 0.643 | +0.000 |
| Budget allocation | 0.357 | 0.071 | -0.286 |

---

## Repository Structure

```
temporal-rag/
    corpus_large/            47 documents across 7 topic areas (2017-2024)
    temporal_rag.py          TemporalRAG class (the library)
    evaluate.py              Benchmark runner with IR metrics
    benchmark.json           99 queries with ground truth document IDs
    requirements.txt         Dependencies
```

---

## Quickstart

```bash
git clone https://github.com/mohit-luthra/temporal-rag.git
cd temporal-rag
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

python evaluate.py --corpus corpus_large --benchmark benchmark.json --half_life 1825
```

Expected output:

```
Corpus: 47 documents  |  Queries: 99  |  Half-life: 1825 days

Metric          Normal RAG    Temporal RAG    Delta
Precision@1       0.4242        0.5657       +0.1414
MRR               0.5152        0.6852       +0.1700
NDCG@3            0.5486        0.7245       +0.1759
```

Run the full half-life scan:

```bash
python evaluate.py --corpus corpus_large --benchmark benchmark.json --half_life 365
python evaluate.py --corpus corpus_large --benchmark benchmark.json --half_life 730
python evaluate.py --corpus corpus_large --benchmark benchmark.json --half_life 1095
python evaluate.py --corpus corpus_large --benchmark benchmark.json --half_life 1825
```

---

## Using the Library

```python
from temporal_rag import TemporalRAG

rag = TemporalRAG(half_life_days=1825)
rag.load_corpus_from_directory("corpus_large/")

results = rag.retrieve(
    query="What is the current influencer marketing strategy?",
    query_date="2024-12-31",
    top_k=3,
    temporal=True
)

for r in results:
    print(f"[{r.date}] {r.doc_id}")
    print(f"  cosine={r.cosine_score:.3f}  decay={r.decay_weight:.3f}  final={r.final_score:.3f}")
```

Each document in the corpus must have `date: YYYY-MM-DD` as its first line:

```
date: 2024-07-15
source: Q2 Campaign Analysis

Your document content here...
```

---

## Half-Life Calibration Guide

| Domain | Recommended h | Rationale |
|---|---|---|
| News / breaking events | 7-30 days | Superseded within weeks |
| Campaign performance reports | 60-90 days | Quarterly reporting cycle |
| Marketing strategy documents | 730-1825 days | Annual planning cycle |
| Methodology / academic papers | 1825+ days | Foundational relevance |

For strategy-class documents, use a longer half-life. The benchmark shows that h = 365 days over-penalises last year's strategy document, while h = 1825 days provides consistent improvement across all metrics.

---

## Why Exponential Decay

Three alternatives were considered.

**Hard cutoff** creates a cliff effect: a document one day past the threshold is treated identically to one ten years past it. The threshold choice is arbitrary and sensitive.

**Linear decay** implies a specific obsolescence date at the zero-weight boundary. There is no theoretical basis for this in knowledge domains.

**Exponential decay** is the unique continuous function satisfying the memoryless property: the probability that information remains current, given it has survived to age t, is independent of t. One interpretable parameter. Degrades gracefully to standard cosine retrieval as h approaches infinity.

---

## Benchmark Design

The benchmark tests temporal discrimination across 7 domains. Each topic has 5 to 8 documents from different years using overlapping vocabulary. A retriever that cannot distinguish by time will score the wrong-year document at similar or higher cosine similarity.

Queries target either the most recent or second most recent document in their topic, with query dates set to December 31st of the ground truth document's year. This operationalises the primary real-world use case: a practitioner querying a knowledge base for current or recent guidance.

---

## Paper

**Temporal Decay Weighting for Time-Aware Retrieval-Augmented Generation**
Mohit Luthra, CSA Havas, New Delhi

Preprint: [arXiv link to be added after submission]

---

## Requirements

```
sentence-transformers
scikit-learn
numpy
```

Python 3.9 or above. No GPU required. The sentence-transformer model (`all-MiniLM-L6-v2`) downloads automatically on first run.

---

## Author

Mohit Luthra
Data Scientist, CSA Havas, New Delhi
