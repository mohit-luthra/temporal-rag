# Temporal RAG

**Exponential temporal decay weighting for time-aware document retrieval.**

Standard RAG scores documents by semantic similarity alone. A 2020 strategy document and a 2024 strategy document on the same topic receive identical retrieval scores for the same query. This repository introduces a single modification to the scoring function that fixes this — no retraining, no architectural changes, no additional models.

---

## The problem

In any domain where knowledge evolves over time — marketing strategy, financial analysis, competitive intelligence, policy — a knowledge base accumulates documents from multiple periods on the same topic. Standard retrieval has no mechanism to prefer the current one.

```
Normal RAG:    score = cosine_similarity(query, document)

               A 2020 document and a 2024 document on the same topic
               receive the same score for the same query.
               The LLM gets outdated evidence.
```

## The fix

```
Temporal RAG:  score = cosine_similarity(query, document)
                       x exp( -log(2) / h * days_since_document )

               h = half-life in days (domain-specific, set by practitioner)
               Recent documents score higher. Old documents decay smoothly.
               No training. No new models. One extra line of math.
```

The decay function is identical to the adstock carry-over formula used in marketing mix modelling, where `h` is the half-life: the age at which a document retains exactly 50% of its original temporal weight.

---

## Results

Evaluated on a 15-query benchmark over a 13-document marketing analytics corpus spanning 2020-2024. Each topic has documents from multiple years; the retriever must surface the temporally appropriate one.

| Metric | Normal RAG | Temporal RAG | Delta |
|---|---|---|---|
| Precision@1 | 0.400 | 0.667 | +26.7 pp |
| MRR | 0.656 | 0.822 | +16.7 pp |
| NDCG@3 | 0.728 | 0.868 | +14.1 pp |
| Failures (of 15) | 9 | 5 | -4 |

Half-life set to 365 days (annual strategy corpus). Both systems use identical TF-IDF vectorisation — the only variable is the scoring function.

### Sensitivity to half-life parameter

| h (days) | Precision@1 |
|---|---|
| 30 | 0.267 |
| 90 | 0.467 |
| 180 | 0.667 |
| 365 | 0.667 |
| 730 | 0.667 |
| no decay | 0.400 |

The method outperforms the baseline across all half-life values above 90 days. For annual planning corpora, precise calibration within the 180-730 day range is not critical.

---

## Repository structure

```
temporal-rag/
    corpus/                      13 documents across 5 topic areas
        influencer_2020_q4.txt   Swiggy influencer strategy FY2021
        influencer_2022_h1.txt   Influencer channel audit H1 FY2023
        influencer_2024_q3.txt   Current strategy Q3 FY2025
        tv_media_2021.txt        TV effectiveness study FY2021
        tv_media_2024.txt        Integrated media review Q2 FY2025
        digital_mix_2021.txt     Digital channel performance H1 FY2022
        digital_mix_2024.txt     Digital media strategy Q3 FY2025
        mmm_methodology_2021.txt MMM technical documentation v1.0
        mmm_methodology_2024.txt Bayesian MMM documentation v3.2
        budget_allocation_2020.txt  Annual media plan FY2021
        budget_allocation_2024.txt  Media investment strategy FY2025
        competitive_2021.txt     Zomato IPO impact assessment
        competitive_2024.txt     Competitive intelligence Q2 FY2025

    temporal_rag.py              TemporalRAG class (the library)
    evaluate.py                  Benchmark runner, computes IR metrics
    benchmark.json               15 queries with ground truth document IDs
    requirements.txt             Dependencies
```

---

## Quickstart

```bash
git clone https://github.com/mohit-luthra/temporal-rag.git
cd temporal-rag
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python evaluate.py
```

Output:

```
Corpus: 13 documents  |  Queries: 15  |  Half-life: 365 days

Query   Normal RAG                                        Temporal RAG
q01     [2020-11-15] influencer_2020_q4 (0.124)          [2024-10-22] influencer_2024_q3 (0.065)
q02     [2022-07-08] influencer_2022_h1 (0.123)          [2024-10-22] influencer_2024_q3 (0.045)
...

Metric          Normal RAG    Temporal RAG    Delta
Precision@1       0.4000        0.6667       +0.2667
MRR               0.6556        0.8222       +0.1667
NDCG@3            0.7278        0.8682       +0.1405
```

Run with a different half-life:

```bash
python evaluate.py --half_life 90
```

---

## Using the library

```python
from temporal_rag import TemporalRAG, Document

rag = TemporalRAG(half_life_days=365)

rag.load_corpus_from_directory("corpus/")

results = rag.retrieve(
    query="What is the current influencer strategy?",
    query_date="2024-12-31",
    top_k=3,
    temporal=True
)

for r in results:
    print(f"[{r.date}] {r.doc_id}")
    print(f"  cosine={r.cosine_score:.3f}  decay={r.decay_weight:.3f}  final={r.final_score:.3f}")
```

To load your own documents, each `.txt` file must have `date: YYYY-MM-DD` as its first line:

```
date: 2024-07-15
source: Q2 Campaign Analysis
author: Analytics team

Your document content here...
```

---

## Half-life calibration guide

| Domain | Recommended h | Rationale |
|---|---|---|
| News / breaking events | 7-30 days | Information superseded within weeks |
| Campaign performance reports | 60-90 days | Quarterly reporting cycle |
| Marketing strategy documents | 180-365 days | Annual planning cycle |
| Methodology / academic papers | 730-1825 days | Foundational methods remain relevant for years |

These mirror the domain-specific adstock half-lives used in marketing mix modelling.

---

## Why exponential decay

Three alternatives were considered.

**Hard cutoff** ("ignore documents older than N days") creates a cliff effect: a document one day past the threshold is treated identically to one ten years past it. The threshold choice is arbitrary and sensitive.

**Linear decay** implies a specific obsolescence event at the zero-weight boundary. There is no theoretical basis for this in knowledge domains.

**Exponential decay** is the unique continuous function satisfying the memoryless property: the probability that information remains current, given it has survived to age t, is independent of t. One interpretable parameter. Degrades gracefully to standard cosine retrieval as h approaches infinity.

---

## Benchmark design

The benchmark is constructed to test temporal discrimination, not lexical matching. Each topic has documents from 2-3 different years using similar vocabulary. A retriever that cannot distinguish by time will score the wrong-year document at similar or higher cosine similarity.

Two queries (q13, q14) use earlier query dates (2021 and 2022) to test that Temporal RAG does not blindly prefer the newest document in the corpus — it must prefer the newest document *relative to the query date*. Both systems fail these queries, which is reported honestly in the paper.

---

## Paper

**Temporal Decay Weighting for Time-Aware Retrieval-Augmented Generation**
Mohit Luthra, CSA (Havas)

Preprint: [arXiv link — to be added after submission]

---

## Requirements

```
scikit-learn
numpy
```

Python 3.9 or above. No GPU required. No API keys required for evaluation.

---

## Author

Mohit Luthra
Data Scientist, CSA - Havas, New Delhi
