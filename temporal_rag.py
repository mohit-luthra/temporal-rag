"""
temporal_rag.py

Retrieval-Augmented Generation with exponential temporal decay weighting.

Retrieval scoring:
    score(q, d) = cosine_similarity(q, d) * exp(-lambda * delta_t)

where delta_t is days between document date and query date, and
lambda = log(2) / half_life ensures 50% weight at exactly half_life days.

This mirrors the adstock decay function used in marketing mix models.
"""

import json
import math
import numpy as np
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Document:
    doc_id: str
    text: str
    date: str          # ISO format: YYYY-MM-DD
    metadata: Dict = field(default_factory=dict)


@dataclass
class RetrievedDocument:
    doc_id: str
    text: str
    date: str
    cosine_score: float
    decay_weight: float
    final_score: float
    metadata: Dict = field(default_factory=dict)


class TemporalRAG:
    """
    RAG retriever with exponential temporal decay.

    Parameters
    ----------
    half_life_days : float
        Number of days at which a document's temporal weight equals 0.5.
        Domain guidance:
            News / campaign reports : 30–90
            Marketing strategy      : 180–365
            Methodology / academic  : 730–1825
    """

    def __init__(self, half_life_days: float = 365):
        self.half_life_days = half_life_days
        self.lam = math.log(2) / half_life_days
        self.vectorizer = TfidfVectorizer(
            min_df=1,
            ngram_range=(1, 2),
            sublinear_tf=True
        )
        self.corpus: List[dict] = []
        self._fitted = False

    def load_corpus_from_directory(self, directory: str) -> int:
        """
        Load .txt documents from a directory. Each file must contain a header
        line of the form 'date: YYYY-MM-DD' as the first line.

        Returns number of documents loaded.
        """
        path = Path(directory)
        documents = []
        for filepath in sorted(path.glob("*.txt")):
            lines = filepath.read_text(encoding="utf-8").strip().splitlines()
            date_str = None
            for line in lines[:5]:
                if line.lower().startswith("date:"):
                    date_str = line.split(":", 1)[1].strip()
                    break
            if date_str is None:
                raise ValueError(f"{filepath.name}: missing 'date:' header on first line")
            text = "\n".join(lines)
            documents.append(Document(
                doc_id=filepath.stem,
                text=text,
                date=date_str,
                metadata={"source_file": filepath.name}
            ))
        self._fit(documents)
        return len(documents)

    def load_corpus(self, documents: List[Document]):
        self._fit(documents)

    def _fit(self, documents: List[Document]):
        texts = [d.text for d in documents]
        vectors = self.vectorizer.fit_transform(texts).toarray()
        self.corpus = [
            {
                "doc_id": d.doc_id,
                "text": d.text,
                "date": d.date,
                "metadata": d.metadata,
                "vector": vectors[i]
            }
            for i, d in enumerate(documents)
        ]
        self._fitted = True

    def _temporal_weight(self, doc_date: str, query_date: str) -> float:
        d_doc = datetime.strptime(doc_date, "%Y-%m-%d")
        d_query = datetime.strptime(query_date, "%Y-%m-%d")
        delta = max(0, (d_query - d_doc).days)
        return math.exp(-self.lam * delta)

    def retrieve(
        self,
        query: str,
        query_date: str,
        top_k: int = 3,
        temporal: bool = True
    ) -> List[RetrievedDocument]:
        if not self._fitted:
            raise RuntimeError("Corpus not loaded. Call load_corpus_from_directory() first.")

        query_vector = self.vectorizer.transform([query]).toarray()
        results = []
        for entry in self.corpus:
            cos = float(cosine_similarity(query_vector, [entry["vector"]])[0][0])
            decay = self._temporal_weight(entry["date"], query_date) if temporal else 1.0
            results.append(RetrievedDocument(
                doc_id=entry["doc_id"],
                text=entry["text"],
                date=entry["date"],
                cosine_score=cos,
                decay_weight=decay,
                final_score=cos * decay,
                metadata=entry["metadata"]
            ))

        results.sort(key=lambda r: r.final_score, reverse=True)
        return results[:top_k]

    def corpus_size(self) -> int:
        return len(self.corpus)
