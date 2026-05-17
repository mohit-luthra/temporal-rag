"""
temporal_rag.py

Retrieval-Augmented Generation with exponential temporal decay weighting.

Retrieval scoring:
    score(q, d) = cosine_similarity(q, d) * exp(-lambda * delta_t)

where delta_t is days between document date and query date, and
lambda = log(2) / half_life ensures 50% weight at exactly half_life days.

Embeddings: sentence-transformers all-MiniLM-L6-v2 (semantic, 384-dim).
Falls back to TF-IDF if sentence-transformers is not installed.
"""

import math
import numpy as np
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict

try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except ImportError:
    _HAS_ST = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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
        Days at which temporal weight equals 0.5.
    model_name : str
        Sentence-transformers model. Ignored if not installed (falls back to TF-IDF).
    """

    def __init__(self, half_life_days: float = 365, model_name: str = "all-MiniLM-L6-v2"):
        self.half_life_days = half_life_days
        self.lam = math.log(2) / half_life_days
        self.corpus: List[dict] = []
        self._fitted = False

        if _HAS_ST:
            print(f"Loading sentence-transformer: {model_name}")
            self._encoder = SentenceTransformer(model_name)
            self._mode = "dense"
            print("Semantic embeddings active.")
        else:
            print("sentence-transformers not found. Using TF-IDF fallback.")
            self._encoder = None
            self._vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 2), sublinear_tf=True)
            self._mode = "tfidf"

    def load_corpus_from_directory(self, directory: str) -> int:
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
                raise ValueError(f"{filepath.name}: missing 'date:' on first line")
            documents.append({
                "doc_id": filepath.stem,
                "text": "\n".join(lines),
                "date": date_str,
                "metadata": {"source_file": filepath.name}
            })
        self._fit(documents)
        return len(documents)

    def _fit(self, documents: List[dict]):
        texts = [d["text"] for d in documents]
        if self._mode == "dense":
            vectors = self._encoder.encode(
                texts, show_progress_bar=True,
                convert_to_numpy=True, normalize_embeddings=True
            )
        else:
            vectors = self._vectorizer.fit_transform(texts).toarray()
        self.corpus = [{**d, "vector": vectors[i]} for i, d in enumerate(documents)]
        self._fitted = True
        print(f"Corpus loaded: {len(self.corpus)} documents ({self._mode}, dim={vectors.shape[1]})")

    def _temporal_weight(self, doc_date: str, query_date: str) -> float:
        d_doc = datetime.strptime(doc_date, "%Y-%m-%d")
        d_query = datetime.strptime(query_date, "%Y-%m-%d")
        delta = max(0, (d_query - d_doc).days)
        return math.exp(-self.lam * delta)

    def retrieve(self, query: str, query_date: str, top_k: int = 3, temporal: bool = True) -> List[RetrievedDocument]:
        if not self._fitted:
            raise RuntimeError("Call load_corpus_from_directory() first.")
        if self._mode == "dense":
            query_vector = self._encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        else:
            query_vector = self._vectorizer.transform([query]).toarray()
        results = []
        for entry in self.corpus:
            cos = float(cosine_similarity(query_vector, [entry["vector"]])[0][0])
            decay = self._temporal_weight(entry["date"], query_date) if temporal else 1.0
            results.append(RetrievedDocument(
                doc_id=entry["doc_id"], text=entry["text"], date=entry["date"],
                cosine_score=cos, decay_weight=decay, final_score=cos * decay,
                metadata=entry["metadata"]
            ))
        results.sort(key=lambda r: r.final_score, reverse=True)
        return results[:top_k]

    def corpus_size(self) -> int:
        return len(self.corpus)
