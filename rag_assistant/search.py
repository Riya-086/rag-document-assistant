import math
from collections import Counter
from dataclasses import dataclass

from .text import tokens


@dataclass(frozen=True)
class Chunk:
    id: int
    document_id: int
    document: str
    text: str


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class BM25Index:
    """Okapi BM25 ranking over document chunks. Small, dependency-free and easy to reason about."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self._chunks: dict[int, Chunk] = {}
        self._terms: dict[int, Counter] = {}
        self._df: Counter = Counter()

    def add(self, chunk: Chunk) -> None:
        terms = Counter(tokens(chunk.text))
        self._chunks[chunk.id], self._terms[chunk.id] = chunk, terms
        self._df.update(terms.keys())

    def remove_document(self, document_id: int) -> None:
        for chunk_id in [c.id for c in self._chunks.values() if c.document_id == document_id]:
            self._df.subtract(self._terms[chunk_id].keys())
            del self._chunks[chunk_id], self._terms[chunk_id]
        self._df = +self._df  # drop zero counts

    def __len__(self) -> int:
        return len(self._chunks)

    def search(self, query: str, k: int = 4, min_score: float = 0.5) -> list[Hit]:
        query_terms = set(tokens(query))
        if not query_terms or not self._chunks:
            return []
        n = len(self._chunks)
        avg_len = sum(sum(t.values()) for t in self._terms.values()) / n
        hits = []
        for chunk_id, terms in self._terms.items():
            length = sum(terms.values())
            score = 0.0
            for term in query_terms & terms.keys():
                idf = math.log(1 + (n - self._df[term] + 0.5) / (self._df[term] + 0.5))
                tf = terms[term]
                score += idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * length / avg_len))
            if score >= min_score:
                hits.append(Hit(self._chunks[chunk_id], round(score, 3)))
        return sorted(hits, key=lambda h: h.score, reverse=True)[:k]
