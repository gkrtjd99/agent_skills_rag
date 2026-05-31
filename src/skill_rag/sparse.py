"""Minimal in-memory BM25 for lexical (keyword) retrieval.

The dense embedding model is cross-lingual but weak on short text and exact
tokens (skill names, "vercel", "EXPLAIN ANALYZE"). BM25 over the full skill
text complements it. With ~100 skills, building the index per query is cheap,
so there is no persistence and no extra dependency.
"""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25:
    """Okapi BM25 over a fixed list of tokenized documents."""

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs = docs
        self.n = len(docs)
        self.doc_len = [len(d) for d in docs]
        self.avgdl = (sum(self.doc_len) / self.n) if self.n else 0.0
        self.tf = [Counter(d) for d in docs]
        df: Counter[str] = Counter()
        for d in docs:
            for term in set(d):
                df[term] += 1
        # Probabilistic idf with +1 so common terms stay non-negative.
        self.idf = {
            term: math.log(1 + (self.n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def scores(self, query_tokens: list[str]) -> list[float]:
        out = [0.0] * self.n
        if not self.n or self.avgdl == 0.0:
            return out
        for term in query_tokens:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i in range(self.n):
                freq = self.tf[i].get(term, 0)
                if not freq:
                    continue
                denom = freq + self.k1 * (
                    1 - self.b + self.b * self.doc_len[i] / self.avgdl
                )
                out[i] += idf * (freq * (self.k1 + 1)) / denom
        return out
