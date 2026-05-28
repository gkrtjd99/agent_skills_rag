from __future__ import annotations

from .corpus import SCORE_THRESHOLD
from .embed import DEFAULT_MODEL, encode_one
from .index import search as _index_search


def search(query: str, k: int = 5, model_name: str = DEFAULT_MODEL) -> dict:
    """Return {status, hits[, message]}.

    status:
      - "ok": at least one hit with score >= SCORE_THRESHOLD
      - "no_match": empty corpus, or all hits below threshold
    """
    query = query.strip()
    if not query:
        return {
            "status": "no_match",
            "hits": [],
            "message": "Empty query. Proceed without using a skill.",
        }
    vec = encode_one(query, name=model_name)
    raw = _index_search(vec, k=k)
    relevant = [h for h in raw if h.score >= SCORE_THRESHOLD]
    if not relevant:
        return {
            "status": "no_match",
            "hits": [],
            "message": "No skill matched this query. Proceed without using a skill.",
        }
    return {
        "status": "ok",
        "hits": [
            {"name": h.name, "description": h.description, "score": round(h.score, 4)}
            for h in relevant
        ],
    }
