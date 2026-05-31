from __future__ import annotations

from .corpus import BM25_THRESHOLD, RRF_K, SCORE_THRESHOLD
from .embed import DEFAULT_MODEL, encode_one
from .index import list_indexed as _list_indexed
from .index import search as _index_search
from .sparse import BM25, tokenize

_NO_MATCH = "No skill matched this query. Proceed without using a skill."


def search(query: str, k: int = 5, model_name: str = DEFAULT_MODEL) -> dict:
    """Hybrid search over the skill corpus: dense cosine + BM25, fused by RRF.

    A candidate is kept if its dense cosine clears ``SCORE_THRESHOLD`` OR its
    BM25 score (normalized by the top BM25 score) clears ``BM25_THRESHOLD``.
    Kept candidates are ordered by reciprocal rank fusion of the two rankings.

    status:
      - "ok": at least one candidate passes either threshold
      - "no_match": empty corpus, or nothing passes
    """
    query = query.strip()
    if not query:
        return {"status": "no_match", "hits": [], "message": "Empty query. Proceed without using a skill."}

    rows = _list_indexed()
    if not rows:
        return {"status": "no_match", "hits": [], "message": _NO_MATCH}

    # --- dense: cosine over the whole corpus so every candidate has a score ---
    vec = encode_one(query, name=model_name)
    dense = _index_search(vec, k=len(rows))
    cosine = {h.name: h.score for h in dense}
    dense_rank = {h.name: i for i, h in enumerate(dense)}
    description = {h.name: h.description for h in dense}
    for r in rows:  # fall back to indexed metadata for anything dense dropped
        description.setdefault(r["name"], r["description"])

    # --- sparse: BM25 over the full indexed text ---
    names = [r["name"] for r in rows]
    bm25 = BM25([tokenize(r["text"]) for r in rows])
    raw_bm25 = bm25.scores(tokenize(query))
    bm25_by_name = dict(zip(names, raw_bm25))
    top_bm25 = max(raw_bm25) if raw_bm25 else 0.0
    norm_bm25 = {
        n: (s / top_bm25 if top_bm25 > 0 else 0.0) for n, s in bm25_by_name.items()
    }
    sparse_order = sorted(
        (n for n in names if bm25_by_name[n] > 0),
        key=lambda n: bm25_by_name[n],
        reverse=True,
    )
    sparse_rank = {n: i for i, n in enumerate(sparse_order)}

    # --- keep candidates that pass either signal, order by RRF ---
    kept = [
        n
        for n in names
        if cosine.get(n, 0.0) >= SCORE_THRESHOLD or norm_bm25.get(n, 0.0) >= BM25_THRESHOLD
    ]
    if not kept:
        return {"status": "no_match", "hits": [], "message": _NO_MATCH}

    def rrf(n: str) -> float:
        score = 0.0
        if n in dense_rank:
            score += 1.0 / (RRF_K + dense_rank[n])
        if n in sparse_rank:
            score += 1.0 / (RRF_K + sparse_rank[n])
        return score

    kept.sort(key=rrf, reverse=True)
    hits = [
        {
            "name": n,
            "description": description.get(n, ""),
            "score": round(cosine.get(n, 0.0), 4),
        }
        for n in kept[:k]
    ]
    return {"status": "ok", "hits": hits}
