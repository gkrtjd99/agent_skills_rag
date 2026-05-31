from skill_rag.sparse import BM25, tokenize


def test_tokenize_lowercases_and_splits():
    assert tokenize("Deploy to Vercel!") == ["deploy", "to", "vercel"]


def test_tokenize_keeps_unicode_word_chars():
    assert tokenize("배포 vercel") == ["배포", "vercel"]


def test_bm25_ranks_matching_doc_first():
    docs = [
        tokenize("deploy a website to vercel preview url"),
        tokenize("review code for bugs and missing tests"),
        tokenize("write failing tests first then implement"),
    ]
    bm25 = BM25(docs)
    scores = bm25.scores(tokenize("vercel deploy"))
    assert len(scores) == 3
    assert scores[0] == max(scores)
    assert scores[0] > 0.0


def test_bm25_unknown_terms_score_zero():
    docs = [tokenize("alpha beta"), tokenize("gamma delta")]
    bm25 = BM25(docs)
    assert bm25.scores(tokenize("zzz nonexistent")) == [0.0, 0.0]


def test_bm25_empty_corpus():
    bm25 = BM25([])
    assert bm25.scores(tokenize("anything")) == []
