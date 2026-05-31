import pytest

from skill_rag import index as index_mod
from skill_rag import retrieve
from skill_rag.models import SkillRecord


@pytest.fixture(autouse=True)
def isolated_index(tmp_path, monkeypatch):
    monkeypatch.setenv("SKILL_RAG_INDEX_PATH", str(tmp_path / "index.lance"))
    import importlib
    importlib.reload(index_mod)
    importlib.reload(retrieve)
    yield
    index_mod.reset()


def _seed():
    index_mod.upsert([
        SkillRecord(
            name="brainstorming",
            description="explore ideas, requirements, and design before implementation",
            path="/x/brainstorming/SKILL.md",
            body="b",
            content_hash="h1",
        ),
        SkillRecord(
            name="tdd",
            description="write failing tests first then implement",
            path="/x/tdd/SKILL.md",
            body="b",
            content_hash="h2",
        ),
    ])


def test_relevant_query_returns_ok(monkeypatch):
    monkeypatch.setattr(retrieve, "SCORE_THRESHOLD", 0.0)
    _seed()
    res = retrieve.search("explore design ideas")
    assert res["status"] == "ok"
    assert len(res["hits"]) >= 1
    assert all("name" in h and "description" in h and "score" in h for h in res["hits"])


def test_threshold_filters_out_low_scores(monkeypatch):
    monkeypatch.setattr(retrieve, "SCORE_THRESHOLD", 0.99)
    _seed()
    res = retrieve.search("totally unrelated random words asdf qwer")
    assert res["status"] == "no_match"
    assert res["hits"] == []
    assert "message" in res


def test_empty_corpus_returns_no_match():
    res = retrieve.search("anything")
    assert res["status"] == "no_match"
    assert res["hits"] == []


def test_lexical_rescue_when_dense_below_threshold(monkeypatch):
    # Dense can never pass; only the BM25 keyword path can rescue a hit.
    monkeypatch.setattr(retrieve, "SCORE_THRESHOLD", 0.99)
    monkeypatch.setattr(retrieve, "BM25_THRESHOLD", 0.1)
    index_mod.upsert([
        SkillRecord(
            name="vercel-deploy",
            description="ship apps",
            path="/x/vercel-deploy/SKILL.md",
            body="use this when you deploy a website to vercel and want a preview url",
            content_hash="h1",
        ),
        SkillRecord(
            name="brainstorming",
            description="explore ideas",
            path="/x/brainstorming/SKILL.md",
            body="design the feature first",
            content_hash="h2",
        ),
    ])
    res = retrieve.search("vercel deploy", k=5)
    assert res["status"] == "ok"
    names = [h["name"] for h in res["hits"]]
    assert names[0] == "vercel-deploy"


def test_body_keyword_matches_when_description_does_not(monkeypatch):
    # The keyword lives only in the body, not the one-line description.
    monkeypatch.setattr(retrieve, "SCORE_THRESHOLD", 0.99)
    monkeypatch.setattr(retrieve, "BM25_THRESHOLD", 0.1)
    index_mod.upsert([
        SkillRecord(
            name="worktrees",
            description="isolate your workspace",
            path="/x/worktrees/SKILL.md",
            body="run git worktree to separate parallel branches cleanly",
            content_hash="h1",
        ),
    ])
    res = retrieve.search("git worktree", k=5)
    assert res["status"] == "ok"
    assert res["hits"][0]["name"] == "worktrees"


def test_no_match_when_neither_signal_passes(monkeypatch):
    monkeypatch.setattr(retrieve, "SCORE_THRESHOLD", 0.99)
    _seed()
    res = retrieve.search("zzz qwerty asdf nonexistent tokens", k=5)
    assert res["status"] == "no_match"
    assert res["hits"] == []
