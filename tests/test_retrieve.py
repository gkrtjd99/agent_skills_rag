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
