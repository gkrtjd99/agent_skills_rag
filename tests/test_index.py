import pytest

from skill_rag import index as index_mod
from skill_rag.embed import encode_one
from skill_rag.models import SkillRecord


@pytest.fixture(autouse=True)
def isolated_index(tmp_path, monkeypatch):
    monkeypatch.setenv("SKILL_RAG_INDEX_PATH", str(tmp_path / "index.lance"))
    import importlib
    importlib.reload(index_mod)
    yield
    index_mod.reset()


def _record(name: str, desc: str = "d", body: str = "b", hash: str = "h") -> SkillRecord:
    return SkillRecord(
        name=name,
        description=desc,
        path=f"/tmp/skills/{name}/SKILL.md",
        body=body,
        content_hash=hash,
    )


def test_empty_index_lists_nothing():
    assert index_mod.list_indexed() == []


def test_upsert_then_list():
    index_mod.upsert([_record("foo"), _record("bar")])
    rows = index_mod.list_indexed()
    assert sorted(r["name"] for r in rows) == ["bar", "foo"]


def test_upsert_same_path_overwrites():
    index_mod.upsert([_record("foo", desc="old", hash="h1")])
    index_mod.upsert([_record("foo", desc="new", hash="h2")])
    rows = index_mod.list_indexed()
    assert len(rows) == 1
    assert rows[0]["description"] == "new"
    assert rows[0]["content_hash"] == "h2"


def test_delete_by_paths():
    r1 = _record("foo")
    r2 = _record("bar")
    index_mod.upsert([r1, r2])
    index_mod.delete_by_paths([r1.path])
    rows = index_mod.list_indexed()
    assert [r["name"] for r in rows] == ["bar"]


def test_search_returns_top_k():
    index_mod.upsert([_record("foo"), _record("bar"), _record("baz")])
    vec = encode_one("foo")
    hits = index_mod.search(vec, k=2)
    assert len(hits) == 2
    assert all(0.0 <= h.score <= 1.0 + 1e-5 for h in hits)


def test_search_on_empty_index():
    vec = encode_one("anything")
    assert index_mod.search(vec, k=5) == []
