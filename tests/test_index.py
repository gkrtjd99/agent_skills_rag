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


def test_search_dim_mismatch_raises_clear_error():
    import numpy as np
    index_mod.upsert([_record("foo")])
    # A query vector of the wrong dimension (e.g. a stale model in a long-lived
    # server process after the index was rebuilt with a different model).
    wrong = np.zeros(7, dtype="float32")
    with pytest.raises(ValueError, match="reindex|dimension|restart"):
        index_mod.search(wrong, k=5)


def test_upsert_stores_full_text_for_lexical_search():
    index_mod.upsert([_record("foo", desc="one liner", body="trigger phrase here")])
    rows = index_mod.list_indexed()
    assert len(rows) == 1
    text = rows[0]["text"]
    assert "foo" in text
    assert "one liner" in text
    assert "trigger phrase here" in text


def test_upsert_stores_agent():
    rec = _record("foo")
    rec.agent = "codex"
    index_mod.upsert([rec])
    rows = index_mod.list_indexed()
    assert rows[0]["agent"] == "codex"


def test_open_table_migrates_legacy_schema(tmp_path):
    import lancedb
    import pyarrow as pa

    # Simulate a pre-`text` (v3) table sitting at the index path.
    db = lancedb.connect(str(index_mod.index_path()))
    legacy = pa.schema(
        [
            pa.field("path", pa.string()),
            pa.field("name", pa.string()),
            pa.field("description", pa.string()),
            pa.field("content_hash", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), 8)),
        ]
    )
    db.create_table(index_mod.TABLE_NAME, schema=legacy)

    tbl = index_mod.open_table()
    assert "text" in tbl.schema.names
    assert tbl.count_rows() == 0
