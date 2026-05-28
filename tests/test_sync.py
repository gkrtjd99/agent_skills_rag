import pytest

from skill_rag import index as index_mod
from skill_rag import sync as sync_mod


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("SKILL_RAG_INDEX_PATH", str(tmp_path / "index.lance"))
    monkeypatch.setenv("SKILL_RAG_CORPUS_PATH", str(tmp_path / "skills"))
    import importlib
    from skill_rag import corpus
    importlib.reload(corpus)
    importlib.reload(index_mod)
    importlib.reload(sync_mod)
    yield
    index_mod.reset()


def _mk(corpus_root, name, desc="d", body="b"):
    d = corpus_root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {desc}\n---\n{body}\n", encoding="utf-8"
    )


def test_sync_adds_skills(tmp_path):
    corpus_root = tmp_path / "skills"
    _mk(corpus_root, "foo")
    _mk(corpus_root, "bar")
    report = sync_mod.run_sync()
    assert sorted(report["added"]) == ["bar", "foo"]
    assert report["updated"] == []
    assert report["removed"] == []
    assert sorted(r["name"] for r in index_mod.list_indexed()) == ["bar", "foo"]


def test_sync_detects_modification(tmp_path):
    corpus_root = tmp_path / "skills"
    _mk(corpus_root, "foo", desc="old")
    sync_mod.run_sync()
    _mk(corpus_root, "foo", desc="new")
    report = sync_mod.run_sync()
    assert report["updated"] == ["foo"]
    rows = index_mod.list_indexed()
    assert rows[0]["description"] == "new"


def test_sync_detects_removal(tmp_path):
    corpus_root = tmp_path / "skills"
    _mk(corpus_root, "foo")
    _mk(corpus_root, "bar")
    sync_mod.run_sync()
    (corpus_root / "foo" / "SKILL.md").unlink()
    (corpus_root / "foo").rmdir()
    report = sync_mod.run_sync()
    assert report["removed"] == ["foo"]
    rows = index_mod.list_indexed()
    assert [r["name"] for r in rows] == ["bar"]


def test_sync_if_stale_skips_within_ttl(monkeypatch, tmp_path):
    corpus_root = tmp_path / "skills"
    _mk(corpus_root, "foo")

    times = [100.0]
    monkeypatch.setattr(sync_mod.time, "monotonic", lambda: times[0])

    sync_mod.sync_if_stale(ttl=30.0)
    assert len(index_mod.list_indexed()) == 1

    _mk(corpus_root, "bar")
    times[0] = 120.0  # within 30s
    sync_mod.sync_if_stale(ttl=30.0)
    assert len(index_mod.list_indexed()) == 1  # not picked up


def test_sync_if_stale_runs_after_ttl(monkeypatch, tmp_path):
    corpus_root = tmp_path / "skills"
    _mk(corpus_root, "foo")

    times = [100.0]
    monkeypatch.setattr(sync_mod.time, "monotonic", lambda: times[0])

    sync_mod.sync_if_stale(ttl=30.0)
    _mk(corpus_root, "bar")
    times[0] = 200.0  # past TTL
    sync_mod.sync_if_stale(ttl=30.0)
    assert sorted(r["name"] for r in index_mod.list_indexed()) == ["bar", "foo"]


def test_sync_if_stale_force_with_ttl_zero(monkeypatch, tmp_path):
    corpus_root = tmp_path / "skills"
    _mk(corpus_root, "foo")
    times = [100.0]
    monkeypatch.setattr(sync_mod.time, "monotonic", lambda: times[0])

    sync_mod.sync_if_stale(ttl=30.0)
    _mk(corpus_root, "bar")
    # No time advance, but ttl=0 forces.
    sync_mod.sync_if_stale(ttl=0)
    assert sorted(r["name"] for r in index_mod.list_indexed()) == ["bar", "foo"]
