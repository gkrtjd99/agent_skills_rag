import pytest

from skill_rag import index as index_mod
from skill_rag import mcp_server
from skill_rag import sync as sync_mod


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("SKILL_RAG_INDEX_PATH", str(tmp_path / "index.lance"))
    monkeypatch.setenv("SKILL_RAG_CORPUS_PATH", str(tmp_path / "skills"))
    monkeypatch.setenv("SKILL_RAG_SCORE_THRESHOLD", "0.0")
    import importlib
    from skill_rag import corpus, retrieve
    importlib.reload(corpus)
    importlib.reload(index_mod)
    importlib.reload(retrieve)
    importlib.reload(sync_mod)
    importlib.reload(mcp_server)
    yield
    index_mod.reset()
    sync_mod.reset_cache()


def _mk(corpus_root, name, desc="d", body="b"):
    d = corpus_root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {desc}\n---\n{body}\n", encoding="utf-8"
    )


def test_full_flow(tmp_path, monkeypatch):
    corpus_root = tmp_path / "skills"
    _mk(corpus_root, "debugging", desc="diagnose bugs systematically")
    _mk(corpus_root, "tdd", desc="write tests first")

    # Patch monotonic before first search so all timestamps use fake clock.
    times = [1000.0]
    monkeypatch.setattr(sync_mod.time, "monotonic", lambda: times[0])

    # 1. search finds something
    res = mcp_server.search_skills("how do I find a bug", k=5)
    assert res["status"] == "ok"
    names = [h["name"] for h in res["hits"]]
    assert "debugging" in names

    # 2. get_skill returns body
    body_res = mcp_server.get_skill("debugging")
    assert body_res["status"] == "ok"
    assert "diagnose bugs" in body_res["body"]

    # 3. add a skill, advance time past TTL → next search picks it up
    _mk(corpus_root, "refactoring", desc="restructure code without changing behavior")
    times[0] = 5000.0  # past TTL
    res2 = mcp_server.search_skills("refactor my code", k=5)
    assert any(h["name"] == "refactoring" for h in res2["hits"])

    # 4. delete a skill, force sync via get_skill → not_found
    (corpus_root / "tdd" / "SKILL.md").unlink()
    (corpus_root / "tdd").rmdir()
    nf = mcp_server.get_skill("tdd")
    assert nf["status"] == "not_found"


def test_bootstrap_skill_never_in_results(tmp_path):
    corpus_root = tmp_path / "skills"
    _mk(corpus_root, "using-skill-rag", desc="the bootstrap meta-skill")
    _mk(corpus_root, "real", desc="real skill")
    res = mcp_server.search_skills("anything", k=5)
    if res["status"] == "ok":
        names = [h["name"] for h in res["hits"]]
        assert "using-skill-rag" not in names
