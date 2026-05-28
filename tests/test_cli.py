import pytest
import json
from typer.testing import CliRunner

from skill_rag import index as index_mod
from skill_rag import sync as sync_mod
from skill_rag.cli import app


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
    yield
    index_mod.reset()
    sync_mod.reset_cache()


def _mk(corpus_root, name, desc="d"):
    d = corpus_root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {desc}\n---\nbody\n", encoding="utf-8"
    )


def test_sync_command(tmp_path):
    runner = CliRunner()
    _mk(tmp_path / "skills", "foo")
    result = runner.invoke(app, ["sync"])
    assert result.exit_code == 0
    assert "added" in result.stdout
    assert "foo" in result.stdout


def test_query_command(tmp_path):
    runner = CliRunner()
    _mk(tmp_path / "skills", "foo", desc="useful skill")
    runner.invoke(app, ["sync"])
    result = runner.invoke(app, ["query", "useful"])
    assert result.exit_code == 0
    assert "foo" in result.stdout


def test_query_no_match(tmp_path, monkeypatch):
    monkeypatch.setenv("SKILL_RAG_SCORE_THRESHOLD", "0.99")
    import importlib
    from skill_rag import corpus, retrieve
    importlib.reload(corpus)
    importlib.reload(retrieve)
    runner = CliRunner()
    _mk(tmp_path / "skills", "foo")
    runner.invoke(app, ["sync"])
    result = runner.invoke(app, ["query", "completely unrelated"])
    assert result.exit_code == 0
    assert "no" in result.stdout.lower()


def test_list_command(tmp_path):
    runner = CliRunner()
    _mk(tmp_path / "skills", "foo")
    _mk(tmp_path / "skills", "bar")
    runner.invoke(app, ["sync"])
    result = runner.invoke(app, ["list-skills"])
    assert result.exit_code == 0
    assert "foo" in result.stdout
    assert "bar" in result.stdout


def test_reset_command(tmp_path):
    runner = CliRunner()
    _mk(tmp_path / "skills", "foo")
    runner.invoke(app, ["sync"])
    result = runner.invoke(app, ["reset"])
    assert result.exit_code == 0
    assert index_mod.list_indexed() == []


def test_eval_command_uses_explicit_corpus_and_dataset(tmp_path):
    runner = CliRunner()
    corpus = tmp_path / "eval-skills"
    _mk(corpus, "foo", desc="alpha beta testing")
    dataset = tmp_path / "queries.jsonl"
    dataset.write_text('{"query": "alpha beta", "expected": ["foo"]}\n')

    result = runner.invoke(
        app,
        ["eval", "--corpus", str(corpus), "--dataset", str(dataset), "--json"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["recall_at_k"] == 1.0
    assert payload["corpus"] == str(corpus.resolve())
    assert payload["dataset"] == str(dataset.resolve())
