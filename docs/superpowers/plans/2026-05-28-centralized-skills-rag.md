# Centralized Skills + Lazy RAG Loading — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild skill-rag to use a single global corpus at `~/.skills/` and expose lazy-loading MCP tools (`search_skills`, `get_skill`) so agents only pull skill bodies they actually need.

**Architecture:** One bootstrap skill stays resident in agent context. Per user message it calls `search_skills` (LanceDB cosine, threshold-filtered, TTL-30s auto-sync) to get metadata, judges fit, then calls `get_skill` to fetch only the relevant bodies. Three explicit response statuses (`ok`/`no_match`/`not_found`) prevent loops. Git history is reset; verified modules preserved.

**Tech Stack:** Python 3.13, uv, LanceDB, sentence-transformers (all-MiniLM-L6-v2), FastMCP (mcp), Typer, pytest.

**Spec:** `docs/superpowers/specs/2026-05-28-centralized-skills-rag-design.md`

---

## File Plan

**Preserve as-is** (from current main, will be re-added after orphan reset):
- `pyproject.toml`, `uv.lock`, `LICENSE`, `.gitignore`
- `src/skill_rag/embed.py`
- `docs/references/*-llms.txt`
- `docs/superpowers/specs/2026-05-28-centralized-skills-rag-design.md`
- `docs/superpowers/plans/2026-05-28-centralized-skills-rag.md` (this file)

**Rewrite**:
- `src/skill_rag/__init__.py` — empty package marker
- `src/skill_rag/models.py` — drop `source`/`allowed_tools`/`extra`; add `body`
- `src/skill_rag/parser.py` — drop `source` param; include `body` in record; hash body
- `src/skill_rag/loader.py` — single `~/.skills/` root; skip `using-skill-rag`
- `src/skill_rag/index.py` — drop `source` column; remove dedup buckets
- `src/skill_rag/retrieve.py` — threshold filter, status response
- `src/skill_rag/sync.py` — TTL cache + diff
- `src/skill_rag/mcp_server.py` — `search_skills` + `get_skill` with statuses
- `src/skill_rag/cli.py` — `sync`/`query`/`list`/`reset`/`mcp` subcommands
- `src/skill_rag/evaluator.py` — adapt to new `SearchHit` (no `sources` list)
- `tests/*` — all tests rewritten against new schema
- `eval/queries.jsonl` — verify expected names still match `~/.skills/` flat layout
- `eval/runner.py` (if present) — see Task 14
- `README.md` (Korean)
- `AGENTS.md`, `ARCHITECTURE.md`
- `docs/product-specs/skill-rag.md`
- `docs/design-docs/{core-beliefs,index}.md`
- `docs/exec-plans/tech-debt-tracker.md`

**New**:
- `src/skill_rag/corpus.py` — single source of corpus path (`~/.skills`) and threshold constant
- `bootstrap-skill/using-skill-rag/SKILL.md`
- `scripts/install.sh`
- `docs/references/mcp-llms.txt`
- `docs/design-docs/mcp-interface.md`
- `docs/design-docs/meta-skill-bootstrap.md`

**Delete**:
- `src/skill_rag/__pycache__/`
- Any active exec plans (`docs/exec-plans/active/*`)
- `docs/exec-plans/completed/*` (cleared, optional — keep if user wants archive)

---

## Constants (referenced by multiple tasks)

```python
# src/skill_rag/corpus.py
from __future__ import annotations
import os
from pathlib import Path

CORPUS_PATH = Path(os.environ.get("SKILL_RAG_CORPUS_PATH", "~/.skills")).expanduser()
BOOTSTRAP_SKILL_NAME = "using-skill-rag"
SCORE_THRESHOLD = float(os.environ.get("SKILL_RAG_SCORE_THRESHOLD", "0.35"))
SYNC_TTL_SECONDS = float(os.environ.get("SKILL_RAG_SYNC_TTL", "30"))
```

---

## Task 1: Reset git history, scaffold preserved files

**Files:**
- Modify: entire working tree (orphan branch)
- Preserve: `pyproject.toml`, `uv.lock`, `LICENSE`, `.gitignore`, `src/skill_rag/embed.py`, `docs/references/*-llms.txt`, `docs/superpowers/`

- [ ] **Step 1: Confirm working tree is clean**

Run: `git status`
Expected: `nothing to commit, working tree clean`

If not clean, stop and ask the user before continuing.

- [ ] **Step 2: Create safety backup branch**

Run: `git branch backup/pre-rewrite`
Expected: silent success. Verify with `git branch --list backup/pre-rewrite`.

- [ ] **Step 3: Create orphan branch and unstage everything**

Run:
```bash
git checkout --orphan fresh-start
git rm --cached -rf .
```
Expected: orphan branch created with no commits; all files unstaged but kept on disk.

- [ ] **Step 4: Delete files we are throwing away**

Run:
```bash
rm -rf src/skill_rag/__pycache__
rm -f src/skill_rag/__init__.py src/skill_rag/models.py src/skill_rag/parser.py \
      src/skill_rag/loader.py src/skill_rag/index.py src/skill_rag/retrieve.py \
      src/skill_rag/sync.py src/skill_rag/mcp_server.py src/skill_rag/cli.py \
      src/skill_rag/evaluator.py
rm -rf tests
rm -f AGENTS.md ARCHITECTURE.md README.md CLAUDE.md
rm -rf docs/product-specs docs/design-docs docs/exec-plans docs/generated
```
Expected: files removed. `src/skill_rag/embed.py` and `docs/references/`, `docs/superpowers/` remain.

- [ ] **Step 5: Stage preserved files**

Run:
```bash
git add pyproject.toml uv.lock LICENSE .gitignore
git add src/skill_rag/embed.py
git add docs/references/ docs/superpowers/
```
Expected: `git status` shows these as new files staged.

- [ ] **Step 6: Add empty package marker**

Create `src/skill_rag/__init__.py`:
```python
```
(empty file)

Run: `git add src/skill_rag/__init__.py`

- [ ] **Step 7: Initial commit**

Run:
```bash
git commit -m "$(cat <<'EOF'
Initial: skill-rag — central ~/.skills corpus, lazy RAG loading

Single global skill corpus at ~/.skills/<name>/SKILL.md served via MCP.
Replaces the prior multi-source (~/.claude + ~/.codex) layout.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```
Expected: commit succeeds.

- [ ] **Step 8: Rename orphan branch to main**

Run: `git branch -M fresh-start main`
Expected: branch renamed. `git log --oneline` shows exactly one commit.

NOTE: do NOT `git push -f origin main`. The user pushes themselves when ready.

---

## Task 2: Add corpus constants module

**Files:**
- Create: `src/skill_rag/corpus.py`
- Create: `tests/test_corpus.py`

- [ ] **Step 1: Write the failing test**

Create `tests/__init__.py` (empty) and `tests/test_corpus.py`:
```python
from pathlib import Path

from skill_rag import corpus


def test_default_corpus_path(monkeypatch):
    monkeypatch.delenv("SKILL_RAG_CORPUS_PATH", raising=False)
    # Re-import to pick up env change
    import importlib
    importlib.reload(corpus)
    assert corpus.CORPUS_PATH == Path("~/.skills").expanduser()


def test_corpus_path_override(monkeypatch, tmp_path):
    monkeypatch.setenv("SKILL_RAG_CORPUS_PATH", str(tmp_path))
    import importlib
    importlib.reload(corpus)
    assert corpus.CORPUS_PATH == tmp_path


def test_constants_exposed():
    assert corpus.BOOTSTRAP_SKILL_NAME == "using-skill-rag"
    assert 0 < corpus.SCORE_THRESHOLD < 1
    assert corpus.SYNC_TTL_SECONDS > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_corpus.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'skill_rag.corpus'`.

- [ ] **Step 3: Implement**

Create `src/skill_rag/corpus.py`:
```python
from __future__ import annotations

import os
from pathlib import Path

CORPUS_PATH = Path(os.environ.get("SKILL_RAG_CORPUS_PATH", "~/.skills")).expanduser()
BOOTSTRAP_SKILL_NAME = "using-skill-rag"
SCORE_THRESHOLD = float(os.environ.get("SKILL_RAG_SCORE_THRESHOLD", "0.35"))
SYNC_TTL_SECONDS = float(os.environ.get("SKILL_RAG_SYNC_TTL", "30"))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_corpus.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/skill_rag/corpus.py tests/__init__.py tests/test_corpus.py
git commit -m "feat: add corpus module with path and threshold constants

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 3: Models — `SkillRecord` and `SearchHit`

**Files:**
- Create: `src/skill_rag/models.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_models.py`:
```python
from skill_rag.models import SearchHit, SkillRecord


def test_skill_record_embed_text():
    r = SkillRecord(
        name="brainstorming",
        description="explore ideas",
        path="/tmp/.skills/brainstorming/SKILL.md",
        body="# body",
        content_hash="abc",
    )
    assert r.embed_text() == "brainstorming\nexplore ideas"


def test_search_hit_fields():
    h = SearchHit(name="x", description="y", score=0.9)
    assert h.name == "x"
    assert h.description == "y"
    assert h.score == 0.9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

Create `src/skill_rag/models.py`:
```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SkillRecord:
    name: str
    description: str
    path: str
    body: str
    content_hash: str

    def embed_text(self) -> str:
        # Stable string we embed. Changing this requires a reindex.
        return f"{self.name}\n{self.description}"


@dataclass(slots=True)
class SearchHit:
    name: str
    description: str
    score: float
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_models.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/skill_rag/models.py tests/test_models.py
git commit -m "feat: add SkillRecord and SearchHit dataclasses

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 4: Parser — SKILL.md text → SkillRecord

**Files:**
- Create: `src/skill_rag/parser.py`
- Create: `tests/test_parser.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_parser.py`:
```python
from pathlib import Path

from skill_rag.parser import parse_skill_file


def _write(tmp_path: Path, name: str, content: str) -> Path:
    d = tmp_path / name
    d.mkdir(parents=True)
    p = d / "SKILL.md"
    p.write_text(content, encoding="utf-8")
    return p


def test_parse_valid(tmp_path):
    p = _write(
        tmp_path,
        "foo",
        "---\nname: foo\ndescription: does foo\n---\nBody here.\n",
    )
    r = parse_skill_file(p)
    assert r is not None
    assert r.name == "foo"
    assert r.description == "does foo"
    assert r.body == "Body here.\n"
    assert r.content_hash  # non-empty
    assert r.path == str(p)


def test_parse_missing_frontmatter(tmp_path):
    p = _write(tmp_path, "foo", "no frontmatter here")
    assert parse_skill_file(p) is None


def test_parse_missing_required_fields(tmp_path):
    p = _write(tmp_path, "foo", "---\nname: foo\n---\nbody")
    assert parse_skill_file(p) is None


def test_parse_malformed_yaml(tmp_path):
    p = _write(tmp_path, "foo", "---\nname: [unclosed\n---\nbody")
    assert parse_skill_file(p) is None


def test_parse_empty_file(tmp_path):
    p = _write(tmp_path, "foo", "")
    assert parse_skill_file(p) is None


def test_hash_changes_with_body(tmp_path):
    p1 = _write(tmp_path / "a", "foo", "---\nname: foo\ndescription: d\n---\nbody1")
    p2 = _write(tmp_path / "b", "foo", "---\nname: foo\ndescription: d\n---\nbody2")
    r1 = parse_skill_file(p1)
    r2 = parse_skill_file(p2)
    assert r1.content_hash != r2.content_hash
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_parser.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

Create `src/skill_rag/parser.py`:
```python
from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from .models import SkillRecord

_FRONTMATTER_DELIM = "---"


def parse_skill_file(path: Path) -> SkillRecord | None:
    """Parse one SKILL.md. Returns None on any parse failure or missing fields.

    content_hash is sha256 of the FULL file text so any change to the body or
    frontmatter triggers re-indexing.
    """
    text = path.read_text(encoding="utf-8")
    fm_text, body = _split_frontmatter(text)
    if fm_text is None:
        return None

    try:
        data = yaml.safe_load(fm_text) or {}
    except yaml.YAMLError:
        return None
    if not isinstance(data, dict):
        return None

    name = str(data.get("name") or "").strip()
    description = str(data.get("description") or "").strip()
    if not name or not description:
        return None

    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return SkillRecord(
        name=name,
        description=description,
        path=str(path),
        body=body,
        content_hash=content_hash,
    )


def _split_frontmatter(text: str) -> tuple[str | None, str]:
    if not text.startswith(_FRONTMATTER_DELIM):
        return None, text
    rest = text[len(_FRONTMATTER_DELIM):].lstrip("\n")
    end = rest.find(f"\n{_FRONTMATTER_DELIM}")
    if end == -1:
        return None, text
    fm = rest[:end]
    body = rest[end + len(_FRONTMATTER_DELIM) + 1:].lstrip("\n")
    return fm, body
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_parser.py -v`
Expected: 6 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/skill_rag/parser.py tests/test_parser.py
git commit -m "feat: add SKILL.md parser

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 5: Loader — scan `~/.skills/`

**Files:**
- Create: `src/skill_rag/loader.py`
- Create: `tests/test_loader.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_loader.py`:
```python
from pathlib import Path

from skill_rag.loader import scan


def _mk(root: Path, name: str, fm_name: str | None = None, desc: str = "d", body: str = "b"):
    d = root / name
    d.mkdir(parents=True)
    nm = fm_name if fm_name is not None else name
    (d / "SKILL.md").write_text(
        f"---\nname: {nm}\ndescription: {desc}\n---\n{body}\n", encoding="utf-8"
    )


def test_scan_empty_dir_returns_empty(tmp_path):
    assert scan(tmp_path) == []


def test_scan_nonexistent_returns_empty(tmp_path):
    assert scan(tmp_path / "missing") == []


def test_scan_finds_skills(tmp_path):
    _mk(tmp_path, "foo")
    _mk(tmp_path, "bar")
    names = sorted(r.name for r in scan(tmp_path))
    assert names == ["bar", "foo"]


def test_scan_skips_bootstrap_skill(tmp_path):
    _mk(tmp_path, "using-skill-rag", desc="bootstrap")
    _mk(tmp_path, "real-skill")
    names = [r.name for r in scan(tmp_path)]
    assert "using-skill-rag" not in names
    assert "real-skill" in names


def test_scan_ignores_dir_without_skill_md(tmp_path):
    (tmp_path / "empty-dir").mkdir()
    _mk(tmp_path, "real-skill")
    names = [r.name for r in scan(tmp_path)]
    assert names == ["real-skill"]


def test_scan_ignores_files_at_root(tmp_path):
    (tmp_path / "loose-file.md").write_text("---\nname: x\ndescription: y\n---\n")
    _mk(tmp_path, "real-skill")
    names = [r.name for r in scan(tmp_path)]
    assert names == ["real-skill"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_loader.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

Create `src/skill_rag/loader.py`:
```python
from __future__ import annotations

from pathlib import Path

from .corpus import BOOTSTRAP_SKILL_NAME
from .models import SkillRecord
from .parser import parse_skill_file


def scan(root: Path) -> list[SkillRecord]:
    """Return a SkillRecord for every <root>/<name>/SKILL.md.

    Flat layout only — no recursion into subdirectories below the skill dir.
    Skips the bootstrap skill so it never appears in search results.
    """
    if not root.exists():
        return []
    records: list[SkillRecord] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if child.name == BOOTSTRAP_SKILL_NAME:
            continue
        skill_md = child / "SKILL.md"
        if not skill_md.is_file():
            continue
        record = parse_skill_file(skill_md)
        if record is not None:
            records.append(record)
    return records
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_loader.py -v`
Expected: 6 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/skill_rag/loader.py tests/test_loader.py
git commit -m "feat: add corpus scanner (flat layout, skips bootstrap)

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 6: Verify preserved `embed.py` still works

**Files:**
- Test: `tests/test_embed.py`

- [ ] **Step 1: Write the test**

Create `tests/test_embed.py`:
```python
import numpy as np

from skill_rag.embed import encode, encode_one, model_dim


def test_model_dim_is_positive():
    assert model_dim() > 0


def test_encode_one_returns_normalized_vector():
    v = encode_one("hello world")
    assert v.shape == (model_dim(),)
    assert abs(np.linalg.norm(v) - 1.0) < 1e-5


def test_encode_batch():
    v = encode(["a", "b", "c"])
    assert v.shape == (3, model_dim())


def test_encode_empty_input():
    v = encode([])
    assert v.shape == (0, model_dim())
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_embed.py -v`
Expected: 4 PASS. (First run downloads the model — may take ~30 s.)

- [ ] **Step 3: Commit**

```bash
git add tests/test_embed.py
git commit -m "test: cover embed module

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 7: Index — LanceDB CRUD (no `source` column)

**Files:**
- Create: `src/skill_rag/index.py`
- Create: `tests/test_index.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_index.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_index.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

Create `src/skill_rag/index.py`:
```python
"""LanceDB-backed index for skill records.

Schema version: 3
- v1: pk=name
- v2: pk=path; added `source` column
- v3: pk=path; removed `source`, `allowed_tools` (single-corpus design)
"""

from __future__ import annotations

import os
from pathlib import Path

import lancedb
import pyarrow as pa

from .embed import DEFAULT_MODEL, encode, model_dim
from .models import SearchHit, SkillRecord

TABLE_NAME = "skills"


def index_path() -> Path:
    return Path(os.environ.get("SKILL_RAG_INDEX_PATH", "./var/index.lance")).expanduser()


def _schema(dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("path", pa.string()),
            pa.field("name", pa.string()),
            pa.field("description", pa.string()),
            pa.field("content_hash", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), dim)),
        ]
    )


def _open_db():
    path = index_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    return lancedb.connect(str(path))


def open_table(model_name: str = DEFAULT_MODEL):
    db = _open_db()
    if TABLE_NAME in db.table_names():
        return db.open_table(TABLE_NAME)
    return db.create_table(TABLE_NAME, schema=_schema(model_dim(model_name)))


def list_indexed() -> list[dict]:
    tbl = open_table()
    if tbl.count_rows() == 0:
        return []
    cols = ["path", "name", "description", "content_hash"]
    return tbl.to_arrow().select(cols).to_pylist()


def upsert(records: list[SkillRecord], model_name: str = DEFAULT_MODEL) -> None:
    if not records:
        return
    tbl = open_table(model_name)
    vectors = encode([r.embed_text() for r in records], name=model_name)
    rows = [
        {
            "path": r.path,
            "name": r.name,
            "description": r.description,
            "content_hash": r.content_hash,
            "vector": vec.tolist(),
        }
        for r, vec in zip(records, vectors)
    ]
    (
        tbl.merge_insert("path")
        .when_matched_update_all()
        .when_not_matched_insert_all()
        .execute(rows)
    )


def delete_by_paths(paths: list[str]) -> int:
    if not paths:
        return 0
    tbl = open_table()
    quoted = ", ".join("'" + p.replace("'", "''") + "'" for p in paths)
    tbl.delete(f"path IN ({quoted})")
    return len(paths)


def reset() -> None:
    db = _open_db()
    if TABLE_NAME in db.table_names():
        db.drop_table(TABLE_NAME)


def search(query_vector, k: int = 5) -> list[SearchHit]:
    tbl = open_table()
    if tbl.count_rows() == 0:
        return []
    rows = tbl.search(query_vector).metric("cosine").limit(k).to_list()
    hits: list[SearchHit] = []
    for row in rows:
        distance = float(row.get("_distance", 0.0))
        score = 1.0 - distance
        hits.append(
            SearchHit(
                name=row["name"],
                description=row["description"],
                score=score,
            )
        )
    return hits
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_index.py -v`
Expected: 6 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/skill_rag/index.py tests/test_index.py
git commit -m "feat: LanceDB index v3 (drop source column, simplify search)

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 8: Retrieve — threshold filter and status response

**Files:**
- Create: `src/skill_rag/retrieve.py`
- Create: `tests/test_retrieve.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_retrieve.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_retrieve.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

Create `src/skill_rag/retrieve.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_retrieve.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/skill_rag/retrieve.py tests/test_retrieve.py
git commit -m "feat: retrieve with threshold filter and status response

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 9: Sync — TTL cache + diff

**Files:**
- Create: `src/skill_rag/sync.py`
- Create: `tests/test_sync.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_sync.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_sync.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

Create `src/skill_rag/sync.py`:
```python
"""Reconcile the LanceDB index against the corpus directory.

Only this module holds mutable state (a single timestamp).
"""

from __future__ import annotations

import time

from . import corpus as corpus_mod
from . import index as index_mod
from . import loader

_last_sync_at: float | None = None


def run_sync() -> dict:
    """Force a sync. Returns {added, updated, removed, unchanged}."""
    global _last_sync_at
    records = loader.scan(corpus_mod.CORPUS_PATH)
    indexed = {row["path"]: row["content_hash"] for row in index_mod.list_indexed()}
    disk_paths = {r.path for r in records}

    added: list[str] = []
    updated: list[str] = []
    to_upsert = []
    unchanged = 0
    for r in records:
        prev_hash = indexed.get(r.path)
        if prev_hash is None:
            added.append(r.name)
            to_upsert.append(r)
        elif prev_hash != r.content_hash:
            updated.append(r.name)
            to_upsert.append(r)
        else:
            unchanged += 1

    removed_paths = [p for p in indexed if p not in disk_paths]
    # Recover names by querying the indexed rows we just listed.
    removed_names: list[str] = []
    if removed_paths:
        for row in index_mod.list_indexed():
            if row["path"] in removed_paths:
                removed_names.append(row["name"])

    if to_upsert:
        index_mod.upsert(to_upsert)
    if removed_paths:
        index_mod.delete_by_paths(removed_paths)

    _last_sync_at = time.monotonic()
    return {
        "added": added,
        "updated": updated,
        "removed": removed_names,
        "unchanged": unchanged,
    }


def sync_if_stale(ttl: float | None = None) -> None:
    """Run sync only if the last run was longer than ``ttl`` seconds ago.

    Pass ``ttl=0`` to force a sync regardless of cache age.
    """
    global _last_sync_at
    if ttl is None:
        ttl = corpus_mod.SYNC_TTL_SECONDS
    now = time.monotonic()
    if ttl > 0 and _last_sync_at is not None and (now - _last_sync_at) < ttl:
        return
    run_sync()


def reset_cache() -> None:
    """Clear the TTL timestamp. Next sync_if_stale call will run."""
    global _last_sync_at
    _last_sync_at = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_sync.py -v`
Expected: 6 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/skill_rag/sync.py tests/test_sync.py
git commit -m "feat: sync module with TTL cache and disk-vs-index diff

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 10: MCP server — `search_skills` and `get_skill`

**Files:**
- Create: `src/skill_rag/mcp_server.py`
- Create: `tests/test_mcp_server.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_mcp_server.py`:
```python
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


def test_search_skills_returns_ok_for_match(tmp_path):
    corpus_root = tmp_path / "skills"
    _mk(corpus_root, "brainstorming", desc="explore ideas before implementation")
    res = mcp_server.search_skills("explore ideas", k=5)
    assert res["status"] == "ok"
    assert any(h["name"] == "brainstorming" for h in res["hits"])


def test_search_skills_returns_no_match_when_empty(tmp_path):
    res = mcp_server.search_skills("anything", k=5)
    assert res["status"] == "no_match"
    assert res["hits"] == []


def test_get_skill_returns_body(tmp_path):
    corpus_root = tmp_path / "skills"
    _mk(corpus_root, "foo", body="hello body")
    mcp_server.search_skills("foo", k=1)  # populate index
    res = mcp_server.get_skill("foo")
    assert res["status"] == "ok"
    assert "hello body" in res["body"]


def test_get_skill_not_found_after_force_sync(tmp_path):
    res = mcp_server.get_skill("does-not-exist")
    assert res["status"] == "not_found"
    assert "again" in res["message"].lower()


def test_get_skill_recovers_via_force_sync(tmp_path, monkeypatch):
    corpus_root = tmp_path / "skills"
    _mk(corpus_root, "foo", body="hi")
    # Do NOT call search first — get_skill must trigger sync itself.
    res = mcp_server.get_skill("foo")
    assert res["status"] == "ok"
    assert "hi" in res["body"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_mcp_server.py -v`
Expected: FAIL — `ModuleNotFoundError` or function-missing.

- [ ] **Step 3: Implement**

Create `src/skill_rag/mcp_server.py`:
```python
"""MCP server exposing two tools: search_skills, get_skill.

search_skills: ranked metadata via vector search (auto-syncs on TTL).
get_skill:     full SKILL.md body for a single skill.

Response shapes (mirrored in bootstrap-skill/using-skill-rag/SKILL.md):
  search_skills -> {"status": "ok", "hits": [...]}
                or {"status": "no_match", "hits": [], "message": "..."}
  get_skill     -> {"status": "ok", "body": "..."}
                or {"status": "not_found", "message": "..."}
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from . import corpus as corpus_mod
from . import retrieve
from . import sync as sync_mod

server = FastMCP("skill-rag")


@server.tool()
def search_skills(query: str, k: int = 5) -> dict:
    """Find skills relevant to ``query``. Call BEFORE responding to any user
    message. Returns metadata only — call ``get_skill`` to fetch the body of
    any skill that fits the task.

    Response:
      - {"status": "ok", "hits": [{"name", "description", "score"}, ...]}
      - {"status": "no_match", "hits": [], "message": "..."}
    """
    sync_mod.sync_if_stale()
    return retrieve.search(query, k=k)


@server.tool()
def get_skill(name: str) -> dict:
    """Fetch the full SKILL.md body for ``name``.

    Response:
      - {"status": "ok", "body": "..."}
      - {"status": "not_found", "message": "..."}
    """
    path = corpus_mod.CORPUS_PATH / name / "SKILL.md"
    if path.exists():
        return {"status": "ok", "body": path.read_text(encoding="utf-8")}
    # File not found — force a sync in case the index is ahead of disk, retry.
    sync_mod.sync_if_stale(ttl=0)
    if path.exists():
        return {"status": "ok", "body": path.read_text(encoding="utf-8")}
    return {
        "status": "not_found",
        "message": (
            f"Skill '{name}' does not exist in the corpus. "
            f"Do not call get_skill or search_skills for this name again. "
            f"Proceed without it."
        ),
    }


def run() -> None:
    server.run()


if __name__ == "__main__":
    run()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_mcp_server.py -v`
Expected: 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/skill_rag/mcp_server.py tests/test_mcp_server.py
git commit -m "feat: MCP server with search_skills and get_skill (status responses)

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 11: CLI — sync / query / list / reset / mcp

**Files:**
- Create: `src/skill_rag/cli.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cli.py`:
```python
import pytest
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

Create `src/skill_rag/cli.py`:
```python
from __future__ import annotations

import json

import typer

from . import index as index_mod
from . import retrieve
from . import sync as sync_mod

app = typer.Typer(no_args_is_help=True, help="skill_rag — local RAG over ~/.skills.")


@app.command()
def sync(json_out: bool = typer.Option(False, "--json")):
    """Reconcile the index against ~/.skills."""
    report = sync_mod.run_sync()
    if json_out:
        typer.echo(json.dumps(report, ensure_ascii=False, indent=2))
        return
    typer.echo(f"added:     {len(report['added'])}")
    for n in report["added"]:
        typer.echo(f"  + {n}")
    typer.echo(f"updated:   {len(report['updated'])}")
    for n in report["updated"]:
        typer.echo(f"  ~ {n}")
    typer.echo(f"removed:   {len(report['removed'])}")
    for n in report["removed"]:
        typer.echo(f"  - {n}")
    typer.echo(f"unchanged: {report['unchanged']}")


@app.command()
def query(
    text: str = typer.Argument(..., help="Natural-language query."),
    k: int = typer.Option(5, "--k", "-k", min=1, max=50),
    json_out: bool = typer.Option(False, "--json"),
):
    """Return top-k skills for the query."""
    res = retrieve.search(text, k=k)
    if json_out:
        typer.echo(json.dumps(res, ensure_ascii=False, indent=2))
        return
    if res["status"] == "no_match":
        typer.echo(f"(no match — {res['message']})")
        return
    for rank, h in enumerate(res["hits"], start=1):
        typer.echo(f"{rank}. [{h['score']:.3f}] {h['name']}")
        typer.echo(f"   {h['description'][:140]}")


@app.command(name="list-skills")
def list_skills(json_out: bool = typer.Option(False, "--json")):
    """List skills currently indexed."""
    rows = index_mod.list_indexed()
    if json_out:
        typer.echo(json.dumps(rows, ensure_ascii=False, indent=2))
        return
    if not rows:
        typer.echo("(index is empty)")
        return
    for r in rows:
        typer.echo(f"- {r['name']}  ({r['path']})")


@app.command()
def reset():
    """Drop the index entirely. Next `sync` will rebuild it."""
    index_mod.reset()
    sync_mod.reset_cache()
    typer.echo("index dropped.")


@app.command()
def mcp():
    """Start the MCP server over stdio."""
    from .mcp_server import run

    run()


if __name__ == "__main__":
    app()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -v`
Expected: 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/skill_rag/cli.py tests/test_cli.py
git commit -m "feat: CLI with sync/query/list-skills/reset/mcp subcommands

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 12: Bootstrap meta-skill

**Files:**
- Create: `bootstrap-skill/using-skill-rag/SKILL.md`

- [ ] **Step 1: Create the file**

Create `bootstrap-skill/using-skill-rag/SKILL.md`:
````markdown
---
name: using-skill-rag
description: Use at the start of every user message to find relevant skills via RAG before responding
---

# Skill RAG — Lazy Skill Loading

You have access to a skill corpus via MCP tools. Skills are NOT loaded into
your context by default. You must search for them per turn.

## Required behavior

**Parent agent**: BEFORE responding to any user message (including clarifying
questions), call `mcp__skill-rag__search_skills(query=<user's message>, k=5)`.

**Subagent**: When invoked, inspect the parent context you were given. If it
describes a substantive task (coding, design, debugging, etc.), call
`search_skills` with a query summarizing your task. If it's a narrow lookup
the parent already framed, skip.

## Handling responses

`search_skills` returns one of:

- `{status: "ok", hits: [{name, description, score}, ...]}`
  - Read each `description`. For any skill that clearly applies, call
    `mcp__skill-rag__get_skill(name)` to fetch its body. Then follow the
    skill exactly.
  - If multiple skills apply, fetch each. Process skills (brainstorming,
    debugging) before implementation skills.
  - If none of the descriptions actually fit despite being returned,
    proceed without a skill. Do not refetch with reworded queries.

- `{status: "no_match", hits: [], message: ...}`
  - No skill applies. Respond directly. **DO NOT call `search_skills` again
    for this turn with a rephrased query.**

`get_skill` returns one of:

- `{status: "ok", body: "..."}`
  - Follow the instructions in `body`.

- `{status: "not_found", message: ...}`
  - The skill was removed. **DO NOT call `get_skill` or `search_skills` for
    this name again this turn.** Proceed without it.

## Anti-patterns

- Calling `search_skills` more than once per user message with reworded
  queries to "try harder". One call. Trust the result.
- Calling `get_skill` for a skill whose description clearly doesn't fit
  just because it appeared in `hits`.
- Skipping `search_skills` because "this looks simple". Always call.
````

- [ ] **Step 2: Commit**

```bash
git add bootstrap-skill/using-skill-rag/SKILL.md
git commit -m "feat: add using-skill-rag bootstrap meta-skill

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 13: Install script

**Files:**
- Create: `scripts/install.sh`

- [ ] **Step 1: Create the script**

Create `scripts/install.sh`:
```bash
#!/usr/bin/env bash
# Install skill-rag for the current user.
# - Creates ~/.skills/ (the central corpus)
# - Installs the bootstrap skill at ~/.skills/using-skill-rag/
# - Symlinks ~/.<harness>/skills/using-skill-rag -> ~/.skills/using-skill-rag
# - Prints MCP server registration instructions per harness

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SKILLS_DIR="${HOME}/.skills"
BOOTSTRAP_SRC="${REPO_ROOT}/bootstrap-skill/using-skill-rag"
BOOTSTRAP_DST="${SKILLS_DIR}/using-skill-rag"

echo "→ Ensuring ${SKILLS_DIR} exists"
mkdir -p "${SKILLS_DIR}"

if [ ! -d "${BOOTSTRAP_DST}" ]; then
  echo "→ Installing bootstrap skill to ${BOOTSTRAP_DST}"
  cp -R "${BOOTSTRAP_SRC}" "${BOOTSTRAP_DST}"
else
  echo "→ Bootstrap skill already present at ${BOOTSTRAP_DST} (skipping copy)"
fi

for harness in claude codex; do
  HARNESS_SKILLS="${HOME}/.${harness}/skills"
  mkdir -p "${HARNESS_SKILLS}"
  LINK="${HARNESS_SKILLS}/using-skill-rag"
  echo "→ Linking ${LINK} → ${BOOTSTRAP_DST}"
  ln -sfn "${BOOTSTRAP_DST}" "${LINK}"
done

cat <<EOF

Done. Next step — register the MCP server in each harness.

Claude Code:
  Add to ~/.claude.json under "mcpServers":
    "skill-rag": {
      "command": "uv",
      "args": ["--directory", "${REPO_ROOT}", "run", "skill-rag", "mcp"]
    }

Codex:
  See the Codex docs for MCP server registration. Use the same command:
    uv --directory ${REPO_ROOT} run skill-rag mcp

After registering, restart the harness so the MCP server is picked up.
EOF
```

- [ ] **Step 2: Make executable and test it parses**

Run:
```bash
chmod +x scripts/install.sh
bash -n scripts/install.sh
```
Expected: no output (syntax OK).

- [ ] **Step 3: Commit**

```bash
git add scripts/install.sh
git commit -m "feat: install script (corpus dir, symlinks, MCP registration help)

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 14: Evaluator + queries.jsonl

**Files:**
- Create: `src/skill_rag/evaluator.py`
- Modify: `eval/queries.jsonl` (verify each `expected` is just a skill name, not path)
- Create: `tests/test_evaluator.py`

- [ ] **Step 1: Inspect existing eval data**

Run: `cat eval/queries.jsonl | head -5`
Expected: JSON lines with `{query, expected}` fields. If `expected` references paths or includes `source::name`, note which lines need editing.

- [ ] **Step 2: Normalize `eval/queries.jsonl` if needed**

For any line where `expected` is not a plain skill name (string or list of strings matching `~/.skills/<name>/`), rewrite it to the plain name. Example:
```json
{"query": "...", "expected": "brainstorming"}
```
or
```json
{"query": "...", "expected": ["brainstorming", "writing-plans"]}
```

- [ ] **Step 3: Write the failing test**

Create `tests/test_evaluator.py`:
```python
import json

import pytest

from skill_rag import index as index_mod
from skill_rag.evaluator import evaluate, load_cases
from skill_rag.models import SkillRecord


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("SKILL_RAG_INDEX_PATH", str(tmp_path / "index.lance"))
    monkeypatch.setenv("SKILL_RAG_SCORE_THRESHOLD", "0.0")
    import importlib
    from skill_rag import corpus, retrieve
    importlib.reload(corpus)
    importlib.reload(index_mod)
    importlib.reload(retrieve)
    yield
    index_mod.reset()


def test_load_cases(tmp_path):
    p = tmp_path / "q.jsonl"
    p.write_text(
        '{"query": "q1", "expected": "foo"}\n'
        '{"query": "q2", "expected": ["bar", "baz"]}\n'
    )
    cases = load_cases(p)
    assert len(cases) == 2
    assert cases[0].expected == ["foo"]
    assert cases[1].expected == ["bar", "baz"]


def test_evaluate_recall(tmp_path):
    index_mod.upsert([
        SkillRecord(name="foo", description="alpha beta",
                    path="/x/foo/SKILL.md", body="", content_hash="h1"),
        SkillRecord(name="bar", description="gamma delta",
                    path="/x/bar/SKILL.md", body="", content_hash="h2"),
    ])
    p = tmp_path / "q.jsonl"
    p.write_text(
        '{"query": "alpha", "expected": "foo"}\n'
        '{"query": "gamma", "expected": "bar"}\n'
    )
    report = evaluate(load_cases(p), k=5)
    assert report.n == 2
    assert report.recall_at_k == 1.0
    assert report.p95_ms >= 0
```

- [ ] **Step 4: Run test to verify it fails**

Run: `uv run pytest tests/test_evaluator.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 5: Implement**

Create `src/skill_rag/evaluator.py`:
```python
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import retrieve


@dataclass(slots=True)
class Case:
    query: str
    expected: list[str]


@dataclass(slots=True)
class Report:
    n: int
    k: int
    recall_at_k: float
    mrr: float
    p50_ms: float
    p95_ms: float
    misses: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "k": self.k,
            "recall_at_k": self.recall_at_k,
            "mrr": self.mrr,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "misses": self.misses,
        }


def load_cases(path: Path) -> list[Case]:
    cases: list[Case] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        expected = obj["expected"]
        if isinstance(expected, str):
            expected = [expected]
        cases.append(Case(query=str(obj["query"]), expected=list(expected)))
    return cases


def evaluate(cases: list[Case], k: int = 5) -> Report:
    if not cases:
        return Report(n=0, k=k, recall_at_k=0.0, mrr=0.0, p50_ms=0.0, p95_ms=0.0)

    hit_count = 0
    rr_sum = 0.0
    latencies: list[float] = []
    misses: list[dict] = []

    for case in cases:
        t0 = time.monotonic()
        res = retrieve.search(case.query, k=k)
        latencies.append((time.monotonic() - t0) * 1000.0)

        names = [h["name"] for h in res.get("hits", [])]
        found = [n for n in case.expected if n in names]
        if found:
            hit_count += 1
            # MRR from first expected name found
            best_rank = min(names.index(n) + 1 for n in found)
            rr_sum += 1.0 / best_rank
        else:
            misses.append({"query": case.query, "expected": case.expected, "got": names})

    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[max(0, int(len(latencies) * 0.95) - 1)]
    return Report(
        n=len(cases),
        k=k,
        recall_at_k=hit_count / len(cases),
        mrr=rr_sum / len(cases),
        p50_ms=p50,
        p95_ms=p95,
        misses=misses,
    )
```

- [ ] **Step 6: Add `eval` subcommand back to CLI**

Modify `src/skill_rag/cli.py` — add at the end (after `mcp` command, before `if __name__ == "__main__":`):
```python
from pathlib import Path as _Path


@app.command()
def eval(
    dataset: _Path = typer.Option(_Path("eval/queries.jsonl"), "--dataset", "-d"),
    k: int = typer.Option(5, "--k", "-k"),
    json_out: bool = typer.Option(False, "--json"),
):
    """Run the evaluation harness."""
    from .evaluator import evaluate as _eval, load_cases

    cases = load_cases(dataset)
    report = _eval(cases, k=k)
    if json_out:
        typer.echo(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        return
    typer.echo(f"n            = {report.n}")
    typer.echo(f"recall@{report.k:<5} = {report.recall_at_k:.3f}")
    typer.echo(f"mrr          = {report.mrr:.3f}")
    typer.echo(f"latency p50  = {report.p50_ms:.1f} ms")
    typer.echo(f"latency p95  = {report.p95_ms:.1f} ms")
    if report.misses:
        typer.echo(f"\nmisses ({len(report.misses)}):")
        for m in report.misses:
            typer.echo(f"  q={m['query']!r}")
            typer.echo(f"    expected={m['expected']}")
            typer.echo(f"    got={m['got']}")
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `uv run pytest tests/test_evaluator.py -v`
Expected: 2 PASS.

- [ ] **Step 8: Commit**

```bash
git add src/skill_rag/evaluator.py src/skill_rag/cli.py tests/test_evaluator.py eval/queries.jsonl
git commit -m "feat: evaluator with recall@k, MRR, latency percentiles

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 15: End-to-end integration test

**Files:**
- Create: `tests/test_e2e.py`

- [ ] **Step 1: Write the test**

Create `tests/test_e2e.py`:
```python
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
    times = [1000.0]
    monkeypatch.setattr(sync_mod.time, "monotonic", lambda: times[0])
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
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/test_e2e.py -v`
Expected: 2 PASS.

- [ ] **Step 3: Run full suite as smoke test**

Run: `uv run pytest -v`
Expected: ALL tests pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_e2e.py
git commit -m "test: end-to-end flow (search -> get -> add -> sync -> delete)

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 16: Rewrite README.md (Korean)

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write README**

Create `README.md`:
````markdown
# skill_rag

`~/.skills/` 에 모아둔 스킬들을 자연어로 검색해서 필요한 것만 에이전트 컨텍스트에 올리는 로컬 RAG.

세션 시작 시 메타-스킬 1개만 자동 로드되고, 나머지 N개는 매 사용자 메시지마다
MCP로 검색해서 적합한 본문만 가져옴. 따라서 처음부터 모든 스킬을 읽느라 컨텍스트
소모하지 않음.

## 핵심 동작

```
사용자 메시지
   │
   ▼
에이전트 → search_skills(query)  ─→ top-k 메타 (name, desc, score)
                                       │
                                       ▼ 적합한 것만
                                  get_skill(name) ─→ SKILL.md 본문
```

- 임베딩: `all-MiniLM-L6-v2` 로컬 모델 (외부 API 호출 없음)
- 벡터 DB: LanceDB
- 인덱스: `search_skills` 호출 시 TTL 30s 캐시로 자동 sync

## 설치

```bash
git clone <this repo>
cd skill_rag
uv sync
bash scripts/install.sh
```

`install.sh`가 하는 일:
1. `~/.skills/` 디렉토리 생성
2. 부트스트랩 메타-스킬 `~/.skills/using-skill-rag/` 설치
3. 각 하네스(`~/.claude/skills/`, `~/.codex/skills/`)에 심볼릭 링크
4. MCP 서버 등록 가이드 출력

각 하네스 설정에 MCP 서버 추가 (안내 메시지 참고) 후 재시작.

## 스킬 추가

`~/.skills/<name>/SKILL.md` 형식으로 파일 작성:

```markdown
---
name: my-skill
description: 한 줄 설명. 검색 정확도가 여기 품질에 좌우됨.
---

# 본문
스킬 사용법을 자세히 적음.
```

다음 `search_skills` 호출 시 30초 이내에 자동 인덱싱됨.

## CLI

| 명령 | 설명 |
| --- | --- |
| `uv run skill-rag sync` | 인덱스 수동 동기화 |
| `uv run skill-rag query "<text>"` | 검색 결과 확인 |
| `uv run skill-rag list-skills` | 인덱스된 스킬 목록 |
| `uv run skill-rag eval` | 평가셋으로 recall@5 측정 |
| `uv run skill-rag reset` | 인덱스 초기화 |
| `uv run skill-rag mcp` | MCP 서버 실행 |

## 환경 변수

| 변수 | 기본 | 설명 |
| --- | --- | --- |
| `SKILL_RAG_CORPUS_PATH` | `~/.skills` | corpus 경로 |
| `SKILL_RAG_INDEX_PATH` | `./var/index.lance` | LanceDB 경로 |
| `SKILL_RAG_MODEL` | `all-MiniLM-L6-v2` | 임베딩 모델 |
| `SKILL_RAG_SCORE_THRESHOLD` | `0.35` | 매칭 임계값 |
| `SKILL_RAG_SYNC_TTL` | `30` | sync 캐시 TTL (초) |

## 문서

- `AGENTS.md` — 에이전트가 첫 작업 전 읽을 순서
- `ARCHITECTURE.md` — 모듈 구조
- `docs/product-specs/skill-rag.md` — 무엇을, 왜
- `docs/design-docs/` — 설계 결정 로그
- `docs/superpowers/specs/` — 기능별 설계 스펙
````

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: rewrite README for central corpus + lazy RAG design

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 17: Rewrite AGENTS.md

**Files:**
- Create: `AGENTS.md`

- [ ] **Step 1: Write AGENTS.md**

Create `AGENTS.md`:
```markdown
# AGENTS

Entry point for any agent joining this repository.
Read documents in the order below before writing code.

## Read Order

1. `ARCHITECTURE.md` — stable system structure and module boundaries.
2. `docs/product-specs/skill-rag.md` — what we are building and why.
3. `docs/design-docs/core-beliefs.md` — non-negotiable principles.
4. `docs/design-docs/mcp-interface.md` — `search_skills` / `get_skill` contract.
5. `docs/design-docs/meta-skill-bootstrap.md` — how the bootstrap skill is installed and behaves.
6. `docs/design-docs/index.md` — design decision log.
7. `docs/superpowers/specs/` and `docs/superpowers/plans/` — active feature work.
8. `docs/references/*-llms.txt` — quick reference for uv, lancedb, sentence-transformers, mcp.

## Repository Map

- `src/skill_rag/` — Python package (cli, models, parser, loader, embed, index, retrieve, sync, mcp_server, evaluator, corpus).
- `bootstrap-skill/using-skill-rag/SKILL.md` — meta-skill installed into each harness via symlink.
- `scripts/install.sh` — set up `~/.skills/`, symlinks, and MCP registration instructions.
- `eval/queries.jsonl` — evaluation queries with gold-standard skill names.
- `tests/` — unit + integration tests.
- `var/` — local-only LanceDB index (gitignored).
- External corpus: `~/.skills/` (read-write by the user; not committed).

## Done When

- `mcp__skill-rag__search_skills` returns `{status: "ok"|"no_match", hits, [message]}`.
- `mcp__skill-rag__get_skill` returns `{status: "ok"|"not_found", body|message}`.
- One bootstrap skill in `~/.skills/`, symlinked into Claude Code + Codex, auto-loads in both.
- File added to `~/.skills/` is reflected in the next `search_skills` call after 30 s.
- `recall@5 ≥ 0.8` on `eval/queries.jsonl`.
- `p95` search latency `< 1 s` on a ~50-skill corpus.
- No cloud API calls at index or query time.

## Operating Rules

- README is the only document allowed in Korean. Everything else stays in English.
- Do not commit the corpus. Treat `~/.skills/` as a user-managed directory.
- Do not add cloud embedding providers. Local-only is a hard constraint (see `core-beliefs.md`).
- When adding a new dependency, update both `pyproject.toml` and `docs/references/<tool>-llms.txt`.
- When the indexing schema changes, bump the schema comment in `src/skill_rag/index.py` and document the migration in `docs/design-docs/`.
- Every behavioral change should be tracked in `docs/superpowers/specs/` and `docs/superpowers/plans/`.
```

- [ ] **Step 2: Commit**

```bash
git add AGENTS.md
git commit -m "docs: rewrite AGENTS.md for new layout and contract

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 18: Rewrite ARCHITECTURE.md

**Files:**
- Create: `ARCHITECTURE.md`

- [ ] **Step 1: Write ARCHITECTURE.md**

Create `ARCHITECTURE.md`:
```markdown
# Architecture

skill_rag serves a single user-global skill corpus (`~/.skills/<name>/SKILL.md`)
to AI agents via an MCP server. Agents start with one bootstrap meta-skill
resident in their context and lazily fetch other skill bodies only when a
RAG search shows they apply.

## Module Boundaries

```
src/skill_rag/
├── corpus.py       # constants: paths, threshold, TTL
├── models.py       # SkillRecord, SearchHit
├── parser.py       # SKILL.md text → SkillRecord
├── loader.py       # ~/.skills/ → [SkillRecord]
├── embed.py        # sentence-transformers wrapper (L2-normalized)
├── index.py        # LanceDB CRUD (schema v3)
├── retrieve.py     # query → top-k with threshold + status response
├── sync.py         # disk↔index diff; TTL cache (only stateful module)
├── mcp_server.py   # FastMCP tools: search_skills, get_skill
├── cli.py          # Typer CLI: sync/query/list-skills/reset/mcp/eval
└── evaluator.py    # recall@k, MRR, latency
```

Each module has one responsibility; only `sync` holds mutable state
(a single timestamp). `mcp_server` is a router with no logic.

## Request Flow

### `search_skills(query, k)`

1. `sync.sync_if_stale()` — runs `loader.scan` and diffs against the index
   only if more than 30 s have elapsed since the last sync.
2. `embed.encode_one(query)` — 384-dim L2-normalized vector.
3. `index.search(vec, k)` — LanceDB cosine, returns `SearchHit[]`.
4. `retrieve.search` filters out hits below `SCORE_THRESHOLD` (default 0.35).
5. Returns `{status: "ok", hits}` or `{status: "no_match", hits: [], message}`.

### `get_skill(name)`

1. Read `~/.skills/<name>/SKILL.md` directly.
2. On miss, force `sync_if_stale(ttl=0)` and retry once.
3. Return `{status: "ok", body}` or `{status: "not_found", message}`.

### Sync (`sync_if_stale`)

- TTL gate (default 30 s, env-overridable).
- `loader.scan(~/.skills)` returns one `SkillRecord` per `<name>/SKILL.md`.
- Skips the `using-skill-rag` directory (it is already loaded by the harness).
- Diff vs. `index.list_indexed()` by `path` + `content_hash`:
  added → upsert; changed → upsert; missing → delete.

## Data Schema

LanceDB table `skills` (schema v3):

| Column | Type | Notes |
| --- | --- | --- |
| `path` | string | Primary key. `<corpus>/<name>/SKILL.md`. |
| `name` | string | From frontmatter. |
| `description` | string | From frontmatter. |
| `content_hash` | string | sha256 of the full SKILL.md. |
| `vector` | list<float32>[384] | Embedding of `name\ndescription`. |

`SkillRecord` and `SearchHit` live in `src/skill_rag/models.py`.

## Loop Prevention

The bootstrap skill's instructions and the server's status responses jointly
prevent runaway calls:

| Tool | Failure shape | Response status | Meta-skill rule |
| --- | --- | --- | --- |
| `search_skills` | No hit above threshold | `no_match` | Respond directly. No retry. |
| `get_skill` | File missing (even after forced sync) | `not_found` | No retry this turn. |
| `search_skills` | Hits returned, none actually fit | `ok` (agent judges) | Proceed without skill. |

## Bootstrap Skill

`~/.skills/using-skill-rag/SKILL.md` is the single source of truth, symlinked
into every supported harness's auto-load directory by `scripts/install.sh`.
`loader.scan` skips it so it never surfaces in search results.

## Constraints

- Local-only: no cloud API calls at index or query time.
- Single user, single corpus.
- Python 3.13 + uv. No `pip` or raw `venv`.
- `~/.skills/` is user-managed and never committed.
```

- [ ] **Step 2: Commit**

```bash
git add ARCHITECTURE.md
git commit -m "docs: rewrite ARCHITECTURE.md for v3 design

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 19: Rewrite product spec and design docs

**Files:**
- Create: `docs/product-specs/skill-rag.md`
- Create: `docs/product-specs/index.md`
- Create: `docs/design-docs/core-beliefs.md`
- Create: `docs/design-docs/mcp-interface.md`
- Create: `docs/design-docs/meta-skill-bootstrap.md`
- Create: `docs/design-docs/index.md`
- Create: `docs/exec-plans/tech-debt-tracker.md`

- [ ] **Step 1: Write product spec**

Create `docs/product-specs/skill-rag.md`:
```markdown
# skill-rag — Product Spec

## What
A local RAG over a user's skill corpus at `~/.skills/`, exposed to AI agents
via an MCP server. Agents call `search_skills` to find relevant skills and
`get_skill` to fetch the body of just the ones that apply.

## Why
Today, every agent session loads its whole skill set into context up front.
Each harness duplicates skills under its own directory. Both waste tokens
and force agents to scan content they will never use this turn.

## Users
A single developer running multiple agent harnesses (Claude Code, Codex, …)
on the same machine, all sharing one skill library.

## Guarantees
- Only one bootstrap skill is loaded by default. Everything else is
  fetched on demand.
- Adding a SKILL.md to `~/.skills/` is searchable within 30 s with no
  manual command.
- `search_skills`/`get_skill` never return a shape that can make a
  conforming agent loop (three explicit terminal statuses).
- Local-only. No cloud calls at index or query time.

## Out of Scope
- Multi-user or shared corpora.
- Re-ranking, BM25, or LLM-based relevance.
- Real-time filesystem watchers.
- Backwards compat with `~/.claude/skills` + `~/.codex/skills` layout.

## Success Metrics
- `recall@5 ≥ 0.8` on `eval/queries.jsonl`.
- `p95 < 1 s` search latency on a ~50-skill corpus.
- No cloud API calls in indexing or query paths.
```

- [ ] **Step 2: Write product spec index**

Create `docs/product-specs/index.md`:
```markdown
# Product Specs

- [`skill-rag.md`](skill-rag.md) — current product
```

- [ ] **Step 3: Write core-beliefs**

Create `docs/design-docs/core-beliefs.md`:
```markdown
# Core Beliefs

Non-negotiable principles. Changes require explicit user buy-in.

## Local-First
No cloud APIs at index or query time. The user owns their machine, their
embeddings, and their corpus. Latency is bounded by local hardware, not
network round trips. Tested via the "no cloud" Done-When criterion.

## Single Global Corpus
One canonical location for skills: `~/.skills/<name>/SKILL.md`. No
multi-source, no per-harness duplication. Harnesses link to the same
files via symlink. Simpler index, simpler mental model, fewer bugs.

## Lazy Loading
Agents do not load skills at session start. The bootstrap skill calls
`search_skills` per user message and `get_skill` only for skills that
clearly apply. Context tokens are spent on relevant skills only.

## Single User
The project assumes one developer on one machine. Concurrency, sharing,
permissions, and ACLs are not designed for and not tested.

## YAGNI
No re-ranking, hybrid search, LLM-based scoring, filesystem watchers,
or other speculative complexity. If `recall@5 ≥ 0.8` is met by plain
cosine over a 384-dim model, we ship that.
```

- [ ] **Step 4: Write mcp-interface**

Create `docs/design-docs/mcp-interface.md`:
````markdown
# MCP Interface

Two tools. Three terminal statuses across them.

## `search_skills(query: str, k: int = 5) -> dict`

Find skills relevant to the query. Auto-runs sync if the cache is stale.

```json
// status: "ok"
{
  "status": "ok",
  "hits": [
    {"name": "brainstorming", "description": "...", "score": 0.82},
    {"name": "writing-plans", "description": "...", "score": 0.74}
  ]
}

// status: "no_match"  (no hits above SCORE_THRESHOLD, or empty corpus)
{
  "status": "no_match",
  "hits": [],
  "message": "No skill matched this query. Proceed without using a skill."
}
```

## `get_skill(name: str) -> dict`

Return the full SKILL.md body. If the file is missing, force a sync and
retry once before returning `not_found`.

```json
// status: "ok"
{"status": "ok", "body": "---\nname: ...\n---\n..."}

// status: "not_found"
{
  "status": "not_found",
  "message": "Skill 'X' does not exist in the corpus. Do not call get_skill or search_skills for this name again. Proceed without it."
}
```

## Threshold

`SCORE_THRESHOLD` (default `0.35`, env `SKILL_RAG_SCORE_THRESHOLD`) filters
out low-similarity matches in `search_skills`. Tune via `eval/queries.jsonl`
to the highest value that still satisfies `recall@5 ≥ 0.8`.

## Loop Prevention Contract

A conforming bootstrap skill must:

- On `search_skills → no_match`: respond directly, not re-call with a
  reworded query.
- On `get_skill → not_found`: not re-call `get_skill` or `search_skills`
  for the same name this turn.
- On `search_skills → ok` where no description actually fits: proceed
  without a skill, not refetch.
````

- [ ] **Step 5: Write meta-skill-bootstrap**

Create `docs/design-docs/meta-skill-bootstrap.md`:
```markdown
# Meta-Skill Bootstrap

## Location

The bootstrap skill lives in two places, both pointing to the same file:

- Source of truth: `~/.skills/using-skill-rag/SKILL.md`
- Each harness's auto-load dir: `~/.<harness>/skills/using-skill-rag/` →
  symlink to the above.

`scripts/install.sh` creates the directories and the symlinks. To update
the skill, edit the canonical file; all harnesses see it immediately.

## Why Symlink

- One file to maintain.
- No drift between harnesses.
- Harness's own auto-load mechanism discovers it as a normal skill.

If a harness blocks symlinks, fall back to copy and re-run `install.sh`
to refresh.

## Why Excluded from Search

`loader.scan` skips the `using-skill-rag` directory because the skill is
always already in the agent's context. Surfacing it in `search_skills`
results would waste a slot.

## Behavior Contract

The skill body (`bootstrap-skill/using-skill-rag/SKILL.md`) spells out:

- Parent agent: call `search_skills` before responding to any user message.
- Subagent: call only if the parent context describes substantive work.
- Status handling: `ok`/`no_match`/`not_found` each have a terminal action.
- Anti-patterns: no multi-call retries with reworded queries.

The MCP server's response shapes enforce the contract; the skill text
makes the contract explicit to the agent.
```

- [ ] **Step 6: Write design-docs index and tech-debt-tracker**

Create `docs/design-docs/index.md`:
```markdown
# Design Decision Log

| Date | Decision | Doc |
| --- | --- | --- |
| 2026-05-28 | Centralize corpus at `~/.skills/`, drop multi-source | `../superpowers/specs/2026-05-28-centralized-skills-rag-design.md` |
| 2026-05-28 | LanceDB schema v3: drop `source` column | `../../src/skill_rag/index.py` (schema comment) |
| 2026-05-28 | `search_skills` + `get_skill` with explicit terminal statuses | `mcp-interface.md` |
| 2026-05-28 | Bootstrap skill via symlink from `~/.skills/` | `meta-skill-bootstrap.md` |
| 2026-05-28 | Local-only, single-user, single-corpus, YAGNI | `core-beliefs.md` |
```

Create `docs/exec-plans/tech-debt-tracker.md`:
```markdown
# Tech Debt Tracker

(empty — fresh rebuild)

## Known follow-ups

- Validate `SCORE_THRESHOLD` default against `eval/queries.jsonl` and
  pin the chosen value in `corpus.py` / `mcp-interface.md`.
- Codex MCP registration instructions in `scripts/install.sh` are a
  placeholder pointing to Codex docs.
```

- [ ] **Step 7: Commit**

```bash
git add docs/product-specs docs/design-docs docs/exec-plans
git commit -m "docs: rewrite product spec, design docs, and decision log

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 20: Add MCP reference and update existing references

**Files:**
- Create: `docs/references/mcp-llms.txt`

- [ ] **Step 1: Write mcp-llms.txt**

Create `docs/references/mcp-llms.txt`:
```
# mcp (Python SDK) — quick reference

Repo: https://github.com/modelcontextprotocol/python-sdk
PyPI: mcp>=1.20.0

## FastMCP server

from mcp.server.fastmcp import FastMCP

server = FastMCP("my-server")

@server.tool()
def my_tool(arg: str) -> dict:
    """Docstring becomes the tool description shown to the agent."""
    return {"status": "ok", "result": arg}

server.run()  # stdio transport

## Tool conventions

- Return JSON-serializable dicts.
- Use explicit "status" field for success vs. error states; agents read it
  to decide next action.
- Keep docstrings short and action-oriented — they appear in the agent's
  prompt as the tool description.

## Registering an MCP server in Claude Code

~/.claude.json
  "mcpServers": {
    "my-server": {
      "command": "uv",
      "args": ["--directory", "/path/to/repo", "run", "my-server", "mcp"]
    }
  }

Restart Claude Code to pick up changes.
```

- [ ] **Step 2: Commit**

```bash
git add docs/references/mcp-llms.txt
git commit -m "docs: add MCP quick reference

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 21: Smoke test the install script + manual MCP check

**Files:** (no files modified — verification only)

- [ ] **Step 1: Dry run the install script with a temp HOME**

Run:
```bash
TMPHOME="$(mktemp -d)"
HOME="${TMPHOME}" bash scripts/install.sh
ls -la "${TMPHOME}/.skills" "${TMPHOME}/.claude/skills" "${TMPHOME}/.codex/skills"
```
Expected:
- `~/.skills/using-skill-rag/SKILL.md` exists.
- `~/.claude/skills/using-skill-rag` is a symlink to `~/.skills/using-skill-rag`.
- `~/.codex/skills/using-skill-rag` is a symlink to `~/.skills/using-skill-rag`.

Clean up: `rm -rf "${TMPHOME}"`.

- [ ] **Step 2: Manual MCP smoke test**

Run:
```bash
SKILL_RAG_CORPUS_PATH="$(mktemp -d)" SKILL_RAG_INDEX_PATH=/tmp/skill-rag-smoke.lance \
  uv run python -c "
from skill_rag import mcp_server
print(mcp_server.search_skills('hello', k=3))
print(mcp_server.get_skill('does-not-exist'))
"
```
Expected:
- First line: `{'status': 'no_match', 'hits': [], 'message': '...'}`
- Second line: `{'status': 'not_found', 'message': '...'}`

Clean up: `rm -rf /tmp/skill-rag-smoke.lance`.

- [ ] **Step 3: Full test suite final pass**

Run: `uv run pytest -v`
Expected: ALL tests pass. Note the total count for the next step.

- [ ] **Step 4: Commit** (verification only — nothing to commit, but record the result)

If everything passes, no commit needed. If the suite failed or any check above
failed, fix in a focused commit before declaring done.

---

## Task 22: Threshold calibration against eval set

**Files:**
- Modify: `src/skill_rag/corpus.py` (set `SCORE_THRESHOLD` default to the calibrated value)
- Modify: `docs/design-docs/mcp-interface.md` (update the default value if changed)

- [ ] **Step 1: Populate corpus from `bootstrap-skill/` + any sample skills available**

For calibration, the index needs real content. Use the user's actual
`~/.skills/` if populated; otherwise create a synthetic set matching
`eval/queries.jsonl` expectations.

Run: `uv run skill-rag sync`
Expected: `added: N` for some N matching the eval set's expected skills.

- [ ] **Step 2: Sweep thresholds and record recall@5**

Run:
```bash
for t in 0.20 0.25 0.30 0.35 0.40 0.45; do
  SKILL_RAG_SCORE_THRESHOLD=$t uv run skill-rag eval --json | \
    python -c "import json,sys; r=json.load(sys.stdin); print(f't=$t recall@5={r[\"recall_at_k\"]:.3f} p95={r[\"p95_ms\"]:.1f}ms')"
done
```
Expected: a table of `(t, recall@5, p95)`. Pick the **largest** `t` that
keeps `recall@5 ≥ 0.8`.

- [ ] **Step 3: Update default**

Edit `src/skill_rag/corpus.py`:
```python
SCORE_THRESHOLD = float(os.environ.get("SKILL_RAG_SCORE_THRESHOLD", "<CALIBRATED_VALUE>"))
```
where `<CALIBRATED_VALUE>` is the chosen number from Step 2.

Edit `docs/design-docs/mcp-interface.md`: replace `default 0.35` with the
calibrated value. Same in `README.md` if mentioned.

- [ ] **Step 4: Re-run tests to confirm nothing broke**

Run: `uv run pytest -v`
Expected: ALL tests pass. (Tests that depend on threshold override it
explicitly via `monkeypatch` so the default change is safe.)

- [ ] **Step 5: Commit**

```bash
git add src/skill_rag/corpus.py docs/design-docs/mcp-interface.md README.md
git commit -m "tune: calibrate SCORE_THRESHOLD against eval set

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Self-Review

The plan has been checked against the spec:

- **Spec coverage** — every section of `2026-05-28-centralized-skills-rag-design.md` maps to one or more tasks:
  - Architecture diagram → Tasks 7–10 (index/retrieve/sync/mcp_server)
  - Components table → Tasks 2–11 (one task per module)
  - Data models → Task 3
  - Data flow (sync, search, get) → Tasks 9, 8, 10
  - Loop prevention → Tasks 8 (no_match), 10 (not_found), 12 (meta-skill text)
  - Bootstrap skill + install → Tasks 12, 13
  - Rebuild plan (orphan branch) → Task 1
  - Documentation rewrite → Tasks 16–20
  - Testing strategy → Tests embedded in Tasks 2–11, plus Task 15 (e2e)
  - Done-When → Task 21 (smoke) + Task 22 (recall@5 calibration)

- **Placeholder scan** — no TBD/TODO/"add appropriate" patterns. The one
  parameterized value (`<CALIBRATED_VALUE>` in Task 22) is explicitly
  derived from a measurement step in the same task.

- **Type consistency** — `SkillRecord` fields (`name`, `description`,
  `path`, `body`, `content_hash`) match across parser/loader/index/sync.
  `SearchHit` fields (`name`, `description`, `score`) match across
  index/retrieve/evaluator. Function names consistent (`run_sync`,
  `sync_if_stale`, `reset_cache`; `search_skills`, `get_skill`).
