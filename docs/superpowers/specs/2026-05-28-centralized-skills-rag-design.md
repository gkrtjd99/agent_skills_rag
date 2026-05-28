# Centralized Skills + Lazy RAG Loading — Design

**Date**: 2026-05-28
**Status**: Approved, pending implementation plan
**Supersedes**: prior multi-source corpus design (`~/.claude/skills` + `~/.codex/skills`)

## Problem

Today every agent session loads its full skill set into context up front. Each
harness (Claude Code, Codex, …) maintains its own copy under
`~/.<harness>/skills/`, which forces duplication and forces agents to spend
context tokens on skills they will never use this turn.

We want a single, user-global skill corpus that agents query on demand,
fetching only the skill bodies that actually apply to the current task.

## Goals

- Single source of truth for skills at `~/.skills/<name>/SKILL.md`.
- Agents start with one tiny bootstrap skill in context. Everything else is
  retrieved per turn via a local MCP server backed by a vector index.
- No cloud calls on any hot path. Local embedding model only.
- Adding a file to `~/.skills/` is reflected in search within 30 seconds, with
  no manual sync command.
- Wrong-skill and missing-skill paths terminate cleanly — never loop.

## Non-Goals

- Multi-user or shared corpora.
- Real-time filesystem watching (TTL-based sync is sufficient).
- Backwards compatibility with the existing `~/.claude/skills` +
  `~/.codex/skills` layout. Users re-install under `~/.skills/`.
- Re-ranking, hybrid (BM25 + vector), or LLM-based relevance scoring.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  Agent                                              │
│  [bootstrap skill: using-skill-rag]                 │
│       │ every user message                          │
│       ▼                                             │
│  search_skills(query, k=5)                          │
│       │                                             │
│       ▼ meta list (name, description, score)        │
│       │                                             │
│       ├─ judge fit ─┐                               │
│       │             │                               │
│       │        ┌────▼────┐         ┌──────────────┐ │
│       │        │ no fit  │────────▶│ respond w/o  │ │
│       │        │         │         │ skill        │ │
│       │        └─────────┘         └──────────────┘ │
│       │                                             │
│       ▼ for each fitting hit                        │
│  get_skill(name)                                    │
│       │ SKILL.md body                               │
│       ▼ follow instructions                         │
└────────────────────────────┬────────────────────────┘
                             │ MCP stdio
              ┌──────────────▼──────────────┐
              │  skill-rag MCP server       │
              │                             │
              │  search_skills:             │
              │   1. sync_if_stale (TTL 30s)│
              │   2. embed(query)           │
              │   3. cosine top-k           │
              │   4. filter score >= 0.35   │
              │                             │
              │  get_skill:                 │
              │   1. read SKILL.md          │
              │   2. on miss: force sync    │
              │      then retry once        │
              └────┬───────────────┬────────┘
                   │               │
        ┌──────────▼──┐  ┌─────────▼────────┐
        │ LanceDB     │  │ ~/.skills/       │
        │ (meta+vec)  │  │  <name>/SKILL.md │
        └─────────────┘  └──────────────────┘
```

Only the bootstrap skill `using-skill-rag` sits in the agent's context at
session start. Every other skill enters context only after `get_skill` returns
its body.

## Components

```
src/skill_rag/
├── cli.py           # skill-rag {index|query|sync|mcp|reset|list}
├── models.py        # SkillRecord, SearchHit
├── parser.py        # SKILL.md text → frontmatter + body
├── loader.py        # ~/.skills/ scan → SkillRecord[]
├── embed.py         # sentence-transformers wrapper
├── index.py         # LanceDB upsert/search/delete
├── sync.py          # disk↔index diff, TTL cache
├── retrieve.py      # query → top-k with threshold filter
└── mcp_server.py    # MCP tools: search_skills, get_skill

bootstrap-skill/
└── using-skill-rag/SKILL.md

scripts/install.sh   # symlink + MCP registration
eval/                # queries.jsonl + runner.py
tests/               # unit + integration
var/                 # gitignored: index.lance
```

**Responsibility per module** (each does one thing):

| Module | Responsibility | Depends on |
|---|---|---|
| `parser` | Parse one SKILL.md string → `{name, description, body, hash}` | — |
| `loader` | Walk corpus dir → `SkillRecord[]`, skip `using-skill-rag` | parser |
| `embed` | Text → L2-normalized vector | sentence-transformers |
| `index` | LanceDB CRUD | embed |
| `sync` | Diff disk vs index, hold TTL timestamp | loader, index |
| `retrieve` | Query → top-k SearchHit with threshold | embed, index |
| `mcp_server` | Route MCP tool calls; no business logic | sync, retrieve, parser |
| `cli` | Human-facing CLI | all of above |

`sync` is the only module with mutable state (a single timestamp). Everything
else is stateless.

### Data Models

```python
@dataclass
class SkillRecord:
    name: str           # frontmatter name
    description: str    # frontmatter description
    path: str           # ~/.skills/<name>/SKILL.md
    body: str           # full SKILL.md body
    content_hash: str   # sha256(body)

@dataclass
class SearchHit:
    name: str
    description: str
    score: float        # cosine similarity, 0..1
```

Removed vs. prior schema: `source` field (single corpus now), and the
`(name, content_hash)` dedup buckets that handled multi-source overlap.

## Data Flow

### Sync (`sync_if_stale`)

```
1. if _last_sync_at and now - _last_sync_at < ttl: return
2. records = loader.scan("~/.skills")
3. db_index = {row.path: row.content_hash for row in index.list_indexed()}
4. changed = [r for r in records if db_index.get(r.path) != r.content_hash]
   removed = [path for path in db_index if path not in {r.path for r in records}]
5. index.upsert(changed); index.delete_by_paths(removed)
6. _last_sync_at = now
```

`_last_sync_at` is a module-level `float | None`. Cleared only when the MCP
process exits or on explicit `skill-rag sync --force`. Default TTL: 30 s.

### `search_skills(query, k=5)`

```python
def search_skills(query: str, k: int = 5) -> dict:
    sync.sync_if_stale()
    vec = embed.encode_one(query)
    hits = index.search(vec, k=k)
    relevant = [h for h in hits if h.score >= SCORE_THRESHOLD]  # 0.35 initial
    if not relevant:
        return {"status": "no_match",
                "message": "No skill matched. Proceed without using a skill.",
                "hits": []}
    return {"status": "ok", "hits": relevant}
```

Threshold tuned via `eval/queries.jsonl` to satisfy recall@5 ≥ 0.8 at the
highest possible value. Initial value 0.35 based on `all-MiniLM-L6-v2`
defaults.

### `get_skill(name)`

```python
def get_skill(name: str) -> dict:
    path = corpus_root / name / "SKILL.md"
    if path.exists():
        return {"status": "ok", "body": path.read_text()}
    sync.sync_if_stale(ttl=0)         # force
    if path.exists():
        return {"status": "ok", "body": path.read_text()}
    return {"status": "not_found",
            "message": f"Skill '{name}' no longer exists. "
                       f"Do not call search_skills or get_skill for it again. "
                       f"Proceed without it."}
```

## Loop Prevention

Three failure shapes, three explicit terminal responses, mirrored in the
bootstrap skill's instructions:

| Tool | Failure | Response status | Meta-skill rule |
|---|---|---|---|
| `search_skills` | All top-k below threshold | `no_match` | Respond directly. Do not retry with rephrased query. |
| `get_skill` | File missing, even after forced sync | `not_found` | Do not call `get_skill` or `search_skills` for this name again this turn. |
| `search_skills` | Returns skills but none actually fit on inspection | `ok` (agent judges) | Proceed without skill. Do not refetch. |

## Bootstrap Skill

**Location**: `~/.skills/using-skill-rag/SKILL.md` (canonical), symlinked into
each harness's auto-load directory.

**Body** (excerpt — full text shipped with the implementation):

```markdown
---
name: using-skill-rag
description: Use at the start of every user message to find relevant skills via RAG before responding
---

# Skill RAG — Lazy Skill Loading

## Required behavior
- Parent agent: BEFORE responding to any user message, call
  search_skills(query=<user's message>, k=5).
- Subagent: inspect the parent context. If it describes a substantive task,
  call search_skills with a query summarizing your task. If it's a narrow
  lookup the parent already framed, skip.

## Handling responses
- search_skills → "ok": read descriptions, call get_skill on those that fit.
- search_skills → "no_match": respond directly, do not retry.
- get_skill → "ok": follow body.
- get_skill → "not_found": do not retry, proceed without it.

## Anti-patterns
- Calling search_skills more than once per turn with reworded queries.
- Fetching a skill whose description clearly doesn't fit.
- Skipping search_skills because "this looks simple".
```

**Installation** (`scripts/install.sh`):

```
1. mkdir -p ~/.skills
2. Create ~/.skills/using-skill-rag/SKILL.md if missing
3. For each harness in {claude, codex}:
     ln -sfn ~/.skills/using-skill-rag ~/.<harness>/skills/using-skill-rag
4. Register skill-rag MCP server in each harness's MCP config
```

Symlinks keep a single source of truth. `loader.scan()` explicitly skips the
`using-skill-rag/` directory so it never appears in search results (it's
always already loaded).

## Rebuild Plan

Git history is reset. Verified-working modules are preserved; everything tied
to the multi-source assumption is rewritten.

**Preserve** (verified, mechanical changes only): `pyproject.toml`, `uv.lock`,
`LICENSE`, `.gitignore`, `embed.py`, `parser.py`, `eval/queries.jsonl`,
`docs/references/*-llms.txt`.

**Modify** (drop `source` field / dedup):
`models.py`, `index.py`, `retrieve.py`.

**Write new**: `sync.py`, `loader.py` (single-path), `mcp_server.py` (with
`get_skill` + status responses), `cli.py` (with `sync` subcommand),
`bootstrap-skill/using-skill-rag/SKILL.md`, `scripts/install.sh`, all tests.

**Procedure**:

```bash
git branch backup/pre-rewrite           # safety net (local only)
git checkout --orphan fresh-start
git rm -rf .
# re-add preserved + new files
git commit -m "Initial: skill-rag with central ~/.skills corpus"
git branch -M fresh-start main
# user runs: git push -f origin main
```

## Documentation Rewrite

| File | Action |
|---|---|
| `README.md` (Korean) | Full rewrite: new concept, one-line install, usage examples |
| `AGENTS.md` | Update Read order, refresh Done-When |
| `ARCHITECTURE.md` | Full rewrite from this spec |
| `docs/product-specs/skill-rag.md` | Full rewrite: lazy loading value, 2-step MCP flow |
| `docs/design-docs/core-beliefs.md` | Add "single global corpus"; keep "local-only" |
| `docs/design-docs/mcp-interface.md` | NEW: tool contracts, status codes, threshold policy |
| `docs/design-docs/meta-skill-bootstrap.md` | NEW: bootstrap behavior + install model |
| `docs/design-docs/index.md` | Restructured decision log |
| `docs/references/mcp-llms.txt` | NEW: MCP reference |
| `docs/exec-plans/active/` | Cleared; new plan created via writing-plans |
| `docs/exec-plans/completed/` | Cleared |
| `docs/exec-plans/tech-debt-tracker.md` | Reset |

## Testing Strategy

**Unit** — one focus per module:

| Module | Cases |
|---|---|
| `parser` | valid frontmatter, missing fields, empty body, malformed YAML, empty file |
| `loader` | empty dir, single skill, N skills, `using-skill-rag` excluded, dirs without SKILL.md ignored |
| `embed` | vector dimension, L2 norm ≈ 1, empty input → shape (0, dim) |
| `index` | upsert→search roundtrip, delete_by_paths, reset, same-path upsert overwrites |
| `sync` | within-TTL skip, post-TTL re-run, add/modify/delete diffs, `ttl=0` forces |
| `retrieve` | above/below threshold split, empty corpus → no_match |
| `mcp_server` | `search_skills` ok/no_match shape, `get_skill` ok/not_found, force-sync-retry on miss |

**Integration** — `tmp_path` fixture with 3–5 fake SKILL.md files:
install → search → get → modify file → bump monotonic clock past TTL →
search → confirm change picked up.

**Evaluation** — `eval/runner.py`:
- recall@5 over `queries.jsonl`
- p95 latency over 1000 calls
- threshold sweep {0.20, 0.25, 0.30, 0.35, 0.40} → select highest value
  satisfying recall@5 ≥ 0.8

**Not tested** (YAGNI): concurrency (MCP serial), network (local only),
multiple embedding models.

## Done-When

- `mcp__skill-rag__search_skills` returns `ok` / `no_match` per spec.
- `mcp__skill-rag__get_skill` returns `ok` / `not_found` per spec.
- One bootstrap skill in `~/.skills/`, symlinked into Claude Code + Codex,
  auto-loads in both sessions.
- File added to `~/.skills/` is reflected in the next `search_skills` call
  after 30 s.
- recall@5 ≥ 0.8 on `eval/queries.jsonl`.
- p95 search latency < 1 s on a ~50-skill corpus.
- No cloud API calls at index or query time.

## Open Questions

None blocking. Threshold value (0.35) is an initial guess; final value
determined during evaluation phase.
