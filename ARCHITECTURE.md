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
4. `retrieve.search` filters out hits below `SCORE_THRESHOLD` (default 0.25).
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
- Embedding model loading is local-only by default; set
  `SKILL_RAG_LOCAL_FILES_ONLY=0` only for an explicit model download/setup run.
