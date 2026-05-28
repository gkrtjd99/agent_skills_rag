# skill_rag

**English** | [한국어](README.ko.md)

A local RAG that searches the skills collected under `~/.skills/` in natural
language and loads only the relevant ones into the agent's context.

At session start only a single meta-skill is auto-loaded; the remaining N
skills are searched per user message via MCP, and only the matching bodies
are fetched. This avoids burning context by reading every skill up front.

## How it works

```
user message
   │
   ▼
agent → search_skills(query)  ─→ top-k metadata (name, desc, score)
                                       │
                                       ▼ only the relevant ones
                                  get_skill(name) ─→ SKILL.md body
```

- Embeddings: `all-MiniLM-L6-v2` running locally (no external API calls)
- Vector DB: LanceDB
- Index: auto-synced with a 30s TTL cache on each `search_skills` call

## Install

### 1) Code + bootstrap skill

```bash
git clone <repo-url>
cd skill_rag
uv sync
bash scripts/install.sh
```

What `install.sh` does:
1. Creates the `~/.skills/` directory
2. Installs the bootstrap meta-skill at `~/.skills/using-skill-rag/`
3. Symlinks it into each harness (`~/.claude/skills/`, `~/.codex/skills/`)
4. Prints the MCP registration guide below to the console

### 2) Register the MCP server

Substitute the repo path (`$REPO`) with the output of `pwd`. The MCP launch
command is the same across every harness:

```
uv --directory $REPO run skill-rag mcp
```

#### Claude Code

One-line CLI registration (user scope, global):

```bash
claude mcp add skill-rag --scope user -- uv --directory "$(pwd)" run skill-rag mcp
```

Or add it directly to `mcpServers` in `~/.claude.json`:

```json
{
  "mcpServers": {
    "skill-rag": {
      "command": "uv",
      "args": ["--directory", "/absolute/path/to/skill_rag", "run", "skill-rag", "mcp"]
    }
  }
}
```

#### Codex

Add to `~/.codex/config.toml`:

```toml
[mcp_servers.skill-rag]
command = "uv"
args = ["--directory", "/absolute/path/to/skill_rag", "run", "skill-rag", "mcp"]
```

#### Other MCP-compatible clients (Cursor, Windsurf, etc.)

Register the same `command` / `args` in each client's MCP settings.

### 3) Restart + sanity check

After restarting the harness, in a fresh session:

- Confirm the `using-skill-rag` meta-skill auto-loads at start
- Confirm the `mcp__skill-rag__search_skills` tool is visible in any message
- Try it directly: `search_skills(query="...", k=5)` → `{status: "ok"|"no_match", hits, ...}`

You can also verify the same search from the CLI:

```bash
uv run skill-rag query "deploy to vercel"
```

## Adding a skill

Write a file at `~/.skills/<name>/SKILL.md`:

```markdown
---
name: my-skill
description: One-line description. Search accuracy depends on this.
---

# Body
Detailed usage of the skill.
```

It is auto-indexed within 30s on the next `search_skills` call.

## CLI

| Command | Description |
| --- | --- |
| `uv run skill-rag sync` | Manually sync the index |
| `uv run skill-rag query "<text>"` | Inspect search results |
| `uv run skill-rag list-skills` | List indexed skills |
| `uv run skill-rag eval` | Measure recall@5 against the public fixture eval |
| `uv run skill-rag reset` | Reset the index |
| `uv run skill-rag mcp` | Run the MCP server |

## Environment variables

| Variable | Default | Description |
| --- | --- | --- |
| `SKILL_RAG_CORPUS_PATH` | `~/.skills` | Corpus path |
| `SKILL_RAG_INDEX_PATH` | `./var/index.lance` | LanceDB path |
| `SKILL_RAG_MODEL` | `all-MiniLM-L6-v2` | Embedding model |
| `SKILL_RAG_LOCAL_FILES_ONLY` | `1` | Load the embedding model from local cache only |
| `SKILL_RAG_SCORE_THRESHOLD` | `0.25` | Match threshold (calibrated against the eval set) |
| `SKILL_RAG_SYNC_TTL` | `30` | Sync cache TTL (seconds) |

`skill-rag eval` defaults to repository-owned fixtures under `eval/fixtures/`
so GitHub users get the same benchmark. To inspect your personal corpus, pass
both paths explicitly:

```bash
uv run skill-rag eval --corpus ~/.skills --dataset eval/queries.jsonl
```

## Docs

- `AGENTS.md` — reading order before an agent's first task
- `ARCHITECTURE.md` — module structure
- `docs/product-specs/skill-rag.md` — what and why
- `docs/design-docs/` — design decision log
- `docs/superpowers/specs/` — per-feature design specs
