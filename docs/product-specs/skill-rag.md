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
- `recall@5 ≥ 0.8` on the public fixture eval shipped in this repository.
- `p95 < 1 s` search latency on a ~50-skill corpus.
- No cloud API calls in indexing or query paths.
