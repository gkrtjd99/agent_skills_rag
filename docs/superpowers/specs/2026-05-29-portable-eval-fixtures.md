# Portable Eval Fixtures

**Date**: 2026-05-29
**Status**: Implemented

## Problem

The original eval dataset assumed a particular user's `~/.skills/` contents.
That made `recall@5` useful as a local diagnostic but not as a repository
benchmark for GitHub users.

## Decision

Ship a small public fixture corpus under `eval/fixtures/skills/` and make
`skill-rag eval` default to that corpus plus `eval/fixtures/queries.jsonl`.
The CLI evaluates in a temporary LanceDB index and calls the MCP
`search_skills` path, so it covers sync and retrieval behavior without
polluting the user's local index.

Users can still evaluate their personal corpus by passing:

```bash
uv run skill-rag eval --corpus ~/.skills --dataset eval/queries.jsonl
```

## Local-Only Runtime

Embedding model loading now defaults to `local_files_only=True` via
`SKILL_RAG_LOCAL_FILES_ONLY=1`. This prevents hidden network calls during
indexing, querying, and eval. A first-time model download should be an
explicit setup action.
