# Portable Eval Fixtures — Implementation Plan

## Goal

Make repository eval portable across machines while preserving a way to test a
user's personal `~/.skills/` corpus.

## Steps

- [x] Add `eval/fixtures/skills/` with representative public `SKILL.md`
  examples.
- [x] Add `eval/fixtures/queries.jsonl` with expected skill names.
- [x] Update evaluator to accept an injectable search function.
- [x] Update `skill-rag eval` to default to fixture corpus and temporary index.
- [x] Route CLI eval through MCP `search_skills` so sync behavior is covered.
- [x] Document how to run personal corpus eval explicitly.
- [x] Default embedding model loading to local cache only at runtime.
