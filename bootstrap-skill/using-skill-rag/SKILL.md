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
