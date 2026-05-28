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

`SCORE_THRESHOLD` (default `0.25`, env `SKILL_RAG_SCORE_THRESHOLD`) filters
out low-similarity matches in `search_skills`. Calibrated against
`eval/queries.jsonl`: 0.25 is the highest value that keeps `recall@5 ≥ 0.8`
(measured recall@5 = 0.864). Re-tune when the eval set or embedding model
changes.

## Loop Prevention Contract

A conforming bootstrap skill must:

- On `search_skills → no_match`: respond directly, not re-call with a
  reworded query.
- On `get_skill → not_found`: not re-call `get_skill` or `search_skills`
  for the same name this turn.
- On `search_skills → ok` where no description actually fits: proceed
  without a skill, not refetch.
