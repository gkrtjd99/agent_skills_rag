# Core Beliefs

Non-negotiable principles. Changes require explicit user buy-in.

## Local-First
No cloud APIs at index or query time. The user owns their machine, their
embeddings, and their corpus. Latency is bounded by local hardware, not
network round trips. Runtime model loading defaults to local cache only; any
model download must be an explicit setup step, not an implicit query/index
side effect. Tested via the "no cloud" Done-When criterion.

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
