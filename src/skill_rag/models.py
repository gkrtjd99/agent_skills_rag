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
        # Stable string we embed AND index for lexical (BM25) search.
        # Body is included because it carries the trigger phrases and
        # examples that the one-line description omits. The embedding model
        # truncates to its max sequence length, so this mainly adds the
        # body's intro to the dense vector while giving BM25 the full text.
        # Changing this requires a reindex.
        parts = [self.name, self.description]
        if self.body.strip():
            parts.append(self.body.strip())
        return "\n".join(parts)


@dataclass(slots=True)
class SearchHit:
    name: str
    description: str
    score: float
