from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from .models import SkillRecord

_FRONTMATTER_DELIM = "---"


def parse_skill_file(path: Path) -> SkillRecord | None:
    """Parse one SKILL.md. Returns None on any parse failure or missing fields.

    content_hash is sha256 of the FULL file text so any change to the body or
    frontmatter triggers re-indexing.
    """
    text = path.read_text(encoding="utf-8")
    fm_text, body = _split_frontmatter(text)
    if fm_text is None:
        return None

    try:
        data = yaml.safe_load(fm_text) or {}
    except yaml.YAMLError:
        return None
    if not isinstance(data, dict):
        return None

    name = str(data.get("name") or "").strip()
    description = str(data.get("description") or "").strip()
    if not name or not description:
        return None

    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return SkillRecord(
        name=name,
        description=description,
        path=str(path),
        body=body,
        content_hash=content_hash,
    )


def _split_frontmatter(text: str) -> tuple[str | None, str]:
    if not text.startswith(_FRONTMATTER_DELIM):
        return None, text
    rest = text[len(_FRONTMATTER_DELIM):].lstrip("\n")
    end = rest.find(f"\n{_FRONTMATTER_DELIM}")
    if end == -1:
        return None, text
    fm = rest[:end]
    body = rest[end + len(_FRONTMATTER_DELIM) + 1:].lstrip("\n")
    return fm, body
