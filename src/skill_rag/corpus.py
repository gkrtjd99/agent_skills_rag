from __future__ import annotations

import os
from pathlib import Path

CORPUS_PATH = Path(os.environ.get("SKILL_RAG_CORPUS_PATH", "~/.skills")).expanduser()
BOOTSTRAP_SKILL_NAME = "using-skill-rag"
SCORE_THRESHOLD = float(os.environ.get("SKILL_RAG_SCORE_THRESHOLD", "0.25"))
SYNC_TTL_SECONDS = float(os.environ.get("SKILL_RAG_SYNC_TTL", "30"))
