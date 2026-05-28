from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

DEFAULT_MODEL = os.environ.get("SKILL_RAG_MODEL", "all-MiniLM-L6-v2")
LOCAL_FILES_ONLY = os.environ.get("SKILL_RAG_LOCAL_FILES_ONLY", "1").lower() not in {
    "0",
    "false",
    "no",
}


@lru_cache(maxsize=4)
def _load_model(name: str):
    # Imported lazily so importing this module is cheap (helps tests).
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name, local_files_only=LOCAL_FILES_ONLY)


def model_dim(name: str = DEFAULT_MODEL) -> int:
    return int(_load_model(name).get_sentence_embedding_dimension())


def encode(texts: list[str], name: str = DEFAULT_MODEL) -> np.ndarray:
    """Encode a batch of texts. Always L2-normalized for cosine search."""
    if not texts:
        return np.zeros((0, model_dim(name)), dtype=np.float32)
    model = _load_model(name)
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vectors.astype(np.float32)


def encode_one(text: str, name: str = DEFAULT_MODEL) -> np.ndarray:
    return encode([text], name=name)[0]
