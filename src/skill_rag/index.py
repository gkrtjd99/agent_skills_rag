"""LanceDB-backed index for skill records.

Schema version: 5
- v1: pk=name
- v2: pk=path; added `source` column
- v3: pk=path; removed `source`, `allowed_tools` (single-corpus design)
- v4: added `text` column (name+description+body) for BM25 lexical search
- v5: added `agent` column (source harness: claude-code, codex, local, ...)
"""

from __future__ import annotations

import os
from pathlib import Path

import lancedb
import pyarrow as pa

from .embed import DEFAULT_MODEL, encode, model_dim
from .models import SearchHit, SkillRecord

TABLE_NAME = "skills"


def index_path() -> Path:
    return Path(os.environ.get("SKILL_RAG_INDEX_PATH", "./var/index.lance")).expanduser()


def _schema(dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("path", pa.string()),
            pa.field("name", pa.string()),
            pa.field("description", pa.string()),
            pa.field("content_hash", pa.string()),
            pa.field("text", pa.string()),
            pa.field("agent", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), dim)),
        ]
    )


def _open_db():
    path = index_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    return lancedb.connect(str(path))


def open_table(model_name: str = DEFAULT_MODEL):
    db = _open_db()
    schema = _schema(model_dim(model_name))
    if TABLE_NAME in db.list_tables().tables:
        tbl = db.open_table(TABLE_NAME)
        # Drop and recreate on any schema drift (e.g. a pre-`text` v3 table).
        # The index is a derived cache — `sync` rebuilds it from the corpus.
        if list(tbl.schema.names) != list(schema.names):
            db.drop_table(TABLE_NAME)
            return db.create_table(TABLE_NAME, schema=schema)
        return tbl
    return db.create_table(TABLE_NAME, schema=schema)


def list_indexed() -> list[dict]:
    tbl = open_table()
    if tbl.count_rows() == 0:
        return []
    cols = ["path", "name", "description", "content_hash", "text", "agent"]
    return tbl.to_arrow().select(cols).to_pylist()


def upsert(records: list[SkillRecord], model_name: str = DEFAULT_MODEL) -> None:
    if not records:
        return
    tbl = open_table(model_name)
    texts = [r.embed_text() for r in records]
    vectors = encode(texts, name=model_name)
    rows = [
        {
            "path": r.path,
            "name": r.name,
            "description": r.description,
            "content_hash": r.content_hash,
            "text": text,
            "agent": r.agent,
            "vector": vec.tolist(),
        }
        for r, text, vec in zip(records, texts, vectors)
    ]
    (
        tbl.merge_insert("path")
        .when_matched_update_all()
        .when_not_matched_insert_all()
        .execute(rows)
    )


def delete_by_paths(paths: list[str]) -> int:
    if not paths:
        return 0
    tbl = open_table()
    quoted = ", ".join("'" + p.replace("'", "''") + "'" for p in paths)
    tbl.delete(f"path IN ({quoted})")
    return len(paths)


def reset() -> None:
    db = _open_db()
    if TABLE_NAME in db.list_tables().tables:
        db.drop_table(TABLE_NAME)


def search(query_vector, k: int = 5) -> list[SearchHit]:
    tbl = open_table()
    if tbl.count_rows() == 0:
        return []
    # Guard the common operational footgun: the index was rebuilt with a
    # different embedding model (different dim) while a long-lived server still
    # holds the old model. Without this, lance raises an opaque internal error.
    col = tbl.schema.field("vector").type
    index_dim = getattr(col, "list_size", None)
    query_dim = len(query_vector)
    if index_dim is not None and query_dim != index_dim:
        raise ValueError(
            f"Query vector dimension ({query_dim}) does not match the index "
            f"({index_dim}). The index was built with a different embedding "
            f"model. Restart the MCP server (or run `skill-rag reset && sync`) "
            f"so the query model and the index agree."
        )
    rows = tbl.search(query_vector).metric("cosine").limit(k).to_list()
    hits: list[SearchHit] = []
    for row in rows:
        distance = float(row.get("_distance", 0.0))
        score = 1.0 - distance
        hits.append(
            SearchHit(
                name=row["name"],
                description=row["description"],
                score=score,
            )
        )
    return hits
