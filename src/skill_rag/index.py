"""LanceDB-backed index for skill records.

Schema version: 3
- v1: pk=name
- v2: pk=path; added `source` column
- v3: pk=path; removed `source`, `allowed_tools` (single-corpus design)
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
            pa.field("vector", pa.list_(pa.float32(), dim)),
        ]
    )


def _open_db():
    path = index_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    return lancedb.connect(str(path))


def open_table(model_name: str = DEFAULT_MODEL):
    db = _open_db()
    if TABLE_NAME in db.table_names():
        return db.open_table(TABLE_NAME)
    return db.create_table(TABLE_NAME, schema=_schema(model_dim(model_name)))


def list_indexed() -> list[dict]:
    tbl = open_table()
    if tbl.count_rows() == 0:
        return []
    cols = ["path", "name", "description", "content_hash"]
    return tbl.to_arrow().select(cols).to_pylist()


def upsert(records: list[SkillRecord], model_name: str = DEFAULT_MODEL) -> None:
    if not records:
        return
    tbl = open_table(model_name)
    vectors = encode([r.embed_text() for r in records], name=model_name)
    rows = [
        {
            "path": r.path,
            "name": r.name,
            "description": r.description,
            "content_hash": r.content_hash,
            "vector": vec.tolist(),
        }
        for r, vec in zip(records, vectors)
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
    if TABLE_NAME in db.table_names():
        db.drop_table(TABLE_NAME)


def search(query_vector, k: int = 5) -> list[SearchHit]:
    tbl = open_table()
    if tbl.count_rows() == 0:
        return []
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
