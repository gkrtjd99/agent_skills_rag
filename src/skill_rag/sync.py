"""Reconcile the LanceDB index against the corpus directory.

Only this module holds mutable state (a single timestamp).
"""

from __future__ import annotations

import time

from . import corpus as corpus_mod
from . import index as index_mod
from . import loader

_last_sync_at: float | None = None


def run_sync() -> dict:
    """Force a sync. Returns {added, updated, removed, unchanged}."""
    global _last_sync_at
    records = loader.scan(corpus_mod.CORPUS_PATH)
    indexed = {row["path"]: row["content_hash"] for row in index_mod.list_indexed()}
    disk_paths = {r.path for r in records}

    added: list[str] = []
    updated: list[str] = []
    to_upsert = []
    unchanged = 0
    for r in records:
        prev_hash = indexed.get(r.path)
        if prev_hash is None:
            added.append(r.name)
            to_upsert.append(r)
        elif prev_hash != r.content_hash:
            updated.append(r.name)
            to_upsert.append(r)
        else:
            unchanged += 1

    removed_paths = [p for p in indexed if p not in disk_paths]
    # Recover names by querying the indexed rows we just listed.
    removed_names: list[str] = []
    if removed_paths:
        for row in index_mod.list_indexed():
            if row["path"] in removed_paths:
                removed_names.append(row["name"])

    if to_upsert:
        index_mod.upsert(to_upsert)
    if removed_paths:
        index_mod.delete_by_paths(removed_paths)

    _last_sync_at = time.monotonic()
    return {
        "added": added,
        "updated": updated,
        "removed": removed_names,
        "unchanged": unchanged,
    }


def sync_if_stale(ttl: float | None = None) -> None:
    """Run sync only if the last run was longer than ``ttl`` seconds ago.

    Pass ``ttl=0`` to force a sync regardless of cache age.
    """
    global _last_sync_at
    if ttl is None:
        ttl = corpus_mod.SYNC_TTL_SECONDS
    now = time.monotonic()
    if ttl > 0 and _last_sync_at is not None and (now - _last_sync_at) < ttl:
        return
    run_sync()


def reset_cache() -> None:
    """Clear the TTL timestamp. Next sync_if_stale call will run."""
    global _last_sync_at
    _last_sync_at = None
