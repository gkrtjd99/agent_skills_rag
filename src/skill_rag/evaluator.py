from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import retrieve


@dataclass(slots=True)
class Case:
    query: str
    expected: list[str]


@dataclass(slots=True)
class Report:
    n: int
    k: int
    recall_at_k: float
    mrr: float
    p50_ms: float
    p95_ms: float
    misses: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "k": self.k,
            "recall_at_k": self.recall_at_k,
            "mrr": self.mrr,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "misses": self.misses,
        }


def load_cases(path: Path) -> list[Case]:
    cases: list[Case] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        expected = obj["expected"]
        if isinstance(expected, str):
            expected = [expected]
        cases.append(Case(query=str(obj["query"]), expected=list(expected)))
    return cases


SearchFn = Callable[[str, int], dict]


def evaluate(cases: list[Case], k: int = 5, search_fn: SearchFn | None = None) -> Report:
    if not cases:
        return Report(n=0, k=k, recall_at_k=0.0, mrr=0.0, p50_ms=0.0, p95_ms=0.0)
    if search_fn is None:
        search_fn = retrieve.search

    hit_count = 0
    rr_sum = 0.0
    latencies: list[float] = []
    misses: list[dict] = []

    for case in cases:
        t0 = time.monotonic()
        res = search_fn(case.query, k)
        latencies.append((time.monotonic() - t0) * 1000.0)

        names = [h["name"] for h in res.get("hits", [])]
        found = [n for n in case.expected if n in names]
        if found:
            hit_count += 1
            best_rank = min(names.index(n) + 1 for n in found)
            rr_sum += 1.0 / best_rank
        else:
            misses.append({"query": case.query, "expected": case.expected, "got": names})

    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[max(0, int(len(latencies) * 0.95) - 1)]
    return Report(
        n=len(cases),
        k=k,
        recall_at_k=hit_count / len(cases),
        mrr=rr_sum / len(cases),
        p50_ms=p50,
        p95_ms=p95,
        misses=misses,
    )
