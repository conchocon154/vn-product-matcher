"""Retrieval metrics and the evaluation loop shared by every experiment."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

K_VALUES = (1, 3, 5, 10)


@dataclass(frozen=True)
class EvalQuery:
    text: str
    sku_id: str
    corruptions: tuple[str, ...]


def load_queries(path: Path) -> list[EvalQuery]:
    queries = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                queries.append(
                    EvalQuery(row["text"], row["sku_id"], tuple(row.get("corruptions", ())))
                )
    return queries


@dataclass
class Metrics:
    retriever: str
    dataset: str
    count: int
    recall: dict[int, float]
    mrr: float

    def row(self) -> str:
        cells = " | ".join(f"{self.recall[k]:6.1%}" for k in K_VALUES)
        return f"{self.retriever:16} | {cells} | {self.mrr:6.3f}"


def evaluate(retriever, queries: Sequence[EvalQuery], dataset: str, k: int = 10) -> Metrics:
    hits = {value: 0 for value in K_VALUES}
    reciprocal_total = 0.0

    # Batched retrievers encode all queries at once; the rest go one by one.
    if hasattr(retriever, "search_many"):
        all_results = retriever.search_many([q.text for q in queries], k)
    else:
        all_results = [retriever.search(q.text, k) for q in queries]

    for query, results in zip(queries, all_results):
        ranked = [sku_id for sku_id, _ in results]
        if query.sku_id in ranked:
            rank = ranked.index(query.sku_id) + 1
            reciprocal_total += 1.0 / rank
            for value in K_VALUES:
                if rank <= value:
                    hits[value] += 1

    total = max(len(queries), 1)
    return Metrics(
        retriever=getattr(retriever, "name", type(retriever).__name__),
        dataset=dataset,
        count=len(queries),
        recall={value: hits[value] / total for value in K_VALUES},
        mrr=reciprocal_total / total,
    )


def header() -> str:
    cells = " | ".join(f"R@{k:<4}" for k in K_VALUES)
    line = f"{'retriever':16} | {cells} | {'MRR':>6}"
    return line + "\n" + "-" * len(line)
