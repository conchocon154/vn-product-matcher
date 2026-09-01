"""Score every retriever on both test sets and write the report table."""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vnmatch.catalog import read_jsonl  # noqa: E402
from vnmatch.evaluate import K_VALUES, evaluate, header, load_queries  # noqa: E402
from vnmatch.retrievers import (  # noqa: E402
    Bm25Match,
    DenseMatch,
    ExactMatch,
    FuzzyMatch,
    HybridMatch,
    TfidfMatch,
)

BASE_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def build_retrievers(products, db_path: Path, model_path: Path, include_base: bool):
    retrievers = [
        ExactMatch(products),
        Bm25Match(db_path),
        FuzzyMatch(products),
        TfidfMatch(products),
    ]
    if include_base:
        retrievers.append(DenseMatch(products, BASE_MODEL, name="dense_base"))
    if model_path.exists():
        dense = DenseMatch(products, str(model_path), name="dense_finetuned")
        retrievers.append(dense)
        retrievers.append(HybridMatch(dense, products))
    return retrievers


def breakdown(retriever, queries) -> dict[str, tuple[int, float]]:
    """R@1 split by which corruption the query carries."""
    totals: collections.Counter = collections.Counter()
    hits: collections.Counter = collections.Counter()
    if hasattr(retriever, "search_many"):
        all_results = retriever.search_many([q.text for q in queries], 1)
    else:
        all_results = [retriever.search(q.text, 1) for q in queries]
    for query, results in zip(queries, all_results):
        correct = bool(results) and results[0][0] == query.sku_id
        for corruption in query.corruptions:
            totals[corruption] += 1
            hits[corruption] += correct
    return {name: (count, hits[name] / count) for name, count in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "vnmatch-minilm")
    parser.add_argument("--include-base", action="store_true",
                        help="also score the untuned encoder (downloads it)")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "results.json")
    args = parser.parse_args()

    products = read_jsonl(ROOT / "data" / "catalog.jsonl")
    retrievers = build_retrievers(
        products, ROOT / "data" / "catalog.db", args.model, args.include_base
    )

    report: dict = {"catalogue_size": len(products), "datasets": {}}
    for split in ("test_seen", "test_unseen"):
        queries = load_queries(ROOT / "data" / f"queries_{split}.jsonl")
        print(f"\n=== {split}  ({len(queries)} queries, {len(products)} SKU index) ===")
        print(header())
        rows = []
        for retriever in retrievers:
            started = time.perf_counter()
            metrics = evaluate(retriever, queries, split)
            elapsed = time.perf_counter() - started
            print(f"{metrics.row()}   {elapsed * 1000 / len(queries):5.1f} ms/q")
            rows.append(
                {
                    "retriever": metrics.retriever,
                    "recall": {str(k): metrics.recall[k] for k in K_VALUES},
                    "mrr": metrics.mrr,
                    "ms_per_query": elapsed * 1000 / len(queries),
                }
            )
        report["datasets"][split] = {"count": len(queries), "results": rows}

    # Where each approach still fails, on the harder of the two sets.
    queries = load_queries(ROOT / "data" / "queries_test_unseen.jsonl")
    print("\n=== R@1 by corruption (test_unseen) ===")
    names = [getattr(r, "name", "?") for r in retrievers if getattr(r, "name", "") in
             {"tfidf_char", "dense_finetuned", "hybrid"}]
    tables = {
        getattr(r, "name"): breakdown(r, queries)
        for r in retrievers
        if getattr(r, "name", "") in {"tfidf_char", "dense_finetuned", "hybrid"}
    }
    if tables:
        any_table = next(iter(tables.values()))
        head = f"{'corruption':16} {'n':>5} " + " ".join(f"{n:>16}" for n in names)
        print(head)
        print("-" * len(head))
        for corruption, (count, _) in sorted(any_table.items(), key=lambda kv: -kv[1][0]):
            cells = " ".join(f"{tables[n][corruption][1]:15.1%}" for n in names)
            print(f"{corruption:16} {count:5}  {cells}")
        report["breakdown_test_unseen"] = {
            name: {k: {"n": v[0], "r@1": v[1]} for k, v in table.items()}
            for name, table in tables.items()
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nreport -> {args.out}")


if __name__ == "__main__":
    main()
