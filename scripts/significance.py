"""Is the gap between two retrievers real, or within sampling noise?

Two retrievers scored on the same queries give paired observations, so a paired
test is the right one.  McNemar looks only at the queries where the two differ,
which is exactly the evidence that distinguishes them; the bootstrap gives a
confidence interval for the size of the gap.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vnmatch.catalog import read_jsonl  # noqa: E402
from vnmatch.evaluate import load_queries  # noqa: E402
from vnmatch.retrievers import DenseMatch, HybridMatch, TfidfMatch  # noqa: E402


def correctness(retriever, queries) -> list[bool]:
    if hasattr(retriever, "search_many"):
        results = retriever.search_many([q.text for q in queries], 1)
    else:
        results = [retriever.search(q.text, 1) for q in queries]
    return [bool(r) and r[0][0] == q.sku_id for q, r in zip(queries, results)]


def mcnemar(a: list[bool], b: list[bool]) -> tuple[int, int, float]:
    """Exact binomial McNemar test on the discordant pairs."""
    from math import comb

    a_only = sum(1 for x, y in zip(a, b) if x and not y)
    b_only = sum(1 for x, y in zip(a, b) if y and not x)
    n = a_only + b_only
    if n == 0:
        return 0, 0, 1.0
    k = min(a_only, b_only)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return a_only, b_only, min(1.0, 2 * tail)


def bootstrap_gap(a: list[bool], b: list[bool], rounds: int, seed: int):
    rng = random.Random(seed)
    n = len(a)
    gaps = []
    for _ in range(rounds):
        indices = [rng.randrange(n) for _ in range(n)]
        gaps.append(sum(b[i] for i in indices) / n - sum(a[i] for i in indices) / n)
    gaps.sort()
    return gaps[int(0.025 * rounds)], gaps[int(0.975 * rounds)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test_unseen")
    parser.add_argument("--rounds", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    products = read_jsonl(ROOT / "data" / "catalog.jsonl")
    queries = load_queries(ROOT / "data" / f"queries_{args.split}.jsonl")
    dense = DenseMatch(products, str(ROOT / "models" / "vnmatch-minilm"), name="dense_finetuned")

    scores = {
        "tfidf_char": correctness(TfidfMatch(products), queries),
        "dense_finetuned": correctness(dense, queries),
        "hybrid": correctness(HybridMatch(dense, products), queries),
    }

    print(f"{args.split}: {len(queries)} queries\n")
    for baseline, contender in (("tfidf_char", "dense_finetuned"),
                                ("dense_finetuned", "hybrid")):
        a, b = scores[baseline], scores[contender]
        a_only, b_only, p_value = mcnemar(a, b)
        low, high = bootstrap_gap(a, b, args.rounds, args.seed)
        print(f"{contender} vs {baseline}")
        print(f"  R@1            : {sum(a)/len(a):.1%} -> {sum(b)/len(b):.1%}")
        print(f"  only {baseline:<15} right: {a_only}")
        print(f"  only {contender:<15} right: {b_only}")
        print(f"  McNemar p      : {p_value:.2e}")
        print(f"  95% CI of gap  : [{low:+.2%}, {high:+.2%}]")
        verdict = "significant" if p_value < 0.05 else "NOT significant"
        print(f"  verdict        : {verdict} at alpha=0.05\n")


if __name__ == "__main__":
    main()
