"""Guard the lexical retriever against regressions.

CI cannot afford to download the encoder, but it can check that the data
pipeline and the character TF-IDF baseline still work end to end.  A drop here
means normalisation or the query generator broke, not that a model got worse.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vnmatch.catalog import read_jsonl  # noqa: E402
from vnmatch.evaluate import evaluate, load_queries  # noqa: E402
from vnmatch.retrievers import TfidfMatch  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-recall1", type=float, default=0.90)
    parser.add_argument("--split", default="test_seen")
    args = parser.parse_args()

    products = read_jsonl(ROOT / "data" / "catalog.jsonl")
    queries = load_queries(ROOT / "data" / f"queries_{args.split}.jsonl")
    metrics = evaluate(TfidfMatch(products), queries, args.split)

    recall1 = metrics.recall[1]
    print(f"tfidf_char R@1 = {recall1:.1%} on {args.split} ({len(queries)} queries)")
    if recall1 < args.min_recall1:
        print(f"FAIL: below the {args.min_recall1:.0%} floor")
        raise SystemExit(1)
    print("OK")


if __name__ == "__main__":
    main()
