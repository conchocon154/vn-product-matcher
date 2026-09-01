"""Split the catalogue into train/validation/test query sets.

Two test sets are produced on purpose:

* `test_seen`   - held-out queries for SKUs whose other queries were trained on.
* `test_unseen` - queries for SKUs the model never saw during training.

Retrieval always runs against the *whole* catalogue, so `test_unseen` measures
whether the model learned Vietnamese hardware naming or merely memorised the
1 827 SKUs it was fitted on.  Reporting only the first number would flatter the
model, which is exactly the mistake this split exists to prevent.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vnmatch.augment import generate  # noqa: E402
from vnmatch.catalog import read_jsonl  # noqa: E402


def write(queries, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for query in queries:
            handle.write(json.dumps(asdict(query), ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=ROOT / "data" / "catalog.jsonl")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--per-product", type=int, default=6)
    parser.add_argument("--unseen-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    products = read_jsonl(args.catalog)
    rng = random.Random(args.seed)

    shuffled = products[:]
    rng.shuffle(shuffled)
    cut = int(len(shuffled) * args.unseen_fraction)
    unseen_products, seen_products = shuffled[:cut], shuffled[cut:]

    # The ambiguity check must see the whole catalogue: a query is only truly
    # unique if no SKU outside the current subset also matches it.
    seen_queries = generate(seen_products, args.per_product, seed=args.seed,
                            catalogue=products)
    unseen_queries = generate(unseen_products, args.per_product, seed=args.seed + 1,
                              catalogue=products)

    # Aggressive shortening can produce a query that fits several SKUs equally
    # well ("son alkyd" names dozens of tins).  Such a query has no single
    # correct answer, so scoring against one gold SKU would punish a retriever
    # for being right.  Drop every query text claimed by more than one SKU.
    owners: dict[str, set[str]] = {}
    for query in seen_queries + unseen_queries:
        owners.setdefault(query.text, set()).add(query.sku_id)
    ambiguous = {text for text, skus in owners.items() if len(skus) > 1}
    before = len(seen_queries) + len(unseen_queries)
    seen_queries = [q for q in seen_queries if q.text not in ambiguous]
    unseen_queries = [q for q in unseen_queries if q.text not in ambiguous]
    dropped = before - len(seen_queries) - len(unseen_queries)
    print(f'Loai {dropped} truy van mo ho ({len(ambiguous)} chuoi trung nhieu SKU)\n')

    # Hold out queries per SKU rather than globally, so every seen SKU keeps at
    # least one training example and contributes one test example.
    by_sku: dict[str, list] = {}
    for query in seen_queries:
        by_sku.setdefault(query.sku_id, []).append(query)

    train, validation, test_seen = [], [], []
    for sku_queries in by_sku.values():
        rng.shuffle(sku_queries)
        if len(sku_queries) >= 3:
            test_seen.append(sku_queries[0])
            validation.append(sku_queries[1])
            train.extend(sku_queries[2:])
        else:
            train.extend(sku_queries)

    for name, rows in (
        ("train", train),
        ("validation", validation),
        ("test_seen", test_seen),
        ("test_unseen", unseen_queries),
    ):
        write(rows, args.out_dir / f"queries_{name}.jsonl")
        print(f"{name:12} {len(rows):6} truy van")

    print(f"\nSKU da hoc    : {len(seen_products)}")
    print(f"SKU chua tung thay: {len(unseen_products)}")
    print(f"Chi muc tim kiem  : {len(products)} SKU (toan bo catalogue)")


if __name__ == "__main__":
    main()
