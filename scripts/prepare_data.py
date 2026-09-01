"""Turn the extracted vendor catalogue into the project's training corpus.

Prices are deliberately dropped: the catalogue is a supplier price list, and the
public dataset only needs names, specs and units for the matching task.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vnmatch.catalog import Product, build_database, read_jsonl, write_jsonl  # noqa: E402

DEFAULT_SOURCE = Path.home() / "ton_catalogue_work" / "catalogue_rows.json"


def load_source(path: Path) -> list[Product]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    products: list[Product] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        name = str(row.get("Tên sản phẩm", "")).strip()
        spec = str(row.get("Quy cách/dòng tiếp theo", "")).strip()
        if not name:
            continue
        product = Product.build(
            sku_id=f"SKU{index:05d}",
            name=name,
            spec=spec,
            unit=str(row.get("ĐVT", "")),
            page=row.get("Trang catalogue"),
        )
        # The source PDF repeats a handful of rows across page breaks.
        if product.canonical in seen:
            continue
        seen.add(product.canonical)
        products.append(product)
    return products


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                        help="raw extracted catalogue (not committed)")
    parser.add_argument("--from-jsonl", type=Path, default=None,
                        help="rebuild the SQLite index from the committed "
                             "data/catalog.jsonl instead of the raw source")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "catalog.jsonl")
    parser.add_argument("--db", type=Path, default=ROOT / "data" / "catalog.db")
    args = parser.parse_args()

    if args.from_jsonl is not None:
        # CI and fresh clones take this path: the raw vendor PDF extract is
        # not in the repository, only the derived price-free catalogue is.
        products = read_jsonl(args.from_jsonl)
    else:
        if not args.source.exists():
            parser.error(f"catalogue not found: {args.source}")
        products = load_source(args.source)
        write_jsonl(products, args.out)
    build_database(products, args.db).close()

    with_spec = sum(1 for p in products if p.spec)
    with_dims = sum(1 for p in products if p.dimensions)
    print(f"{len(products)} SKU -> {args.out}")
    print(f"  co quy cach : {with_spec}")
    print(f"  co kich thuoc: {with_dims}")
    print(f"  co vat lieu  : {sum(1 for p in products if p.material)}")
    print(f"SQLite       -> {args.db}")


if __name__ == "__main__":
    main()
