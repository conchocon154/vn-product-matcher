"""Catalogue storage: JSONL on disk, SQLite for lookup and serving."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path

from .normalize import extract_dimensions, extract_material, normalize, normalize_unit

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS products (
    sku_id      TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    spec        TEXT NOT NULL DEFAULT '',
    unit        TEXT NOT NULL DEFAULT '',
    full_name   TEXT NOT NULL,
    canonical   TEXT NOT NULL,
    material    TEXT,
    dimensions  TEXT NOT NULL DEFAULT '',
    page        INTEGER
);
CREATE INDEX IF NOT EXISTS idx_products_canonical ON products (canonical);
CREATE INDEX IF NOT EXISTS idx_products_material  ON products (material);

-- Full-text index over the canonical form, used by the lexical retrieval stage.
CREATE VIRTUAL TABLE IF NOT EXISTS products_fts USING fts5 (
    canonical,
    content='products',
    content_rowid='rowid'
);
"""


@dataclass(frozen=True)
class Product:
    sku_id: str
    name: str
    spec: str
    unit: str
    full_name: str
    canonical: str
    material: str | None
    dimensions: str
    page: int | None

    @classmethod
    def build(cls, sku_id: str, name: str, spec: str, unit: str, page: int | None) -> "Product":
        full_name = f"{name} {spec}".strip()
        return cls(
            sku_id=sku_id,
            name=name.strip(),
            spec=spec.strip(),
            unit=normalize_unit(unit),
            full_name=full_name,
            canonical=normalize(full_name),
            material=extract_material(full_name),
            dimensions=" ".join(extract_dimensions(full_name)),
            page=page,
        )


def write_jsonl(products: list[Product], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for product in products:
            handle.write(json.dumps(asdict(product), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[Product]:
    with path.open(encoding="utf-8") as handle:
        return [Product(**json.loads(line)) for line in handle if line.strip()]


def build_database(products: list[Product], db_path: Path) -> sqlite3.Connection:
    """Create a fresh SQLite catalogue, including its FTS5 mirror."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    connection = sqlite3.connect(db_path)
    connection.executescript(SCHEMA_SQL)
    connection.executemany(
        """INSERT INTO products
           (sku_id, name, spec, unit, full_name, canonical, material, dimensions, page)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (p.sku_id, p.name, p.spec, p.unit, p.full_name, p.canonical,
             p.material, p.dimensions, p.page)
            for p in products
        ],
    )
    # Populate the FTS mirror from the base table it shadows.
    connection.execute(
        "INSERT INTO products_fts (rowid, canonical) SELECT rowid, canonical FROM products"
    )
    connection.commit()
    return connection
