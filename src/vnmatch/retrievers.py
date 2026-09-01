"""Retrieval strategies, from cheap lexical baselines to the tuned encoder.

Every retriever exposes the same `search(query, k)` contract so that
`evaluate.py` can score them side by side without special cases.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from .catalog import Product
from .normalize import extract_dimensions, extract_material, normalize


class Retriever(Protocol):
    name: str

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Return up to `k` (sku_id, score) pairs, best score first."""


class ExactMatch:
    """Lower bound: the canonical strings must be identical."""

    name = "exact"

    def __init__(self, products: Sequence[Product]) -> None:
        self._index: dict[str, str] = {}
        for product in products:
            self._index.setdefault(product.canonical, product.sku_id)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        hit = self._index.get(normalize(query))
        return [(hit, 1.0)] if hit else []


class FuzzyMatch:
    """Edit-distance matching - what a shop app would ship without any ML."""

    name = "rapidfuzz"

    def __init__(self, products: Sequence[Product]) -> None:
        self._sku_ids = [p.sku_id for p in products]
        self._canonicals = [p.canonical for p in products]

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        from rapidfuzz import fuzz, process

        matches = process.extract(
            normalize(query), self._canonicals, scorer=fuzz.token_set_ratio, limit=k
        )
        return [(self._sku_ids[index], score / 100.0) for _, score, index in matches]


class TfidfMatch:
    """Character n-gram TF-IDF: robust to typos, blind to word meaning."""

    name = "tfidf_char"

    def __init__(self, products: Sequence[Product]) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._sku_ids = [p.sku_id for p in products]
        self._vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
        self._matrix = self._vectorizer.fit_transform(p.canonical for p in products)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        vector = self._vectorizer.transform([normalize(query)])
        scores = (self._matrix @ vector.T).toarray().ravel()
        top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        top = top[np.argsort(-scores[top])]
        return [(self._sku_ids[i], float(scores[i])) for i in top if scores[i] > 0]


class Bm25Match:
    """BM25 through SQLite FTS5 - the ranking a plain SQL deployment gets free."""

    name = "bm25_fts5"

    def __init__(self, db_path: Path) -> None:
        self._connection = sqlite3.connect(str(db_path), check_same_thread=False)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        tokens = [t for t in normalize(query).split() if t]
        if not tokens:
            return []
        # Quote each token so digits and `x` separators cannot be parsed as
        # FTS5 operators, and OR them so a partial query still retrieves.
        expression = " OR ".join(f'"{token}"' for token in tokens)
        try:
            rows = self._connection.execute(
                """SELECT p.sku_id, -bm25(products_fts) AS score
                     FROM products_fts
                     JOIN products p ON p.rowid = products_fts.rowid
                    WHERE products_fts MATCH ?
                 ORDER BY score DESC
                    LIMIT ?""",
                (expression, k),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [(sku_id, float(score)) for sku_id, score in rows]


class DenseMatch:
    """Sentence-transformer embeddings, either off-the-shelf or fine-tuned."""

    def __init__(self, products: Sequence[Product], model_path: str, name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.name = name
        self._sku_ids = [p.sku_id for p in products]
        self._model = SentenceTransformer(model_path)
        self._embeddings = self._model.encode(
            [p.canonical for p in products],
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=256,
            show_progress_bar=False,
        )

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return self._model.encode(
            [normalize(t) for t in texts],
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=256,
            show_progress_bar=False,
        )

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        return self.search_many([query], k)[0]

    def search_many(self, queries: Sequence[str], k: int) -> list[list[tuple[str, float]]]:
        """Batched search - encoding one query at a time wastes most of the GPU."""
        vectors = self.encode(queries)
        scores = vectors @ self._embeddings.T
        results = []
        for row in scores:
            top = np.argpartition(-row, min(k, len(row) - 1))[:k]
            top = top[np.argsort(-row[top])]
            results.append([(self._sku_ids[i], float(row[i])) for i in top])
        return results


class HybridMatch:
    """Dense retrieval reranked by the signals encoders handle badly.

    Embedding models compress `10x50x17` and `10x60x17` to nearly the same
    vector, yet in a hardware catalogue those are different parts at different
    prices.  This reranker keeps the encoder for the descriptive words and hands
    the dimensions and material grade to exact comparison.
    """

    name = "hybrid"

    def __init__(
        self,
        dense: DenseMatch,
        products: Sequence[Product],
        *,
        dimension_weight: float = 0.35,
        material_weight: float = 0.15,
        candidates: int = 50,
    ) -> None:
        self._dense = dense
        self._by_sku = {p.sku_id: p for p in products}
        self._dimension_weight = dimension_weight
        self._material_weight = material_weight
        self._candidates = candidates

    def _bonus(self, query: str, product: Product) -> float:
        score = 0.0
        query_dimensions = set(extract_dimensions(query))
        product_dimensions = set(product.dimensions.split()) if product.dimensions else set()
        if query_dimensions and product_dimensions:
            overlap = query_dimensions & product_dimensions
            if overlap:
                score += self._dimension_weight
            else:
                # A partial run: staff typed 10x50, the SKU is 10x50x17.
                prefixes = {d for d in product_dimensions
                            if any(d.startswith(q) or q.startswith(d) for q in query_dimensions)}
                if prefixes:
                    score += self._dimension_weight * 0.6
                else:
                    score -= self._dimension_weight
        query_material = extract_material(query)
        if query_material and product.material:
            score += self._material_weight if query_material == product.material else -self._material_weight
        return score

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        return self.search_many([query], k)[0]

    def search_many(self, queries: Sequence[str], k: int) -> list[list[tuple[str, float]]]:
        dense_results = self._dense.search_many(queries, self._candidates)
        reranked = []
        for query, candidates in zip(queries, dense_results):
            scored = [
                (sku_id, score + self._bonus(query, self._by_sku[sku_id]))
                for sku_id, score in candidates
            ]
            scored.sort(key=lambda pair: pair[1], reverse=True)
            reranked.append(scored[:k])
        return reranked
