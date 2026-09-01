"""Assemble the retrieval stack the API serves."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from .catalog import Product, read_jsonl
from .retrievers import DenseMatch, HybridMatch, TfidfMatch

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG = ROOT / "data" / "catalog.jsonl"
DEFAULT_MODEL = ROOT / "models" / "vnmatch-minilm"


class MatcherService:
    """Fine-tuned encoder in front, lexical retriever as the fallback.

    The encoder is optional on purpose: the API must still start on a machine
    where the fine-tuned weights were never downloaded, so a deployment without
    the model degrades to TF-IDF rather than failing to boot.

    The hybrid reranker is available but off by default.  It measured no better
    than the encoder alone (McNemar p = 1.00) while costing about four times the
    latency, so serving it would be paying for nothing - see the README.
    """

    def __init__(
        self,
        catalog_path: Path,
        model_path: Path | None,
        *,
        use_hybrid: bool = False,
    ) -> None:
        self.products: list[Product] = read_jsonl(catalog_path)
        self._by_sku = {p.sku_id: p for p in self.products}
        self._lexical = TfidfMatch(self.products)
        self._dense: DenseMatch | HybridMatch | None = None
        self.model_loaded = False

        if model_path is not None and model_path.exists():
            dense = DenseMatch(self.products, str(model_path), name="dense_finetuned")
            self._dense = HybridMatch(dense, self.products) if use_hybrid else dense
            self.model_loaded = True

    @property
    def backend(self) -> str:
        return getattr(self._dense, "name", "tfidf_char")

    def search(self, query: str, k: int = 5) -> list[dict]:
        retriever = self._dense or self._lexical
        results = retriever.search(query, k)
        payload = []
        for sku_id, score in results:
            product = self._by_sku[sku_id]
            payload.append(
                {
                    "sku_id": product.sku_id,
                    "name": product.name,
                    "spec": product.spec,
                    "unit": product.unit,
                    "full_name": product.full_name,
                    "material": product.material,
                    "score": round(float(score), 4),
                }
            )
        return payload


@lru_cache(maxsize=1)
def get_service() -> MatcherService:
    catalog = Path(os.environ.get("VNMATCH_CATALOG", DEFAULT_CATALOG))
    model_env = os.environ.get("VNMATCH_MODEL", str(DEFAULT_MODEL))
    model = Path(model_env) if model_env else None
    use_hybrid = os.environ.get("VNMATCH_HYBRID", "").lower() in {"1", "true", "yes"}
    return MatcherService(catalog, model, use_hybrid=use_hybrid)
