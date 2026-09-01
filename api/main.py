"""HTTP interface over the product matcher."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastapi import FastAPI, HTTPException, Query  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from vnmatch.service import get_service  # noqa: E402

app = FastAPI(
    title="Vietnamese Product Matcher",
    description="Match free-text Vietnamese hardware names onto catalogue SKUs.",
    version="1.0.0",
)


class Match(BaseModel):
    sku_id: str
    name: str
    spec: str
    unit: str
    full_name: str
    material: str | None
    score: float


class SearchResponse(BaseModel):
    query: str
    backend: str
    took_ms: float
    results: list[Match]


@app.get("/health")
def health() -> dict:
    service = get_service()
    return {
        "status": "ok",
        "catalogue_size": len(service.products),
        "backend": service.backend,
        "model_loaded": service.model_loaded,
    }


@app.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., min_length=1, description="What the user typed"),
    k: int = Query(5, ge=1, le=50, description="How many matches to return"),
) -> SearchResponse:
    if not q.strip():
        raise HTTPException(status_code=422, detail="query is empty")
    service = get_service()
    started = time.perf_counter()
    results = service.search(q, k)
    return SearchResponse(
        query=q,
        backend=service.backend,
        took_ms=round((time.perf_counter() - started) * 1000, 2),
        results=[Match(**row) for row in results],
    )
