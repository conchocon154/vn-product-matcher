import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(monkeypatch_module=None):
    # Force the lexical fallback so the test suite never needs the encoder.
    import os

    os.environ["VNMATCH_CATALOG"] = str(ROOT / "data" / "catalog.jsonl")
    os.environ["VNMATCH_MODEL"] = ""
    from vnmatch.service import get_service

    get_service.cache_clear()
    sys.path.insert(0, str(ROOT / "api"))
    from api.main import app  # noqa: F401

    return TestClient(app)


def test_health_reports_the_backend(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["catalogue_size"] > 0
    assert body["model_loaded"] is False
    assert body["backend"] == "tfidf_char"


def test_search_returns_ranked_matches(client):
    body = client.get("/search", params={"q": "bl lgn 10x50 i304", "k": 5}).json()
    assert body["query"] == "bl lgn 10x50 i304"
    assert 1 <= len(body["results"]) <= 5
    scores = [row["score"] for row in body["results"]]
    assert scores == sorted(scores, reverse=True)


def test_search_rejects_an_empty_query(client):
    assert client.get("/search", params={"q": "   "}).status_code == 422


def test_k_is_bounded(client):
    assert client.get("/search", params={"q": "bu long", "k": 999}).status_code == 422
