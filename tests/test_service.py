import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vnmatch.service import MatcherService

CATALOG = ROOT / "data" / "catalog.jsonl"


def test_falls_back_to_lexical_without_a_model():
    service = MatcherService(CATALOG, model_path=None)
    assert service.model_loaded is False
    assert service.backend == "tfidf_char"
    assert service.search("bu long luc giac ngoai 10x50", 3)


def test_missing_model_directory_is_not_fatal():
    service = MatcherService(CATALOG, model_path=ROOT / "models" / "does-not-exist")
    assert service.model_loaded is False
    assert service.search("son xit", 1)


def test_results_carry_the_catalogue_fields():
    service = MatcherService(CATALOG, model_path=None)
    top = service.search("bl lgn 10x50 i304", 1)[0]
    assert set(top) == {"sku_id", "name", "spec", "unit", "full_name", "material", "score"}
    assert top["sku_id"].startswith("SKU")
