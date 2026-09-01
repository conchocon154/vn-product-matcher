import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from vnmatch.catalog import Product, build_database
from vnmatch.retrievers import Bm25Match, ExactMatch, FuzzyMatch, TfidfMatch

CATALOGUE = [
    Product.build("A", "Bu lông lục giác ngoài", "10x50x17 inox 304", "Con", 1),
    Product.build("B", "Bu lông lục giác ngoài", "10x60x17 inox 304", "Con", 1),
    Product.build("C", "Tán XD M12", "sắt xi xám", "Con", 2),
    Product.build("D", "Sơn xịt WIN 225 Black Board", "", "Lon", 3),
]


def test_exact_match_needs_the_canonical_form():
    retriever = ExactMatch(CATALOGUE)
    assert retriever.search("Bu lông lục giác ngoài 10x50x17 inox 304", 5)[0][0] == "A"
    assert retriever.search("bu long lg ngoai 10x50", 5) == []


def test_shorthand_query_reaches_the_right_sku():
    # `BL LGN 10*50 i304` normalises onto the catalogue wording.
    for retriever in (TfidfMatch(CATALOGUE), FuzzyMatch(CATALOGUE)):
        assert retriever.search("BL LGN 10*50*17 i304", 3)[0][0] == "A", retriever.name


def test_retrievers_respect_k():
    retriever = TfidfMatch(CATALOGUE)
    assert len(retriever.search("bu long inox", 2)) <= 2


def test_bm25_over_sqlite(tmp_path):
    db_path = tmp_path / "catalog.db"
    build_database(CATALOGUE, db_path).close()
    retriever = Bm25Match(db_path)
    results = retriever.search("son xit win", 3)
    assert results and results[0][0] == "D"


def test_bm25_survives_punctuation(tmp_path):
    db_path = tmp_path / "catalog.db"
    build_database(CATALOGUE, db_path).close()
    # A raw `10x50x17 "` would be an FTS5 syntax error if tokens were not quoted.
    assert Bm25Match(db_path).search('10x50x17 " OR ', 3) is not None
