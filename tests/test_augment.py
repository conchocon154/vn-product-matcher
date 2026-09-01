import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vnmatch.augment import AmbiguityChecker, corrupt, generate
from vnmatch.catalog import Product
from vnmatch.normalize import normalize


def make(sku_id: str, name: str, spec: str = "") -> Product:
    return Product.build(sku_id, name, spec, "Con", page=1)


CATALOGUE = [
    make("A", "Bu lông lục giác ngoài", "10x50x17 inox 304"),
    make("B", "Bu lông lục giác ngoài", "10x60x17 inox 304"),
    make("C", "Tán XD M12", "sắt xi xám"),
    make("D", "Tán XD M22", "sắt xi xám"),
    make("E", "Sơn xịt WIN 225 Black Board"),
]


def test_checker_flags_a_query_that_fits_two_skus():
    checker = AmbiguityChecker(CATALOGUE)
    # Drops the size, so it fits both the M12 and the M22 nut.
    assert checker.matching_skus("tan xd sat xi xam") == 2
    assert checker.matching_skus("tan xd m12 sat xi xam") == 1


def test_checker_ignores_unknown_tokens():
    checker = AmbiguityChecker(CATALOGUE)
    assert checker.matching_skus("khong co trong catalogue") == 0


def test_corrupt_rejects_an_ambiguous_result():
    checker = AmbiguityChecker(CATALOGUE)
    rng = random.Random(0)
    rejected = False
    for _ in range(400):
        if corrupt("Tán XD M12 sắt xi xám", rng, checker) is None:
            rejected = True
            break
    assert rejected, "a lossy corruption should eventually be rejected"


def test_generated_queries_are_answerable_and_labelled():
    queries = generate(CATALOGUE, per_product=3, seed=1, catalogue=CATALOGUE)
    assert queries
    sku_ids = {p.sku_id for p in CATALOGUE}
    for query in queries:
        assert query.sku_id in sku_ids
        assert query.text.strip()
        assert query.corruptions


def test_generate_is_deterministic_for_a_seed():
    first = generate(CATALOGUE, per_product=3, seed=7, catalogue=CATALOGUE)
    second = generate(CATALOGUE, per_product=3, seed=7, catalogue=CATALOGUE)
    assert [q.text for q in first] == [q.text for q in second]


def test_queries_are_not_just_the_clean_name():
    queries = generate(CATALOGUE, per_product=4, seed=3, catalogue=CATALOGUE)
    by_sku = {p.sku_id: p for p in CATALOGUE}
    trivial = [q for q in queries
               if normalize(q.text) == by_sku[q.sku_id].canonical and len(q.corruptions) == 1]
    assert not trivial
