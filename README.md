# Vietnamese Product Matcher

Match free-text Vietnamese hardware names onto catalogue SKUs.

*Tiếng Việt: [README.vi.md](README.vi.md)*

[![CI](https://github.com/conchocon154/vn-product-matcher/actions/workflows/ci.yml/badge.svg)](https://github.com/conchocon154/vn-product-matcher/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A shop assistant types `oc lgn 10x50 i304`. The catalogue calls the same part
**Bu lông lục giác ngoài 10x50x17 inox 304**. No character the two strings share
in common is enough: `ốc` and `bu lông` are different words for the same bolt,
`lgn` is an abbreviation, `i304` is a steel grade, and the wrench size `17` was
never typed at all.

This repository builds and measures a matcher for that problem over a real
1 827-SKU Vietnamese hardware catalogue.

```
$ curl 'localhost:8000/search?q=oc+lgn+10x50+i304&k=1'
{
  "query": "oc lgn 10x50 i304",
  "backend": "dense_finetuned",
  "took_ms": 26.25,
  "results": [
    {
      "sku_id": "SKU00260",
      "full_name": "Bu lông lục giác ngoài 10x50x17 inox 304",
      "unit": "con",
      "material": "inox 304",
      "score": 0.9689
    }
  ]
}
```

## Results

Retrieval over the full 1 827-SKU catalogue. `test_unseen` holds queries for
274 SKUs that were **removed from training entirely**, so it measures whether
the model learned how Vietnamese hardware is named rather than memorising the
catalogue.

| retriever | R@1 (seen) | R@1 (unseen) | MRR (unseen) | ms/query |
|---|---|---|---|---|
| exact string match | 8.4% | 8.3% | 0.083 | 0.0 |
| BM25 (SQLite FTS5) | 82.2% | 83.4% | 0.853 | 0.6 |
| RapidFuzz `token_set_ratio` | 89.6% | 88.2% | 0.920 | 2.4 |
| TF-IDF character 3–5-grams | 94.8% | 95.0% | 0.970 | 0.6 |
| encoder, off the shelf | 65.0% | 63.8% | 0.721 | 0.6 |
| **encoder, fine-tuned** | **97.1%** | **96.9%** | **0.980** | **0.5** |
| encoder + dimension reranker | 96.9% | 96.8% | 0.980 | 2.4 |

`seen` and `unseen` land within 0.2 points of each other, so the fine-tuned
encoder generalises to SKUs it never saw.

The `ms/query` column is *batched* throughput, measured while scoring the
whole test set at once. A single request through the API costs about 26 ms,
because one query cannot fill the encoder's batch. Serving many queries at a
time is where that column applies.

### Three findings worth stating plainly

**1. Fine-tuning is what matters, not the choice of encoder.**
Off the shelf, `paraphrase-multilingual-MiniLM-L12-v2` scores 63.8% — far
*worse* than a TF-IDF baseline that takes six lines of scikit-learn. The same
weights after fine-tuning score 96.9%. Reaching for an embedding model without
adapting it to the domain would have made this system worse, not better.

**2. The win over a good lexical baseline is real but modest.**
Character n-gram TF-IDF is a strong baseline here: 95.0%. The fine-tuned
encoder beats it by 1.9 points.

```
dense_finetuned vs tfidf_char   (test_unseen, n = 1634, paired)
  R@1                 95.0% -> 96.9%
  only tfidf right    26
  only encoder right  56
  McNemar p           1.22e-03
  95% CI of the gap   [+0.80%, +2.94%]
```

The gap is statistically significant, and it is 1.9 points — not the order of
magnitude that "we added deep learning" usually implies. Where the encoder
earns its keep is the synonym queries it was built for: 93.6% → 97.4%.

**3. My hybrid reranker was a wasted idea, and the measurement says so.**
The design assumed encoders garble numbers, so a reranker re-scored candidates
on exact dimension and material overlap. It changed nothing:

```
hybrid vs dense_finetuned   (test_unseen, n = 1634, paired)
  R@1                 96.9% -> 96.8%
  only encoder right  7
  only hybrid right   6
  McNemar p           1.00
  95% CI of the gap   [-0.49%, +0.37%]
```

Fine-tuning had already taught the encoder to read `10x50x17`. The reranker is
still in `retrievers.py` as a documented negative result, but the API serves the
plain encoder: the hybrid costs about four times the latency for no measurable
accuracy.

### Where every method still fails

R@1 on `test_unseen`, split by the corruption each query carries:

| corruption | n | TF-IDF | fine-tuned encoder |
|---|---|---|---|
| shorthand (`bl` → `bu lông`) | 480 | 97.1% | 98.8% |
| word swapped | 318 | 96.2% | 97.2% |
| no diacritics | 1315 | 94.7% | 96.7% |
| typo | 587 | 94.7% | 96.1% |
| word dropped | 392 | 94.1% | 96.7% |
| `*` or `×` for `x` | 357 | 94.7% | 96.6% |
| **synonym** (`ốc` → `bu lông`) | 608 | 93.6% | **97.4%** |
| **partial dimensions** (`10x50` for `10x50x17`) | 262 | 83.2% | **88.5%** |
| **terse** (adjectives dropped) | 67 | **79.1%** | 73.1% |

Partial dimensions are the standing weakness of both: `10x50` genuinely fits
several SKUs and the ranking has to guess. Terse queries are the one place
lexical matching still wins.

## Limitations

Read these before trusting the numbers.

- **The queries are synthetic.** The shop keeps no log of what staff typed, so
  every query here was generated by corrupting catalogue names
  (`src/vnmatch/augment.py`). The corruptions were chosen from how names degrade
  on real order slips, but they are still a model of user behaviour, not a
  sample of it.
- **The synonym result is partly circular.** Training queries and test queries
  draw synonyms from the same dictionary, so the encoder is rewarded for
  learning a mapping the generator already knew. Against synonyms absent from
  that list, expect closer to the TF-IDF number than to 97.4%.
- **Prices are excluded.** The source is a supplier price list; only names,
  specs and units are published here.
- **One catalogue, one trade.** 1 827 SKUs of hardware and plumbing supplies.
  Nothing here has been tested on another domain.

## How it works

```
raw catalogue extract        (not in this repo - vendor PDF)
        │  scripts/prepare_data.py     drop prices, dedupe, canonicalise
        ▼
data/catalog.jsonl  +  data/catalog.db      1827 SKU, SQLite + FTS5
        │  scripts/build_queries.py    two-stage corruption + ambiguity filter
        ▼
queries_{train,validation,test_seen,test_unseen}.jsonl
        │  scripts/train.py            MultipleNegativesRankingLoss
        ▼                              with TF-IDF-mined hard negatives
models/vnmatch-minilm
        │  scripts/evaluate_all.py     R@k, MRR, per-corruption breakdown
        │  scripts/significance.py     McNemar + paired bootstrap
        ▼
api/main.py                            FastAPI /search, /health
```

### Query generation, and why it is split in two stages

Corruptions fall into two kinds, and conflating them silently corrupts the
labels:

- **Surface corruptions** — typos, dropped diacritics, `*` for `x`,
  abbreviations, synonyms — change the spelling and keep the query answerable.
- **Lossy corruptions** — dropping a word, truncating `10x50x17` to `10x50`,
  stripping every adjective — can leave a query that fits several SKUs.

`tán xd sắt xi xám` is a real example: dropping the size makes it fit both the
M12 and the M22 nut. Scored against a single gold SKU, that query punishes a
retriever for returning a perfectly correct answer.

So lossy corruptions run first, and the result is checked against an inverted
index over the whole catalogue: if more than one SKU's tokens cover the query,
the sample is discarded. Surface corruptions run afterwards, where they cannot
create ambiguity. 77 ambiguous queries were dropped this way.

### Hard negatives

In-batch negatives from a random batch are mostly unrelated products —
distinguishing a bolt from a tin of paint teaches the encoder little. The
negatives that matter are the ones a lexical retriever already confuses with the
right answer: the same bolt one length longer, the same bracket in a different
steel grade. `scripts/train.py` mines two of those per query from the TF-IDF
retriever before training.

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Rebuild the derived artefacts from the committed catalogue:

```bash
python scripts/prepare_data.py --from-jsonl data/catalog.jsonl
python scripts/build_queries.py
```

Score the lexical baselines — no model download, a few seconds:

```bash
python scripts/evaluate_all.py
```

Fine-tune the encoder (about 7 minutes on an M-series Mac via MPS):

```bash
python scripts/train.py
python scripts/evaluate_all.py --include-base
python scripts/significance.py
```

Serve it:

```bash
uvicorn api.main:app --app-dir . --reload
```

`GET /search?q=<text>&k=<n>` returns ranked matches; `GET /health` reports
which backend is live. With no model in `models/`, the API starts anyway and
serves TF-IDF, so a deployment that never ran training still works.

Set `VNMATCH_HYBRID=1` to serve the dimension reranker instead — kept for
reproducing finding 3, not recommended.

## Layout

```
src/vnmatch/
  normalize.py    Vietnamese canonicalisation: diacritics, abbreviations, dimensions
  augment.py      two-stage query corruption + the ambiguity checker
  catalog.py      Product model, JSONL storage, SQLite schema with FTS5
  retrievers.py   exact, BM25, RapidFuzz, TF-IDF, dense, hybrid
  evaluate.py     R@k and MRR
  service.py      the stack the API serves, with lexical fallback
scripts/          prepare_data, build_queries, train, evaluate_all, significance
api/main.py       FastAPI
tests/            27 tests, no model or network required
```

## Tests

```bash
python -m pytest tests -q
```

27 tests covering normalisation edge cases, the ambiguity checker, retriever
contracts, the lexical fallback and the HTTP layer. CI runs them on every push
and fails if TF-IDF R@1 drops below 90%, which catches a broken normaliser or
query generator without needing the encoder.

## License

MIT — see [LICENSE](LICENSE).
