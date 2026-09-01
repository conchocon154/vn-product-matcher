"""Fine-tune a multilingual sentence encoder on the noisy-query task.

Training uses MultipleNegativesRankingLoss with mined hard negatives.  The
in-batch negatives a plain pair setup provides are mostly unrelated products,
which teaches the model little; the negatives that matter are the SKUs a lexical
retriever already confuses with the right answer - the same bolt in a different
length, or the same part in a different steel grade.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vnmatch.catalog import read_jsonl  # noqa: E402
from vnmatch.evaluate import load_queries  # noqa: E402
from vnmatch.normalize import normalize  # noqa: E402

BASE_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def mine_hard_negatives(products, queries, per_query: int, seed: int):
    """Pick confusable wrong answers using the character TF-IDF retriever."""
    from vnmatch.retrievers import TfidfMatch

    retriever = TfidfMatch(products)
    by_sku = {p.sku_id: p for p in products}
    rng = random.Random(seed)
    examples = []
    for query in queries:
        gold = by_sku.get(query.sku_id)
        if gold is None:
            continue
        candidates = [
            sku_id for sku_id, _ in retriever.search(query.text, per_query + 4)
            if sku_id != query.sku_id
        ]
        if not candidates:
            continue
        for sku_id in candidates[:per_query]:
            examples.append((normalize(query.text), gold.canonical, by_sku[sku_id].canonical))
    rng.shuffle(examples)
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--negatives-per-query", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=ROOT / "models" / "vnmatch-minilm")
    args = parser.parse_args()

    from datasets import Dataset
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
    from sentence_transformers import SentenceTransformerTrainingArguments as TrainingArguments
    from sentence_transformers.losses import MultipleNegativesRankingLoss

    products = read_jsonl(ROOT / "data" / "catalog.jsonl")
    train_queries = load_queries(ROOT / "data" / "queries_train.jsonl")
    print(f"{len(products)} SKU, {len(train_queries)} truy van huan luyen")

    print("Dang khai thac hard negatives...")
    triplets = mine_hard_negatives(
        products, train_queries, args.negatives_per_query, args.seed
    )
    print(f"{len(triplets)} bo ba (query, dung, sai-de-nham)")

    dataset = Dataset.from_dict(
        {
            "anchor": [t[0] for t in triplets],
            "positive": [t[1] for t in triplets],
            "negative": [t[2] for t in triplets],
        }
    )

    model = SentenceTransformer(args.base_model)
    loss = MultipleNegativesRankingLoss(model)

    trainer = SentenceTransformerTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(ROOT / "models" / "_checkpoints"),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            warmup_ratio=0.1,
            seed=args.seed,
            logging_steps=50,
            save_strategy="no",
            report_to=[],
        ),
        train_dataset=dataset,
        loss=loss,
    )
    trainer.train()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.out))
    print(f"\nDa luu model -> {args.out}")


if __name__ == "__main__":
    main()
