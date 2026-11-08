"""Score unigram, bigram and trigram models (k = 1) on the validation split.

Writes one prediction file per model and results/ngram/metrics.json.
"""

import json

from lm.data import ROOT, load_split, load_vocab
from lm.metrics import metrics_from_file
from lm.ngram import NGramModel

OUTPUT_DIR = ROOT / "results/ngram"


def main():
    train_ids = load_split("train")
    valid_ids = load_split("valid")
    vocab = load_vocab()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {}

    for n in (1, 2, 3):
        print(f"Training {n}-gram model...", flush=True)

        model = NGramModel(n=n, vocab_size=len(vocab), k=1)
        model.fit(train_ids)

        path = OUTPUT_DIR / f"ngram_{n}_k1_valid.npz"
        model.score_split(valid_ids, path)

        metrics = metrics_from_file(path)
        results[f"{n}-gram"] = metrics

        print(f"Perplexity: {metrics['perplexity']:.4f}", flush=True)

        del model

    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
