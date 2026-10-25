
import json
from pathlib import Path

from lm.data import load_split, load_vocab
from lm.metrics import metrics_from_file
from lm.ngram import NGramModel


def main():
    train_ids = load_split("train")
    valid_ids = load_split("valid")
    vocab = load_vocab()

    output_dir = Path("results/ngram")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for n in (1, 2, 3):
        print(f"Training {n}-gram model...", flush=True)

        model = NGramModel(n=n, vocab_size=len(vocab), k=1)
        model.fit(train_ids)

        path = output_dir / f"ngram_{n}_k1_valid.npz"
        model.score_split(valid_ids, path)

        metrics = metrics_from_file(path)
        results[f"{n}-gram"] = metrics

        print(f"Perplexity: {metrics['perplexity']:.4f}", flush=True)

        del model

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
