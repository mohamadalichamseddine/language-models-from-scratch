"""Score unigram, bigram and trigram models (k = 1) on the validation split.

Each model gets a run folder, results/ngram/n{n}_k1/, with its config and its
per-token predictions. Then compute the metrics and the comparison table with:
    python scripts/evaluate.py results/ngram/* --out results/ngram/comparison
"""

from lm.data import load_split, load_vocab
from lm.evaluation import RESULTS_DIR, predictions_path, save_config
from lm.ngram import NGramModel

K = 1


def main():
    train_ids = load_split("train")
    valid_ids = load_split("valid")
    vocab_size = len(load_vocab())

    for n in (1, 2, 3):
        print(f"Scoring the {n}-gram model...", flush=True)

        run_dir = RESULTS_DIR / "ngram" / f"n{n}_k{K}"
        save_config(run_dir, {
            "model": "ngram",
            "n": n,
            "k": K,
            "train_split": "train",
            "vocab_size": vocab_size,
        })

        model = NGramModel(n=n, vocab_size=vocab_size, k=K).fit(train_ids)
        model.score_split(valid_ids, predictions_path(run_dir, "valid"))
        del model

    print(f"Predictions saved under {RESULTS_DIR / 'ngram'}")


if __name__ == "__main__":
    main()
