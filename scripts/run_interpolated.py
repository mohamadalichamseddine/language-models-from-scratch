"""Interpolated trigram baseline, with its weights tuned on the validation split.

P(w | h) = l1 * P_uni(w) + l2 * P_bi(w | h) + l3 * P_tri(w | h), where the three
parts are unsmoothed (MLE, k = 0) n-gram models. Every weight triple on a grid
with step LAMBDA_STEP is tried, and the one with the lowest validation
perplexity is kept. The tuned model is scored as the run folder
results/ngram/interpolated/ (config, predictions and metrics).

Afterwards, rebuild the comparison table with:
    python scripts/evaluate.py results/ngram/* --out results/ngram/comparison
"""

import numpy as np

from lm.data import load_split, load_vocab
from lm.evaluation import RESULTS_DIR, evaluate_run, predictions_path, save_config
from lm.ngram import InterpolatedNGram, NGramModel, tune_lambdas

RUN_DIR = RESULTS_DIR / "ngram" / "interpolated"
LAMBDA_STEP = 0.01


def main():
    train_ids = load_split("train")
    valid_ids = load_split("valid")
    vocab_size = len(load_vocab())

    print("Fitting the unigram, bigram and trigram parts...", flush=True)
    parts = [NGramModel(n=n, vocab_size=vocab_size, k=0).fit(train_ids) for n in (1, 2, 3)]

    # The probability each part gives every validation token (0 where a part
    # has never seen the token after its history).
    part_probs = np.stack([np.exp(part.score_split(valid_ids)[0]) for part in parts], axis=1)

    lambdas, perplexity = tune_lambdas(part_probs, step=LAMBDA_STEP)
    print(f"Best weights: {lambdas} (validation perplexity {perplexity:.2f})", flush=True)

    save_config(RUN_DIR, {
        "model": "interpolated_ngram",
        "parts": "MLE unigram, bigram and trigram (k = 0)",
        "lambdas": lambdas,
        "lambda_step": LAMBDA_STEP,
        "train_split": "train",
        "vocab_size": vocab_size,
    })

    print("Scoring the interpolated model...", flush=True)
    model = InterpolatedNGram(parts, lambdas)
    model.score_split(valid_ids, predictions_path(RUN_DIR, "valid"))

    metrics = evaluate_run(RUN_DIR, "valid", len(valid_ids))
    print(f"Validation perplexity: {metrics['perplexity']:.2f}, top-1 accuracy: {metrics['top1_accuracy']:.2%}")


if __name__ == "__main__":
    main()
