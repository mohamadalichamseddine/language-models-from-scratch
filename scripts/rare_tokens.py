"""T22: cross-entropy and top-1 accuracy by how often the target word occurs in train.

Needs the T21 test predictions: unzip results/t21_results.zip from the repo
root, or run scripts/final_evaluation.py. Writes results/rare_tokens/
(rare_tokens.md and .csv) and prints rare words on which the k=128
Transformer and the interpolated trigram disagree most.
"""

import argparse
import sys
from pathlib import Path

from lm.data import load_split, load_vocab
from lm.disagreement import format_row, top_disagreements
from lm.evaluation import RESULTS_DIR, predictions_path
from lm.final_evaluation import ablation_dir
from lm.rare_tokens import frequency_groups, rare_token_table, write_rare_token_table
from lm.reproducibility import read_numeric_csv


def main():
    parser = argparse.ArgumentParser(description="CE and top-1 accuracy by train frequency of the target word.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "rare_tokens")
    parser.add_argument("--report-only", action="store_true", help="render saved CSV; does not recompute scores or examples")
    args = parser.parse_args()

    # WikiText has characters that Windows' default encoding can't print to a file or pipe.
    sys.stdout.reconfigure(encoding="utf-8")

    vocab = load_vocab()
    test_ids = load_split("test")
    groups = frequency_groups(load_split("train"), test_ids, len(vocab), vocab.unk_id)
    if args.report_only:
        rows = read_numeric_csv(args.results_dir / "rare_tokens/rare_tokens.csv", {"model"})
        print(write_rare_token_table(rows, groups, args.out))
        return
    print(write_rare_token_table(rare_token_table(args.results_dir, groups), groups, args.out))

    transformer = predictions_path(ablation_dir(args.results_dir, 128, 0), "test")
    trigram = predictions_path(args.results_dir / "ngram/interpolated", "test")
    examples = top_disagreements(transformer, trigram, test_ids, vocab, top=3, mask=groups == "1-10")
    print("Rare words (1-10 times in train) that the Transformer k=128 (seed 0) predicts much better")
    print("than the interpolated trigram, with the log-prob difference:")
    for row in examples["a_wins"]:
        print(format_row(row))
    print("\nRare words the interpolated trigram predicts better:")
    for row in examples["b_wins"][:2]:
        print(format_row(row))


if __name__ == "__main__":
    main()
