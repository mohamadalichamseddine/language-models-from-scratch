"""T23: where the k=128 Transformer beats the k=6 one, by context and word class.

Needs the T21 test predictions: unzip results/t21_results.zip from the repo
root, or run scripts/final_evaluation.py. Writes results/long_context/
(long_context.md and .csv) and prints examples.
"""

import argparse
import sys
from pathlib import Path

from lm.data import load_split, load_vocab
from lm.disagreement import format_row, top_disagreements
from lm.evaluation import RESULTS_DIR, predictions_path
from lm.final_evaluation import ablation_dir
from lm.long_context import CONTEXT_GROUPS, LONG, SHORT, context_groups, long_context_tables, write_long_context_tables


def main():
    parser = argparse.ArgumentParser(description="Where a longer context helps: Transformer k=128 vs k=6 on test.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "long_context")
    args = parser.parse_args()

    # WikiText has characters that Windows' default encoding can't print to a file or pipe.
    sys.stdout.reconfigure(encoding="utf-8")

    vocab = load_vocab()
    test_ids = load_split("test")
    print(write_long_context_tables(long_context_tables(args.results_dir, test_ids, vocab), args.out))

    long = predictions_path(ablation_dir(args.results_dir, LONG, 0), "test")
    short = predictions_path(ablation_dir(args.results_dir, SHORT, 0), "test")
    copy_like = context_groups(test_ids) == CONTEXT_GROUPS[1]
    examples = top_disagreements(long, short, test_ids, vocab, top=4, context=20, mask=copy_like)
    print(f"Copy-like tokens where k={LONG} helps most (seed 0; log-prob k={LONG} minus k={SHORT}):")
    for row in examples["a_wins"]:
        print(format_row(row))

    examples = top_disagreements(long, short, test_ids, vocab, top=3, context=20)
    print(f"\nTokens where k={SHORT} does better than k={LONG}:")
    for row in examples["b_wins"]:
        print(format_row(row))


if __name__ == "__main__":
    main()
