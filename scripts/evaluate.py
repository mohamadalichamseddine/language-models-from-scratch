"""Compute metrics for one or more runs and build one comparison table.

Each run folder needs a <split>.npz prediction file (see lm.evaluation). This
saves metrics.json in every run folder, then writes the table as Markdown and
CSV and prints it.

Example, after scripts/run_ngram.py:
    python scripts/evaluate.py results/ngram/* --out results/ngram/comparison
"""

import argparse
import glob
from pathlib import Path

from lm.data import SPLITS, load_split
from lm.evaluation import evaluate_run, run_name, write_table


def find_run_dirs(patterns):
    """Expand the run folder arguments.

    Wildcards are expanded here, because PowerShell doesn't expand them.
    Only folders count, so a table saved next to the runs is skipped.
    """
    run_dirs = []
    for pattern in patterns:
        matches = [Path(p) for p in sorted(glob.glob(pattern)) if Path(p).is_dir()]
        if not matches:
            raise SystemExit(f"No run folders match {pattern}")
        run_dirs.extend(matches)
    return run_dirs


def main():
    parser = argparse.ArgumentParser(description="Compute metrics for runs and build a comparison table.")
    parser.add_argument("runs", nargs="+", help="run folders, wildcards allowed (e.g. results/ngram/*)")
    parser.add_argument("--split", default="valid", choices=SPLITS, help="split to evaluate (default: valid)")
    parser.add_argument("--out", type=Path, required=True, help="table path without extension; writes .md and .csv")
    args = parser.parse_args()

    num_tokens = len(load_split(args.split))

    rows = []
    for run_dir in find_run_dirs(args.runs):
        metrics = evaluate_run(run_dir, args.split, num_tokens)
        rows.append((run_name(run_dir), metrics))

    title = f"Metrics on the {args.split} split ({num_tokens:,} tokens)."
    print(write_table(rows, args.out, title))


if __name__ == "__main__":
    main()
