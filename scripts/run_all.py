"""T26: rerun every step after training, from the committed checkpoints.

Usage:
    python scripts/run_all.py

Prepares the data, runs the tests, and then reruns the experiment scripts in
order: n-gram tuning and interpolation, validation scoring of every committed
Transformer checkpoint, context truncation, generation, the plots, and the
T21-T23 test-set analyses. Each script writes its results into results/ as
usual, so check `git diff` afterwards. On a different machine the scores can
change in the last digits.

Nothing is trained: training needs a GPU and its commands are in the README.
The test split is not scored again either. T21-T23 are rebuilt from the saved
test predictions (unzip results/t21_results.zip) and skipped without them.

On a CPU it takes about two hours, mostly scoring the Transformers.
"""

import argparse
import os
import subprocess
import sys

import torch

from lm.data import ROOT
from lm.evaluation import RESULTS_DIR, predictions_path
from lm.final_evaluation import final_rows

# WikiText has characters the Windows console can't print, and the plots
# need no window.
ENV = {**os.environ, "PYTHONUTF8": "1", "MPLBACKEND": "Agg"}


def run(*args):
    """Run one step from the repo root and stop at the first failure."""
    print(f"\n>>> python {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, *args], cwd=ROOT, env=ENV, check=True)


def has_test_predictions():
    """True when every model in the final table has its T21 test predictions."""
    return all(predictions_path(run_dir, "test").exists()
               for _, run_dirs in final_rows(RESULTS_DIR) for run_dir in run_dirs)


def main():
    parser = argparse.ArgumentParser(description="Rerun every step after training, from the committed checkpoints.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64, help="Transformer windows per forward pass")
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()
    scoring = ["--device", args.device, "--batch-size", str(args.batch_size)]

    # Data (T02, T03)
    run("scripts/prepare_data.py")
    if not args.skip_tests:
        run("-m", "pytest", "-q")
    run("scripts/dataset_stats.py")

    # N-gram baselines (T05-T08)
    run("scripts/run_ngram.py")
    run("scripts/run_interpolated.py")
    run("scripts/evaluate.py", "results/ngram/*", "--out", "results/ngram/comparison")

    # The T13 Transformer on valid (T12-T17). The sweep table comes from the
    # training logs, since only the best of the nine sweep models is committed.
    run("scripts/score_transformer.py", "--run", "transformer/best", *scoring)
    run("scripts/sweep_transformer.py", "--report-only")
    run("scripts/build_figures.py")
    run("scripts/model_disagreement.py", "results/transformer/best/valid.npz", "results/ngram/n3_k0.001/valid.npz",
        "--split", "valid", "--top", "20", "--context", "10", "--out", "results/figures/disagreements.csv")
    run("scripts/generate.py", "--device", args.device)
    run("scripts/generation_metrics.py")

    # Context ablation (T18-T20). The trained runs are kept and only scored.
    run("scripts/run_ablation.py", *scoring)
    for seed in (0, 1):
        run("scripts/score_context_truncation.py", "--run", f"ablation/k128_seed{seed}", *scoring)
    run("scripts/plot_ablation.py")

    # Test set (T21-T23), from the saved test predictions
    if has_test_predictions():
        run("scripts/final_evaluation.py")
        run("scripts/rare_tokens.py")
        run("scripts/long_context.py")
    else:
        print("\nSkipping T21-T23: the test predictions are missing. Unzip results/t21_results.zip "
              "from the repo root to rebuild them; the test split is never scored again here.")

    # Full generation analysis (T24)
    run("scripts/generation_analysis.py", "--device", args.device)
    print("\nDone. Check `git diff` to see what changed.")


if __name__ == "__main__":
    main()
