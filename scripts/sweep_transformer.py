"""T13: the first full Transformer runs, with a small learning-rate x dropout sweep.

Trains the Transformer at context 128 for every pair in LEARNING_RATES x
DROPOUTS (all other settings are the defaults of scripts/train_transformer.py)
and scores each run on valid. Each run gets its own folder,
results/transformer/sweep/lr{lr}_drop{dropout}/. Runs that are already done
are skipped, so after a Colab disconnect the script can simply be started
again.

The best run is the one with the lowest validation perplexity. At the end the
script writes:

    results/transformer/sweep/sweep.md, sweep.json   every run: settings, metrics, training time
    results/transformer/best/   a copy of the best run's folder, with its model
        (best.pt): the Transformer later tickets use
    configs/transformer.json   the best settings, frozen for the ablation runs (T18)
    results/comparison.md, comparison.csv   n-grams at their best k, the
        interpolated trigram, and the best Transformer

The model in results/transformer/best/ is the only one that goes into the
repo; the notebook leaves the sweep runs' models on Google Drive.

Needs a GPU (see notebooks/train_on_colab.ipynb):
    python scripts/sweep_transformer.py
"""

import json
import shutil
import subprocess
import sys

from lm.data import ROOT, load_split
from lm.evaluation import RESULTS_DIR, run_name, write_table

SWEEP_DIR = RESULTS_DIR / "transformer" / "sweep"
BEST_DIR = RESULTS_DIR / "transformer" / "best"
FROZEN_CONFIG = ROOT / "configs" / "transformer.json"

CONTEXT = 128
LEARNING_RATES = [3e-4, 6e-4, 1e-3]
DROPOUTS = [0.1, 0.2, 0.3]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def run_script(script, *args):
    subprocess.run([sys.executable, str(ROOT / "scripts" / script), *args], check=True)


def train_and_score(lr, dropout):
    """Train and score one setting, unless that was already done. Returns its run summary."""
    run = f"transformer/sweep/lr{lr:g}_drop{dropout:g}"
    run_dir = RESULTS_DIR / run

    # training.json is written when training finishes, so a run that was cut
    # off is trained again from the start.
    if not (run_dir / "training.json").exists():
        print(f"\n=== Training {run} ===", flush=True)
        run_script("train_transformer.py", "--run", run, "--context", str(CONTEXT), "--lr", str(lr), "--dropout", str(dropout))

    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists() or "valid" not in read_json(metrics_path):
        print(f"\n=== Scoring {run} ===", flush=True)
        run_script("score_transformer.py", "--run", run)

    training = read_json(run_dir / "training.json")
    metrics = read_json(metrics_path)["valid"]
    return {
        "run": run,
        "lr": lr,
        "dropout": dropout,
        "valid_perplexity": metrics["perplexity"],
        "valid_top1_accuracy": metrics["top1_accuracy"],
        "best_step": training["best_step"],
        "steps": training["steps"],
        "stopped_early": training["stopped_early"],
        "train_seconds": training["train_seconds"],
        "device": training["device"],
    }


def write_sweep_table(runs, best):
    lines = [
        f"# Transformer sweep at context {CONTEXT}",
        "",
        "Learning rate x dropout; every other setting is the default of scripts/train_transformer.py.",
        "Validation perplexity comes from sliding-window scoring. The best run is in bold.",
        "",
        "| Learning rate | Dropout | Val perplexity | Top-1 | Best step | Steps trained | Minutes |",
        "|---|---|---|---|---|---|---|",
    ]
    for run in runs:
        perplexity = f"{run['valid_perplexity']:.2f}"
        if run is best:
            perplexity = f"**{perplexity}**"
        lines.append(
            f"| {run['lr']:g} | {run['dropout']:g} | {perplexity} | {run['valid_top1_accuracy']:.2%} "
            f"| {run['best_step']} | {run['steps']} | {run['train_seconds'] / 60:.1f} |"
        )
    devices = sorted({run["device"] for run in runs})
    lines += ["", f"Trained on: {', '.join(devices)}."]
    (SWEEP_DIR / "sweep.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    write_json(SWEEP_DIR / "sweep.json", {
        "context": CONTEXT,
        "learning_rates": LEARNING_RATES,
        "dropouts": DROPOUTS,
        "best_run": best["run"],
        "runs": runs,
    })


def freeze_best_config(best):
    """Save the best run's settings in the format scripts/train_transformer.py --config reads."""
    config = read_json(RESULTS_DIR / best["run"] / "config.json")
    settings = {key: value for key, value in config["model_config"].items() if key != "vocab_size"}
    settings.update(config["training"])
    write_json(FROZEN_CONFIG, settings)


def copy_best_run(best):
    """Copy the best run's folder, model included, to results/transformer/best/."""
    if BEST_DIR.exists():
        shutil.rmtree(BEST_DIR)  # left over from an earlier sweep
    shutil.copytree(RESULTS_DIR / best["run"], BEST_DIR)


def write_comparison(best):
    """One table: each n-gram at its best k, the interpolated trigram and the best Transformer."""
    best_k = read_json(RESULTS_DIR / "ngram" / "tuning.json")["best_k"]
    run_dirs = [RESULTS_DIR / "ngram" / f"n{n}_k{best_k[name]}" for n, name in [(1, "unigram"), (2, "bigram"), (3, "trigram")]]
    run_dirs += [RESULTS_DIR / "ngram" / "interpolated", RESULTS_DIR / best["run"]]

    rows = [(run_name(run_dir), read_json(run_dir / "metrics.json")["valid"]) for run_dir in run_dirs]
    title = f"Metrics on the valid split ({len(load_split('valid')):,} tokens). N-grams use their best k."
    print(write_table(rows, RESULTS_DIR / "comparison", title))


def main():
    runs = [train_and_score(lr, dropout) for lr in LEARNING_RATES for dropout in DROPOUTS]
    best = min(runs, key=lambda run: run["valid_perplexity"])

    write_sweep_table(runs, best)
    freeze_best_config(best)
    copy_best_run(best)
    print(f"Best run: {best['run']} (val perplexity {best['valid_perplexity']:.2f}), copied to {BEST_DIR}\n")
    write_comparison(best)


if __name__ == "__main__":
    main()
