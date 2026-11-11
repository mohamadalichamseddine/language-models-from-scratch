
"""T16: one command to build the headline validation table and figures.

Reads saved validation metrics for the best n-grams, optional interpolation,
and the chosen Transformer run. Does not invent metrics for absent runs.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lm.evaluation import RESULTS_DIR, TABLE_COLUMNS, write_table


def read_valid(run):
    path = run / "metrics.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("valid")


def build_figures(results_dir, transformer_run, out):
    out.mkdir(parents=True, exist_ok=True)
    tuning_path = results_dir / "ngram/tuning.json"
    tuning = json.loads(tuning_path.read_text(encoding="utf-8"))
    rows = []
    missing = []
    for n, name in enumerate(("unigram", "bigram", "trigram"), 1):
        k = tuning["best_k"][name]
        run = results_dir / "ngram" / f"n{n}_k{k:g}"
        metrics = read_valid(run)
        if metrics is None:
            missing.append(str(run / "metrics.json"))
        else:
            rows.append((f"{name} (k={k:g})", metrics))
    interp = results_dir / "ngram/interpolated"
    if interp.exists():
        metrics = read_valid(interp)
        if metrics is not None:
            rows.append(("interpolated trigram", metrics))
        else:
            missing.append(str(interp / "metrics.json"))
    trans = results_dir / transformer_run
    metrics = read_valid(trans)
    if metrics is None:
        missing.append(str(trans / "metrics.json"))
    else:
        rows.append(("Transformer", metrics))
    if not rows:
        raise ValueError("No validation metrics found")
    write_table(rows, out / "comparison", "# Validation set comparison (only available scored models)")

    fig, ax = plt.subplots(figsize=(8, 4.7))
    labels = [name for name, _ in rows]
    vals = [metric["perplexity"] for _, metric in rows]
    bars = ax.bar(labels, vals)
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
    ax.set_ylabel("Validation perplexity (lower is better)")
    ax.set_title("Validation perplexity by model")
    ax.tick_params(axis="x", rotation=22)
    ax.set_ylim(0, max(vals) * 1.18)
    fig.tight_layout()
    fig.savefig(out / "perplexity.png", dpi=150)
    plt.close(fig)

    curves = trans / "curves.csv"
    if curves.exists():
        with curves.open(newline="", encoding="utf-8") as f:
            points = list(csv.DictReader(f))
        if points:
            keys = set(points[0])
            fig, ax = plt.subplots(figsize=(8, 4.7))
            for key in ("train_loss", "val_loss"):
                if key in keys:
                    xy = [(float(p["step"]), float(p[key])) for p in points
                          if p.get(key) not in (None, "", "nan") and np.isfinite(float(p[key]))]
                    if xy:
                        ax.plot(*zip(*xy), label=key.replace("_", " "))
            ax.set_xlabel("Training step")
            ax.set_ylabel("Cross-entropy loss (nats)")
            ax.set_title("Transformer training curves")
            ax.legend()
            fig.tight_layout()
            fig.savefig(out / "transformer_loss.png", dpi=150)
            plt.close(fig)
    else:
        missing.append(str(curves))
    status = {"models_included": labels, "missing_inputs": missing,
              "complete": not missing}
    (out / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    for item in missing:
        print(f"Missing input (not fabricated): {item}")
    print(f"Wrote comparison and available figures to {out}")
    return status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--transformer-run", default="transformer/best")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "figures")
    args = parser.parse_args()
    build_figures(args.results_dir, args.transformer_run, args.out)


if __name__ == "__main__":
    main()
