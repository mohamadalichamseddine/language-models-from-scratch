"""Run folders, per-run metrics and the comparison table.

Every model run has its own folder under results/, for example
results/ngram/n3_k1/, holding:

    config.json     the run's settings, saved by the script that made the run
    <split>.npz     per-token predictions on a split, in the lm.metrics format
                    (not committed: rebuilt by the scoring scripts)
    metrics.json    metrics for each scored split, saved by evaluate_run()
"""

import csv
import json

from lm.data import ROOT
from lm.metrics import compute_metrics, load_predictions

RESULTS_DIR = ROOT / "results"

# Columns of the comparison table: metric name -> (header, format).
TABLE_COLUMNS = {
    "cross_entropy_nats": ("CE (nats)", "{:.4f}"),
    "cross_entropy_bits": ("CE (bits)", "{:.4f}"),
    "perplexity": ("Perplexity", "{:.2f}"),
    "top1_accuracy": ("Top-1", "{:.2%}"),
    "top5_accuracy": ("Top-5", "{:.2%}"),
    "top10_accuracy": ("Top-10", "{:.2%}"),
}


def save_config(run_dir, config):
    """Save a run's settings as config.json in its folder."""
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "config.json", config)


def predictions_path(run_dir, split):
    """Return where a run's predictions on a split are stored."""
    return run_dir / f"{split}.npz"


def evaluate_run(run_dir, split, num_tokens):
    """Compute a run's metrics on a split and save them in its metrics.json.

    num_tokens is the length of the split. Every model must predict every
    token of the split, otherwise the scores of different models can't be
    compared. Metrics already saved for other splits are kept.
    """
    path = predictions_path(run_dir, split)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing. Run the script that scores this model first.")

    logprob, rank = load_predictions(path)
    if len(logprob) != num_tokens:
        raise ValueError(
            f"{path} has {len(logprob):,} predictions, but the {split} split has "
            f"{num_tokens:,} tokens. Every model must predict every token."
        )

    metrics = compute_metrics(logprob, rank)

    metrics_path = run_dir / "metrics.json"
    all_metrics = _read_json(metrics_path) if metrics_path.exists() else {}
    all_metrics[split] = metrics
    _write_json(metrics_path, all_metrics)
    return metrics


def run_name(run_dir):
    """Name a run by its folder relative to results/, e.g. ngram/n3_k1."""
    run_dir = run_dir.resolve()
    if run_dir.is_relative_to(RESULTS_DIR):
        return run_dir.relative_to(RESULTS_DIR).as_posix()
    return run_dir.name


def write_table(rows, out, title):
    """Write (run name, metrics) rows to out.md and out.csv.

    The Markdown table is rounded for reading; the CSV keeps full precision.
    Returns the Markdown text.
    """
    out.parent.mkdir(parents=True, exist_ok=True)

    headers = ["Run"] + [header for header, _ in TABLE_COLUMNS.values()]
    lines = [title, "", "| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for name, metrics in rows:
        cells = [name] + [fmt.format(metrics[key]) for key, (_, fmt) in TABLE_COLUMNS.items()]
        lines.append("| " + " | ".join(cells) + " |")
    markdown = "\n".join(lines) + "\n"

    out.with_name(out.name + ".md").write_text(markdown, encoding="utf-8")

    with open(out.with_name(out.name + ".csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["run"] + list(TABLE_COLUMNS))
        for name, metrics in rows:
            writer.writerow([name] + [metrics[key] for key in TABLE_COLUMNS])

    return markdown


def _read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
