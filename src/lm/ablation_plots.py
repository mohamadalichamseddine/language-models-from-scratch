"""Validate comparable ablation runs and plot mean/sample standard deviation."""

import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lm.ablation import CONTEXTS, SEEDS, read_json, write_json

METRICS = {
    "perplexity": "Validation perplexity",
    "cross_entropy_nats": "Cross-entropy (nats/token)",
    "top1_accuracy": "Top-1 accuracy",
}


def collect_runs(root, contexts=CONTEXTS, seeds=SEEDS, allow_partial=False):
    """Reject mixed settings/data/budgets; identify missing grid cells explicitly."""
    rows, missing = [], []
    reference = None
    for context in contexts:
        for seed in seeds:
            run = root / f"k{context}_seed{seed}"
            required = [run / name for name in ("config.json", "training.json", "metrics.json", "scoring.json")]
            if not all(path.exists() for path in required):
                missing.append(run.name)
                continue
            config, training, metrics, scoring = [read_json(path) for path in required]
            if "valid" not in metrics:
                missing.append(run.name)
                continue
            model_settings = config["model_config"].copy()
            training_settings = config["training"].copy()
            if model_settings.pop("context") != context or training_settings.pop("seed") != seed:
                raise ValueError(f"{run}: context or seed disagrees with its folder")
            if training_settings.get("early_stopping", True):
                raise ValueError(f"{run}: equal-budget runs must disable early stopping")
            budget = training_settings["max_steps"] * training_settings["tokens_per_batch"]
            if (training["tokens_seen"] != budget or training["stopped_early"]
                    or training["steps"] != training_settings["max_steps"] or config["token_budget"] != budget):
                raise ValueError(f"{run}: incomplete or inconsistent training budget")
            signature = scoring["signature"]
            if (signature["split"] != "valid" or signature["num_tokens"] != config["data"]["valid_tokens"]
                    or signature["data"] != config["data"] or signature["scoring_context"] != context
                    or signature["trained_context"] != context):
                raise ValueError(f"{run}: scoring metadata does not match the training experiment")
            comparable = (model_settings, training_settings, config["data"])
            if reference is not None and comparable != reference:
                raise ValueError(f"{run}: settings or data differ beyond context and seed")
            reference = comparable
            values = {key: float(metrics["valid"][key]) for key in METRICS}
            if not all(np.isfinite(value) for value in values.values()):
                raise ValueError(f"{run}: nonfinite metrics")
            rows.append({"context": context, "seed": seed, "tokens_seen": budget, **values})
    if missing and not allow_partial:
        raise ValueError(f"Missing runs: {', '.join(missing)}. Use --allow-partial for a labeled preview.")
    if not rows:
        raise ValueError("No completed, scored ablation runs found")
    return rows, missing


def summarize_runs(rows):
    """Macro-average seeds; sample SD (ddof=1), undefined for a single seed."""
    summaries = []
    for context in sorted({row["context"] for row in rows}):
        group = [row for row in rows if row["context"] == context]
        if len({row["seed"] for row in group}) != len(group):
            raise ValueError("Duplicate context/seed measurements")
        summary = {"context": context, "n_seeds": len(group), "seeds": [row["seed"] for row in group]}
        for metric in METRICS:
            values = [row[metric] for row in group]
            summary[f"{metric}_mean"] = float(np.mean(values))
            summary[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        summaries.append(summary)
    return summaries


def load_baselines(results_dir):
    best = read_json(results_dir / "ngram/tuning.json")["best_k"]
    baselines = []
    for n, name in enumerate(("unigram", "bigram", "trigram"), 1):
        run = results_dir / "ngram" / f"n{n}_k{best[name]:g}"
        metrics = read_json(run / "metrics.json")["valid"]
        unit = "token" if n == 2 else "tokens"
        baselines.append((f"{name} ({n - 1} context {unit})", metrics))
    interpolated = results_dir / "ngram/interpolated/metrics.json"
    if interpolated.exists():
        baselines.append(("interpolated trigram (2 context tokens)", read_json(interpolated)["valid"]))
    return baselines


def build_ablation_figures(root, results_dir, out, contexts=CONTEXTS, seeds=SEEDS, allow_partial=False):
    rows, missing = collect_runs(root, contexts, seeds, allow_partial)
    summaries = summarize_runs(rows)
    baselines = load_baselines(results_dir)
    out.mkdir(parents=True, exist_ok=True)
    status = {
        "complete_requested_grid": not missing,
        "contexts": list(contexts), "seeds": list(seeds), "missing_runs": missing,
        "at_least_two_seeds_per_context": all(row["n_seeds"] >= 2 for row in summaries) and not missing,
        "standard_deviation": "sample (ddof=1); undefined for one seed",
    }
    write_json(out / "ablation_status.json", status)
    write_json(out / "ablation_summary.json", summaries)
    for filename, records in (("ablation_runs.csv", rows), ("ablation_summary.csv", summaries)):
        with (out / filename).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    note = "Partial grid. " if missing else ""
    note += "Mean +/- sample SD across seeds; n/a means only one seed."
    lines = ["# Validation context ablation", "", note, "",
             "| Context | Seeds | Perplexity | CE (nats/token) | Top-1 |",
             "|---:|---:|---:|---:|---:|"]
    for row in summaries:
        cells = []
        for metric in METRICS:
            mean, std = row[f"{metric}_mean"], row[f"{metric}_std"]
            cells.append(f"{mean:.6g} +/- {std:.6g}" if std is not None else f"{mean:.6g} (SD n/a)")
        lines.append(f"| {row['context']} | {row['n_seeds']} | " + " | ".join(cells) + " |")
    if missing:
        lines += ["", "Missing runs: " + ", ".join(missing)]
    (out / "ablation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for metric, label in METRICS.items():
        fig, ax = plt.subplots(figsize=(8, 5))
        x = [row["context"] for row in summaries]
        means = [row[f"{metric}_mean"] for row in summaries]
        ax.plot(x, means, "o-", color="#2a78d6", label="Transformer (trained at each context)")
        multiple = [row for row in summaries if row["n_seeds"] > 1]
        if multiple:
            ax.errorbar([row["context"] for row in multiple], [row[f"{metric}_mean"] for row in multiple],
                        yerr=[row[f"{metric}_std"] for row in multiple], fmt="none", capsize=4, color="#2a78d6")
        for row in summaries:
            ax.annotate(f"n={row['n_seeds']}", (row["context"], row[f"{metric}_mean"]),
                        xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8)
        for (name, values), color in zip(baselines, ("#eb6834", "#1b8c5a", "#8b5cb6", "#777777")):
            ax.axhline(values[metric], linestyle="--", linewidth=1, color=color, label=name)
        ax.set_xticks(sorted(contexts))
        ax.set_xlabel("Training and scoring context (tokens)")
        ax.set_ylabel(label)
        if metric == "perplexity":
            ax.set_yscale("log")
        if metric == "top1_accuracy":
            from matplotlib.ticker import PercentFormatter
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_title("Validation context ablation" + (" (partial grid)" if missing else ""))
        ax.margins(y=0.12)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, framealpha=1.0)
        fig.tight_layout()
        fig.savefig(out / f"ablation_{metric}.png", dpi=150)
        plt.close(fig)
    return status
