"""Score add-k n-gram models and tune k on the validation split.

For n = 1, 2, 3 and every k in K_GRID, the model is scored on valid in its own
run folder, results/ngram/n{n}_k{k}/, with its config, predictions and
metrics. The best k for each n is the one with the lowest validation
perplexity. The script also measures the share of validation tokens that get
zero probability without smoothing (k = 0, the MLE model).

Writes results/ngram/tuning.json, tuning.md and perplexity_vs_k.png. The
comparison table across all runs then comes from:
    python scripts/evaluate.py results/ngram/* --out results/ngram/comparison
"""

import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from lm.data import load_split, load_vocab
from lm.evaluation import RESULTS_DIR, evaluate_run, predictions_path, save_config
from lm.ngram import NGramModel

NGRAM_DIR = RESULTS_DIR / "ngram"
NAMES = {1: "unigram", 2: "bigram", 3: "trigram"}
K_GRID = [1, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001]

# First three slots of the project's categorical palette, checked for colorblind safety.
COLORS = {"unigram": "#2a78d6", "bigram": "#eb6834", "trigram": "#1baf7a"}
TEXT_COLOR = "#52514e"


def score_run(n, k, train_ids, valid_ids, vocab_size):
    """Score one add-k model on valid in its own run folder and return its metrics."""
    run_dir = NGRAM_DIR / f"n{n}_k{k}"
    save_config(run_dir, {
        "model": "ngram",
        "n": n,
        "k": k,
        "train_split": "train",
        "vocab_size": vocab_size,
    })

    model = NGramModel(n=n, vocab_size=vocab_size, k=k).fit(train_ids)
    model.score_split(valid_ids, predictions_path(run_dir, "valid"))
    return evaluate_run(run_dir, "valid", len(valid_ids))


def zero_probability_share(n, train_ids, valid_ids, vocab_size):
    """Return the share of valid tokens that get probability 0 when k = 0."""
    model = NGramModel(n=n, vocab_size=vocab_size, k=0).fit(train_ids)
    logprob, _ = model.score_split(valid_ids)
    return float(np.mean(np.isneginf(logprob)))


def write_report(tuning, path):
    names = list(NAMES.values())
    perplexity = tuning["valid_perplexity"]

    lines = [
        "# Add-k tuning on the validation split",
        "",
        "Validation perplexity for each k. The best k for each model is in bold.",
        "",
        "| k | " + " | ".join(name.capitalize() for name in names) + " |",
        "|---|" + "---|" * len(names),
    ]
    for i, k in enumerate(K_GRID):
        cells = []
        for name in names:
            cell = f"{perplexity[name][i]:.2f}"
            cells.append(f"**{cell}**" if k == tuning["best_k"][name] else cell)
        lines.append(f"| {k:g} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "| Model | Best k | Validation perplexity | Tokens with zero probability at k = 0 |",
        "|---|---|---|---|",
    ]
    for name in names:
        lines.append(
            f"| {name.capitalize()} | {tuning['best_k'][name]:g} | {min(perplexity[name]):.2f} "
            f"| {tuning['zero_probability_share'][name]:.2%} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_perplexity(tuning, path):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    for name, color in COLORS.items():
        values = tuning["valid_perplexity"][name]
        ax.plot(K_GRID, values, color=color, linewidth=1.5, marker="o", markersize=4,
                markeredgecolor="white", markeredgewidth=0.8, label=name.capitalize())

        # Mark and label the best k. The unigram line is flat and its best k sits
        # at the left edge, so its label goes above the line instead of below.
        best_k = tuning["best_k"][name]
        best = min(values)
        ax.plot(best_k, best, marker="o", markersize=8, color=color,
                markeredgecolor="white", markeredgewidth=1.2)
        if name == "unigram":
            offset, ha, va = (4, 8), "left", "bottom"
        else:
            offset, ha, va = (0, -11), "center", "top"
        ax.annotate(f"best k = {best_k:g} ({best:,.0f})", (best_k, best), xytext=offset,
                    textcoords="offset points", ha=ha, va=va, fontsize=8, color=TEXT_COLOR)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    ax.yaxis.set_major_locator(FixedLocator([300, 500, 1000, 2000, 5000, 10000]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y / 1000:g}K" if y >= 1000 else f"{y:g}"))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_ylim(300, 20000)

    ax.set_title("Validation perplexity by add-k smoothing")
    ax.set_xlabel("k (log scale)")
    ax.set_ylabel("Validation perplexity (log scale)")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(True, color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    train_ids = load_split("train")
    valid_ids = load_split("valid")
    vocab_size = len(load_vocab())

    perplexity, best_k, zero_share = {}, {}, {}
    for n, name in NAMES.items():
        perplexity[name] = []
        for k in K_GRID:
            print(f"Scoring the {name} model with k = {k}...", flush=True)
            metrics = score_run(n, k, train_ids, valid_ids, vocab_size)
            perplexity[name].append(metrics["perplexity"])

        best_k[name] = K_GRID[int(np.argmin(perplexity[name]))]
        zero_share[name] = zero_probability_share(n, train_ids, valid_ids, vocab_size)
        print(f"Best k for the {name} model: {best_k[name]}", flush=True)

    tuning = {
        "k_grid": K_GRID,
        "valid_perplexity": perplexity,
        "best_k": best_k,
        "zero_probability_share": zero_share,
    }
    with open(NGRAM_DIR / "tuning.json", "w", encoding="utf-8") as f:
        json.dump(tuning, f, indent=2)
        f.write("\n")

    write_report(tuning, NGRAM_DIR / "tuning.md")
    plot_perplexity(tuning, NGRAM_DIR / "perplexity_vs_k.png")
    print(f"Tuning results saved under {NGRAM_DIR}")


if __name__ == "__main__":
    main()
