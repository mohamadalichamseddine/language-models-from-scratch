"""Basic statistics for the WikiText-2 splits.

Reads data/raw/wikitext-2/ and writes a Markdown summary (stats.md) and a
rank-frequency plot (zipf.png) to results/data_stats/. Tokens here are the
whitespace-separated words in each file; <eos> is not counted.
"""

import re
from collections import Counter

import matplotlib.pyplot as plt

from lm.data import RAW_DIR, ROOT, SPLITS

OUT_DIR = ROOT / "results/data_stats"


def read_lines(split):
    with open(RAW_DIR / f"wiki.{split}.tokens", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def count_articles(lines):
    # An article title is a line like " = Title = " with a blank line on each side.
    # The blank-line check matters: some table rows also start and end with "=".
    count = 0
    for i in range(1, len(lines) - 1):
        if lines[i - 1] == "" and lines[i + 1] == "" and re.fullmatch(r" = [^=]+ = ", lines[i]):
            count += 1
    return count


def split_table(lines, counts):
    train_vocab = set(counts["train"])
    rows = [
        "| Split | Lines | Articles | Tokens | Unique tokens | `<unk>` rate | Tokens not in train vocab |",
        "|---|---|---|---|---|---|---|",
    ]
    for split in SPLITS:
        split_counts = counts[split]
        n_tokens = sum(split_counts.values())
        n_articles = count_articles(lines[split])
        unk_rate = split_counts["<unk>"] / n_tokens
        n_unseen = sum(n for token, n in split_counts.items() if token not in train_vocab)
        rows.append(
            f"| {split} | {len(lines[split]):,} | {n_articles:,} | {n_tokens:,} "
            f"| {len(split_counts):,} | {unk_rate:.2%} | {n_unseen:,} |"
        )
    return rows


def top_tokens_table(train_counts, n=20):
    total = sum(train_counts.values())
    rows = ["| Rank | Token | Count | Share |", "|---|---|---|---|"]
    for rank, (token, count) in enumerate(train_counts.most_common(n), start=1):
        rows.append(f"| {rank} | `{token}` | {count:,} | {count / total:.2%} |")
    return rows


def plot_zipf(train_counts, path):
    frequencies = sorted(train_counts.values(), reverse=True)
    ranks = range(1, len(frequencies) + 1)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(ranks, frequencies, color="#2a78d6", linewidth=1.5)
    ax.set_title("WikiText-2 train tokens: frequency by rank")
    ax.set_xlabel("Rank (log scale)")
    ax.set_ylabel("Frequency (log scale)")
    ax.grid(True, color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    lines = {split: read_lines(split) for split in SPLITS}
    counts = {split: Counter(token for line in lines[split] for token in line.split()) for split in SPLITS}

    report = [
        "# WikiText-2 statistics",
        "",
        "Tokens are the whitespace-separated words in each file; `<eos>` is not counted.",
        "",
        *split_table(lines, counts),
        "",
        f"Train vocabulary: {len(counts['train']):,} distinct tokens.",
        "",
        "## Top 20 train tokens",
        "",
        *top_tokens_table(counts["train"]),
        "",
        "## Rank-frequency plot",
        "",
        "![Rank-frequency plot of train tokens](zipf.png)",
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "stats.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    plot_zipf(counts["train"], OUT_DIR / "zipf.png")

    print("\n".join(report))
    print(f"\nSaved stats.md and zipf.png to {OUT_DIR}")


if __name__ == "__main__":
    main()
