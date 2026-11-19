"""T22: how well each model predicts rare words (RQ3).

Every target token of the test split goes into a group by how often its word
occurs in train: 1-10, 11-100, 101-1k or more than 1k times. WikiText-2
already replaced the words seen fewer than 3 times with <unk>, so in practice
the rarest group starts at 3. <unk> stands for all those words at once, so it
is reported on its own. For every model and group, the table gives the
cross-entropy (nats per token) and the top-1 accuracy. Transformer rows
average their two seeds.
"""

import csv

import numpy as np

from lm.final_evaluation import load_test_predictions, model_rows

FREQUENCY_GROUPS = {"1-10": (1, 10), "11-100": (11, 100), "101-1k": (101, 1000), ">1k": (1001, np.inf)}
UNK_GROUP = "<unk>"
GROUPS = [*FREQUENCY_GROUPS, UNK_GROUP]
METRICS = {"cross_entropy_nats": "Cross-entropy (nats per token)", "top1_accuracy": "Top-1 accuracy"}


def frequency_groups(train_ids, target_ids, vocab_size, unk_id):
    """Return the group of every target token, from how often it occurs in train."""
    counts = np.bincount(train_ids, minlength=vocab_size)[target_ids]
    groups = np.empty(len(target_ids), dtype=object)
    for name, (low, high) in FREQUENCY_GROUPS.items():
        groups[(counts >= low) & (counts <= high)] = name
    groups[np.asarray(target_ids) == unk_id] = UNK_GROUP
    return groups


def group_metrics(logprob, rank, groups):
    """CE (nats) and top-1 accuracy of the targets in each group."""
    return {group: {"cross_entropy_nats": float(-logprob[groups == group].mean()),
                    "top1_accuracy": float(np.mean(rank[groups == group] == 0))}
            for group in GROUPS}


def rare_token_table(results_dir, groups):
    """One row per model, with its CE and top-1 accuracy in every group (averaged over seeds)."""
    rows = []
    for name, run_dirs in model_rows(results_dir):
        runs = [group_metrics(*load_test_predictions(run_dir), groups) for run_dir in run_dirs]
        row = {"model": name}
        for metric in METRICS:
            for group in GROUPS:
                row[f"{metric} {group}"] = float(np.mean([run[group][metric] for run in runs]))
        rows.append(row)
    return rows


def write_rare_token_table(rows, groups, out):
    """Write the table as out/rare_tokens.md and .csv. Returns the Markdown text."""
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "rare_tokens.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Markdown hides <unk> as an unknown HTML tag unless it is in backticks.
    header = "| Model | " + " | ".join(f"`{group}`" if group == UNK_GROUP else group for group in GROUPS) + " |"
    rule = "|---|" + "---:|" * len(GROUPS)
    sizes = [f"{np.sum(groups == group):,} ({np.mean(groups == group):.1%})" for group in GROUPS]
    lines = [
        "# Rare-token analysis (test split)",
        "",
        "Test targets grouped by how often the word occurs in train. WikiText-2 already replaced the words",
        "seen fewer than 3 times with `<unk>`, so the rarest group starts at 3. Transformer rows average two seeds.",
        "",
        header.replace("Model", "Group"),
        rule,
        "| Test tokens | " + " | ".join(sizes) + " |",
    ]
    for metric, title in METRICS.items():
        lines += ["", f"## {title}", "", header, rule]
        for row in rows:
            values = [row[f"{metric} {group}"] for group in GROUPS]
            cells = [f"{value:.2%}" if metric == "top1_accuracy" else f"{value:.3f}" for value in values]
            lines.append(f"| {row['model']} | " + " | ".join(cells) + " |")

    markdown = "\n".join(lines) + "\n"
    (out / "rare_tokens.md").write_text(markdown, encoding="utf-8")
    return markdown
