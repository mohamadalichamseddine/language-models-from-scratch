"""T23: where a longer context helps (RQ2, RQ3).

Compares the Transformers trained with 128 and with 6 tokens of context on
every test token. A token's gain is its loss (negative log-prob, nats) with
k=6 minus its loss with k=128, each averaged over the two seeds, so a positive
gain means the longer context helped. Tokens are grouped in two ways:

- by where the target word last appeared before it: within the previous 6
  tokens (both models can see it), only 7-128 tokens back (only the k=128
  model can see it: copy-like cases, such as a name repeated in an article),
  or not in the previous 128 tokens at all;
- by word class: function words (a fixed list of closed-class English words),
  content words (all other words), <unk>, and punctuation, numbers and <eos>.
"""

import csv

import numpy as np

from lm.ablation import SEEDS
from lm.data import EOS, UNK
from lm.final_evaluation import ablation_dir, load_test_predictions

SHORT, LONG = 6, 128
CONTEXT_GROUPS = ["within previous 6", "7-128 back only", "not in previous 128"]
WORD_CLASSES = ["function word", "content word", "<unk>", "punctuation, numbers, <eos>"]

# Closed-class English words: articles and determiners, pronouns, prepositions,
# conjunctions, auxiliary and modal verbs, a few particles, and the clitics
# that WikiText splits off ('s, n't, ...).
FUNCTION_WORDS = set("""
a an the this that these those each every either neither some any no all both another such
what which whose much many more most few several other
i me my mine myself you your yours yourself yourselves he him his himself she her hers herself
it its itself we us our ours ourselves they them their theirs themselves who whom whoever
whatever whichever
about above across after against along among around as at before behind below beneath beside
besides between beyond by despite down during except for from in inside into like near of off
on onto out outside over past since through throughout to toward towards under underneath until
up upon via with within without
and but or nor so yet because although though while whereas if unless whether than when where
whereby once
be am is are was were been being have has had having do does did doing will would shall should
can could may might must ought
not also only then there here very too just even how why ever never again still
's ' n't 're 've 'll 'd 'm
""".split())


def distance_to_previous(ids):
    """How many tokens back each token last appeared (inf if it did not appear before)."""
    last_seen = {}
    distance = np.full(len(ids), np.inf)
    for i, token in enumerate(np.asarray(ids).tolist()):
        if token in last_seen:
            distance[i] = i - last_seen[token]
        last_seen[token] = i
    return distance


def context_groups(ids):
    """The context group of every token, from how far back its word last appeared."""
    distance = distance_to_previous(ids)
    groups = np.full(len(ids), CONTEXT_GROUPS[2], dtype=object)
    groups[distance <= LONG] = CONTEXT_GROUPS[1]
    groups[distance <= SHORT] = CONTEXT_GROUPS[0]
    return groups


def word_class(token):
    if token == UNK:
        return WORD_CLASSES[2]
    if token == EOS or not any(char.isalpha() for char in token):
        return WORD_CLASSES[3]
    return WORD_CLASSES[0] if token.lower() in FUNCTION_WORDS else WORD_CLASSES[1]


def word_classes(ids, vocab):
    """The word class of every token."""
    classes = np.array([word_class(token) for token in vocab.tokens], dtype=object)
    return classes[np.asarray(ids)]


def mean_test_loss(results_dir, context):
    """Per-token test loss (nats) of the Transformer trained at this context, averaged over seeds."""
    return np.mean([-load_test_predictions(ablation_dir(results_dir, context, seed))[0] for seed in SEEDS], axis=0)


def group_rows(groups, names, loss_short, loss_long):
    """One row per group: its size, both losses, the gain and its share of the total gain."""
    gain = loss_short - loss_long
    rows = []
    for name in names:
        member = groups == name
        rows.append({
            "group": name,
            "tokens": int(member.sum()),
            "share": float(member.mean()),
            f"loss_k{SHORT}": float(loss_short[member].mean()),
            f"loss_k{LONG}": float(loss_long[member].mean()),
            "gain": float(gain[member].mean()),
            f"k{LONG}_better": float(np.mean(gain[member] > 0)),
            "share_of_total_gain": float(gain[member].sum() / gain.sum()),
        })
    return rows


def long_context_tables(results_dir, ids, vocab):
    """The rows of the three tables: by context group, by word class, and by both."""
    loss_short, loss_long = mean_test_loss(results_dir, SHORT), mean_test_loss(results_dir, LONG)
    contexts, classes = context_groups(ids), word_classes(ids, vocab)
    both = np.array([f"{word} / {context}" for word, context in zip(classes, contexts)], dtype=object)
    both_names = [f"{word} / {context}" for word in WORD_CLASSES for context in CONTEXT_GROUPS]
    return {
        "context": group_rows(contexts, CONTEXT_GROUPS, loss_short, loss_long),
        "word class": group_rows(classes, WORD_CLASSES, loss_short, loss_long),
        "word class and context": group_rows(both, both_names, loss_short, loss_long),
    }


def write_long_context_tables(tables, out):
    """Write the tables as out/long_context.md and .csv. Returns the Markdown text."""
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "long_context.csv", "w", newline="", encoding="utf-8") as f:
        fields = ["grouping"] + list(tables["context"][0])
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for grouping, rows in tables.items():
            writer.writerows({"grouping": grouping, **row} for row in rows)

    total_gain = sum(row["gain"] * row["tokens"] for row in tables["context"]) / sum(
        row["tokens"] for row in tables["context"])
    lines = [
        f"# Where a longer context helps: Transformer k={LONG} vs k={SHORT} (test split)",
        "",
        f"Gain = loss with k={SHORT} minus loss with k={LONG} (nats per token, each averaged over two seeds);",
        f"positive means the longer context helps. Over all test tokens the gain is {total_gain:.4f} nats.",
        f"\"7-128 back only\" are the copy-like cases: the target word appeared in the previous {LONG} tokens",
        f"but not in the previous {SHORT}, so only the k={LONG} model could see it.",
    ]
    titles = {"context": "By where the target word last appeared", "word class": "By word class",
              "word class and context": "By word class and where the word last appeared"}
    for grouping, rows in tables.items():
        lines += [
            "", f"## {titles[grouping]}", "",
            f"| Group | Tokens | Share | Loss k={SHORT} | Loss k={LONG} | Gain | k={LONG} better | Share of total gain |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            # Markdown hides <unk> and <eos> as unknown HTML tags unless they are in backticks.
            group = row["group"].replace(UNK, f"`{UNK}`").replace(EOS, f"`{EOS}`")
            lines.append(
                f"| {group} | {row['tokens']:,} | {row['share']:.1%} | {row[f'loss_k{SHORT}']:.3f} "
                f"| {row[f'loss_k{LONG}']:.3f} | {row['gain']:+.3f} | {row[f'k{LONG}_better']:.1%} "
                f"| {row['share_of_total_gain']:.1%} |"
            )

    markdown = "\n".join(lines) + "\n"
    (out / "long_context.md").write_text(markdown, encoding="utf-8")
    return markdown
