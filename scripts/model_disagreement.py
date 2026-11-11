
"""T17: inspect the largest per-token log-prob differences between two runs.

Example:
 python scripts/model_disagreement.py results/transformer/k128/valid.npz results/ngram/n3_k0.001/valid.npz --top 20
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from lm.data import load_split, load_vocab
from lm.metrics import load_predictions


def disagreement_rows(a_file, b_file, ids, vocab, top=20, context=10):
    """Returns the biggest advantages of model A and B, ordered by margin.

    Positive margin = model A assigns larger log-probability to the gold token.
    """
    if top < 1 or context < 0:
        raise ValueError("top must be positive and context nonnegative")
    a, _ = load_predictions(a_file)
    b, _ = load_predictions(b_file)
    ids = np.asarray(ids)
    if len(a) != len(b) or len(a) != len(ids):
        raise ValueError("Prediction lengths must match each other and the token split")
    if np.any(np.isnan(a)) or np.any(np.isnan(b)):
        raise ValueError("NaN log probabilities are invalid")
    margin = a - b
    if np.any(np.isnan(margin)):
        raise ValueError("Undefined margin (-inf minus -inf); use smoothed models")
    rows = {}
    for label, indices in (("a_wins", np.argsort(-margin, kind="stable")[:top]),
                           ("b_wins", np.argsort(margin, kind="stable")[:top])):
        rows[label] = []
        for index in indices:
            i = int(index)
            rows[label].append({"position": i, "token": vocab.tokens[int(ids[i])],
                                "context_before": " ".join(vocab.decode(ids[max(0, i - context):i])),
                                "context_after": " ".join(vocab.decode(ids[i + 1:i + context + 1])),
                                "a_logprob": float(a[i]), "b_logprob": float(b[i]),
                                "a_minus_b": float(margin[i])})
    return rows


def main():
    parser = argparse.ArgumentParser(description="Compare two per-token score files, positive margin means A wins")
    parser.add_argument("a", type=Path, help="Transformer's valid.npz (model A)")
    parser.add_argument("b", type=Path, help="Trigram's valid.npz (model B)")
    parser.add_argument("--split", default="valid", choices=("train", "valid", "test"))
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--context", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None, help="Use first N validation tokens for aligned subset score files")
    parser.add_argument("--out", type=Path, help="optional CSV output")
    args = parser.parse_args()
    ids = load_split(args.split)
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        ids = ids[:args.limit]
    rows = disagreement_rows(args.a, args.b, ids, load_vocab(), args.top, args.context)
    for group, heading in (("a_wins", "MODEL A BEATS MODEL B"), ("b_wins", "MODEL B BEATS MODEL A")):
        print(f"\n{heading} (A - B log-prob):")
        for r in rows[group]:
            print(f"[{r['position']}] {r['a_minus_b']:+.4f}: ... {r['context_before']} [ {r['token']} ] {r['context_after']} ...")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["winner"] + list(rows["a_wins"][0]))
            writer.writeheader()
            for group in ("a_wins", "b_wins"):
                for row in rows[group]:
                    writer.writerow({"winner": group, **row})


if __name__ == "__main__":
    main()
