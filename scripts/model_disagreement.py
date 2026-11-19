"""T17: print the tokens where two models' log-probs differ most.

Both prediction files must be on the full split. For the Transformer vs the
trigram on valid, first build them with:
    python scripts/score_transformer.py --run transformer/best
    python scripts/run_ngram.py
then run:
    python scripts/model_disagreement.py results/transformer/best/valid.npz \
        results/ngram/n3_k0.001/valid.npz --out results/figures/disagreements.csv
"""

import argparse
import csv
import sys
from pathlib import Path

from lm.data import SPLITS, load_split, load_vocab
from lm.disagreement import top_disagreements


def main():
    parser = argparse.ArgumentParser(description="Compare two per-token prediction files; a positive margin means A wins.")
    parser.add_argument("a", type=Path, help="predictions of model A, e.g. the Transformer's valid.npz")
    parser.add_argument("b", type=Path, help="predictions of model B, e.g. the trigram's valid.npz")
    parser.add_argument("--split", default="valid", choices=SPLITS)
    parser.add_argument("--top", type=int, default=20, help="tokens to list for each model")
    parser.add_argument("--context", type=int, default=10, help="tokens of context shown on each side")
    parser.add_argument("--out", type=Path, help="optional CSV output")
    args = parser.parse_args()

    # WikiText has characters (e.g. non-breaking hyphens) that Windows' default
    # encoding can't print when the output goes to a file or a pipe.
    sys.stdout.reconfigure(encoding="utf-8")

    rows = top_disagreements(args.a, args.b, load_split(args.split), load_vocab(), args.top, args.context)
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
