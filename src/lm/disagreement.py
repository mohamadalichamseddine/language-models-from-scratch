"""Find the tokens where two models disagree most (T17).

Both models' predictions on the same split are read from their lm.metrics
files. A token's margin is model A's log-prob of the target minus model B's,
so a large positive margin means A predicted that token much better than B.
"""

import numpy as np

from lm.metrics import load_predictions


def top_disagreements(a_path, b_path, ids, vocab, top=20, context=10, mask=None):
    """Return the tokens where A beats B most ("a_wins") and where B beats A most ("b_wins").

    Each list is sorted by how much the winner wins. A row holds the token's
    position in the split, the token, `context` tokens on each side, both
    log-probs and the margin (A minus B). If mask is given, only the
    positions where it is True are considered.
    """
    if top < 1 or context < 0:
        raise ValueError("top must be positive and context nonnegative")
    a, _ = load_predictions(a_path)
    b, _ = load_predictions(b_path)
    ids = np.asarray(ids)
    if not len(a) == len(b) == len(ids):
        raise ValueError("Both prediction files must cover every token of the split")
    margin = a - b
    if np.any(np.isnan(margin)):
        raise ValueError("Undefined margin: a log-prob is NaN, or both models give a token zero probability")

    def row(i):
        return {
            "position": i,
            "token": vocab.tokens[ids[i]],
            "context_before": " ".join(vocab.decode(ids[max(0, i - context):i])),
            "context_after": " ".join(vocab.decode(ids[i + 1:i + 1 + context])),
            "a_logprob": float(a[i]),
            "b_logprob": float(b[i]),
            "a_minus_b": float(margin[i]),
        }

    positions = np.arange(len(ids)) if mask is None else np.flatnonzero(mask)
    a_wins = positions[np.argsort(-margin[positions], kind="stable")[:top]]
    b_wins = positions[np.argsort(margin[positions], kind="stable")[:top]]
    return {
        "a_wins": [row(int(i)) for i in a_wins],
        "b_wins": [row(int(i)) for i in b_wins],
    }


def format_row(row):
    """One line for printing: position, margin, and the token in brackets inside its context."""
    return (f"[{row['position']}] {row['a_minus_b']:+.4f}: ... {row['context_before']} "
            f"[ {row['token']} ] {row['context_after']} ...")
