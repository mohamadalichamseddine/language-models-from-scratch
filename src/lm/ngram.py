"""Unigram, bigram and trigram language models with add-k smoothing.

P(w | h) = (c(h, w) + k) / (c(h) + k * V), where h is the previous n - 1 tokens
and V is the vocabulary size. k = 0 gives the MLE model, and a history never
seen in train gives a uniform distribution.
"""

from bisect import bisect_left
from collections import Counter, defaultdict

import numpy as np

from lm.metrics import save_predictions


class NGramModel:
    """Counts n-grams on the train ids and predicts the next token from them."""

    def __init__(self, n, vocab_size, k=1.0):
        if n not in (1, 2, 3):
            raise ValueError("n must be 1, 2, or 3")
        if vocab_size < 1 or k < 0:
            raise ValueError("Invalid vocabulary size or k")

        self.n = n
        self.vocab_size = vocab_size
        self.k = k
        self.counts = defaultdict(Counter)
        self.totals = Counter()
        self.unigrams = Counter()
        self.priority = None
        self._rank_cache = {}

    def fit(self, train_ids):
        """Count every n-gram in the train ids and return the model."""
        ids = np.asarray(train_ids, dtype=np.int64)

        if ids.ndim != 1 or len(ids) == 0:
            raise ValueError("Training data must be a nonempty 1D array")
        if np.any(ids < 0) or np.any(ids >= self.vocab_size):
            raise ValueError("Token ID outside vocabulary")

        self.counts.clear()
        self.totals.clear()
        self.unigrams = Counter(ids.tolist())
        self._rank_cache.clear()

        for i, token in enumerate(ids):
            history = tuple(ids[max(0, i - self.n + 1):i])
            token = int(token)
            self.counts[history][token] += 1
            self.totals[history] += 1

        # priority[token] is the token's place in unigram order (most frequent
        # first). It breaks ties when ranking.
        order = sorted(
            range(self.vocab_size),
            key=lambda token: (-self.unigrams[token], token),
        )

        self.priority = np.empty(self.vocab_size, dtype=np.int64)
        self.priority[order] = np.arange(self.vocab_size)
        return self

    def _history(self, history):
        """Return the last n - 1 tokens of a history as a tuple of ints."""
        return tuple(int(x) for x in history[-(self.n - 1):]) if self.n > 1 else ()

    def probability(self, token, history=()):
        """Return P(token | history) with add-k smoothing."""
        if self.priority is None:
            raise RuntimeError("Model has not been fitted")

        token = int(token)
        if not 0 <= token < self.vocab_size:
            raise ValueError("Token ID outside vocabulary")

        history = self._history(history)
        total = self.totals[history]

        if total == 0:
            return 1.0 / self.vocab_size

        count = self.counts[history][token]
        return (count + self.k) / (total + self.k * self.vocab_size)

    def _rank_info(self, history):
        """Return how the tokens seen after this history are ranked.

        The result is (ranks, positions, observed): the rank of each seen token,
        the sorted unigram positions of the seen tokens, and how many there are.
        It is None for a history never seen in train. Results are cached per
        history, because scoring asks for the same histories again and again.
        """
        if history in self._rank_cache:
            return self._rank_cache[history]

        counts = self.counts.get(history)
        info = None

        if counts:
            ordered = sorted(
                counts,
                key=lambda token: (-counts[token], self.priority[token]),
            )
            ranks = {token: i for i, token in enumerate(ordered)}
            positions = sorted(int(self.priority[token]) for token in counts)
            info = (ranks, positions, len(counts))

        self._rank_cache[history] = info
        return info

    def rank(self, token, history=()):
        """Return the zero-based rank of token in P(. | history).

        Tokens are ordered by probability, and ties are broken by unigram
        frequency in train. Tokens seen after the history come first; all the
        unseen ones share the same probability, so they follow in unigram order.
        """
        if self.priority is None:
            raise RuntimeError("Model has not been fitted")

        token = int(token)
        if not 0 <= token < self.vocab_size:
            raise ValueError("Token ID outside vocabulary")

        history = self._history(history)
        info = self._rank_info(history)

        # Unseen history: the distribution is uniform, so only unigram order counts.
        if info is None:
            return int(self.priority[token])

        ranks, positions, observed = info

        if token in ranks:
            return ranks[token]

        # Its place among the unseen tokens is its unigram position minus the
        # seen tokens that come before it in unigram order.
        position = int(self.priority[token])
        return observed + position - bisect_left(positions, position)

    def score_split(self, ids, path=None):
        """Score every token of a split and return (logprob, rank).

        Each token is predicted from the previous n - 1 tokens of the split
        (fewer at the very start). If path is given, the predictions are also
        saved in the lm.metrics format.
        """
        if self.priority is None:
            raise RuntimeError("Model has not been fitted")

        ids = np.asarray(ids, dtype=np.int64)
        if ids.ndim != 1:
            raise ValueError("Split must be a 1D array")
        if np.any(ids < 0) or np.any(ids >= self.vocab_size):
            raise ValueError("Token ID outside vocabulary")

        logprob = np.empty(len(ids), dtype=np.float64)
        rank = np.empty(len(ids), dtype=np.int64)

        for i, token in enumerate(ids):
            history = ids[max(0, i - self.n + 1):i]
            p = self.probability(token, history)

            logprob[i] = np.log(p) if p > 0 else -np.inf
            rank[i] = self.rank(token, history)

        if path is not None:
            save_predictions(path, logprob, rank)

        return logprob, rank
