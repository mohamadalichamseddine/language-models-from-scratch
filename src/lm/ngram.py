
from bisect import bisect_left
from collections import Counter, defaultdict
from functools import lru_cache

import numpy as np

from lm.metrics import save_predictions


class NGramModel:
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

    def fit(self, train_ids):
        ids = np.asarray(train_ids, dtype=np.int64)

        if ids.ndim != 1 or len(ids) == 0:
            raise ValueError("Training data must be a nonempty 1D array")
        if np.any(ids < 0) or np.any(ids >= self.vocab_size):
            raise ValueError("Token ID outside vocabulary")

        self.counts.clear()
        self.totals.clear()
        self.unigrams = Counter(ids.tolist())
        self._rank_info.cache_clear()

        for i, token in enumerate(ids):
            history = tuple(ids[max(0, i - self.n + 1):i])
            token = int(token)
            self.counts[history][token] += 1
            self.totals[history] += 1

        order = sorted(
            range(self.vocab_size),
            key=lambda token: (-self.unigrams[token], token),
        )

        self.priority = np.empty(self.vocab_size, dtype=np.int64)
        self.priority[order] = np.arange(self.vocab_size)
        return self

    def _history(self, history):
        return tuple(int(x) for x in history[-(self.n - 1):]) if self.n > 1 else ()

    def probability(self, token, history=()):
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

    @lru_cache(maxsize=2048)
    def _rank_info(self, history):
        counts = self.counts.get(history)

        if not counts:
            return None

        ordered = sorted(
            counts,
            key=lambda token: (-counts[token], self.priority[token]),
        )

        ranks = {token: i for i, token in enumerate(ordered)}
        positions = sorted(int(self.priority[token]) for token in counts)

        return ranks, positions, len(counts)

    def rank(self, token, history=()):
        if self.priority is None:
            raise RuntimeError("Model has not been fitted")

        token = int(token)
        if not 0 <= token < self.vocab_size:
            raise ValueError("Token ID outside vocabulary")

        history = self._history(history)
        info = self._rank_info(history)

        if info is None or self.k == 0 and self.totals[history] == 0:
            return int(self.priority[token])

        ranks, positions, observed = info

        if token in ranks:
            return ranks[token]

        position = int(self.priority[token])
        return observed + position - bisect_left(positions, position)

    def score_split(self, ids, path=None):
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
