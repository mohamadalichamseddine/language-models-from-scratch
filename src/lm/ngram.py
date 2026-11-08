"""Unigram, bigram and trigram language models with add-k smoothing.

P(w | h) = (c(h, w) + k) / (c(h) + k * V), where h is the previous n - 1 tokens
and V is the vocabulary size. k = 0 gives the MLE model, and a history never
seen in train gives a uniform distribution.

InterpolatedNGram mixes a unigram, a bigram and a trigram model with weights
that sum to 1, and tune_lambdas picks those weights on held-out tokens.
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
        self._followers_cache = {}

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
        self._followers_cache.clear()

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

    def distribution(self, history=()):
        """Return P(w | history) for every token w, as one array over the vocabulary."""
        if self.priority is None:
            raise RuntimeError("Model has not been fitted")

        history = self._history(history)
        total = self.totals[history]

        if total == 0:
            return np.full(self.vocab_size, 1.0 / self.vocab_size)

        tokens, counts = self._followers(history)
        denominator = total + self.k * self.vocab_size
        probs = np.full(self.vocab_size, self.k / denominator)
        probs[tokens] = (counts + self.k) / denominator
        return probs

    def _followers(self, history):
        """Return the tokens seen after a history and their counts, as arrays.

        Cached per history, because scoring asks for the same histories often.
        """
        if history not in self._followers_cache:
            counts = self.counts[history]
            tokens = np.fromiter(counts.keys(), dtype=np.int64, count=len(counts))
            values = np.fromiter(counts.values(), dtype=np.float64, count=len(counts))
            self._followers_cache[history] = (tokens, values)
        return self._followers_cache[history]

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


class InterpolatedNGram:
    """Mixes n-gram models: P(w | h) = l1 * P_uni(w) + l2 * P_bi(w | h) + l3 * P_tri(w | h)."""

    def __init__(self, models, lambdas):
        if len(models) != len(lambdas) or min(lambdas) < 0 or not np.isclose(sum(lambdas), 1):
            raise ValueError("Need one nonnegative weight per model, and the weights must sum to 1")

        self.models = models
        self.lambdas = lambdas
        self.context = max(model.n for model in models) - 1
        # All models are fitted on the same train ids, so they share the unigram
        # order that breaks rank ties.
        self.priority = models[0].priority

    def distribution(self, history=()):
        """Return the mixed P(w | history) for every token w."""
        return sum(weight * model.distribution(history) for weight, model in zip(self.lambdas, self.models))

    def score_split(self, ids, path=None):
        """Score every token of a split and return (logprob, rank).

        Works like NGramModel.score_split: ranks are taken in the mixed
        distribution, and ties are broken by unigram frequency in train.
        """
        ids = np.asarray(ids, dtype=np.int64)
        logprob = np.empty(len(ids), dtype=np.float64)
        rank = np.empty(len(ids), dtype=np.int64)

        for i, token in enumerate(ids):
            probs = self.distribution(ids[max(0, i - self.context):i])
            p = probs[token]
            logprob[i] = np.log(p) if p > 0 else -np.inf

            # Tokens ahead of the target: higher probability, or the same
            # probability and earlier in unigram order.
            ties_before = (probs == p) & (self.priority < self.priority[token])
            rank[i] = np.count_nonzero(probs > p) + np.count_nonzero(ties_before)

        if path is not None:
            save_predictions(path, logprob, rank)

        return logprob, rank


def tune_lambdas(part_probs, step=0.01):
    """Pick interpolation weights for three models by perplexity on held-out tokens.

    part_probs has one row per token and one column per model: the probability
    each model gives that token. Every weight triple on a grid with this step
    (summing to 1) is tried, and the one with the lowest perplexity wins.
    Returns (lambdas, perplexity).
    """
    steps = round(1 / step)
    best_lambdas, best_perplexity = None, np.inf

    for i in range(steps + 1):
        for j in range(steps + 1 - i):
            lambdas = np.array([i, j, steps - i - j]) / steps
            with np.errstate(divide="ignore"):
                perplexity = float(np.exp(-np.mean(np.log(part_probs @ lambdas))))
            if perplexity < best_perplexity:
                best_lambdas, best_perplexity = lambdas, perplexity

    if best_lambdas is None:
        raise ValueError("Every choice of weights gives some token zero probability")
    return [float(weight) for weight in best_lambdas], best_perplexity
