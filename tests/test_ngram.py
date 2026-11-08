from collections import Counter

import numpy as np
import pytest

from lm.ngram import InterpolatedNGram, NGramModel, tune_lambdas
from lm.metrics import load_predictions, compute_metrics

# Token 5 never appears in train, and (5, 5) is a history never seen.
TRAIN = [0, 1, 0, 2, 0, 1, 3, 4, 1, 0, 2, 2]
VOCAB_SIZE = 6
HISTORIES = [(), (0,), (1, 0), (0, 2), (5, 5)]


def mle_parts():
    return [NGramModel(n=n, vocab_size=VOCAB_SIZE, k=0).fit(TRAIN) for n in (1, 2, 3)]


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("k", [0, 1])
def test_probabilities_sum_to_one(n, k):
    model = NGramModel(n=n, vocab_size=4, k=k)
    model.fit([0, 1, 0, 2, 0, 1, 3])

    for history in [(), (0,), (0, 1), (3, 3)]:
        total = sum(model.probability(w, history) for w in range(4))
        assert total == pytest.approx(1.0)


def test_bigram_probabilities():
    model = NGramModel(n=2, vocab_size=3, k=1)
    model.fit([0, 1, 0, 1, 0, 2])

    assert model.probability(1, [0]) == pytest.approx(3 / 6)
    assert model.probability(2, [0]) == pytest.approx(2 / 6)
    assert model.probability(0, [0]) == pytest.approx(1 / 6)


def test_unseen_history():
    model = NGramModel(n=2, vocab_size=4, k=1)
    model.fit([0, 1, 0, 2])

    for token in range(4):
        assert model.probability(token, [3]) == pytest.approx(0.25)


def test_mle_zero_probability():
    model = NGramModel(n=2, vocab_size=4, k=0)
    model.fit([0, 1, 0, 2])

    assert model.probability(3, [0]) == 0.0


def test_rank_ties():
    model = NGramModel(n=2, vocab_size=4, k=1)
    model.fit([0, 1, 0, 2, 0, 1, 3])

    assert model.rank(1, [0]) == 0
    assert model.rank(2, [0]) == 1
    assert model.rank(0, [0]) == 2
    assert model.rank(3, [0]) == 3


def test_unseen_history_rank():
    model = NGramModel(n=2, vocab_size=4, k=1)
    model.fit([0, 1, 0, 2])

    assert model.rank(0, [3]) == 0
    assert model.rank(1, [3]) == 1


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("k", [0, 0.5])
def test_rank_matches_full_sort(n, k):
    unigrams = Counter(TRAIN)
    model = NGramModel(n=n, vocab_size=VOCAB_SIZE, k=k).fit(TRAIN)

    for history in HISTORIES:
        probs = [model.probability(w, history) for w in range(VOCAB_SIZE)]
        expected = sorted(range(VOCAB_SIZE), key=lambda w: (-probs[w], -unigrams[w], w))

        for rank, token in enumerate(expected):
            assert model.rank(token, history) == rank


def test_score_split(tmp_path):
    model = NGramModel(n=2, vocab_size=3, k=1)
    model.fit([0, 1, 0, 2])

    path = tmp_path / "bigram.npz"
    logprob, rank = model.score_split([0, 1, 2], path)

    saved_logprob, saved_rank = load_predictions(path)

    np.testing.assert_allclose(saved_logprob, logprob)
    np.testing.assert_array_equal(saved_rank, rank)

    metrics = compute_metrics(logprob, rank)

    assert len(logprob) == 3
    assert np.isfinite(metrics["perplexity"])


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("k", [0, 0.5])
def test_distribution_matches_probability(n, k):
    model = NGramModel(n=n, vocab_size=VOCAB_SIZE, k=k).fit(TRAIN)

    for history in HISTORIES:
        expected = [model.probability(w, history) for w in range(VOCAB_SIZE)]
        np.testing.assert_array_equal(model.distribution(history), expected)


def test_interpolated_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        InterpolatedNGram(mle_parts(), [0.5, 0.5, 0.5])


def test_interpolated_distribution_sums_to_one():
    model = InterpolatedNGram(mle_parts(), [0.2, 0.5, 0.3])

    for history in HISTORIES:
        assert model.distribution(history).sum() == pytest.approx(1.0)


def test_interpolated_with_one_part_matches_that_model():
    parts = mle_parts()
    bigram_only = InterpolatedNGram(parts, [0, 1, 0])
    ids = [0, 1, 0, 2, 5, 4, 1]

    logprob, rank = bigram_only.score_split(ids)
    expected_logprob, expected_rank = parts[1].score_split(ids)

    np.testing.assert_array_equal(logprob, expected_logprob)
    np.testing.assert_array_equal(rank, expected_rank)


def test_interpolated_rank_matches_full_sort():
    model = InterpolatedNGram(mle_parts(), [0.2, 0.5, 0.3])
    unigrams = Counter(TRAIN)
    ids = [0, 1, 0, 2, 5, 4, 1, 3]

    _, rank = model.score_split(ids)

    for i, token in enumerate(ids):
        probs = model.distribution(ids[max(0, i - 2):i])
        expected = sorted(range(VOCAB_SIZE), key=lambda w: (-probs[w], -unigrams[w], w))
        assert rank[i] == expected.index(token)


def test_tune_lambdas_picks_the_best_model():
    # The third model gives every token the highest probability, so it gets all the weight.
    part_probs = np.array([[0.1, 0.2, 0.9], [0.3, 0.1, 0.8]])

    lambdas, perplexity = tune_lambdas(part_probs, step=0.1)

    assert lambdas == pytest.approx([0, 0, 1])
    assert perplexity == pytest.approx(np.exp(-np.mean(np.log([0.9, 0.8]))))


def test_tune_lambdas_mixes_models():
    # Each of the first two models is sure about a different token, so an even mix wins.
    part_probs = np.array([[0.9, 0.1, 0.0], [0.1, 0.9, 0.0]])

    lambdas, _ = tune_lambdas(part_probs, step=0.1)

    assert lambdas == pytest.approx([0.5, 0.5, 0])
