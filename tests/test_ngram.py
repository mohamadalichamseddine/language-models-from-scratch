
import numpy as np
import pytest

from lm.ngram import NGramModel
from lm.metrics import load_predictions, compute_metrics


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
