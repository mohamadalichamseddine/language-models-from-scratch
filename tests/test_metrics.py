
import numpy as np
import pytest

from lm.metrics import (
    save_predictions,
    load_predictions,
    compute_metrics,
    metrics_from_file,
)


def test_save_and_load(tmp_path):
    path = tmp_path / "predictions.npz"
    logprob = np.log([0.5, 0.25, 0.125])
    rank = np.array([0, 2, 7])

    save_predictions(path, logprob, rank)

    loaded_logprob, loaded_rank = load_predictions(path)

    np.testing.assert_allclose(loaded_logprob, logprob)
    np.testing.assert_array_equal(loaded_rank, rank)


def test_cross_entropy_and_perplexity():
    logprob = np.log([0.5, 0.25])
    rank = np.array([0, 1])

    metrics = compute_metrics(logprob, rank)

    assert metrics["cross_entropy_nats"] == pytest.approx(1.5 * np.log(2))
    assert metrics["cross_entropy_bits"] == pytest.approx(1.5)
    assert metrics["perplexity"] == pytest.approx(2 ** 1.5)


def test_topk_accuracy():
    rank = np.array([0, 2, 7, 11])
    logprob = np.log([0.5, 0.25, 0.1, 0.01])

    metrics = compute_metrics(logprob, rank)

    assert metrics["top1_accuracy"] == pytest.approx(0.25)
    assert metrics["top5_accuracy"] == pytest.approx(0.50)
    assert metrics["top10_accuracy"] == pytest.approx(0.75)


def test_metrics_from_file(tmp_path):
    path = tmp_path / "test.npz"

    save_predictions(path, np.log([0.5, 0.5]), [0, 0])

    metrics = metrics_from_file(path)

    assert metrics["perplexity"] == pytest.approx(2.0)
    assert metrics["top1_accuracy"] == pytest.approx(1.0)


def test_zero_probability():
    metrics = compute_metrics(
        [np.log(0.5), -np.inf],
        [0, 9],
    )

    assert np.isinf(metrics["cross_entropy_nats"])
    assert np.isinf(metrics["perplexity"])


def test_empty_predictions():
    with pytest.raises(ValueError):
        compute_metrics([], [])


def test_mismatched_lengths(tmp_path):
    with pytest.raises(ValueError):
        save_predictions(tmp_path / "bad.npz", [-1.0, -2.0], [0])
