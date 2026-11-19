import numpy as np
import pytest

from lm.evaluation import predictions_path
from lm.final_evaluation import BLOCK_TOKENS, block_totals, bootstrap_resamples, resampled_metrics, summarize_row
from lm.metrics import compute_metrics, save_predictions


def toy_predictions(num_tokens, seed):
    rng = np.random.default_rng(seed)
    return -rng.exponential(5.0, num_tokens), rng.integers(0, 20, num_tokens)


def test_a_resample_scores_the_tokens_of_its_blocks():
    logprob, rank = toy_predictions(2 * BLOCK_TOKENS + 300, seed=0)  # the last block is short
    resample = np.array([[2, 0, 2]])
    tokens = np.r_[0:300] + 2 * BLOCK_TOKENS
    chosen = np.concatenate([tokens, np.arange(BLOCK_TOKENS), tokens])

    metrics = resampled_metrics(block_totals(logprob, rank), resample)
    expected = compute_metrics(logprob[chosen], rank[chosen])
    for name, value in expected.items():
        assert metrics[name][0] == pytest.approx(value)


def test_interval_collapses_when_every_block_is_the_same():
    block_logprob, block_rank = toy_predictions(BLOCK_TOKENS, seed=1)
    logprob, rank = np.tile(block_logprob, 5), np.tile(block_rank, 5)
    metrics = resampled_metrics(block_totals(logprob, rank), bootstrap_resamples(len(logprob)))
    assert np.allclose(metrics["perplexity"], compute_metrics(logprob, rank)["perplexity"])


def test_row_averages_seeds_and_reports_their_sd(tmp_path):
    num_tokens = 3 * BLOCK_TOKENS
    runs = []
    for seed in (0, 1):
        run_dir = tmp_path / f"seed{seed}"
        save_predictions(predictions_path(run_dir, "test"), *toy_predictions(num_tokens, seed))
        runs.append(run_dir)

    summary = summarize_row(runs, bootstrap_resamples(num_tokens), num_tokens)
    perplexities = [compute_metrics(*toy_predictions(num_tokens, seed))["perplexity"] for seed in (0, 1)]
    assert summary["perplexity"] == pytest.approx(np.mean(perplexities))
    assert summary["perplexity_seed_sd"] == pytest.approx(np.std(perplexities, ddof=1))
    assert summary["perplexity_ci_low"] < summary["perplexity"] < summary["perplexity_ci_high"]
