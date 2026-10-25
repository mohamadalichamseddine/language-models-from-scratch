
"""Prediction storage and evaluation metrics for language models."""

from pathlib import Path

import numpy as np


def save_predictions(path, logprob, rank):
    """Save per-token natural log-probabilities and zero-based ranks, including <eos>."""
    logprob = np.asarray(logprob, dtype=np.float64)
    rank = np.asarray(rank)

    if logprob.ndim != 1 or rank.ndim != 1:
        raise ValueError("Inputs must be 1D arrays")

    if len(logprob) != len(rank):
        raise ValueError("Inputs must have equal lengths")

    if not np.issubdtype(rank.dtype, np.integer) or np.any(rank < 0):
        raise ValueError("Ranks must be nonnegative integers")

    if np.any(np.isnan(logprob)) or np.any(logprob > 0):
        raise ValueError("Invalid log-probabilities")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        path,
        logprob=logprob,
        rank=rank.astype(np.int64),
    )


def load_predictions(path):
    """Load per-token log-probabilities and zero-based ranks."""
    with np.load(path, allow_pickle=False) as data:
        return data["logprob"], data["rank"]


def compute_metrics(logprob, rank):
    """Compute metrics over all targets, including <eos>.

    CE (nats/token) = -mean(logprob).
    CE (bits/token) = CE_nats / ln(2).
    Perplexity = exp(CE_nats).
    Top-k accuracy = mean(rank < k), for k = 1, 5, 10.
    Ranks are zero-based. Zero probability gives infinite CE and perplexity.
    """
    logprob = np.asarray(logprob, dtype=np.float64)
    rank = np.asarray(rank)

    if logprob.ndim != 1 or rank.ndim != 1:
        raise ValueError("Inputs must be 1D arrays")

    if len(logprob) != len(rank):
        raise ValueError("Inputs must have equal lengths")

    if len(logprob) == 0:
        raise ValueError("Empty predictions")

    if not np.issubdtype(rank.dtype, np.integer) or np.any(rank < 0):
        raise ValueError("Invalid ranks")

    if np.any(np.isnan(logprob)) or np.any(logprob > 0):
        raise ValueError("Invalid log-probabilities")

    ce_nats = float(-np.mean(logprob))
    ce_bits = float(ce_nats / np.log(2.0))
    perplexity = float(np.exp(ce_nats))

    return {
        "cross_entropy_nats": ce_nats,
        "cross_entropy_bits": ce_bits,
        "perplexity": perplexity,
        "top1_accuracy": float(np.mean(rank < 1)),
        "top5_accuracy": float(np.mean(rank < 5)),
        "top10_accuracy": float(np.mean(rank < 10)),
    }


def metrics_from_file(path):
    """Compute evaluation metrics from a saved prediction file."""
    logprob, rank = load_predictions(path)
    return compute_metrics(logprob, rank)
