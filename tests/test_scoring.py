import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from lm.evaluation import evaluate_run, predictions_path
from lm.metrics import load_predictions
from lm.transformer.model import TransformerConfig, TransformerLM
from lm.transformer.scoring import score_split

VOCAB_SIZE = 30
EOS_ID = 0
CONTEXT = 8
IDS = np.random.default_rng(0).integers(0, VOCAB_SIZE, 45)


def small_model():
    torch.manual_seed(0)
    config = TransformerConfig(vocab_size=VOCAB_SIZE, context=CONTEXT, n_layers=2, n_heads=2, d_model=16)
    return TransformerLM(config).eval()


def score_token_by_token(model, ids):
    """The slow, obvious way: one forward pass per target."""
    stream = torch.tensor([EOS_ID] + list(ids))
    logprob, rank = [], []
    with torch.no_grad():
        for i, target in enumerate(ids):
            history = stream[max(0, i + 1 - CONTEXT) : i + 1]
            log_probs = F.log_softmax(model(history[None])[0, -1].double(), dim=-1)
            p = log_probs[target]
            logprob.append(p.item())
            rank.append(int((log_probs > p).sum() + (log_probs[:target] == p).sum()))
    return np.array(logprob), np.array(rank)


@pytest.mark.parametrize("batch_size", [1, 4, 256])
def test_sliding_window_matches_token_by_token(batch_size):
    model = small_model()
    logprob, rank = score_split(model, IDS, EOS_ID, batch_size=batch_size)
    expected_logprob, expected_rank = score_token_by_token(model, IDS)

    np.testing.assert_allclose(logprob, expected_logprob, rtol=0, atol=1e-5)
    np.testing.assert_array_equal(rank, expected_rank)


def test_split_shorter_than_the_context():
    model = small_model()
    logprob, rank = score_split(model, IDS[:5], EOS_ID)
    expected_logprob, expected_rank = score_token_by_token(model, IDS[:5])

    np.testing.assert_allclose(logprob, expected_logprob, rtol=0, atol=1e-5)
    np.testing.assert_array_equal(rank, expected_rank)


def test_ties_are_broken_by_lower_id():
    # With zero embeddings every token gets the same logit, so a target's rank
    # is the number of tokens with a lower id.
    model = small_model()
    with torch.no_grad():
        model.token_embedding.weight.zero_()

    logprob, rank = score_split(model, IDS, EOS_ID)
    np.testing.assert_allclose(logprob, -math.log(VOCAB_SIZE))
    np.testing.assert_array_equal(rank, IDS)


def test_evaluator_reads_the_saved_predictions(tmp_path):
    model = small_model()
    path = predictions_path(tmp_path, "valid")
    logprob, rank = score_split(model, IDS, EOS_ID, path=path)

    saved_logprob, saved_rank = load_predictions(path)
    np.testing.assert_array_equal(saved_logprob, logprob)
    np.testing.assert_array_equal(saved_rank, rank)

    metrics = evaluate_run(tmp_path, "valid", num_tokens=len(IDS))
    assert metrics["perplexity"] == pytest.approx(math.exp(-logprob.mean()))
