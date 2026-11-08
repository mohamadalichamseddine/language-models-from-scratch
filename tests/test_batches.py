import numpy as np
import pytest
import torch

from lm.transformer.batches import BatchSampler

# Token i has id i, so every window is a run of consecutive numbers.
IDS = np.arange(1000, dtype=np.int32)


def test_targets_are_inputs_shifted_by_one():
    inputs, targets = BatchSampler(IDS, context=8, tokens_per_batch=64).next_batch()

    assert inputs.shape == targets.shape == (8, 8)
    assert torch.equal(targets[:, :-1], inputs[:, 1:])
    assert torch.equal(targets[:, -1], inputs[:, -1] + 1)


def test_windows_are_contiguous_pieces_of_the_stream():
    inputs, _ = BatchSampler(IDS, context=8, tokens_per_batch=64).next_batch()
    assert torch.equal(inputs - inputs[:, :1], torch.arange(8).expand(8, 8))


def test_every_context_gets_the_same_tokens_per_batch():
    for context in [6, 32, 64, 128]:
        inputs, _ = BatchSampler(IDS, context=context, tokens_per_batch=384).next_batch()
        assert inputs.numel() == 384


def test_tokens_per_batch_must_be_a_multiple_of_context():
    with pytest.raises(ValueError):
        BatchSampler(IDS, context=6, tokens_per_batch=64)


def test_same_seed_gives_same_batches():
    first = BatchSampler(IDS, context=8, tokens_per_batch=64, seed=1)
    second = BatchSampler(IDS, context=8, tokens_per_batch=64, seed=1)
    other = BatchSampler(IDS, context=8, tokens_per_batch=64, seed=2)

    for _ in range(3):
        a, b, c = first.next_batch()[0], second.next_batch()[0], other.next_batch()[0]
        assert torch.equal(a, b)
        assert not torch.equal(a, c)


def test_windows_can_reach_the_last_token():
    # With only one possible window, it must be the whole stream.
    inputs, targets = BatchSampler(np.arange(9), context=8, tokens_per_batch=8).next_batch()
    assert inputs.tolist() == [list(range(8))]
    assert targets.tolist() == [list(range(1, 9))]


def test_sampling_does_not_touch_the_global_seed():
    # Dropout uses the global generator, so drawing batches must not change it.
    torch.manual_seed(0)
    expected = torch.rand(1)

    torch.manual_seed(0)
    BatchSampler(IDS, context=8, tokens_per_batch=64).next_batch()
    assert torch.equal(torch.rand(1), expected)
