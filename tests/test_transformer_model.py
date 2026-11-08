import pytest
import torch

from lm.transformer.model import (
    TransformerConfig,
    TransformerLM,
    load_checkpoint,
    save_checkpoint,
)

SMALL = TransformerConfig(vocab_size=50, context=16, n_layers=2, n_heads=4, d_model=32, dropout=0.1)


def small_model(seed=0):
    torch.manual_seed(seed)
    return TransformerLM(SMALL).eval()


def test_output_shape():
    model = small_model()
    ids = torch.randint(0, SMALL.vocab_size, (3, 10))
    assert model(ids).shape == (3, 10, SMALL.vocab_size)


def test_future_tokens_never_change_earlier_logits():
    model = small_model()
    ids = torch.randint(0, SMALL.vocab_size, (2, SMALL.context))
    changed = ids.clone()
    changed[:, 9] = (ids[:, 9] + 1) % SMALL.vocab_size

    with torch.no_grad():
        before, after = model(ids), model(changed)

    assert torch.equal(before[:, :9], after[:, :9])
    assert not torch.allclose(before[:, 9:], after[:, 9:])


def test_next_token_logits_match_last_position():
    model = small_model()
    ids = torch.randint(0, SMALL.vocab_size, (2, 12))
    with torch.no_grad():
        torch.testing.assert_close(model.next_token_logits(ids), model(ids)[:, -1])


def test_input_longer_than_context_is_rejected():
    model = small_model()
    with pytest.raises(ValueError):
        model(torch.zeros(1, SMALL.context + 1, dtype=torch.long))


def test_embeddings_are_tied():
    # The only vocabulary-sized weight is the token embedding, used for input and output.
    model = small_model()
    vocab_sized = [name for name, p in model.named_parameters() if p.shape[0] == SMALL.vocab_size]
    assert vocab_sized == ["token_embedding.weight"]


def test_parameter_count_of_the_planned_model():
    model = TransformerLM(TransformerConfig(vocab_size=33_278))
    count = model.num_parameters()
    print(f"\nParameters (4 layers, 4 heads, d_model 256, 33,278 tokens): {count:,}")

    embeddings = 33_278 * 256 + 128 * 256
    per_block = (256 * 768 + 768) + (256 * 256 + 256) + (256 * 1024 + 1024) + (1024 * 256 + 256) + 4 * 256
    assert count == embeddings + 4 * per_block + 2 * 256


def test_initial_loss_is_close_to_uniform():
    # Before training, every token should get about 1 / vocab_size probability.
    torch.manual_seed(0)
    config = TransformerConfig(vocab_size=33_278, context=16, n_layers=2, n_heads=2, d_model=32)
    model = TransformerLM(config).eval()
    ids = torch.randint(0, config.vocab_size, (4, 16))

    with torch.no_grad():
        logits = model(ids[:, :-1])
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, config.vocab_size), ids[:, 1:].reshape(-1))

    assert loss.item() == pytest.approx(torch.log(torch.tensor(33_278.0)).item(), abs=0.1)


def test_checkpoint_round_trip(tmp_path):
    model = small_model()
    save_checkpoint(model, tmp_path / "best.pt", step=7)
    loaded = load_checkpoint(tmp_path / "best.pt")

    # Weights come back as float32, rounded to half precision.
    for name, tensor in model.state_dict().items():
        assert loaded.state_dict()[name].dtype == torch.float32
        assert torch.equal(loaded.state_dict()[name], tensor.half().float())

    ids = torch.randint(0, SMALL.vocab_size, (2, 8))
    with torch.no_grad():
        torch.testing.assert_close(loaded(ids), model(ids), rtol=0, atol=1e-3)
    assert torch.load(tmp_path / "best.pt")["step"] == 7


def test_checkpoint_is_stored_at_half_precision(tmp_path):
    save_checkpoint(small_model(), tmp_path / "best.pt")
    weights = torch.load(tmp_path / "best.pt")["state_dict"]
    assert all(tensor.dtype == torch.float16 for tensor in weights.values())
