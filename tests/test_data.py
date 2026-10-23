import pytest

from lm.data import EOS, UNK, Vocab, build_vocab, lines_to_tokens, read_tokens

PUBLISHED_TOKEN_COUNTS = {"train": 2_088_628, "valid": 217_646, "test": 245_569}
PUBLISHED_VOCAB_SIZE = 33_278


def test_eos_after_every_line_including_empty_ones():
    lines = ["the cat \n", "\n", "sat\n"]
    assert lines_to_tokens(lines) == ["the", "cat", EOS, EOS, "sat", EOS]


def test_encode_then_decode_gives_back_the_tokens():
    tokens = "the cat sat on the mat".split()
    vocab = build_vocab(tokens)
    assert vocab.decode(vocab.encode(tokens)) == tokens


def test_unknown_tokens_become_unk():
    vocab = build_vocab("the cat sat".split())
    assert vocab.decode(vocab.encode(["the", "dog"])) == ["the", UNK]


def test_vocab_is_sorted_by_frequency_then_alphabetically():
    vocab = build_vocab("b a b c b a d".split())
    assert vocab.tokens[:4] == ["b", "a", "c", "d"]


def test_vocab_always_has_eos_and_unk():
    vocab = build_vocab(["hello"])
    assert EOS in vocab.ids and UNK in vocab.ids


def test_vocab_save_and_load(tmp_path):
    vocab = build_vocab("the cat sat".split())
    vocab.save(tmp_path / "vocab.json")
    assert Vocab.load(tmp_path / "vocab.json").tokens == vocab.tokens


@pytest.mark.parametrize("split", PUBLISHED_TOKEN_COUNTS)
def test_token_counts_match_published_numbers(split):
    assert len(read_tokens(split)) == PUBLISHED_TOKEN_COUNTS[split]


def test_vocab_size_matches_published_number():
    assert len(build_vocab(read_tokens("train"))) == PUBLISHED_VOCAB_SIZE
