"""WikiText-2 tokens, vocabulary and token ids.

Every model reads the same token stream: each line of a raw file becomes its
words followed by <eos>, and an empty line becomes a lone <eos>. This is the
standard WikiText-2 convention and reproduces the published token counts
(2,088,628 train / 217,646 valid / 245,569 test).
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data/raw/wikitext-2"
PROCESSED_DIR = ROOT / "data/processed"
VOCAB_PATH = PROCESSED_DIR / "vocab.json"

SPLITS = ["train", "valid", "test"]
EOS = "<eos>"
UNK = "<unk>"


def lines_to_tokens(lines):
    tokens = []
    for line in lines:
        tokens.extend(line.split())
        tokens.append(EOS)
    return tokens


def read_tokens(split):
    """Return every token of a raw split, with <eos> at the end of each line."""
    with open(RAW_DIR / f"wiki.{split}.tokens", encoding="utf-8") as f:
        return lines_to_tokens(f)


class Vocab:
    """Maps tokens to ids and back. A token's id is its index in `tokens`."""

    def __init__(self, tokens):
        self.tokens = list(tokens)
        self.ids = {token: i for i, token in enumerate(self.tokens)}
        self.unk_id = self.ids[UNK]

    def __len__(self):
        return len(self.tokens)

    def encode(self, tokens):
        """Turn tokens into an int32 array of ids. Unknown tokens become <unk>."""
        ids = [self.ids.get(token, self.unk_id) for token in tokens]
        return np.array(ids, dtype=np.int32)

    def decode(self, ids):
        return [self.tokens[i] for i in ids]

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.tokens, f, ensure_ascii=False, indent=0)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))


def build_vocab(tokens):
    """Build a vocabulary from training tokens, most frequent token first.

    Ties are broken alphabetically so the ids are the same on every machine.
    <eos> and <unk> are always included.
    """
    counts = Counter(tokens)
    vocab = sorted(counts, key=lambda token: (-counts[token], token))
    for special in [EOS, UNK]:
        if special not in counts:
            vocab.append(special)
    return Vocab(vocab)


def load_vocab():
    _check_prepared(VOCAB_PATH)
    return Vocab.load(VOCAB_PATH)


def load_split(split):
    """Return the token ids of a split as an int32 array."""
    path = PROCESSED_DIR / f"{split}.npy"
    _check_prepared(path)
    return np.load(path)


def _check_prepared(path):
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing. Run `python scripts/prepare_data.py` first.")
