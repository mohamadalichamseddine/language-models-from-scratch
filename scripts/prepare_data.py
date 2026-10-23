"""Build the vocabulary and encode each split as an array of token ids.

Writes data/processed/vocab.json and data/processed/{train,valid,test}.npy.
Run once after cloning; models then load the files with lm.data.load_split().
The vocabulary is built from the train split only.

Each split also gets a short checksum; if teammates see the same checksums,
they have identical data.
"""

import hashlib

import numpy as np

from lm.data import PROCESSED_DIR, SPLITS, VOCAB_PATH, build_vocab, read_tokens


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    vocab = build_vocab(read_tokens("train"))
    vocab.save(VOCAB_PATH)
    print(f"vocab: {len(vocab):,} tokens -> {VOCAB_PATH}")

    for split in SPLITS:
        ids = vocab.encode(read_tokens(split))
        np.save(PROCESSED_DIR / f"{split}.npy", ids)

        n_unk = int(np.sum(ids == vocab.unk_id))
        checksum = hashlib.sha256(ids.tobytes()).hexdigest()[:12]
        print(f"{split}: {len(ids):,} tokens, {n_unk:,} <unk>, checksum {checksum}")


if __name__ == "__main__":
    main()
