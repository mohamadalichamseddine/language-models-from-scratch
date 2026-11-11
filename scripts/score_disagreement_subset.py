
"""Generate aligned T04 per-token outputs for a validation subset for T17.

To use the full validation split instead, run the project's full score_transformer.py
and run_ngram.py scripts, then compare their matching valid.npz files.
"""

import argparse
from pathlib import Path

from lm.data import EOS, load_split, load_vocab
from lm.ngram import NGramModel
from lm.transformer.model import load_checkpoint
from lm.transformer.scoring import score_split


def main():
    p = argparse.ArgumentParser()

    p.add_argument('--limit', type=int, default=2048)
    p.add_argument('--batch-size', type=int, default=64)

    p.add_argument(
        '--checkpoint',
        type=Path,
        default=Path('results/transformer/best/best.pt')
    )

    p.add_argument(
        '--out',
        type=Path,
        default=Path('results/disagreement_subset')
    )

    args = p.parse_args()

    ids = load_split('valid')[:args.limit]

    if len(ids) == 0:
        raise ValueError('Empty selection')

    args.out.mkdir(parents=True, exist_ok=True)

    vocab = load_vocab()

    print('Fitting trigram...', flush=True)

    tri = NGramModel(
        3,
        len(vocab),
        k=0.001
    ).fit(load_split('train'))

    tri.score_split(
        ids,
        args.out / 'trigram_valid.npz'
    )

    print('Scoring Transformer...', flush=True)

    model = load_checkpoint(args.checkpoint)

    score_split(
        model,
        ids,
        vocab.ids[EOS],
        batch_size=args.batch_size,
        path=args.out / 'transformer_valid.npz'
    )

    print('Saved aligned subset outputs at', args.out)


if __name__ == '__main__':
    main()
