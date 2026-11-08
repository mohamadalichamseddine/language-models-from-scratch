"""Score a trained Transformer on a split and compute its metrics.

Example, after scripts/train_transformer.py --run transformer/k128:
    python scripts/score_transformer.py --run transformer/k128

This loads checkpoints/<run>/best.pt, saves every token's log-prob and rank as
results/<run>/<split>.npz (see lm.transformer.scoring), and adds the metrics
to results/<run>/metrics.json. To compare it with other runs, use
scripts/evaluate.py.
"""

import argparse
import time
from pathlib import Path

import torch

from lm.data import EOS, SPLITS, load_split, load_vocab
from lm.evaluation import RESULTS_DIR, evaluate_run, predictions_path
from lm.transformer.model import CHECKPOINTS_DIR, load_checkpoint
from lm.transformer.scoring import score_split


def main():
    parser = argparse.ArgumentParser(description="Score a trained Transformer on a split.")
    parser.add_argument("--run", required=True, help="run name used for training, e.g. transformer/k128")
    parser.add_argument("--split", default="valid", choices=SPLITS, help="split to score (default: valid)")
    parser.add_argument("--batch-size", type=int, default=256, help="windows per forward pass")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINTS_DIR)
    args = parser.parse_args()

    model = load_checkpoint(args.checkpoint_dir / args.run / "best.pt", args.device)
    ids = load_split(args.split)
    eos_id = load_vocab().ids[EOS]
    run_dir = args.results_dir / args.run

    print(f"Scoring {len(ids):,} {args.split} tokens with context {model.config.context} on {args.device}...", flush=True)
    start = time.time()
    score_split(model, ids, eos_id, args.batch_size, path=predictions_path(run_dir, args.split))
    print(f"Done in {time.time() - start:.0f}s")

    metrics = evaluate_run(run_dir, args.split, len(ids))
    print(f"Perplexity {metrics['perplexity']:.2f}, top-1 accuracy {metrics['top1_accuracy']:.2%}")


if __name__ == "__main__":
    main()
