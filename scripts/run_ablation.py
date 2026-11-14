"""T18: train and score the frozen context/seed grid on validation only."""

import argparse
import json
from pathlib import Path

import torch

from lm.ablation import CONTEXTS, SEEDS, data_identity, experiment_configs, read_json, train_ablation_run
from lm.data import ROOT, load_split, load_vocab
from lm.evaluation import RESULTS_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/transformer.json")
    parser.add_argument("--contexts", type=int, nargs="+", default=list(CONTEXTS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "ablation")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=256, help="scoring windows per batch")
    parser.add_argument("--dry-run", action="store_true", help="print effective settings without training")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    runs = experiment_configs(read_json(args.config), args.contexts, args.seeds)
    if args.dry_run:
        print(json.dumps(runs, indent=2))
        return
    train_ids, valid_ids, vocab = load_split("train"), load_split("valid"), load_vocab()
    identity = data_identity(train_ids, valid_ids, vocab)
    for run in runs:
        print(f"\n{run['run']}: {run['token_budget']:,} training tokens on {args.device}", flush=True)
        train_ablation_run(run, args.out / run["run"], train_ids, valid_ids, vocab, identity,
                           args.device, args.batch_size)


if __name__ == "__main__":
    main()
