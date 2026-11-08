"""Train the Transformer on WikiText-2 and log the run.

Example:
    python scripts/train_transformer.py --run transformer/k128 --context 128

Settings can also come from a JSON file; flags given on the command line
override it:
    python scripts/train_transformer.py --config configs/transformer.json --run transformer/k128

The run folder results/<run>/ gets config.json, curves.csv and training.json
(see lm.transformer.training), and the best checkpoint is saved as
checkpoints/<run>/best.pt. Score it afterwards with:
    python scripts/score_transformer.py --run transformer/k128
"""

import argparse
import json
from dataclasses import asdict, fields
from pathlib import Path

import torch

from lm.data import load_split, load_vocab
from lm.evaluation import RESULTS_DIR, save_config
from lm.transformer.model import CHECKPOINTS_DIR, TransformerConfig, TransformerLM
from lm.transformer.training import TrainingConfig, train
from lm.utils import set_seed


def parse_args():
    model, training = TransformerConfig(vocab_size=0), TrainingConfig()
    parser = argparse.ArgumentParser(description="Train the Transformer and log the run.")
    parser.add_argument("--run", required=True, help="run name, e.g. transformer/k128")
    parser.add_argument("--config", type=Path, help="JSON file of settings; command-line flags override it")

    parser.add_argument("--context", type=int, default=model.context, help="context size k in tokens")
    parser.add_argument("--n-layers", type=int, default=model.n_layers)
    parser.add_argument("--n-heads", type=int, default=model.n_heads)
    parser.add_argument("--d-model", type=int, default=model.d_model)
    parser.add_argument("--dropout", type=float, default=model.dropout)

    parser.add_argument("--tokens-per-batch", type=int, default=training.tokens_per_batch)
    parser.add_argument("--max-steps", type=int, default=training.max_steps)
    parser.add_argument("--lr", type=float, default=training.lr, help="peak learning rate")
    parser.add_argument("--min-lr-ratio", type=float, default=training.min_lr_ratio)
    parser.add_argument("--warmup-steps", type=int, default=training.warmup_steps)
    parser.add_argument("--weight-decay", type=float, default=training.weight_decay)
    parser.add_argument("--grad-clip", type=float, default=training.grad_clip)
    parser.add_argument("--eval-every", type=int, default=training.eval_every)
    parser.add_argument("--patience", type=int, default=training.patience)
    parser.add_argument("--seed", type=int, default=training.seed)

    parser.add_argument("--eval-tokens", type=int, help="use only the first N val tokens for the val loss (quick runs)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINTS_DIR)
    args = parser.parse_args()

    # Settings from --config become the new defaults, then the flags are read again.
    if args.config:
        settings = json.loads(args.config.read_text(encoding="utf-8"))
        unknown = set(settings) - set(vars(args))
        if unknown:
            parser.error(f"Unknown settings in {args.config}: {', '.join(sorted(unknown))}")
        parser.set_defaults(**settings)
        args = parser.parse_args()
    return args


def settings_for(config_class, args):
    """Pick the command-line values that belong to a config dataclass."""
    return {f.name: getattr(args, f.name) for f in fields(config_class) if hasattr(args, f.name)}


def main():
    args = parse_args()
    set_seed(args.seed)

    model_config = TransformerConfig(vocab_size=len(load_vocab()), **settings_for(TransformerConfig, args))
    training_config = TrainingConfig(**settings_for(TrainingConfig, args))
    train_ids = load_split("train")
    val_ids = load_split("valid")[: args.eval_tokens]

    model = TransformerLM(model_config)
    run_dir = args.results_dir / args.run
    save_config(run_dir, {
        "model": "transformer",
        "model_config": asdict(model_config),
        "training": asdict(training_config),
        "parameters": model.num_parameters(),
        "train_split": "train",
        "val_tokens_for_val_loss": len(val_ids),
    })

    print(f"Training {model.num_parameters():,} parameters on {args.device}, context {model_config.context}", flush=True)
    summary = train(model, train_ids, val_ids, training_config, run_dir, args.checkpoint_dir / args.run / "best.pt", args.device)

    minutes = summary["train_seconds"] / 60
    print(f"Best val loss {summary['best_val_loss']:.4f} at step {summary['best_step']} ({minutes:.1f} min on {summary['device']})")


if __name__ == "__main__":
    main()
