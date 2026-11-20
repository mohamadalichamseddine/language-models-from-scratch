"""T19: score a trained k=128 checkpoint with shorter visible histories."""

import argparse
from pathlib import Path

import torch

from lm.ablation import CONTEXTS, data_identity, read_json, score_checkpoint
from lm.data import load_split, load_vocab
from lm.evaluation import RESULTS_DIR, write_table
from lm.transformer.model import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="ablation/k128_seed0", help="source run relative to results/")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--contexts", type=int, nargs="+", default=list(CONTEXTS))
    parser.add_argument("--out", type=Path, help="default: results/context_truncation/<run>/")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--report-only", action="store_true", help="render saved metrics without loading or scoring checkpoints")
    args = parser.parse_args()
    if args.batch_size < 1 or len(set(args.contexts)) != len(args.contexts):
        parser.error("Use a positive batch size and distinct contexts")
    if any(k not in CONTEXTS for k in args.contexts):
        parser.error(f"Choose contexts from {CONTEXTS}")
    out = args.out or args.results_dir / "context_truncation" / args.run
    if args.report_only:
        source = args.results_dir / "context_truncation" / args.run
        rows = [(f"trained k=128, scored k={context}", read_json(source / f"k{context}/metrics.json")["valid"])
                for context in args.contexts]
        print(write_table(rows, out / "comparison", f"# Validation context truncation: {args.run}"))
        return
    checkpoint = args.results_dir / args.run / "best.pt"
    if load_checkpoint(checkpoint).config.context != 128:
        parser.error("T19 requires a model trained at context 128")
    vocab, ids = load_vocab(), load_split("valid")
    identity = data_identity(load_split("train"), ids, vocab)
    rows = []
    for context in args.contexts:
        metrics = score_checkpoint(checkpoint, out / f"k{context}", ids, vocab.ids["<eos>"],
                                   identity, context, args.device, args.batch_size)
        rows.append((f"trained k=128, scored k={context}", metrics))
    print(write_table(rows, out / "comparison", f"# Validation context truncation: {args.run}"))


if __name__ == "__main__":
    main()
