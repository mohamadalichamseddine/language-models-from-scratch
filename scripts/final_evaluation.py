"""T21: score the test split once with every final model and build the final table.

Run this only once every choice is frozen; nothing gets tuned after it:
    python scripts/final_evaluation.py --device cuda

On Colab, use notebooks/final_test_on_colab.ipynb. Models whose test
predictions already exist (test.npz in their run folder) are not scored
again: after a disconnect the command carries on where it stopped, and with
every prediction in place it only rebuilds the table in results/final/.
"""

import argparse
from pathlib import Path

import torch

from lm.ablation import data_identity, read_json, write_json
from lm.data import EOS, load_split, load_vocab
from lm.evaluation import RESULTS_DIR
from lm.final_evaluation import BLOCK_TOKENS, RESAMPLES, build_final_table, score_ngrams, score_transformers, write_final_table


def main():
    parser = argparse.ArgumentParser(description="Score the test split once with every final model.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=256, help="Transformer windows per forward pass")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "final")
    parser.add_argument("--report-only", action="store_true", help="render the saved final.json summaries without test scoring")
    args = parser.parse_args()

    if args.report_only:
        source = args.results_dir / "final"
        config = read_json(source / "config.json")
        if config["bootstrap"]["block_tokens"] != BLOCK_TOKENS or config["bootstrap"]["resamples"] != RESAMPLES:
            raise ValueError("Saved bootstrap settings differ from the report's protocol")
        print(write_final_table(read_json(source / "final.json"), args.out, config["test_tokens"]))
        return

    vocab = load_vocab()
    train_ids, valid_ids, test_ids = load_split("train"), load_split("valid"), load_split("test")
    trained_on = read_json(args.results_dir / "ablation/k6_seed0/config.json")["data"]
    if data_identity(train_ids, valid_ids, vocab) != trained_on:
        raise SystemExit("The prepared data differs from the data the models were trained on.")

    records = score_transformers(args.results_dir, test_ids, vocab.ids[EOS], args.device, args.batch_size)
    records |= score_ngrams(args.results_dir, train_ids, test_ids, len(vocab))

    # Keep the records of earlier invocations (e.g. before a disconnect).
    log_path = args.out / "scoring.json"
    log = read_json(log_path) if log_path.exists() else {}
    write_json(log_path, {**log, **records})

    print(build_final_table(args.results_dir, args.out, test_ids))


if __name__ == "__main__":
    main()
