"""T24: generate matched continuations for all frozen n-grams and ablation models."""

import argparse
from pathlib import Path

from lm.ablation import read_json
from lm.data import ROOT, load_split, load_vocab
from lm.evaluation import RESULTS_DIR
from lm.generation_analysis import GenerationConfig, run_generation, write_generation_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/generation.json")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "generation/ablation")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--samples", type=Path, help="rebuild reports from saved samples without generating")
    args = parser.parse_args()
    if args.samples:
        write_generation_report(read_json(args.samples), args.out)
    else:
        run_generation(GenerationConfig(**read_json(args.config)), args.results_dir, args.out,
                       load_split("train"), load_split("valid"), load_vocab(), args.device)
    print(f"Wrote full generation analysis to {args.out}")


if __name__ == "__main__":
    main()
