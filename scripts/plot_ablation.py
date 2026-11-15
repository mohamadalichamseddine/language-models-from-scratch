"""T20: plot validation metrics across contexts, with seed SD and n-gram references."""

import argparse
from pathlib import Path

from lm.ablation import CONTEXTS, SEEDS
from lm.ablation_plots import build_ablation_figures
from lm.evaluation import RESULTS_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=RESULTS_DIR / "ablation")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR, help="source of n-gram reference metrics")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "figures")
    parser.add_argument("--contexts", type=int, nargs="+", default=list(CONTEXTS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--allow-partial", action="store_true", help="label incomplete results as a preview")
    args = parser.parse_args()
    if (len(set(args.contexts)) != len(args.contexts) or any(k not in CONTEXTS for k in args.contexts)
            or len(set(args.seeds)) != len(args.seeds) or any(s < 0 for s in args.seeds)):
        parser.error("Choose distinct supported contexts and distinct nonnegative seeds")
    status = build_ablation_figures(args.runs, args.results_dir, args.out, args.contexts, args.seeds, args.allow_partial)
    print(f"Wrote ablation figures to {args.out}; missing runs: {status['missing_runs']}")


if __name__ == "__main__":
    main()
