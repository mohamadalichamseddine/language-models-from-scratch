"""T26: bundle public source code into one UTF-8 text file."""

import argparse
from pathlib import Path

from lm.data import ROOT
from lm.reproducibility import bundle_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results/reproducibility/source_bundle.txt")
    args = parser.parse_args()
    count = bundle_sources(ROOT, args.out)
    print(f"Bundled {count} source files into {args.out}")


if __name__ == "__main__":
    main()
