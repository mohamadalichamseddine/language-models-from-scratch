
"""T15: compute macro-averaged generation diversity metrics per model."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from lm.evaluation import RESULTS_DIR
from lm.generation import aggregate_generation


def build_table(samples_path, out):
    data = json.loads(Path(samples_path).read_text(encoding="utf-8"))
    grouped = defaultdict(list)

    for sample in data["samples"]:
        grouped[sample["model"]].append(sample["generated_ids"])

    if not grouped:
        raise ValueError("No samples in generation file")

    rows = [
        {
            "model": model,
            "samples": len(samples),
            **aggregate_generation(samples)
        }
        for model, samples in sorted(grouped.items())
    ]

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with (out / "metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0])
        )
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Generation metrics",
        "",
        "Macro-average of per-sample scores over generated tokens only.",
        "",
        "| Model | Samples | Distinct-1 | Distinct-2 | Repetition (4-gram) |",
        "|---|---:|---:|---:|---:|"
    ]

    for row in rows:
        lines.append(
            f"| {row['model']} | {row['samples']} | "
            f"{row['distinct_1']:.4f} | "
            f"{row['distinct_2']:.4f} | "
            f"{row['repetition_4']:.4f} |"
        )

    (out / "metrics.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8"
    )

    print("\n".join(lines))
    return rows


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--samples",
        type=Path,
        default=RESULTS_DIR / "generation/samples.json"
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=RESULTS_DIR / "generation"
    )

    args = parser.parse_args()
    build_table(args.samples, args.out)


if __name__ == "__main__":
    main()
