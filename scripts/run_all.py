"""T26: prepare data, test, and reproduce reports from the committed artifacts.

No training or test scoring is performed. Use --regenerate-samples to also
repeat T24 generation with the frozen checkpoints and generation config.
"""

import argparse
import os
import platform
import subprocess
import sys
from importlib.metadata import version

from lm.ablation import write_json
from lm.data import ROOT

REPORT_COMMANDS = [
    ["scripts/dataset_stats.py"],
    ["scripts/run_ngram.py", "--report-only"],
    ["scripts/evaluate.py", "results/ngram/*", "--saved-metrics", "--out", "results/ngram/comparison"],
    ["scripts/sweep_transformer.py", "--report-only"],
    ["scripts/build_figures.py"],
    ["scripts/generation_metrics.py"],
    ["scripts/score_context_truncation.py", "--run", "ablation/k128_seed0", "--report-only"],
    ["scripts/score_context_truncation.py", "--run", "ablation/k128_seed1", "--report-only"],
    ["scripts/plot_ablation.py"],
    ["scripts/final_evaluation.py", "--report-only"],
    ["scripts/rare_tokens.py", "--report-only"],
    ["scripts/long_context.py", "--report-only"],
    ["scripts/generation_analysis.py", "--samples", "results/generation/ablation/samples.json"],
    ["scripts/bundle_source.py"],
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-tests", action="store_true", help="only prepare data and rebuild reports")
    parser.add_argument("--regenerate-samples", action="store_true", help="also regenerate T24 samples from checkpoints")
    parser.add_argument("--device", default="cpu", help="device for --regenerate-samples")
    args = parser.parse_args()
    commands = [["scripts/prepare_data.py"]]
    if not args.skip_tests:
        commands.append(["-m", "pytest", "-q"])
    if args.regenerate_samples:
        commands.append(["scripts/generation_analysis.py", "--config", "configs/generation.json", "--device", args.device])
    commands.extend(REPORT_COMMANDS)
    manifest = {"mode": "reports from saved artifacts", "regenerated_samples": args.regenerate_samples,
                "python": platform.python_version(), "platform": platform.system(),
                "packages": {name: version(name) for name in ("numpy", "torch", "datasets", "matplotlib", "pytest")},
                "commands": [], "complete": False}
    log = ROOT / "results/reproducibility/run_all.json"
    environment = {**os.environ, "PYTHONUTF8": "1", "MPLBACKEND": "Agg"}
    for command in commands:
        print("Running: python " + " ".join(command), flush=True)
        result = subprocess.run([sys.executable, *command], cwd=ROOT, env=environment)
        manifest["commands"].append({"args": command, "returncode": result.returncode})
        write_json(log, manifest)
        if result.returncode:
            raise SystemExit(result.returncode)
    manifest["complete"] = True
    write_json(log, manifest)
    print(f"Completed {len(commands)} commands. Run log: {log}")


if __name__ == "__main__":
    main()
