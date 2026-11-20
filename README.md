# N-Grams vs Decoder-Only Transformer

Classical n-gram language models and a small decoder-only Transformer, both written from scratch and compared on WikiText-2. The plan and ticket status are in [docs/roadmap.md](docs/roadmap.md).

## Setup

Requires Python 3.11 or newer. The pinned environment is tested on Python 3.12; historical Colab runs used their recorded environments.

```bash
git clone https://github.com/mohamadalichamseddine/language-models-from-scratch.git
cd language-models-from-scratch
python -m venv .venv
source .venv/bin/activate        # PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/prepare_data.py   # builds data/processed/: vocabulary and token ids
python -m pytest -q
```

`pip install -r requirements.txt` also installs this repo's package (`src/lm`) in editable mode, so scripts and tests can `import lm` and pick up code changes without reinstalling.

The requirements pin the tested direct dependency versions, including `torch==2.14.0`. For GPU use, install a matching platform-specific PyTorch 2.14.0 build before the requirements. A local build suffix (such as `+rocm10.1.0`) satisfies this pin. Check `torch.cuda.is_available()` and device names before choosing `--device`; CPU report reproduction needs no GPU.

For AMD GPUs, see the [ROCm setup notes](docs/context-ablation.md#setup-and-device-check).

## Where things run

- **Data processing and n-gram models:** any laptop; CPU is enough.
- **Transformer training:** Open [notebooks/train_on_colab.ipynb](https://colab.research.google.com/github/mohamadalichamseddine/language-models-from-scratch/blob/main/notebooks/train_on_colab.ipynb) in Colab: it clones the repo into Google Drive (so runs survive a disconnect), runs the tests and the T13 sweep, and downloads the results.

  The scripts also work on their own, on any machine with a GPU:

  ```bash
  python scripts/train_transformer.py --run transformer/k128 --context 128   # train, log curves, keep the best checkpoint
  python scripts/score_transformer.py --run transformer/k128                 # score valid with a sliding window
  python scripts/sweep_transformer.py                                        # T13: learning rate x dropout sweep
  ```
- **Final test-set evaluation (T21):** Open [notebooks/final_test_on_colab.ipynb](https://colab.research.google.com/github/mohamadalichamseddine/language-models-from-scratch/blob/main/notebooks/final_test_on_colab.ipynb) in Colab (about 25 minutes on a T4), or run `python scripts/final_evaluation.py` on any machine (about 2 hours on a CPU). It scores the test split once with every final model and writes the table with 95% bootstrap confidence intervals to `results/final/`.

## Layout

For T18-T20 training, context truncation, and plots, follow
[the context-ablation runbook](docs/context-ablation.md).

```
src/lm/               shared code (data, metrics, evaluation) and the n-gram models
src/lm/transformer/   the Transformer
scripts/              entry points (download data, train, evaluate)
configs/              run configurations
notebooks/            Colab notebook for training the Transformer on a GPU
tests/                unit tests (run with pytest)
results/              metrics, tables, figures and the trained models later tickets need
data/                 WikiText-2: raw text is committed, data/processed/ is generated
docs/                 roadmap and notes
```

Training uses `lm.utils.set_seed`; batch sampling and generation use explicitly seeded generators. Runs save their settings, but hardware and runtime differences can still produce small numerical differences. See the [context-ablation runbook](docs/context-ablation.md) for how much repeated scoring differs across runs and devices.

## Reproduce the reports (T26)

After installing requirements, run from the repository root:

```bash
python scripts/run_all.py
```

This prepares the committed raw data, runs the tests, rebuilds reports from saved
metrics/samples, and bundles the public source. It does not train or score models.
The run log is `results/reproducibility/run_all.json`; any failed command stops
execution. `--skip-tests` skips only the tests. To also regenerate all T24 samples:

```bash
python scripts/run_all.py --regenerate-samples --device cpu
```

Use `--device cuda:1` for the RX 9070 XT on the machine used for T18/T24, after
checking the device list. See the [generation protocol and findings](docs/generation-analysis.md).
Generation settings and seeds are frozen in `configs/generation.json`; the full
samples, data fingerprints, checkpoint hashes and runtime are saved in
`results/generation/ablation/`.

The following commands rebuild every published table/figure from its recorded
inputs. Paths in the first column are relative to `results/`. `--report-only`
formats saved results: for T21 it preserves the existing bootstrap summaries;
for T22/T23 it renders the existing numerical CSV. It does not independently
recompute those scientific analyses. Recalculation commands are below the table.

| Artifact | Exact command (from repository root) |
|---|---|
| `data_stats/stats.md`, `data_stats/zipf.png` | `python scripts/dataset_stats.py` |
| `ngram/tuning.md`, `ngram/perplexity_vs_k.png` | `python scripts/run_ngram.py --report-only` |
| `ngram/comparison.md`, `.csv` | `python scripts/evaluate.py "results/ngram/*" --saved-metrics --out results/ngram/comparison` |
| `transformer/sweep/sweep.md`, `comparison.md`, `comparison.csv` | `python scripts/sweep_transformer.py --report-only` |
| `figures/comparison.md`, `.csv`, `perplexity.png`, `transformer_loss.png` | `python scripts/build_figures.py` |
| `generation/metrics.md`, `.csv` | `python scripts/generation_metrics.py` |
| `context_truncation/ablation/k128_seed0/comparison.md`, `.csv` | `python scripts/score_context_truncation.py --run ablation/k128_seed0 --report-only` |
| `context_truncation/ablation/k128_seed1/comparison.md`, `.csv` | `python scripts/score_context_truncation.py --run ablation/k128_seed1 --report-only` |
| `figures/ablation_*.png`, `ablation_runs.csv`, `ablation_summary.md`, `.csv` | `python scripts/plot_ablation.py` |
| `final/final.md`, `.csv` | `python scripts/final_evaluation.py --report-only` |
| `rare_tokens/rare_tokens.md`, `.csv` | `python scripts/rare_tokens.py --report-only` |
| `long_context/long_context.md`, `.csv` | `python scripts/long_context.py --report-only` |
| `generation/ablation/metrics.md`, `.csv`, `representative_samples.md` | `python scripts/generation_analysis.py --samples results/generation/ablation/samples.json` |
| `figures/disagreements.csv` (requires validation predictions) | `python scripts/model_disagreement.py results/transformer/best/valid.npz results/ngram/n3_k0.001/valid.npz --split valid --top 20 --context 10 --out results/figures/disagreements.csv` |

`run_all` runs the commands above except the disagreement export, which needs
ignored per-token predictions. Training `curves.csv` files are source logs for
figures, not reports that can be reconstructed from aggregate metrics.

### Recompute the underlying experiments

These are separate from report reproduction. The original frozen n-gram grid
and selected values are in `results/ngram/tuning.json`; interpolation weights
are in `results/ngram/interpolated/config.json`. To recreate their validation
predictions and tuning results, use `python scripts/run_ngram.py` and
`python scripts/run_interpolated.py`. These recompute the historical validation
searches; do not change their grids or use test scores to select settings.

T13's original training/sweep command is `python scripts/sweep_transformer.py`.
It can train nine models, replaces `results/transformer/best/`, and rewrites the
frozen config. Use a separate experiment checkout if repeating it. All run
settings, training seeds and curves are stored under `results/transformer/`.
T18-T20 commands and their equal-budget policy are in the
[context-ablation runbook](docs/context-ablation.md). The default eight-run command
is `python scripts/run_ablation.py --config configs/transformer.json --contexts 6 32 64 128 --seeds 0 1 --device cpu`.
To recreate T19 scores, run the two context-truncation commands above without
`--report-only`, adding `--device cpu` or your checked GPU device.

To rebuild T14's original samples use `python scripts/generate.py --device cpu`;
then run the generation-metrics command. To recreate T24's complete sample set
use `python scripts/generation_analysis.py --config configs/generation.json --device cpu`.
Fixed seeds reproduce sampling for identical probabilities; changed hardware,
PyTorch builds or numerical libraries can change generated sequences.

For T17, restore archived validation predictions, or recreate the n-gram
validation predictions as above and run
`python scripts/score_transformer.py --run transformer/best --split valid --device cpu`.
Then run the disagreement command in the table.

For T21-T23, restore the archived T21 predictions into their original run folders
(the `results/t21_results.zip` download is ignored). With those predictions present,
`python scripts/final_evaluation.py --device cpu` rebuilds the bootstrap table;
`python scripts/rare_tokens.py` and `python scripts/long_context.py` recompute the
analyses and print examples. The final evaluator scores any missing test files,
so it is not a routine reproduction command for a fresh clone. Test scoring was
already performed for T21 after freezing the models; do not retune against it.
The bootstrap seed (0), block size (1,000), resamples (10,000), and model mapping
are recorded in `results/final/config.json`.

### Bundle source and verify a fresh clone

```bash
python scripts/bundle_source.py --out results/reproducibility/source_bundle.txt
```

The deterministic UTF-8 bundle includes Python under `src/`, `scripts/` and
`tests/`, notebook code cells, JSON configs and packaging files. It omits notebook
outputs, data, checkpoints, private documentation, environments and secrets.
The generated bundle and machine-specific run log are ignored by Git.

For a fresh-clone check, install requirements in a new virtual environment and run:

```bash
python scripts/generation_analysis.py --samples results/generation/ablation/samples.json --out results/reproduced-generation
python -c "from pathlib import Path; a=Path('results/generation/ablation/metrics.csv'); b=Path('results/reproduced-generation/metrics.csv'); assert a.read_bytes()==b.read_bytes(); print('Generation table matches exactly')"
python scripts/run_all.py
```

The table check recomputes all three metrics from the committed token IDs, with
no prepared dataset, model checkpoint loading or GPU required. See
[the verification record](results/reproducibility/verification.json) for the
actual environment and checks completed. Automated verification is recorded
separately from the roadmap's final teammate reproduction sign-off.
