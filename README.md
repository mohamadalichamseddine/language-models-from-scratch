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

The requirements pin the tested direct dependency versions, including `torch==2.14.0`. For GPU use, install a matching platform-specific PyTorch 2.14.0 build before the requirements. A local build suffix (such as `+rocm10.1.0`) satisfies this pin. Check `torch.cuda.is_available()` and device names before choosing `--device`. Reproducing the results with `scripts/run_all.py` needs no GPU.

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

## Reproduce the results (T26)

One command reruns every step after training, starting from the committed checkpoints:

```bash
python scripts/run_all.py
```

It prepares the data, runs the tests, and then runs the scripts in the table below in order. Nothing is trained, and the test split is not scored again. On a CPU it takes about two hours, most of it scoring the Transformers. Options: `--device` (default: the GPU if there is one), `--batch-size` (default 64, which fits in laptop memory) and `--skip-tests`.

Every script writes into `results/`, so run it on a clean checkout and look at `git diff` afterwards. Scores computed on different hardware can change in the last digits: the T18 models were scored on an AMD GPU, and repeated scoring differs at rounding precision (see the [context-ablation runbook](docs/context-ablation.md)).

The T21–T23 steps need the T21 test predictions (`results/t21_results.zip`, from the T21 Colab run, unzipped from the repo root). Without them `run_all` skips those three steps instead of scoring the test split again.

| Results (under `results/`) | Command |
|---|---|
| `data_stats/` | `python scripts/dataset_stats.py` |
| `ngram/n*_k*/`, `ngram/tuning.md`, `ngram/perplexity_vs_k.png` | `python scripts/run_ngram.py` |
| `ngram/interpolated/` | `python scripts/run_interpolated.py` |
| `ngram/comparison.md`, `.csv` | `python scripts/evaluate.py "results/ngram/*" --out results/ngram/comparison` |
| `transformer/best/metrics.json` | `python scripts/score_transformer.py --run transformer/best` |
| `transformer/sweep/sweep.md`, `comparison.md`, `.csv` | `python scripts/sweep_transformer.py --report-only` |
| `figures/comparison.md`, `.csv`, `perplexity.png`, `transformer_loss.png` | `python scripts/build_figures.py` |
| `figures/disagreements.csv` | `python scripts/model_disagreement.py results/transformer/best/valid.npz results/ngram/n3_k0.001/valid.npz --split valid --top 20 --context 10 --out results/figures/disagreements.csv` |
| `generation/samples.json` | `python scripts/generate.py` |
| `generation/metrics.md`, `.csv` | `python scripts/generation_metrics.py` |
| `ablation/k*_seed*/metrics.json` | `python scripts/run_ablation.py` |
| `context_truncation/ablation/k128_seed0/` (and `seed1`) | `python scripts/score_context_truncation.py --run ablation/k128_seed0` |
| `figures/ablation_*.png`, `ablation_runs.csv`, `ablation_summary.md`, `.csv` | `python scripts/plot_ablation.py` |
| `final/final.md`, `.csv` | `python scripts/final_evaluation.py` |
| `rare_tokens/` | `python scripts/rare_tokens.py` |
| `long_context/` | `python scripts/long_context.py` |
| `generation/ablation/` | `python scripts/generation_analysis.py` |

Notes on a few rows:

- `sweep_transformer.py --report-only` rebuilds the sweep table from `results/transformer/sweep/sweep.json`, because only the best of the nine sweep models is committed.
- `run_ablation.py` keeps runs whose training is complete (all eight are committed) and only scores them.
- `final_evaluation.py` scores the test split for any model whose test predictions are missing. Restore `t21_results.zip` first so it only rebuilds the table. The bootstrap settings are in `results/final/config.json`.

### Training (not part of `run_all`)

Training needs a GPU. The T13 sweep is `python scripts/sweep_transformer.py` (or [notebooks/train_on_colab.ipynb](notebooks/train_on_colab.ipynb)), and the T18 runs are described in the [context-ablation runbook](docs/context-ablation.md). Both write over the committed models, so use a separate checkout.

### Quick check from a fresh clone

After setup, this rebuilds the T24 generation table from the committed token ids in a few seconds. `git diff` should then show no changes:

```bash
python scripts/generation_analysis.py --samples results/generation/ablation/samples.json
git diff --stat
```
