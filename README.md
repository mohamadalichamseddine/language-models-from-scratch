# N-Grams vs Decoder-Only Transformer

Classical n-gram language models and a small decoder-only Transformer, both written from scratch and compared on WikiText-2. The plan and ticket status are in [docs/roadmap.md](docs/roadmap.md).

## Setup

Requires Python 3.11 or newer (Colab uses 3.13).

```bash
git clone https://github.com/mohamadalichamseddine/language-models-from-scratch.git
cd language-models-from-scratch
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/prepare_data.py   # builds data/processed/: vocabulary and token ids
pytest
```

`pip install -r requirements.txt` also installs this repo's package (`src/lm`) in editable mode, so scripts and tests can `import lm` and pick up code changes without reinstalling.

On Windows, the default `torch` from pip runs on CPU only. To use an NVIDIA GPU, first install torch with the command from [pytorch.org](https://pytorch.org/get-started/locally/), then run `pip install -r requirements.txt`.

## Where things run

- **Data processing and n-gram models:** any laptop; CPU is enough.
- **Transformer training:** Open [notebooks/train_on_colab.ipynb](https://colab.research.google.com/github/mohamadalichamseddine/language-models-from-scratch/blob/main/notebooks/train_on_colab.ipynb) in Colab: it clones the repo into Google Drive (so runs survive a disconnect), runs the tests and the T13 sweep, and downloads the results.

  The scripts also work on their own, on any machine with a GPU:

  ```bash
  python scripts/train_transformer.py --run transformer/k128 --context 128   # train, log curves, keep the best checkpoint
  python scripts/score_transformer.py --run transformer/k128                 # score valid with a sliding window
  python scripts/sweep_transformer.py                                        # T13: learning rate x dropout sweep
  ```

## Layout

```
src/lm/               shared code (data, metrics, evaluation) and the n-gram models
src/lm/transformer/   the Transformer
scripts/              entry points (download data, train, evaluate)
configs/              run configurations
notebooks/            Colab notebook for training the Transformer on a GPU
tests/                unit tests (run with pytest)
results/              metrics, tables and figures
checkpoints/          trained models that later tickets need (trained on Colab)
data/                 WikiText-2: raw text is committed, data/processed/ is generated
docs/                 roadmap and notes
```

Every script calls `lm.utils.set_seed` first, so runs can be repeated exactly.
