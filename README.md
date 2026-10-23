# N-Grams vs Decoder-Only Transformer

Classical n-gram language models and a small decoder-only Transformer, both written from scratch and compared on WikiText-2. The plan and ticket status are in [docs/roadmap.md](docs/roadmap.md).

## Setup

Requires Python 3.10 or newer.

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
- **Transformer training:** Google Colab with a GPU runtime (Runtime → Change runtime type → T4 GPU), because not every team laptop has an NVIDIA GPU. In a Colab notebook:

  ```
  !git clone https://github.com/mohamadalichamseddine/language-models-from-scratch.git
  %cd language-models-from-scratch
  !pip install -r requirements.txt
  !python scripts/prepare_data.py
  ```

## Layout

```
src/lm/     package code (data, models, metrics)
scripts/    entry points (download data, train, evaluate)
configs/    run configurations
tests/      unit tests (run with pytest)
results/    metrics, tables and figures
data/       WikiText-2: raw text is committed, data/processed/ is generated
docs/       roadmap and notes
```

Every script calls `lm.utils.set_seed` first, so runs can be repeated exactly.
